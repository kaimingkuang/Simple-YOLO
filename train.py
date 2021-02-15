import argparse
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnext50_32x4d
from tqdm import tqdm

import config as cfg
import transforms as aug
from dataset import VOCDataset, get_dataloader
from losses import DetectLoss
from metrics import calculate_map
from model import YOLOResNeXt
from utils import calculate_detect_prediction


warnings.filterwarnings("ignore")


def _parse_cmd_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", required=True,
        help="Root directory of data and logs.")
    parser.add_argument("--log_dir", default="logs",
        help="Sub-directory of tensorboard logs.")
    parser.add_argument("--gpu", default="0", help="GPU device ID.")

    args = parser.parse_args()

    return args


def _setup_dataloaders(root_dir, return_dataset=False):
    """
    Setup dataloaders.
    """
    preprocessing = [
        aug.NormalizeBboxes(cfg.grid_size),
        aug.Bboxes2Matrices(cfg.grid_size, cfg.num_classes),
        aug.Resize(cfg.target_size),
        aug.Normalize(cfg.mean, cfg.std, 1. / 255),
        aug.ToTensor()
    ]
    transforms_train = preprocessing
    transforms_val = preprocessing

    ds_train = VOCDataset(root_dir, image_set="train")
    dl_train = get_dataloader(ds_train, transforms_train, cfg.batch_size,
        num_workers=4)
    ds_val = VOCDataset(root_dir, image_set="val")
    dl_val = get_dataloader(ds_val, transforms_val, cfg.batch_size)

    if return_dataset:
        return dl_train, dl_val, ds_train, ds_val

    return dl_train, dl_val


def _get_description_str(epoch_idx):
    """
    Get progress bar description string.
    """
    total_digits = len(str(cfg.epochs))
    cur_digits = len(str(epoch_idx))
    sup_digits = total_digits - cur_digits
    desc_str = f"Epoch {'0' * (sup_digits) + str(epoch_idx + 1)}/{cfg.epochs}"

    return desc_str


def _train_epoch(model, dl_train, criterion, optimizer, scheduler, epoch_idx):
    """
    Train the model for an epoch.
    """
    model.train()

    progress = tqdm(total=len(dl_train), ncols=150)
    progress.set_description(_get_description_str(epoch_idx))

    loss_train = 0
    cls_loss_train = 0
    reg_loss_train = 0
    len_train = 0

    for _, sample in enumerate(dl_train):
        optimizer.zero_grad()

        images, cls_target, reg_target = sample
        images = images.cuda()
        cls_target = cls_target.cuda()
        reg_target = reg_target.cuda()

        cls_output, reg_output = model(images)
        total_loss, cls_loss, reg_loss = criterion(cls_output, reg_output,
            cls_target, reg_target)

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_train += total_loss.detach().cpu().item() * images.size(0)
        cls_loss_train += cls_loss.detach().cpu().item() * images.size(0)
        reg_loss_train += reg_loss.detach().cpu().item() * images.size(0)
        len_train += images.size(0)

        progress.set_postfix_str(f"loss={total_loss.cpu().item():.4f}, "\
            f"cls_loss={cls_loss.cpu().item():.4f}, "\
            f"reg_loss={reg_loss.cpu().item():.4f}")
        progress.update()

    progress.close()

    loss_train /= len_train
    cls_loss_train /= len_train
    reg_loss_train /= len_train

    return loss_train, cls_loss_train, reg_loss_train


def _eval_epoch(model, dl_val, criterion):
    """
    Evaluate the model at the end of an epoch.
    """
    model.eval()

    loss_val = 0
    cls_loss_val = 0
    reg_loss_val = 0
    len_val = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for i, sample in enumerate(dl_val):
            sys.stdout.write("\r")

            images, cls_target, reg_target = sample
            images = images.cuda()
            cls_target = cls_target.cuda()
            reg_target = reg_target.cuda()

            cls_output, reg_output = model(images)
            total_loss, cls_loss, reg_loss = criterion(cls_output, reg_output,
                cls_target, reg_target)

            loss_val += total_loss.cpu().item() * images.size(0)
            cls_loss_val += cls_loss.cpu().item() * images.size(0)
            reg_loss_val += reg_loss.cpu().item() * images.size(0)
            len_val += images.size(0)

            y_true += calculate_detect_prediction(cls_target, reg_target,
                cfg.target_size, cfg.prob_thresh, cfg.overlap_iou_thresh,
                cfg.num_classes)
            y_pred += calculate_detect_prediction(cls_output, reg_output,
                cfg.target_size, cfg.prob_thresh, cfg.overlap_iou_thresh)

            sys.stdout.write(f"Validation: {i + 1}/{len(dl_val)}")

    loss_val /= len_val
    cls_loss_val /= len_val
    reg_loss_val /= len_val

    map_score = calculate_map(y_true, y_pred, cfg.num_classes - 1)

    sys.stdout.flush()

    return loss_val, cls_loss_val, reg_loss_val, map_score


def _print_eval_results(loss_train, cls_loss_train, reg_loss_train, loss_val,
        cls_loss_val, reg_loss_val, map_score):
    """
    Print training and evaluation results.
    """
    sys.stdout.write("\r")
    sys.stdout.write(f"loss={loss_train:.4f}/{loss_val:.4f} | ")
    sys.stdout.write(f"cls_loss={cls_loss_train:.4f}/{cls_loss_val:.4f} | ")
    sys.stdout.write(f"reg_loss={reg_loss_train:.4f}/{reg_loss_val:.4f} | ")
    sys.stdout.write(f"mAP={map_score:.4f}\n")


def _save_model_weights(model_weights, model_weights_dir, epoch_idx):
    """
    Save model weights under the log directory.
    """
    model_weights_path = os.path.join(model_weights_dir,
        f"model_epoch_{epoch_idx}.pth")
    torch.save(model_weights, model_weights_path)


def _update_tensorboard(tb_writer, loss_train, cls_loss_train, reg_loss_train,
        loss_val, cls_loss_val, reg_loss_val, map_score, epoch_idx):
    """
    Update tensorboard logs.
    """
    tb_writer.add_scalars("loss", {"train": loss_train, "val": loss_val},
        epoch_idx)
    tb_writer.add_scalars("cls_loss", {"train": cls_loss_train,
        "val": cls_loss_val}, epoch_idx)    
    tb_writer.add_scalars("reg_loss", {"train": reg_loss_train,
        "val": reg_loss_val}, epoch_idx)   
    tb_writer.add_scalar("mAP", map_score, epoch_idx)
    tb_writer.flush()


def main():
    # parse command line arguments
    args = _parse_cmd_args()

    # set visible GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # set data directory and log directory
    root_dir = args.root
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(root_dir, args.log_dir, cur_time)

    # set up tensorboard_writer
    tb_writer = SummaryWriter(log_dir)

    # set up model weights directory
    model_weights_dir = os.path.join(log_dir, "model_weights")
    if not os.path.exists(model_weights_dir):
        os.mkdir(model_weights_dir)

    # get dataloaders
    dl_train, dl_val = _setup_dataloaders(root_dir)

    # model, criterion, optimizer and scheduler setup
    model = YOLOResNeXt(resnext50_32x4d, cfg.num_classes).cuda()
    criterion = DetectLoss(cfg.w_cls, cfg.w_reg, cfg.w_pos, cfg.w_neg)
    optimizer = optim.AdamW(model.parameters())
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, cfg.max_lr,
        total_steps=cfg.epochs * len(dl_train), pct_start=cfg.pct_start,
        final_div_factor=cfg.final_div_factor)

    # training loop
    for i in range(cfg.epochs):
        loss_train, cls_loss_train, reg_loss_train = _train_epoch(model,
            dl_train, criterion, optimizer, scheduler, i)
        loss_val, cls_loss_val, reg_loss_val, map_score = _eval_epoch(model,
            dl_val, criterion)
        _print_eval_results(loss_train, cls_loss_train, reg_loss_train,
            loss_val, cls_loss_val, reg_loss_val, map_score)
        
        _update_tensorboard(tb_writer, loss_train, cls_loss_train,
            reg_loss_train, loss_val, cls_loss_val, reg_loss_val,
            map_score, i)
        _save_model_weights(model.state_dict(), model_weights_dir, i)


if __name__ == "__main__":
    main()
