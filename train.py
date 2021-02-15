import argparse
import os
import warnings

import numpy as np
import torch
from torch import optim
from torchvision.models import resnext50_32x4d
from tqdm import tqdm

import config as cfg
import transforms as aug
from dataset import VOCDataset, get_dataloader
from losses import DetectLoss
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
        False)
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
    desc_str = f"Epoch {'0' * (sup_digits) + str(epoch_idx)}/{cfg.epochs}"

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

    loss_train /= len_train
    cls_loss_train /= len_train
    reg_loss_train /= len_train

    progress.close()

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
        for _, sample in enumerate(dl_val):
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

            y_true += calculate_detect_prediction(cls_target, reg_target,
                cfg.target_size, cfg.prob_thresh, cfg.overlap_iou_thresh)
            y_pred += calculate_detect_prediction(cls_output, reg_output,
                cfg.target_size, cfg.prob_thresh, cfg.overlap_iou_thresh)

    loss_val /= len_val
    cls_loss_val /= len_val
    reg_loss_val /= len_val


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
    log_dir = os.path.join(root_dir, args.log_dir)

    # get dataloaders
    dl_train, dl_val = _setup_dataloaders(root_dir)

    # model, criterion, optimizer and scheduler setup
    model = YOLOResNeXt(resnext50_32x4d, cfg.num_classes).cuda()
    criterion = DetectLoss(cfg.w_cls, cfg.w_reg, cfg.w_pos, cfg.w_neg)
    optimizer = optim.AdamW(model.parameters())
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, cfg.max_lr,
        total_steps=cfg.epochs * len(dl_train), pct_start=cfg.pct_start,
        final_div_factor=cfg.final_div_factor)

    for i in range(cfg.epochs):
        _train_epoch(model, dl_train, criterion, optimizer, scheduler, i)


if __name__ == "__main__":
    main()
