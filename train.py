import argparse
import os
import warnings

import numpy as np
import torch
from tqdm import tqdm

import config as cfg
import transforms as aug
from dataset import VOCDataset, get_dataloader


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


def _setup_dataloaders(root_dir):
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
        True, 4)
    ds_val = VOCDataset(root_dir, image_set="val")
    dl_val = get_dataloader(ds_val, transforms_val, cfg.batch_size)

    return dl_train, dl_val


def _train_epoch(model, dl_train):
    """
    Train the model for an epoch.
    """
    n_digits = len(str(cfg.epochs))
    progress = tqdm(total=len(dl_train), ncols=150)

    for _, sample in enumerate(dl_train):
        images, cls_targets, reg_targets = sample
        


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

    for i in range(cfg.epochs):
        progress = tqdm(total=len(dl_train), ncols=150)
        progress.set_description


if __name__ == "__main__":
    main()
