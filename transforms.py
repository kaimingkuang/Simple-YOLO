import cv2
import numpy as np
import torch

from dataset import xyxy2xywh


NINF = -9e6


class NormalizeBboxes:
    """
    Normalize bounding boxes.

    Parameters
    ----------
    grid_size : tuple of int
        Size of the grid in the last feature map.
    """

    def __init__(self, grid_size):
        self.grid_size = np.array(grid_size)
    
    def __call__(self, **kwargs):
        reg_targets_xyxy = kwargs["reg_targets"]
        image_shapes = [np.array(image.shape[-2::-1])
            for image in kwargs["images"]]

        # calculate the size of each cell in the grid
        cell_sizes = [shape / self.grid_size for shape in image_shapes]

        # convert the regression targets to xywh format
        reg_targets_xywh = [xyxy2xywh(xyxy) for xyxy in reg_targets_xyxy]

        # calculate regression targets' centers
        reg_targets_centers = [np.array(xywh[:, :2])
            for xywh in reg_targets_xywh]

        # normalize centers' coordinates
        centers_norm = [center / cell_size for center, cell_size
            in zip(reg_targets_centers, cell_sizes)]

        # calculate centers indices in the grid
        centers_indices = [np.floor(center).astype(int)
            for center in centers_norm]

        # calculate normalized centers and sizes
        bboxes_centers_norm = [norm - idx for norm, idx
            in zip(centers_norm, centers_indices)]
        bboxes_sizes_norm = [xywh[:, 2:] / cell_size for xywh, cell_size
            in zip(reg_targets_xywh, cell_sizes)]

        # concatenate normalized centers and sizes
        bboxes = [np.concatenate((center, size), axis=1) for center, size
            in zip(bboxes_centers_norm, bboxes_sizes_norm)]

        kwargs["reg_targets"] = bboxes
        kwargs["obj_indices"] = centers_indices

        return kwargs


class Bboxes2Matrices:
    """
    Convert bounding boxes to matrices.

    Parameters
    ----------
    grid_size : tuple of int
        Size of the grid in the last feature map.
    """

    def __init__(self, grid_size, num_classes):
        self.grid_size = grid_size
        self.num_classes = num_classes

    def __call__(self, **kwargs):
        cls_targets = kwargs["cls_targets"]
        reg_targets = kwargs["reg_targets"]
        b = len(cls_targets)

        # construct empty target matrices
        cls_mat = np.ones((b, ) + self.grid_size, dtype=np.int16) * -1
        reg_mat = np.ones((b, 4) + self.grid_size) * NINF

        # fill each object in matrices
        for i in range(b):
            obj_indices = kwargs["obj_indices"][i].T.tolist()
            cls_mat[i, obj_indices[1], obj_indices[0]] = cls_targets[i]
            reg_mat[i, :, obj_indices[1], obj_indices[0]] = reg_targets[i]

        cls_mat = cls_mat + 1
        kwargs["cls_targets"] = cls_mat
        kwargs["reg_targets"] = reg_mat

        return kwargs


class Resize:
    """
    Resize input images.

    Parameters
    ----------
    target_szie : tuple of int
        The image target size.
    """

    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, **kwargs):
        kwargs["images"] = [cv2.resize(image, self.target_size) for image
            in kwargs["images"]]

        return kwargs


class Normalize:
    """
    Normalize image pixel values.

    Parameters
    ----------
    mean : tuple of float
        Image pixel means.
    std : tuple of float
        Image pixel standard deviations.
    scale : float
        Image pixel scale.
    """

    def __init__(self, mean, std, scale):
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.scale = scale

    def __call__(self, **kwargs):
        kwargs["images"] = [(image * self.scale - self.mean) / self.std
            for image in kwargs["images"]]

        return kwargs


class ToTensor:
    """
    Convert inputs to tensors.
    """

    def __call__(self, **kwargs):
        kwargs["images"] = np.stack(kwargs["images"]).transpose(0, 3, 1, 2)
        kwargs["cls_targets"] = np.stack(kwargs["cls_targets"])
        kwargs["reg_targets"] = np.stack(kwargs["reg_targets"])

        kwargs["images"] = torch.from_numpy(kwargs["images"]).float()
        kwargs["cls_targets"] = torch.from_numpy(kwargs["cls_targets"]).long()
        kwargs["reg_targets"] = torch.from_numpy(kwargs["reg_targets"])\
            .float()

        return kwargs


if __name__ == "__main__":
    from dataset import VOCDataset, get_dataloader


    ds_train = VOCDataset(root="/mnt/storage/kaiming/etc/voc2012",
        image_set="train")
    ds_val = VOCDataset(root="/mnt/storage/kaiming/etc/voc2012",
        image_set="val")
    grid_size = (7, 7)
    num_classes = 21
    target_size = (224, 224)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transforms = [
        NormalizeBboxes(grid_size),
        Bboxes2Matrices(grid_size, num_classes),
        Resize(target_size),
        Normalize(mean, std, 1. / 255),
        ToTensor()
    ]
    dl = get_dataloader(ds_val, transforms, 4, False)
    for i, sample in enumerate(dl):
        images, cls_targets, reg_targets = sample
        print(images.size())
        print(cls_targets.size())
        print(reg_targets.size())
        break
