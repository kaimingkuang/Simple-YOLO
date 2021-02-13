import numpy as np

from dataset import xyxy2xywh


class NormalizeBoundingBoxes:
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

        # calculate normalized centers' coordinates within each grid cell
        bboxes_centers_norm = [norm - idx for norm, idx
            in zip(centers_norm, centers_indices)]
        
        # calculate normalized widths and heights
        bboxes_sizes_norm = [xywh[:, 2:] / cell_size for xywh, cell_size
            in zip(reg_targets_xywh, cell_sizes)]
        
        # concatenate normalized centers and sizes
        bboxes = [np.concatenate((center, size), axis=1) for center, size
            in zip(bboxes_centers_norm, bboxes_sizes_norm)]
        
        kwargs["reg_targets"] = bboxes

        return kwargs


if __name__ == "__main__":
    from dataset import _unzip_samples, VOCDataset


    ds = VOCDataset(root="/mnt/storage/kaiming/etc/", image_set="trainval")
    samples = [ds[i] for i in range(4)]
    images, cls_targets, reg_targets = _unzip_samples(samples)
    data = NormalizeBoundingBoxes((7, 7))(images=images,
        reg_targets=reg_targets)
    print([x.shape for x in data["reg_targets"]])
