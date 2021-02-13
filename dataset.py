import numpy as np
from torchvision.datasets import VOCDetection


# VOC detection classes
CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]


def xyxy2xywh(xyxy):
    """
    Convert xyxy bouding boxes to xywh format.

    Parameters
    ----------
    xyxy : numpy.ndarray
        Bounding boxes in xyxy format.
    
    Returns
    -------
    xywh : numpy.ndarray
        Bounding boxes in xywh format.
    """
    x_min, y_min = xyxy[:, 0], xyxy[:, 1]
    x_max, y_max = xyxy[:, 2], xyxy[:, 3]
    center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
    width, height = x_max - x_min, y_max - y_min
    xywh = np.stack((center_x, center_y, width, height), axis=1)

    return xywh


def xywh2xyxy(xywh):
    """
    Convert xywh bouding boxes to xyxy format.

    Parameters
    ----------
    xywh : numpy.ndarray
        Bounding boxes in xywh format.
    
    Returns
    -------
    xyxy : numpy.ndarray
        Bounding boxes in xyxy format.
    """
    center_x, center_y = xywh[:, 0], xywh[:, 1]
    width, height = xywh[:, 2], xywh[:, 3]
    x_min, y_min = center_x - width / 2, center_y - height / 2
    x_max, y_max = center_x + width / 2, center_y + height / 2
    xyxy = np.stack((x_min, y_min, x_max, y_max), axis=1)

    return xyxy


class VOCDataset(VOCDetection):

    @staticmethod
    def _parse_annotations(annots):
        """
        Parse VOC XML annotations.
        """
        objects = annots["annotation"]["object"]
        labels = [(CLASSES.index(obj["name"]),
            [int(dim) for dim in obj["bndbox"].values()]) for obj in objects]
        cls_targets = np.array([x[0] for x in labels])
        reg_targets = np.array([x[1] for x in labels])

        return cls_targets, reg_targets

    def __getitem__(self, idx):
        image, annots = super().__getitem__(idx)
        image = np.array(image)
        cls_targets, reg_targets = self._parse_annotations(annots)

        return image, cls_targets, reg_targets


def _unzip_samples(samples):
    """
    Unzip images and targets in samples.
    """
    images = [x[0] for x in samples]
    cls_targets = [x[1] for x in samples]
    reg_targets = [x[2] for x in samples]

    return images, cls_targets, reg_targets


def _apply_transforms(data, transforms):
    """
    Apply transforms on data.
    """
    for t in transforms:
        data = t(**data)
    
    return data


def collate_factory(transforms):
    def collate_fn(samples):
        # unzip images and labels
        images, cls_targets, reg_targets = _unzip_samples(samples)

        # apply transforms
        data = {
            "images": images,
            "cls_targets": cls_targets,
            "reg_targets": reg_targets
        }
        data = _apply_transforms(data, transforms)

        return data["images"], data["cls_targets"], data["reg_targets"]

    return collate_fn
