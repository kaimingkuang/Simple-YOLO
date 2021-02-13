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
    "people",
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
    xyxy : sequence of float
        Bounding boxes in xyxy format.
    
    Returns
    -------
    xywh : sequence of float
        Bounding boxes in xywh format.
    """
    x_min, y_min, x_max, y_max = xyxy
    center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
    width, height = x_max - x_min, y_max - y_min
    xywh = (center_x, center_y, width, height)

    return xywh


def xywh2xyxy(xywh):
    """
    Convert xywh bouding boxes to xyxy format.

    Parameters
    ----------
    xywh : sequence of float
        Bounding boxes in xywh format.
    
    Returns
    -------
    xyxy : sequence of float
        Bounding boxes in xyxy format.
    """
    center_x, center_y, width, height = xywh
    x_min, y_min = center_x - width / 2, center_y - height / 2
    x_max, y_max = center_x + width / 2, center_y + height / 2
    xyxy = (x_min, y_min, x_max, y_max)

    return xyxy


class VOCDataset(VOCDetection):

    @staticmethod
    def _parse_annotations(annots):
        """
        Parse VOC XML annotations.
        """
        objects = annots["annotation"]["object"]
        labels = [(obj["name"], [int(dim) for dim in obj["bndbox"].values()])
            for obj in objects]

        return labels

    def __getitem__(self, idx):
        image, annots = super().__getitem__(idx)
        image = np.array(image)
        labels = self._parse_annotations(annots)

        return image, labels
