import numpy as np


EPS = 1e-8


def _intersection(bboxes_0, bboxes_1):
    """
    Calculate intersections between two lists of bboxes.
    """
    inter_x_min = np.maximum(bboxes_0[:, 0], bboxes_1[:, 0])
    inter_x_max = np.minimum(bboxes_0[:, 2], bboxes_1[:, 2])
    inter_y_min = np.maximum(bboxes_0[:, 1], bboxes_1[:, 1])
    inter_y_max = np.minimum(bboxes_0[:, 3], bboxes_1[:, 3])
    inter_x_diff = np.where(inter_x_max - inter_x_min <= 0,
        0, inter_x_max - inter_x_min)
    inter_y_diff = np.where(inter_y_max - inter_y_min <= 0,
        0, inter_y_max - inter_y_min)
    intersections = inter_x_diff * inter_y_diff

    return intersections


def _union(bboxes_0, bboxes_1, intersections):
    """
    Calculate intersections between two lists of bboxes.
    """
    # union(A, B) = (A + B) - intersection(A, B)
    areas_0 = (bboxes_0[:, 2] - bboxes_0[:, 0])\
        * (bboxes_0[:, 3] - bboxes_0[:, 1])
    areas_1 = (bboxes_1[:, 2] - bboxes_1[:, 0])\
        * (bboxes_1[:, 3] - bboxes_1[:, 1])
    unions = areas_0 + areas_1 - intersections

    return unions


def iou(bboxes_0, bboxes_1):
    """
    Calculate IoUs between two lists of bboxes.

    Parameters
    ----------
    bboxes_0, bboxes_1 : numpy.ndarray
        Bounding boxes in shape of N x 4.
    
    Returns
    -------
    ious : numpy.ndarray
        IoU scores between two lists of bboxes.
    """
    inters = _intersection(bboxes_0, bboxes_1)
    unions = _union(bboxes_0, bboxes_1, inters)
    ious = inters / (unions + EPS)

    return ious
