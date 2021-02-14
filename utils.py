import numpy as np
import torch

from .metrics import iou


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


def _output2bbox(cls_output, reg_output, image_size):
    """
    Convert raw output to bounding boxes and probabilities.
    """
    cls_max_indices = np.argmax(cls_output, axis=1)
    cls_max_probs = np.max(cls_output, axis=1)
    b = cls_output.shape[0]
    cell_size = (np.array(image_size[::-1]) / cls_output.shape[2:])\
        [np.newaxis, :]

    bboxes = []

    for i in range(b):
        # filter the positive predictions
        cls_pos_indices = np.where(cls_max_indices[i] > 0)
        reg_output_pos = reg_output[i, :, cls_pos_indices[0],
            cls_pos_indices[1]]

        # denormalize xy
        reg_output_pos[:, 0] += cls_pos_indices[1] + 0.5
        reg_output_pos[:, 1] += cls_pos_indices[0] + 0.5
        reg_output_pos[:, :2] *= cell_size

        # denormalize wh
        reg_output_pos[:, 2:] *= cell_size

        # get classes and probabilities of bboxes
        cls_indices_pos = cls_max_indices[i, cls_pos_indices[0],
            cls_pos_indices[1]][:, np.newaxis]
        cls_output_pos = cls_max_probs[i, cls_pos_indices[0],
            cls_pos_indices[1]][:, np.newaxis]

        # append new bboxes
        bboxes.append(np.concatenate((cls_output_pos, cls_indices_pos,
            reg_output_pos), axis=1))

    return bboxes


def _remove_overlaps(bboxes, iou_thresh):
    """
    Remove overlapping bboxes.
    """
    bboxes_xyxy = xywh2xyxy(bboxes[:, 2:])
    probs = bboxes[:, 1]
    iou_mat = iou(bboxes_xyxy, bboxes_xyxy)

    n_boxes = bboxes.shape[0]
    removed_indices = []
    for i in range(n_boxes):
        if i not in removed_indices:
            overlap_indices = np.where(iou_mat[i, :] > iou_thresh)[0]
            overlap_probs = probs[overlap_indices]
            max_prob_idx = np.argmax(overlap_probs).tolist()
            removed_indices += [idx for idx in overlap_indices
                if idx != max_prob_idx]

    kept_indices = [idx for idx in range(n_boxes)
        if idx not in removed_indices]
    bboxes = bboxes[kept_indices, :]

    return bboxes


def _nms(bboxes, prob_thresh, iou_thresh):
    """
    Apply non-maximum supression.
    """
    b = len(bboxes)

    for i in range(b):
        # filter out lower probability predictions
        bboxes[i] = bboxes[i][bboxes[i][:, 1] >= prob_thresh]

        # remove overlaps according to IoU between boxes
        bboxes[i] = _remove_overlaps(bboxes[i], iou_thresh)

    return bboxes


def calculate_detect_result(cls_output, reg_output, cls_target, reg_target,
        image_size, prob_thresh, iou_thresh):
    """
    Calculate the detection result of a single batch.
    """
    # convert torch.Tensor to np.array
    cls_output = torch.softmax(cls_output, dim=1)
    cls_output = cls_output.cpu().numpy()
    reg_output = reg_output.cpu().numpy()
    cls_target = cls_target.cpu().numpy()
    reg_target = reg_target.cpu().numpy()

    # convert output to bboxes and probabilities
    bboxes = _output2bbox(cls_output, reg_output, image_size)

    # NMS
    bboxes = _nms(bboxes, prob_thresh, iou_thresh)

    # calculate the mAP

    pass