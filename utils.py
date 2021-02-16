import numpy as np
import torch
from torch.nn.functional import one_hot

from metrics import iou


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


def _output2prediction(cls_output, reg_output, prob_thresh, image_size):
    """
    Convert raw output to bounding boxes and probabilities.
    """
    b = cls_output.shape[0]
    cell_size = (np.array(image_size[::-1]) / cls_output.shape[2:])\
        [np.newaxis, :]

    bboxes = []

    for i in range(b):
        # filter the positive predictions
        cls_pos_indices = np.where(cls_output[i, 0] > prob_thresh)
        reg_output_pos = reg_output[i, :, cls_pos_indices[0],
            cls_pos_indices[1]]

        # denormalize xy
        reg_output_pos[:, 0] += cls_pos_indices[1] + 0.5
        reg_output_pos[:, 1] += cls_pos_indices[0] + 0.5
        reg_output_pos[:, :2] *= cell_size

        # denormalize wh
        reg_output_pos[:, 2:] *= cell_size

        # convert xywh to xyxy
        reg_output_pos = xywh2xyxy(reg_output_pos)

        # get classes and probabilities of bboxes
        cls_indices_pos = np.argmax(cls_output[i, 1:, cls_pos_indices[0],
            cls_pos_indices[1]], axis=1)[:, np.newaxis]
        cls_prob_pos = cls_output[i, 0, cls_pos_indices[0],
            cls_pos_indices[1]][:, np.newaxis]

        # append new bboxes
        bboxes.append(np.concatenate((cls_indices_pos, cls_prob_pos,
            reg_output_pos), axis=1))

    return bboxes


def _remove_overlaps(bboxes, iou_thresh):
    """
    Remove overlapping bboxes.
    """
    n_bboxes = bboxes.shape[0]
    if n_bboxes == 1:
        return bboxes

    bboxes_xyxy = xywh2xyxy(bboxes[:, 2:])
    probs = bboxes[:, 1]
    iou_mat = iou(bboxes_xyxy, bboxes_xyxy)

    kept_indices = []
    all_indices = np.argsort(probs).tolist()

    while len(all_indices) > 0:
        # push the index with the largest prob into the stack
        cur_idx = all_indices.pop(0)
        kept_indices.append(cur_idx)

        # calculate overlapping indices
        overlap_indices = np.where(iou_mat[cur_idx, :] > iou_thresh)\
            [0].tolist()
        overlap_indices = [idx for idx in overlap_indices
            if idx != cur_idx]
        overlap_indices = set(overlap_indices).intersection(set(all_indices))

        # remove overlapping indices
        for idx in overlap_indices:
            all_indices.remove(idx)

    bboxes = bboxes[kept_indices, :]

    return bboxes


def _nms(predictions, prob_thresh, iou_thresh):
    """
    Apply non-maximum supression.
    """
    b = len(predictions)

    for i in range(b):
        # filter out lower probability predictions
        predictions[i] = predictions[i][predictions[i][:, 1] >= prob_thresh]

        # remove overlaps according to IoU between boxes
        predictions[i] = _remove_overlaps(predictions[i], iou_thresh)

    return predictions


def _onehot_encode(labels, num_classes):
    """
    Apply one-hot encoding.
    """
    b, h, w = labels.size()
    labels = labels.view(b, h, w, 1)
    onehot_labels = torch.zeros(b, h, w, num_classes, dtype=torch.float,
        device=labels.device)
    onehot_labels.scatter_(-1, labels, 1)
    onehot_labels = onehot_labels.permute(0, 3, 1, 2)

    return onehot_labels


def calculate_detect_prediction(cls_output, reg_output, image_size,
        prob_thresh, iou_thresh, num_classes=None):
    """
    Calculate the detection prediction of a single batch.

    Parameters
    ----------
    cls_output : torch.Tensor
        Classification output.
    reg_output : torch.Tensor
        Regression output.
    imaeg_size : tuple of int
        Input image size in format of H x W.
    prob_thresh : float
        Probability threshold for positive predictions.
    iou_thresh : float
        IoU threshold in NMS.
    num_classes : int
        Number of classes.

    Returns
    -------
    predictions : list of numpy.ndarray
        List of predictions containing predicted class, probability and bbox.
    """
    # convert torch.Tensor to np.array
    if cls_output.size(1) == 2:
        gt_flag = True
        cls_output = torch.cat((cls_output[:, 0:1],
            _onehot_encode(cls_output[:, 1], num_classes)), dim=1)
    else:
        gt_flag = False
        cls_output[:, 0] = torch.sigmoid(cls_output[:, 0])
        cls_output[:, 1:] = torch.softmax(cls_output[:, 1:], dim=1)
    cls_output = cls_output.cpu().numpy()
    reg_output = reg_output.cpu().numpy()

    # convert output to bboxes and probabilities
    predictions = _output2prediction(cls_output, reg_output, prob_thresh,
        image_size)

    # NMS
    if not gt_flag:
        predictions = _nms(predictions, prob_thresh, iou_thresh)

    return predictions
