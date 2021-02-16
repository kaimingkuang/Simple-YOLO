import numpy as np


EPS = 1e-8


def _intersection(bbox_0, bbox_1):
    """
    Calculate intersections between two bboxes.
    """
    inter_x_min = np.maximum(bbox_0[0], bbox_1[0])
    inter_x_max = np.minimum(bbox_0[2], bbox_1[2])
    inter_y_min = np.maximum(bbox_0[1], bbox_1[1])
    inter_y_max = np.minimum(bbox_0[3], bbox_1[3])
    inter_x_diff = np.where(inter_x_max - inter_x_min <= 0,
        0, inter_x_max - inter_x_min)
    inter_y_diff = np.where(inter_y_max - inter_y_min <= 0,
        0, inter_y_max - inter_y_min)
    intersection = inter_x_diff * inter_y_diff

    return intersection


def _union(bbox_0, bbox_1, intersection):
    """
    Calculate intersections between two bboxes.
    """
    # union(A, B) = (A + B) - intersection(A, B)
    area_0 = (bbox_0[2] - bbox_0[0])\
        * (bbox_0[3] - bbox_0[1])
    area_1 = (bbox_1[2] - bbox_1[0])\
        * (bbox_1[3] - bbox_1[1])
    unions = area_0 + area_1 - intersection

    return unions


def _iou_single(bbox_0, bbox_1):
    """
    Calculate the IoU between two bboxes.
    """
    inter = _intersection(bbox_0, bbox_1)
    union = _union(bbox_0, bbox_1, inter)
    iou = inter / (union + EPS)

    return iou


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
    ious = np.zeros((len(bboxes_0), len(bboxes_1)))
    for i in range(len(bboxes_0)):
        for j in range(len(bboxes_1)):
            ious[i, j] = _iou_single(bboxes_0[i], bboxes_1[j])

    return ious


def _calculate_ap_per_class(results, n_positive):
    """
    Calculate AP for a single class according to evaluation results.
    """
    # if results are empty, return 0 AP
    if len(results) == 0:
        return 0

    # sort the results by probability in descending order
    results_sort_indices = np.flip(np.argsort(results[:, 1], axis=0))
    results = results[results_sort_indices]

    # calculate TP, FP and FN
    tp_cnts = np.cumsum(results[:, -1])
    fp_cnts = np.cumsum(1 - results[:, -1])
    fn_cnts = n_positive - tp_cnts

    # calculate recalls and precisions
    recalls = tp_cnts / (tp_cnts + fn_cnts + EPS)
    precisions = tp_cnts / (tp_cnts + fp_cnts + EPS)

    # get interpolated precisions at specified recalls
    recall_threshes = np.linspace(0, 1, 11)
    precision_at_recalls = np.interp(recall_threshes, recalls, precisions)

    ap = np.mean(precision_at_recalls)

    return ap


def calculate_map(y_true, y_pred, num_classes, iou_thresh=0.5):
    """
    Calculate mAP.

    Parameters
    ----------
    y_true : list of numpy.ndarray
        List of GTs containing predicted class, probability and bbox.
    y_pred : list of numpy.ndarray
        List of predictions containing predicted class, probability and bbox.
    num_classes : int
        Number of classes.
    iou_thresh : float, optional
        IoU threshold for a prediction to be considered as hit.
        The default value is 0.5.

    Returns
    -------
    map_score : float
        mAP score.
    """
    iou_mats = [iou(true[:, 2:], pred[:, 2:]) for true, pred
        in zip(y_true, y_pred)]
    # results: pred_class | pred_prob | hit_or_not
    results = [np.zeros((len(y_pred[i]), 3)) for i in range(len(y_pred))]

    # get all GTs hit by each prediction
    for i in range(len(y_true)):
        if len(y_pred[i]) == 0:
            continue

        hit_indices_pair = np.argwhere(iou_mats[i] > iou_thresh)
        hit_pred_indices = np.unique(hit_indices_pair[:, 1])
        results[i][:, :2] = y_pred[i][:, :2]
        for pred_idx in hit_pred_indices:
            cur_hits = hit_indices_pair[hit_indices_pair[:, 1] == pred_idx, 0]
            pred_class = y_pred[i][pred_idx, 0]
            gt_classes = y_true[i][cur_hits, 0]
            results[i][pred_idx, -1] = 1 if pred_class in gt_classes else 0

    # calculate AP for each class
    results = np.concatenate([res for res in results if len(res) > 0])
    n_positive_per_class = [sum([len(y_true[i][y_true[i][:, 0] == cls_idx])
        for i in range(len(y_true))]) for cls_idx
        in range(num_classes)]
    ap_per_class = [_calculate_ap_per_class(results[
        results[:, 0] == cls_idx], n_positive_per_class[cls_idx])
        for cls_idx in range(num_classes)]

    # calculate mAP
    map_score = np.mean(ap_per_class)

    return map_score
