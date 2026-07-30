import numpy as np
from typing import List


def precision_score(y_true: List[int], y_pred: List[int]):
    """
    Compute the precision score operating on two lists of offsets (e.g., character).
    >>> assert precision_score([[0, 1, 4, 5]], [[0, 1, 6]]) == 0.6666666666666666

    Args:
    - y_true: a list of offsets serving as the ground truth
    - y_pred: a list of predicted offsets

    Returns:
    - precision: the precision score
    """
    scores = []
    for pred, true in zip(y_pred, y_true):
        if len(true) == 0 and len(pred) == 0:
            scores.append(1)
        elif len(true) == 0 and len(pred) != 0:
            scores.append(0)
        elif len(true) != 0 and len(pred) == 0:
            scores.append(0)
        else:
            pred_set = set(pred)
            true_set = set(true)
            scores.append(len(pred_set.intersection(true_set)) / len(pred_set))

    return np.mean(scores)


def recall_score(y_true: List[int], y_pred: List[int]):
    """
    Compute the recall score operating on two lists of offsets (e.g., character).
    >>> assert recall_score([[0, 1, 4, 5]], [[0, 1, 6]]) == 0.5

    Args:
    - y_true: a list of offsets serving as the ground truth
    - y_pred: a list of predicted offsets

    Returns:
    - recall: the recall score
    """
    scores = []
    for pred, true in zip(y_pred, y_true):
        if len(true) == 0 and len(pred) == 0:
            scores.append(1)
        elif len(true) == 0 and len(pred) != 0:
            scores.append(0)
        elif len(true) != 0 and len(pred) == 0:
            scores.append(0)
        else:
            pred_set = set(pred)
            true_set = set(true)
            scores.append(len(pred_set.intersection(true_set)) / len(true_set))
    return np.mean(scores)


def f1_score(y_true: List[int], y_pred: List[int]):
    """
    Compute the F1 score operating on two lists of offsets (e.g., character).
    >>> assert f1_score([[0, 1, 4, 5]], [[0, 1, 6]]) == 0.5714285714285715

    Args:
    - y_true: a list of offsets serving as the ground truth
    - y_pred: a list of predicted offsets

    Returns:
    - f1: the F1 score
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
