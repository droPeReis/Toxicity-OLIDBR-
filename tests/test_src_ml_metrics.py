import numpy as np
from transformers import EvalPrediction
from src.ml.metrics.utils import compute_metrics
from src.ml.metrics.spans import precision_score, recall_score, f1_score


def test_compute_metrics():
    preds = np.array(
        [
            [
                -0.1586,
                0.2220,
                -0.0367,
                -0.2293,
                0.3756,
                -0.3352,
                -0.3230,
                -0.0227,
                -0.0658,
                0.1569,
            ],
            [
                -0.1576,
                0.1573,
                -0.0312,
                -0.2690,
                0.3792,
                -0.3307,
                -0.3826,
                0.0392,
                -0.1152,
                0.1577,
            ],
        ]
    )

    labels = np.array(
        [[0, 1, 0, 0, 1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 0, 0, 1, 0, 1]]
    )

    metrics = compute_metrics(
        EvalPrediction(predictions=preds, label_ids=labels),
        problem_type="multi-label",
    )
    assert metrics == {
        "accuracy": 1.0,
        "f1": 1.0,
        "precision": 1.0,
        "recall": 1.0,
    }


def test_precision_score():
    assert precision_score([[0, 1, 4, 5]], [[0, 1, 6]]) == 0.6666666666666666


def test_recall_score():
    assert recall_score([[0, 1, 4, 5]], [[0, 1, 6]]) == 0.5


def test_f1_score():
    assert f1_score([[0, 1, 4, 5]], [[0, 1, 6]]) == 0.5714285714285715
