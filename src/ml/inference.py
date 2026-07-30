import torch
import numpy as np
from typing import Union
from transformers import EvalPrediction
from transformers.trainer_utils import PredictionOutput


def predict(
    predictions: Union[EvalPrediction, PredictionOutput, np.ndarray],
    return_proba: bool = False,
    threshold: float = 0.5,
    problem_type: str = None,
):
    """Predict the labels of a batch of samples.

    Args:
    - predictions: The predictions of the model.
    - return_proba: Whether to return the probability of each label.
    - threshold: The threshold to be used to convert the logits to labels.
    - problem_type: The type of the problem. Can be "binary", "multi-class" or "multi-label".

    Returns:
    - The predicted labels.
    """
    if problem_type not in [None, "binary", "multi-class", "multi-label"]:
        raise ValueError(
            f"Invalid problem type: {problem_type}. "
            "It must be one of 'binary', 'multi-class' or 'multi-label'."
        )

    if isinstance(predictions, (EvalPrediction, PredictionOutput)):
        if isinstance(predictions.predictions, tuple):
            predictions = predictions.predictions[0]
        else:
            predictions = predictions.predictions

    if problem_type == "multi-label":
        act_fn = torch.nn.Sigmoid()
    else:  # binary or multi-class
        act_fn = torch.nn.Softmax(dim=1)

    probs: np.ndarray = act_fn(torch.Tensor(predictions))

    if return_proba:
        return probs
    else:
        if problem_type == "multi-label":
            return np.array(probs >= threshold, dtype=int)
        else:
            if predictions.shape[1] == 2:
                return np.array(probs[:, 1] > threshold, dtype=int)
            else:
                return np.argmax(probs, axis=1)
