import os
from src.ml.utils import (
    compute_pos_weight,
    remaining_args_to_env,
    flatten_dict,
)


def test_compute_pos_weight():
    y = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    pos_weight = compute_pos_weight(y)
    assert pos_weight == [2.0, 2.0, 2.0]


def test_remaining_args_to_env():
    args = ["--aws_profile_name", "default"]
    remaining_args_to_env(args)
    assert os.environ["AWS_PROFILE_NAME"] == "default"


def test_flatten_dict():
    d = {"a": {"b": 1, "c": 2}, "d": 3}
    flattened = flatten_dict(d)
    assert flattened == {"a_b": 1, "a_c": 2, "d": 3}
