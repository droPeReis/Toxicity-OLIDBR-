import os
import json
import boto3
import torch
import numpy as np
from typing import List, Union
from collections.abc import MutableMapping


def flatten_dict(
    obj: MutableMapping, parent_key: str = "", sep: str = "_"
) -> MutableMapping:
    """Flatten a nested dictionary.

    Args:
    - obj (MutableMapping): The dictionary to be flattened.
    - parent_key (str, optional): The parent key. Defaults to ''.
    - sep (str, optional): The separator. Defaults to '_'.

    Returns:
    MutableMapping: The flattened dictionary.
    """
    items = []
    for k, v in obj.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def compute_pos_weight(
    y: Union[np.ndarray, List[List[int]], torch.Tensor]
) -> List[float]:
    """Compute positive weight for class imbalance.

    Args:
    - y: The labels.

    Returns:
    - pos_weight: array-like of shape (n_classes,)
    """
    if isinstance(y, list):
        y = np.array(y)
    elif isinstance(y, torch.Tensor):
        y = y.numpy()

    pos_weight = []
    for i in range(len(y[0])):
        positives = y[:, i].sum()
        negatives = len(y[:, i]) - positives
        pos_weight.append(negatives / positives)
    return pos_weight


def remaining_args_to_env(args: List[str]):
    """Convert the remaining arguments to environment variables.
    Workaround for https://github.com/boto/boto3/issues/3488

    Args:
    - args: The arguments.
    """
    if len(args) > 0:
        for arg in range(len(args)):
            if args[arg].startswith("--") and args[arg] != "--MLFLOW_TAGS":
                os.environ[args[arg].strip("--").upper()] = args[arg + 1]


def remove_checkpoints(
    bucket_name: str, checkpoint_prefix: str, aws_profile_name: str = "default"
):
    """
    Remove all checkpoints from the specified S3 bucket and prefix.

    Args:
    - bucket_name: The name of the S3 bucket.
    - checkpoint_prefix: The prefix of the checkpoints to remove.
    - aws_profile_name: The name of the AWS profile to use.
    """
    session = boto3.Session(profile_name=aws_profile_name)
    s3 = session.resource("s3")
    bucket = s3.Bucket(bucket_name)
    response = bucket.objects.filter(Prefix=checkpoint_prefix).delete()

    if len(response) == 0:
        print("No checkpoints found.")
    elif response[0]["ResponseMetadata"]["HTTPStatusCode"] == 200:
        count = len(response[0]["Deleted"])
        print(
            f"Deleted {count} checkpoints from {bucket_name}/{checkpoint_prefix}."
        )
