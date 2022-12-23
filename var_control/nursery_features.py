import torch
from torchtyping import TensorType  # type: ignore
from torchtyping import patch_typeguard
from typeguard import typechecked
import pickle
from os import listdir
from os.path import isfile, join
from os.path import exists
import numpy as np
from torch import Tensor

batch = None
classes = None
size = None

patch_typeguard()


@typechecked
def load_features(
        file_name: str,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    with open(file_name, 'rb') as handle:
        outputs = pickle.load(handle)
        status = torch.tensor(outputs['status'])
        z_pos = torch.tensor(outputs['z'][status == 1])[:2000]
        y_pos = torch.tensor(outputs['y'][status == 1])[:2000]
        z_neg = torch.tensor(outputs['z'][status == 0])[:2000]
        y_neg = torch.tensor(outputs['y'][status == 0])[:2000]

    return z_pos, y_pos, z_neg, y_neg


@typechecked
def get_label_dict(
    file_name="data/nursery/data.pkl"
) -> dict():
    with open(file_name, 'rb') as handle:
        outputs = pickle.load(handle)
    return outputs['label_dict']


@typechecked
def labels_to_one_hot(
        labels: TensorType["batch", int],
        n_classes: int
) -> TensorType["batch", "classes", int]:
    one_hot_mat = torch.zeros((labels.shape[0], n_classes), dtype=int)
    one_hot_mat[torch.arange(one_hot_mat.shape[0]), labels.tolist()] = 1
    return one_hot_mat


@typechecked
def load_features_and_ground_truth(
        feature_key: str
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    match feature_key:
        case "logreg":
            file_name = "data/nursery/data.pkl"
            z_pos, y_pos, z_neg, y_neg = load_features(file_name)
        case _:
            raise ValueError(f"unknown feature_key: {feature_key}")
    n_classes = z_pos.shape[1]
    y_pos = labels_to_one_hot(y_pos, n_classes)
    y_neg = labels_to_one_hot(y_neg, n_classes)
    return z_pos, y_pos, z_neg, y_neg
