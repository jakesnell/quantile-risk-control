import torch
from torchtyping import TensorType  # type: ignore
from torchtyping import patch_typeguard
from typeguard import typechecked
import pickle

batch = None
classes = None

patch_typeguard()

data_root = "data/clip_cifar_100"


def load_features():
    logits = torch.load("{}/clip_b32_Z.pt".format(data_root), map_location=torch.device('cpu'))
    labels = torch.load("{}/clip_b32_y.pt".format(data_root), map_location=torch.device('cpu'))
    return logits, labels


def labels_to_one_hot(
    labels: TensorType["batch", int],
    n_classes: int
):
    one_hot_mat = torch.zeros((labels.shape[0], n_classes), dtype=int)
    one_hot_mat[torch.arange(one_hot_mat.shape[0]), labels.tolist()] = 1
    return one_hot_mat


def load_features_and_ground_truth(feature_key: str):
    match feature_key:
        case "clip_b32":
            features, labels = load_features()
        case _:
            raise ValueError(f"unknown feature_key: {feature_key}")
    return features, labels_to_one_hot(labels, features.shape[-1])
