import torch
from torchtyping import TensorType  # type: ignore
from torchtyping import patch_typeguard
from typeguard import typechecked
import pickle

batch = None
classes = None

patch_typeguard()

data_root = "data/go_emotions"


def load_features():
    logits = torch.load("{}/go_emotions_Z.pt".format(data_root))
    labels = torch.load("{}/go_emotions_y.pt".format(data_root))
    return logits, labels


def load_features_and_ground_truth(feature_key: str):
    match feature_key:
        case "bert":
            features, labels = load_features()
        case _:
            raise ValueError(f"unknown feature_key: {feature_key}")
    return features, labels
