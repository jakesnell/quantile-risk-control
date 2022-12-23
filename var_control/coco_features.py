import torch
from torchtyping import TensorType  # type: ignore
from torchtyping import patch_typeguard
from typeguard import typechecked
from var_control.coco_loader import CocoLoader

batch = None
classes = None

patch_typeguard()


@typechecked
def load_features(
    file_name: str,
) -> tuple[TensorType["batch", int], TensorType["batch", "classes", float]]:
    match torch.jit.load(file_name).state_dict():
        case {"image_ids": ids, "output": features}:
            return ids, features
        case _:
            raise ValueError("unknown format")


@typechecked
def load_features_and_ground_truth(
    feature_key: str,
) -> tuple[TensorType["batch", "classes", float], TensorType["batch", "classes", int]]:
    match feature_key:
        case "tresnet_m":
            file_name = "output/load_and_run_tresnet/tresnet_m_coco_all.pt"
        case _:
            raise ValueError(f"unknown feature_key: {feature_key}")

    loader = CocoLoader()
    ids, features = load_features(file_name)
    return features, loader.one_hot_labels(ids)
