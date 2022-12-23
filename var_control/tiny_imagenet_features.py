import torch
from torchtyping import TensorType  # type: ignore
from torchtyping import patch_typeguard
from typeguard import typechecked
import pickle

batch = None
classes = None

patch_typeguard()

tiny_imagenet_data_path = "output/load_and_run_resnet50/resnet50_tiny-imagenet.pkl"
tiny_imagenet_c_data_path = "output/load_and_run_resnet50/resnet50_tiny_imagenet_c_fog_4.pkl"

@typechecked
def load_features(
    file_name: str,
) -> tuple[TensorType["batch", "classes", float], TensorType["batch", int]]:
    with open(file_name, 'rb') as handle:
        outputs = pickle.load(handle)
        logits = outputs['outputs']
        labels = outputs['labels']

    return logits, labels


def load_mixed_features(
    file_name1: str,
    file_name2: str,
    mix_key: str = "imbalanced"
) -> tuple[TensorType["batch", "classes", float], TensorType["batch", int]]:

    assert mix_key in ["balanced", "imbalanced"]
    if mix_key == "imbalanced":
        print("mix imbalanced")
        n_reg = 1500
        n_corr = 500
    else:
        print("mix balanced")
        n_reg = n_corr = 1000

    with open(file_name1, 'rb') as handle:
        outputs = pickle.load(handle)
        logits1 = outputs['outputs'][:n_reg]
        labels1 = outputs['labels'][:n_reg]

    with open(file_name2, 'rb') as handle:
        outputs = pickle.load(handle)
        logits2 = outputs['outputs'][:n_corr]
        labels2 = outputs['labels'][:n_corr]

    labels = torch.concat([labels1, labels2], 0)
    logits = torch.vstack([logits1, logits2])
    
    indices = torch.randperm(labels.size()[0])
    labels=labels[indices]
    logits=logits[indices]

    return logits, labels


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
    feature_key: str,
    mix_key: str = "imbalanced",
) -> tuple[TensorType[float], TensorType[int]]:
    match feature_key:
        case "resnet50":
            file_name = tiny_imagenet_data_path
            features, labels = load_features(file_name)
        case "resnet50-corrupt":
            file_name = tiny_imagenet_c_data_path
            features, labels = load_features(file_name)
        case "resnet50-mixed":
            file_name1 = tiny_imagenet_data_path
            file_name2 = tiny_imagenet_c_data_path
            features, labels = load_mixed_features(file_name1, file_name2, mix_key)
        case _:
            raise ValueError(f"unknown feature_key: {feature_key}")

    n_classes = features.shape[1]
    one_hot_labels = labels_to_one_hot(labels, n_classes)
    return features[:5000], one_hot_labels[:5000]
