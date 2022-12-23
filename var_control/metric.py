import torch
from torch import Tensor


def sensitivity(input: Tensor, target: Tensor) -> Tensor:
    """Compute the sensitivity (recall).

    Args:
        input (Tensor) [..., C]: Predictions.
        target (Tensor) [..., C]: Targets.

    Returns:
        Tensor [...]: The per-example sensitivity.
    """
    numer = torch.sum(input * target, -1)
    denom = torch.sum(target, -1)
    ratio = numer / denom
    return torch.where(denom.gt(0), ratio, torch.ones_like(ratio))


def specificity(input: Tensor, target: Tensor) -> Tensor:
    return sensitivity(1 - input, 1 - target)


def balanced_accuracy(input: Tensor, target: Tensor) -> Tensor:
    return 0.5 * (sensitivity(input, target) + specificity(input, target))


def weighted_loss(y_hat: Tensor, y: Tensor) -> Tensor:
    y_hat = y_hat.permute(1, 0, 2)
    label_weights = torch.rand(1000)
    n = y.shape[0]
    # compute unweighted loss
    fixed_ls = 1 - sensitivity(y_hat, y)
    # convert one-hot matrix to label vector
    y_labels = torch.argmax(y, dim=1).tolist()
    # get vector of per sample loss weights
    weight_per_sample = label_weights[y_labels].unsqueeze(0)
    # apply class specific loss to each sample
    ls = (fixed_ls > 0.0).float().view(-1, n) * weight_per_sample
    return ls.permute(1, 0)


def custom_loss(input: Tensor, labels: Tensor) -> Tensor:
    label_weights = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])
    y_labels = torch.argmax(labels, dim=1).tolist()

    weight_per_sample = label_weights[y_labels].unsqueeze(1)
    sens = 1 - sensitivity(input, labels.unsqueeze(1))
    spec = 1 - sensitivity(1 - input, 1 - labels.unsqueeze(1))
    return weight_per_sample * sens + (1 - weight_per_sample) * spec


def mIoU(pred_mask, mask, smooth=1e-10, n_classes=2):

    pred_mask = pred_mask.permute(1, 0, 2)
    with torch.no_grad():

        if len(list(pred_mask.size())) <= 2:
            pred_mask = pred_mask.unsqueeze(0)
        mask = mask.view(1, mask.shape[0], -1)

        iou_per_class = []
        for clas in range(1, n_classes):  # loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0:  # no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = (
                    torch.logical_and(true_class, true_label).sum(axis=2).float()
                )
                union = torch.logical_or(true_class, true_label).sum(axis=2).float()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou.unsqueeze(0))
        miou = torch.concat(iou_per_class, 0)
        miou = torch.mean(miou, axis=0).squeeze(0)

        return miou.permute(1, 0)
