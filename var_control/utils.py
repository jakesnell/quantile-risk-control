from torch import Tensor
import torch
from dataclasses import asdict, dataclass
from collections import OrderedDict
import math
from tqdm import tqdm


def quantile_indices(x: Tensor, beta: float) -> Tensor:
    assert x.dim() == 1
    n = x.size(0)
    n_trunc = math.floor(n * beta)
    _, inds = torch.sort(x)
    return inds[:n_trunc]


class Batch:
    def __init__(self, args: list[Tensor], num_hypotheses: int):
        assert len(args) == 2
        pred_size, loss = args
        assert pred_size.dim() == 2
        assert pred_size.size(0) == loss.size(0) == num_hypotheses
        assert pred_size.size(1) == loss.size(1)
        self.pred_size = pred_size
        self.loss = loss


def predict(thresholds: Tensor, scores: Tensor) -> Tensor:
    """Use the provided thresholds to create prediction sets from the given
    scores.
    Args:
        thresholds (Tensor) [H] (float): Input thresholds.
        scores (Tensor) [...] (float): Scores to be thresholded.
    Returns:
        Tensor [H, ...] (long): Binary mask of predictions, constructed by thresholding
        the scores.
    """
    preds = torch.gt(scores.unsqueeze(-1), thresholds)
    return torch.permute(preds, [-1] + list(range(scores.dim()))).long()


def predict_batch(thresholds: Tensor, scores: Tensor, batch_size=32, device="cpu") -> Tensor:
    """Use the provided thresholds to create prediction sets from the given
    scores.
    Args:
        thresholds (Tensor) [H] (float): Input thresholds.
        scores (Tensor) [...] (float): Scores to be thresholded.
    Returns:
        Tensor [H, ...] (long): Binary mask of predictions, constructed by thresholding
        the scores.
    """
    preds = torch.Tensor().to(device)
    for i in tqdm(range(scores.shape[0] // batch_size + 1)):
        s = scores[i * batch_size : (i + 1) * batch_size]
        batch_preds = torch.gt(s.unsqueeze(-1), thresholds)
        preds = torch.cat([preds, batch_preds], 0)
    print(preds.shape)
    # preds = torch.concat(preds, 0)
    return torch.permute(preds, [-1] + list(range(scores.dim()))).long()


@dataclass(frozen=True)
class MonotonicTrialResult:
    quantile_loss: float
    mean_in_quantile_loss: float
    mean_in_quantile_pred_size: float
    quantile_satisfied: bool
    # mean_loss: float
    # mean_pred_size: float
    # mean_satisfied: bool


def run_monotonic_trial(
    train_batch: Batch,
    test_batch: Batch,
    method_dict: OrderedDict,
    alpha: float,
    beta: float,
    delta: float,
    markov_scaling: bool,
) -> OrderedDict[str, MonotonicTrialResult]:
    def get_result(method):
        if hasattr(method, "fit_target_var"):
            hypothesis_ind = method.fit_target_var(train_batch.loss, delta, beta, alpha)
        else:
            # use Markov scaling
            if markov_scaling:
                scaled_alpha = alpha * (1 - beta)
            else:
                scaled_alpha = alpha
            hypothesis_ind = method.fit_target_risk(
                train_batch.loss, delta, scaled_alpha
            )

        # compute the test loss
        test_loss = test_batch.loss[hypothesis_ind]
        test_pred_size = test_batch.pred_size[hypothesis_ind]

        # mean_loss = test_loss.mean(-1).item()
        # mean_pred_size = test_pred_size.float().mean(-1).item()
        # mean_satisfied = mean_loss <= alpha

        quantile_loss = torch.quantile(test_loss, beta).item()
        quantile_satisfied = quantile_loss <= alpha

        inds = quantile_indices(test_loss, beta)
        mean_in_quantile_loss = test_loss[inds].mean(-1).item()
        mean_in_quantile_pred_size = test_pred_size[inds].float().mean(-1).item()

        return MonotonicTrialResult(
            # mean_loss=mean_loss,
            # mean_pred_size=mean_pred_size,
            # mean_satisfied=mean_satisfied,
            quantile_loss=quantile_loss,
            mean_in_quantile_loss=mean_in_quantile_loss,
            mean_in_quantile_pred_size=mean_in_quantile_pred_size,
            quantile_satisfied=quantile_satisfied,
        )

    return OrderedDict([(k, get_result(v)) for k, v in method_dict.items()])


@dataclass(frozen=True)
class GeneralTrialResult:
    quantile_loss: float
    # mean_in_quantile_loss: float
    # mean_in_quantile_pred_size: float
    # quantile_satisfied: bool
    # mean_loss: float
    # mean_pred_size: float
    # mean_satisfied: bool
    # mean_alpha: float
    quantile_alpha: float


def run_general_trial(
    train_batch: Batch,
    test_batch: Batch,
    method_dict: OrderedDict,
    alpha: float,
    beta: float,
    delta: float,
    markov_scaling: bool,
) -> OrderedDict[str, GeneralTrialResult]:
    def get_result(method):
        if hasattr(method, "fit_var"):
            hypothesis_ind, quantile_alpha = method.fit_var(
                train_batch.loss, delta, beta
            )
            mean_alpha = math.nan
        else:
            if markov_scaling:
                hypothesis_ind = method.fit_risk(
                    train_batch.loss, alpha * (1 - beta), delta
                )
                mean_alpha = math.nan
                quantile_alpha = alpha
            else:
                hypothesis_ind, mean_alpha = method.fit_risk(train_batch.loss, alpha, delta)
                if not mean_alpha:
                    mean_alpha = alpha
                # mean_alpha = alpha
                # quantile_alpha = math.nan
                quantile_alpha = mean_alpha/(1-beta)

        # compute the test loss
        test_loss = test_batch.loss[hypothesis_ind]
        test_pred_size = test_batch.pred_size[hypothesis_ind]

        # mean_loss = test_loss.mean(-1).item()
        # mean_pred_size = test_pred_size.float().mean(-1).item()
        # mean_satisfied = mean_loss <= mean_alpha

        quantile_loss = torch.quantile(test_loss, beta).item()
        quantile_satisfied = quantile_loss <= quantile_alpha

        inds = quantile_indices(test_loss, beta)
        mean_in_quantile_loss = test_loss[inds].mean(-1).item()
        mean_in_quantile_pred_size = test_pred_size[inds].float().mean(-1).item()

        return GeneralTrialResult(
            # mean_loss=mean_loss,
            # mean_pred_size=mean_pred_size,
            # mean_satisfied=mean_satisfied,
            quantile_loss=quantile_loss,
            # mean_in_quantile_loss=mean_in_quantile_loss,
            # mean_in_quantile_pred_size=mean_in_quantile_pred_size,
            # quantile_satisfied=quantile_satisfied,
            # mean_alpha=mean_alpha,
            quantile_alpha=quantile_alpha,
        )

    return OrderedDict([(k, get_result(v)) for k, v in method_dict.items()])


def calculate_oracle(test_loss, beta, loss_type="general", alpha=None):

    quantile = torch.quantile(test_loss, beta, dim=-1, interpolation="higher")
    if loss_type == "general":
        hypothesis_ind = torch.argmin(quantile, -1)
        oracle_var = quantile[hypothesis_ind].item()
    else:
        hypothesis_ind = int(
            torch.nonzero(quantile <= alpha).view(-1).max().item()
        )
        oracle_var = quantile[hypothesis_ind].item()
    return hypothesis_ind, oracle_var
