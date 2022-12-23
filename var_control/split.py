import torch
from torch import Tensor
from torch.utils.data import (
    Dataset,
    TensorDataset,
    DataLoader,
    random_split,
    RandomSampler,
)


def collate(dataset: Dataset, batch_size: int) -> list[Tensor]:
    """Collate the first batch into a list of tensors.

    Args:
        dataset (TensorDataset): Dataset representing the data to be collated.
    """
    return next(iter(DataLoader(dataset, batch_size=batch_size)))


def even_split(*args):
    """Construct a 50/50 split of data.

    Args:
        args ([Tensor]): Inputs, where each args[i].size(0) = N

    Returns:
        tuple[[Tensor], [Tensor]]: The split data into train_data, test_data
    """
    num_examples = args[0].size(0)
    num_train = num_examples // 2
    num_test = num_examples - num_train

    dataset = TensorDataset(*args)

    train_data, test_data = random_split(dataset, [num_train, num_test])

    train_batch = collate(train_data, len(train_data))
    test_batch = collate(test_data, len(test_data))

    if len(args) == 1:
        return train_batch[0], test_batch[0]
    else:
        return train_batch, test_batch


def even_split_iterator(dataset: TensorDataset, num_trials: int):
    num_examples = len(dataset)
    num_train = num_examples // 2
    loader = DataLoader(
        dataset, batch_size=len(dataset), sampler=RandomSampler(dataset)
    )
    for _ in range(num_trials):
        for sample in loader:
            yield [s[:num_train] for s in sample], [s[num_train:] for s in sample]
            break


def fixed_split_iterator(dataset: TensorDataset, num_trials: int, num_train=1000):
    # num_examples = len(dataset)
    # num_train = num_examples // 2
    loader = DataLoader(
        dataset, batch_size=len(dataset), sampler=RandomSampler(dataset)
    )
    for _ in range(num_trials):
        for sample in loader:
            yield [s[:num_train] for s in sample], [s[num_train:] for s in sample]
            break


def fixed_split(
    z: Tensor, y: Tensor, calib_size: int
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    num_examples = z.size(0)

    perm = torch.randperm(num_examples)
    train_perm = perm[:calib_size]
    test_perm = perm[calib_size:]

    return (z[train_perm], y[train_perm]), (z[test_perm], y[test_perm])
