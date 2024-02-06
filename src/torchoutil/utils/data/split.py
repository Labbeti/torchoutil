#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from typing import Callable, Iterable, List, Union

import torch
from torch import Generator, Tensor

from torchoutil.utils.collections import flat_list


def random_split(
    num_samples: int,
    lengths: Iterable[float],
    generator: Union[int, Generator, None] = None,
) -> List[List[int]]:
    """
    Args:
        targets: List of class indices of size (N,).
        lengths: Ratios of the target splits.
        generator: Torch Generator or seed to make this function deterministic. defaults to None.
    """
    lengths = _round_lengths(num_samples, lengths, math.floor)
    if isinstance(generator, int):
        generator = Generator().manual_seed(generator)

    indices = torch.randperm(num_samples, generator=generator)
    start = 0
    splits = []
    for length in lengths:
        end = start + length
        split = indices[start:end].tolist()
        splits.append(split)
    return splits


def balanced_random_split(
    targets: Tensor,
    num_classes: int,
    lengths: Iterable[float],
    generator: Union[int, Generator, None] = None,
) -> List[List[int]]:
    """
    Args:
        targets: List of class indices of size (N,).
        num_classes: Number of classes.
        lengths: Ratios of the target splits.
        generator: Torch Generator or seed to make this function deterministic. defaults to None.
    """
    lengths = list(lengths)
    if isinstance(generator, int):
        generator = Generator().manual_seed(generator)

    indices_per_class = []
    for class_idx in range(num_classes):
        indices = torch.where(targets.eq(class_idx))[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        indices = indices.tolist()
        indices_per_class.append(indices)

    splits = _split_indices_per_class(indices_per_class, lengths, math.floor)
    flatten = [flat_list(split)[0] for split in splits]
    return flatten


def _round_lengths(
    n: int,
    lengths: Iterable[Union[int, float]],
    round_fn: Callable[[float], int],
) -> List[int]:
    int_lenghts = []
    for length in lengths:
        if isinstance(length, float):
            length = round_fn(n * length)
        int_lenghts.append(length)
    return int_lenghts


def _split_indices_per_class(
    indices_per_class: List[List[int]],
    lengths: List[float],
    round_fn: Callable[[float], int],
) -> List[List[List[int]]]:
    """
    Split distinct indices per class.

    Example:
    ```
    >>> split_indices_per_class(indices_per_class=[[1, 2], [3, 4], [5, 6]], ratios=[0.5, 0.5])
    ... [[[1], [3], [5]], [[2], [4], [6]]]
    ```

    Args:
        indices_per_class: List of indices of each class.
        lengths: The ratios of each indices split.
    """
    assert 0.0 <= sum(lengths) <= 1.0, "Ratio sum cannot be greater than 1.0."

    num_classes = len(indices_per_class)
    num_splits = len(lengths)

    indices_per_ratio_per_class = [
        [[] for _ in range(num_classes)] for _ in range(num_splits)
    ]

    current_starts = [0 for _ in range(num_classes)]
    for i, length in enumerate(lengths):
        for j, indices in enumerate(indices_per_class):
            current_start = current_starts[j]
            current_end = current_start + round_fn(length * len(indices))
            sub_indices = indices[current_start:current_end]
            indices_per_ratio_per_class[i][j] = sub_indices
            current_starts[j] = current_end

    return indices_per_ratio_per_class
