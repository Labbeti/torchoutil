#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from typing import Callable, Iterable, List, Optional, Union

import torch
from torch import Tensor

from torchoutil.core.make import GeneratorLike, make_generator
from torchoutil.nn.functional.transform import as_tensor
from torchoutil.pyoutil.collections import flat_list_of_list


def random_split(
    num_samples_or_indices: Union[int, List[int], Tensor],
    lengths: Iterable[float],
    generator: GeneratorLike = None,
    round_fn: Callable[[float], int] = math.floor,
) -> List[List[int]]:
    """Generate indices for a random dataset split.

    Args:
        num_samples: Number of total samples.
        lengths: Ratios of the target splits. Values should be in range [0, 1].
        generator: Torch Generator or seed to make this function deterministic. defaults to None.
        round_fn: Function to round ratios to integer sizes. defaults to math.floor.
    """
    generator = make_generator(generator)

    if isinstance(num_samples_or_indices, int):
        num_samples = num_samples_or_indices
        indices = torch.randperm(num_samples, generator=generator)
    else:
        indices = as_tensor(num_samples_or_indices)
        num_samples = len(indices)
    del num_samples_or_indices

    lengths = _round_lengths(num_samples, lengths, round_fn)
    generator = make_generator(generator)

    start = 0
    splits = []
    for length in lengths:
        end = start + length
        split = indices[start:end].tolist()
        splits.append(split)
    return splits


def balanced_monolabel_split(
    targets_indices: Union[Tensor, List[int]],
    num_classes: int,
    lengths: Iterable[float],
    generator: GeneratorLike = None,
    round_fn: Callable[[float], int] = math.floor,
) -> List[List[int]]:
    """Generate indices for a random dataset split while keeping the same multiclass distribution.

    Args:
        targets: List of class indices of size (N,).
        num_classes: Number of classes.
        lengths: Ratios of the target splits. Values should be in range [0, 1].
        generator: Torch Generator or seed to make this function deterministic. defaults to None.
        round_fn: Function to round ratios to integer sizes. defaults to math.floor.
    """
    if isinstance(targets_indices, list):
        targets_indices = as_tensor(targets_indices)

    assert (0 <= targets_indices).all(), "Target classes indices must be positive."
    assert (
        targets_indices < num_classes
    ).all(), f"Target classes indices must be lower than {num_classes=}."

    lengths = list(lengths)
    generator = make_generator(generator)

    indices_per_class = []
    for class_idx in range(num_classes):
        mask = targets_indices.eq(class_idx)
        indices = torch.where(mask)[0]
        indices = indices.tolist()
        indices_per_class.append(indices)

    indices_per_class = _shuffle_indices_per_class(indices_per_class, generator)
    splits = _split_indices_per_class(indices_per_class, lengths, round_fn)
    flatten = [flat_list_of_list(split)[0] for split in splits]
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


def _shuffle_indices_per_class(
    indices_per_class: List[List[int]],
    generator: Optional[torch.Generator],
) -> List[List[int]]:
    shuffled = []
    for indices in indices_per_class:
        perm = torch.randperm(len(indices), generator=generator)
        indices = torch.as_tensor(indices)
        indices = indices[perm].tolist()
        shuffled.append(indices)
    return indices_per_class


def _split_indices_per_class(
    indices_per_class: List[List[int]],
    lengths: List[float],
    round_fn: Callable[[float], int],
) -> List[List[List[int]]]:
    """
    Split distinct indices per class.

    Example:
    --------
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
