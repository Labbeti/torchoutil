#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
from functools import lru_cache
from itertools import combinations
from typing import Optional, overload

import torch
from torch import Tensor
from torch.nn import functional as F

from torchoutil.nn.functional.multiclass import probs_to_onehot
from torchoutil.types.tensor_subclasses import Tensor2D, Tensor3D


@overload
def multilabel_to_powerset(
    multilabel: Tensor,
    *,
    mapping: Tensor,
) -> Tensor3D:
    ...


@overload
def multilabel_to_powerset(
    multilabel: Tensor,
    *,
    num_classes: int,
    max_set_size: int,
) -> Tensor3D:
    ...


def multilabel_to_powerset(
    multilabel: Tensor,
    *,
    mapping: Optional[Tensor] = None,
    num_classes: Optional[int] = None,
    max_set_size: Optional[int] = None,
) -> Tensor3D:
    """
    Args:
        multilabel: (batch_size, num_frames, num_classes) Tensor

    Returns:
        powerset: (batch_size, num_frames, num_powerset_classes) Tensor
    """
    if mapping is not None:
        num_powerset_classes, _num_classes = mapping.shape
    elif num_classes is not None and max_set_size is not None:
        mapping = build_mapping(num_classes, max_set_size)
        num_powerset_classes, _ = mapping.shape
    else:
        msg = "Either mapping or (num_classes and max_set_size) must be provided as arguments, but all of them are None."
        raise ValueError(msg)

    powerset = F.one_hot(
        torch.argmax(torch.matmul(multilabel, mapping.T), dim=-1),
        num_classes=num_powerset_classes,
    )
    return powerset  # type: ignore


@overload
def powerset_to_multilabel(
    powerset: Tensor,
    soft: bool = False,
    *,
    mapping: Tensor,
) -> Tensor3D:
    ...


@overload
def powerset_to_multilabel(
    powerset: Tensor,
    soft: bool = False,
    *,
    num_classes: int,
    max_set_size: int,
) -> Tensor3D:
    ...


def powerset_to_multilabel(
    powerset: Tensor,
    soft: bool = False,
    *,
    mapping: Optional[Tensor] = None,
    num_classes: Optional[int] = None,
    max_set_size: Optional[int] = None,
) -> Tensor3D:
    """
    Args:
        powerset:  Powerset logits, probabilities or onehot tensor of shape (batch_size, num_frames, num_powerset_classes).

    Returns:
        multilabel: (batch_size, num_frames, num_classes) Tensor
    """
    if mapping is not None:
        pass
    elif num_classes is not None and max_set_size is not None:
        mapping = build_mapping(num_classes, max_set_size)
    else:
        msg = "Either mapping or (num_classes and max_set_size) must be provided as arguments, but all of them are None."
        raise ValueError(msg)

    if soft:
        powerset_probs = powerset.exp()
    else:
        powerset_probs = probs_to_onehot(powerset, dim=-1, dtype=torch.float)

    multilabel = torch.matmul(powerset_probs, mapping)
    return multilabel  # type: ignore


@lru_cache(maxsize=None)
def build_mapping(num_classes: int, max_set_size: int) -> Tensor2D:
    """Build powerset mapping matrix of shape (num_powerset_classes, num_classes)."""
    num_powerset_classes = get_num_powerset_classes(num_classes, max_set_size)
    mapping = torch.zeros(num_powerset_classes, num_classes)
    powerset_k = 0

    for set_size in range(0, max_set_size + 1):
        for current_set in combinations(range(num_classes), set_size):
            mapping[powerset_k, current_set] = 1
            powerset_k += 1

    return mapping  # type: ignore


@lru_cache(maxsize=None)
def get_num_powerset_classes(num_classes: int, max_set_size: int) -> int:
    binom_coefficients = (math.comb(num_classes, i) for i in range(0, max_set_size + 1))
    return int(sum(binom_coefficients))
