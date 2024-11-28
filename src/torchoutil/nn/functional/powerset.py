#!/usr/bin/env python
# -*- coding: utf-8 -*-

# MIT License
#
# Copyright (c) 2023- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - https://herve.niderb.fr
# Alexis PLAQUET

# This code is based on PyAnnote source code from: https://github.com/pyannote/pyannote-audio/blob/4407a66023cb42fd74450ab83b802a09ffa27d52/pyannote/audio/utils/powerset.py

import math
from functools import lru_cache
from itertools import combinations
from typing import Optional, overload

import torch
from torch import Tensor
from torch.nn import functional as F

from torchoutil.core.get import DTypeLike, get_dtype
from torchoutil.nn.functional.multiclass import probs_to_onehot
from torchoutil.pyoutil.logging import warn_once
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
        mapping = build_powerset_mapping(num_classes, max_set_size)
        num_powerset_classes, _ = mapping.shape
    else:
        msg = "Either mapping or (num_classes and max_set_size) must be provided as arguments, but all of them are None."
        raise ValueError(msg)

    if not multilabel.is_floating_point():
        tgt_dtype = mapping.dtype
        msg = f"Implicit multilabel conversion from {multilabel.dtype} to {tgt_dtype} in multilabel_to_powerset fn."
        warn_once(msg, __name__)
        multilabel = multilabel.to(dtype=tgt_dtype)

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
        mapping = build_powerset_mapping(num_classes, max_set_size)
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
def build_powerset_mapping(
    num_classes: int,
    max_set_size: int,
    dtype: DTypeLike = None,
) -> Tensor2D:
    """Build powerset mapping matrix of shape (num_powerset_classes, num_classes)."""
    dtype = get_dtype(dtype)
    num_powerset_classes = get_num_powerset_classes(num_classes, max_set_size)
    mapping = torch.zeros(num_powerset_classes, num_classes, dtype=dtype)
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
