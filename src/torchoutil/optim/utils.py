#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List

from torch.optim.optimizer import Optimizer


def get_lr(optim: Optimizer, idx: int = 0) -> float:
    """
    Get the learning rate of the first group of an optimizer.

    Args:
        optim: The optimizer to get.
        idx: The group index of the learning rate in the optimizer. defaults to 0.
    """
    return get_lrs(optim)[idx]


def get_lrs(optim: Optimizer) -> List[float]:
    """
    Get the learning rates in all groups of an optimizer.

    Args:
        optim: The optimizer to get.
    """
    return [group["lr"] for group in optim.param_groups]
