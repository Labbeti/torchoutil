#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Any, Iterable, List, Optional, Tuple, Union

from torch import nn
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer

pylog = logging.getLogger(__name__)


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


def create_params_groups_bias(
    model: Union[nn.Module, Iterable[Tuple[str, Parameter]]],
    weight_decay: float,
    skip_list: Optional[Iterable[str]] = (),
    verbose: int = 2,
) -> list[dict[str, Any]]:
    if isinstance(model, nn.Module):
        params = model.named_parameters()
    else:
        params = model
    del model

    decay: list[Parameter] = []
    no_decay: list[Parameter] = []

    if skip_list is None:
        skip_list = {}
    else:
        skip_list = dict.fromkeys(skip_list)

    for name, param in params:
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
            if verbose >= 2:
                pylog.debug(f"No wd for {name}")
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
