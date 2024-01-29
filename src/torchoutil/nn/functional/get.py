#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Union

import torch

_DEVICE_CUDA_IF_AVAILABLE = "cuda_if_available"


def get_device(
    device: Union[str, torch.device, None] = _DEVICE_CUDA_IF_AVAILABLE
) -> Optional[torch.device]:
    if device == _DEVICE_CUDA_IF_AVAILABLE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)
    return device
