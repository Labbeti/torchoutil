#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import Tensor, nn

from torchoutil.utils.packaging import _NUMPY_AVAILABLE

if _NUMPY_AVAILABLE:
    import numpy as np

    class ToNumpy(nn.Module):
        def forward(self, x: Tensor) -> np.ndarray:
            return x.cpu().numpy()

    class FromNumpy(nn.Module):
        def forward(self, x: np.ndarray) -> Tensor:
            return torch.from_numpy(x)
