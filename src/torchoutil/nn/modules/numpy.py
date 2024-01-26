#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import Tensor

from torchoutil.nn.modules.typed import TModule
from torchoutil.utils.packaging import _NUMPY_AVAILABLE

if _NUMPY_AVAILABLE:
    import numpy as np

    class ToNumpy(TModule[Tensor, np.ndarray]):
        def forward(self, x: Tensor) -> np.ndarray:
            return x.cpu().numpy()

    class FromNumpy(TModule[np.ndarray, Tensor]):
        def forward(self, x: np.ndarray) -> Tensor:
            return torch.from_numpy(x)
