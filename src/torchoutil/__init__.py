#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Collection of functions and modules to help development in PyTorch.
"""

__name__ = "torchoutil"
__author__ = "Étienne Labbé (Labbeti)"
__author_email__ = "labbeti.pub@gmail.com"
__license__ = "MIT"
__maintainer__ = "Étienne Labbé (Labbeti)"
__status__ = "Development"
__version__ = "0.4.0"


from torch import *  # type: ignore

# Re-import for language servers
from . import hub as hub
from . import nn as nn
from . import optim as optim
from . import utils as utils
from .nn.functional import *
from .types.classes import Device, DeviceLike, DTypeLike, GeneratorLike
from .types.tensor_typing import (
    BoolTensor,
    BoolTensor0D,
    BoolTensor1D,
    BoolTensor2D,
    BoolTensor3D,
    ByteTensor,
    ByteTensor0D,
    ByteTensor1D,
    ByteTensor2D,
    ByteTensor3D,
    CharTensor,
    CharTensor0D,
    CharTensor1D,
    CharTensor2D,
    CharTensor3D,
    DoubleTensor,
    DoubleTensor0D,
    DoubleTensor1D,
    DoubleTensor2D,
    DoubleTensor3D,
    FloatTensor,
    FloatTensor0D,
    FloatTensor1D,
    FloatTensor2D,
    FloatTensor3D,
    HalfTensor,
    HalfTensor0D,
    HalfTensor1D,
    HalfTensor2D,
    HalfTensor3D,
    IntTensor,
    IntTensor0D,
    IntTensor1D,
    IntTensor2D,
    IntTensor3D,
    LongTensor,
    LongTensor0D,
    LongTensor1D,
    LongTensor2D,
    LongTensor3D,
    ShortTensor,
    ShortTensor0D,
    ShortTensor1D,
    ShortTensor2D,
    ShortTensor3D,
    Tensor0D,
    Tensor1D,
    Tensor2D,
    Tensor3D,
)
