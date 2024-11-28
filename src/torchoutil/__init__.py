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
__version__ = "0.5.0"


from torch import *  # type: ignore

# Re-import for language servers
from . import core as core
from . import extras as extras
from . import fft as fft
from . import hub as hub
from . import nn as nn
from . import optim as optim
from . import utils as utils
from .core.dtype_enum import DTypeEnum
from .core.dtype_enum import DTypeEnum as dtype_enum
from .core.get import (  # noqa: F401
    CUDA_IF_AVAILABLE,
    DeviceLike,
    DTypeLike,
    GeneratorLike,
    get_device,
    get_dtype,
    get_generator,
)
from .core.semver import Version
from .nn.functional import *
from .types.tensor_subclasses import (
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
    CDoubleTensor,
    CDoubleTensor0D,
    CDoubleTensor1D,
    CDoubleTensor2D,
    CDoubleTensor3D,
    CFloatTensor,
    CFloatTensor0D,
    CFloatTensor1D,
    CFloatTensor2D,
    CFloatTensor3D,
    CHalfTensor,
    CHalfTensor0D,
    CHalfTensor1D,
    CHalfTensor2D,
    CHalfTensor3D,
    CharTensor,
    CharTensor0D,
    CharTensor1D,
    CharTensor2D,
    CharTensor3D,
    ComplexFloatingTensor,
    ComplexFloatingTensor0D,
    ComplexFloatingTensor1D,
    ComplexFloatingTensor2D,
    ComplexFloatingTensor3D,
    DoubleTensor,
    DoubleTensor0D,
    DoubleTensor1D,
    DoubleTensor2D,
    DoubleTensor3D,
    FloatingTensor,
    FloatingTensor0D,
    FloatingTensor1D,
    FloatingTensor2D,
    FloatingTensor3D,
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
    SignedIntegerTensor,
    SignedIntegerTensor0D,
    SignedIntegerTensor1D,
    SignedIntegerTensor2D,
    SignedIntegerTensor3D,
    Tensor0D,
    Tensor1D,
    Tensor2D,
    Tensor3D,
)
from .types.variable_fns import as_tensor, empty, full, ones, rand, zeros
from .utils.saving.common import to_builtin
from .utils.saving.csv import load_csv, to_csv
from .utils.saving.json import load_json, to_json
from .utils.saving.pickle import load_pickle, to_pickle

version = __version__
version_info = Version(__version__)

del Version
