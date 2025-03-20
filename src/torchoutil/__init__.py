#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Collection of functions and modules to help development in PyTorch."""


__name__ = "torchoutil"
__author__ = "Étienne Labbé (Labbeti)"
__author_email__ = "labbeti.pub@gmail.com"
__license__ = "MIT"
__maintainer__ = "Étienne Labbé (Labbeti)"
__status__ = "Development"
__version__ = "0.6.0"

# Import global functions and classes from torch
from torch import *  # type: ignore

# Re-import torchoutil modules for language servers
from . import core as core
from . import extras as extras
from . import hub as hub
from . import nn as nn
from . import optim as optim
from . import pyoutil as pyoutil
from . import types as types
from . import utils as utils

# Global imports
from .core.dtype_enum import DTypeEnum
from .core.dtype_enum import DTypeEnum as dtype_enum
from .core.make import (  # noqa: F401
    CUDA_IF_AVAILABLE,
    DeviceLike,
    DTypeLike,
    GeneratorLike,
)
from .hub.download import download_file
from .nn.functional import *
from .pyoutil.semver import Version
from .pyoutil.typing.guards import isinstance_guard
from .serialization.common import to_builtin
from .serialization.csv import dump_csv, load_csv
from .serialization.dump_fn import dump, save
from .serialization.json import dump_json, load_json
from .serialization.load_fn import load
from .serialization.pickle import dump_pickle, load_pickle
from .serialization.torch import load_torch, to_torch
from .types.guards import (
    is_builtin_number,
    is_builtin_scalar,
    is_number_like,
    is_scalar_like,
)
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
    IntegralTensor,
    IntegralTensor0D,
    IntegralTensor1D,
    IntegralTensor2D,
    IntegralTensor3D,
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
    UnsignedIntegerTensor,
    UnsignedIntegerTensor0D,
    UnsignedIntegerTensor1D,
    UnsignedIntegerTensor2D,
    UnsignedIntegerTensor3D,
)

version = __version__
version_info = Version(__version__)
