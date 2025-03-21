#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules (classes) interface."""

from torch.nn.modules import *  # type: ignore

from .activation import LogSoftmaxMultidim, SoftmaxMultidim
from .container import (
    EModuleDict,
    EModuleList,
    EModulePartial,
    ESequential,
    ModuleDict,
    ModuleList,
    ModulePartial,
    Sequential,
)
from .crop import CropDim, CropDims
from .layer import PositionalEncoding
from .mask import MaskedMean, MaskedSum
from .module import EModule, Module
from .multiclass import (
    IndexToName,
    IndexToOnehot,
    NameToIndex,
    NameToOnehot,
    OnehotToIndex,
    OnehotToName,
    ProbsToIndex,
    ProbsToName,
    ProbsToOnehot,
)
from .multilabel import (
    IndicesToMultihot,
    IndicesToMultinames,
    MultihotToIndices,
    MultihotToMultinames,
    MultinamesToIndices,
    MultinamesToMultihot,
    ProbsToIndices,
    ProbsToMultihot,
    ProbsToMultinames,
)
from .numpy import NumpyToTensor, TensorToNumpy, ToNumpy
from .pad import PadAndStackRec, PadDim, PadDims
from .powerset import MultilabelToPowerset, PowersetToMultilabel
from .tensor import (
    FFT,
    IFFT,
    Abs,
    Angle,
    Exp,
    Exp2,
    Imag,
    Log,
    Log2,
    Log10,
    Max,
    Mean,
    Min,
    Normalize,
    Permute,
    Pow,
    Real,
    Repeat,
    RepeatInterleave,
    Reshape,
    TensorTo,
    ToList,
    Transpose,
    View,
)
from .transform import (
    AsTensor,
    Flatten,
    Identity,
    MoveToRec,
    PadAndCropDim,
    RepeatInterleaveNd,
    ResampleNearestFreqs,
    ResampleNearestRates,
    ResampleNearestSteps,
    Shuffled,
    Squeeze,
    ToItem,
    Topk,
    TopP,
    ToTensor,
    TransformDrop,
    Unsqueeze,
    ViewAsComplex,
    ViewAsReal,
)
