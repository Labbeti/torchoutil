#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.nn.modules import *

from .activation import LogSoftmaxMultidim, SoftmaxMultidim
from .crop import CropDim, CropDims
from .layer import PositionalEncoding
from .mask import MaskedMean, MaskedSum
from .mixins import EModule, EModuleDict, EModuleList, EModulePartial, ESequential
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
    AsTensor,
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
    Squeeze,
    TensorTo,
    ToItem,
    ToList,
    Topk,
    Transpose,
    Unsqueeze,
    View,
)
from .transform import (
    Flatten,
    Identity,
    PadAndCropDim,
    RepeatInterleaveNd,
    ResampleNearestFreqs,
    ResampleNearestRates,
    ResampleNearestSteps,
    Shuffled,
    ToTensor,
    TransformDrop,
)
