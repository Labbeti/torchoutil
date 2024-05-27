#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.nn.modules import *

from .activation import SoftmaxMultidim
from .crop import CropDim, CropDims
from .layer import PositionalEncoding
from .mask import MaskedMean, MaskedSum
from .mixins import EModule, ESequential
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
    IndicesToNames,
    MultihotToIndices,
    MultihotToNames,
    NamesToIndices,
    NamesToMultihot,
    ProbsToIndices,
    ProbsToMultihot,
    ProbsToNames,
)
from .numpy import NumpyToTensor, TensorToNumpy, ToNumpy
from .pad import PadAndStackRec, PadDim, PadDims
from .tensor import (
    FFT,
    IFFT,
    Abs,
    Angle,
    AsTensor,
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
    Transpose,
    Unsqueeze,
)
from .transform import (
    RepeatInterleaveNd,
    ResampleNearestFreqs,
    ResampleNearestRates,
    ResampleNearestSteps,
    TransformDrop,
)
