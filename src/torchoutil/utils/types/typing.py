#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, ClassVar, Dict, Final, Protocol, Tuple, Union, runtime_checkable

import torch
from torch import Tensor
from typing_extensions import Annotated

from torchoutil.utils.packaging import _NUMPY_AVAILABLE


@runtime_checkable
class DataclassInstance(Protocol):
    # Class meant for typing purpose only
    __dataclass_fields__: ClassVar[Dict[str, Any]]


@runtime_checkable
class NamedTupleInstance(Protocol):
    # Class meant for typing purpose only
    _fields: Tuple[str, ...]
    _field_defaults: Dict[str, Any]

    def _asdict(self) -> Dict[str, Any]:
        ...

    def __getitem__(self, idx):
        ...

    def __len__(self) -> int:
        ...


if not _NUMPY_AVAILABLE:
    from torchoutil.utils.types import _numpy_placeholder as np  # noqa: F401
    from torchoutil.utils.types import _numpy_placeholder as numpy

    ACCEPTED_NUMPY_DTYPES = ()

else:
    import numpy  # noqa: F401
    import numpy as np

    # Numpy dtypes that can be converted to tensor
    ACCEPTED_NUMPY_DTYPES = (
        np.float64,
        np.float32,
        np.float16,
        np.complex64,
        np.complex128,
        np.int64,
        np.int32,
        np.int16,
        np.int8,
        np.uint8,
        bool,
    )


Tensor0D = Annotated[Tensor, "0D"]
Tensor1D = Annotated[Tensor, "1D"]
Tensor2D = Annotated[Tensor, "2D"]
Tensor3D = Annotated[Tensor, "3D"]
Tensor4D = Annotated[Tensor, "4D"]
Tensor5D = Annotated[Tensor, "5D"]

BuiltinScalar = Union[int, float, bool, complex]
NumpyScalar = Union[np.ndarray, np.number]
Scalar = Union[BuiltinScalar, NumpyScalar, Tensor0D]


TORCH_DTYPES: Final[Dict[str, torch.dtype]] = {
    "float32": torch.float32,
    "float": torch.float,
    "float64": torch.float64,
    "double": torch.double,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "half": torch.half,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "short": torch.short,
    "int32": torch.int32,
    "int": torch.int,
    "int64": torch.int64,
    "long": torch.long,
    "complex32": torch.complex32,
    "complex64": torch.complex64,
    "cfloat": torch.cfloat,
    "complex128": torch.complex128,
    "cdouble": torch.cdouble,
    "quint8": torch.quint8,
    "qint8": torch.qint8,
    "qint32": torch.qint32,
    "bool": torch.bool,
    "quint4x2": torch.quint4x2,
    "quint2x4": torch.quint2x4,
}
