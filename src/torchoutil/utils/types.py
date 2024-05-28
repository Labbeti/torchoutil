#!/usr/bin/env python
# -*- coding: utf-8 -*-

from types import NoneType
from typing import (
    Any,
    ClassVar,
    Dict,
    NewType,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

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


Tensor0D = Annotated[Tensor, "0D"]
Tensor1D = Annotated[Tensor, "1D"]
Tensor2D = Annotated[Tensor, "2D"]
Tensor3D = Annotated[Tensor, "3D"]
Tensor4D = Annotated[Tensor, "4D"]
Tensor5D = Annotated[Tensor, "5D"]


if not _NUMPY_AVAILABLE:

    class _Placeholder:
        generic = NewType("generic", float)
        ndarray = NewType("ndarray", float)
        dtype = NoneType

    np = _Placeholder()

else:
    import numpy as np  # type: ignore

BuiltinScalar = Union[int, float, bool, complex]
NumpyScalar = Union[np.generic, np.ndarray]
TorchScalar = Tensor0D
Scalar = Union[BuiltinScalar, NumpyScalar, TorchScalar]
