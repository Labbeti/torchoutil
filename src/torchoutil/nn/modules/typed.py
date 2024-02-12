#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Generic, OrderedDict, TypeVar, overload

from torch import nn

InType = TypeVar("InType", covariant=False, contravariant=True)
OutType = TypeVar("OutType", covariant=True, contravariant=False)
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")
T4 = TypeVar("T4")
T5 = TypeVar("T5")
T6 = TypeVar("T6")


class TModule(Generic[InType, OutType], nn.Module):
    """Typed version of torch.nn.Module. Can specify an input and output type."""

    def __call__(self, *args: InType, **kwargs: InType) -> OutType:
        return super().__call__(*args, **kwargs)

    def forward(self, *args: InType, **kwargs: InType) -> OutType:
        return super().forward(*args, **kwargs)

    @overload
    def compose(self, other: "TModule[Any, T1]") -> "TSequential[InType, T1]":
        ...

    @overload
    def compose(self, other: nn.Module) -> "TSequential[InType, Any]":
        ...

    def compose(self, other) -> "TSequential[InType, Any]":
        return TSequential(self, other)

    def duplicate(self, num: int) -> "TSequential[InType, OutType]":
        duplicated = [self] * num
        return TSequential(*duplicated)


class TSequential(Generic[InType, OutType], TModule[InType, OutType], nn.Sequential):
    """Typed version of torch.nn.Sequential, designed to work with torchoutil.nn.TModules."""

    @overload
    def __init__(
        self,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TModule[InType, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TModule[InType, T1],
        arg1: TModule[T1, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TModule[InType, T1],
        arg1: TModule[T1, T2],
        arg2: TModule[T2, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TModule[InType, T1],
        arg1: TModule[T1, T2],
        arg2: TModule[T2, T3],
        arg3: TModule[T3, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TModule[InType, T1],
        arg1: TModule[T1, T2],
        arg2: TModule[T2, T3],
        arg3: TModule[T3, T4],
        arg4: TModule[T4, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TModule[InType, T1],
        arg1: TModule[T1, T2],
        arg2: TModule[T2, T3],
        arg3: TModule[T3, T4],
        arg4: TModule[T4, T5],
        arg5: TModule[T5, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TModule[InType, T1],
        arg1: TModule[T1, T2],
        arg2: TModule[T2, T3],
        arg3: TModule[T3, T4],
        arg4: TModule[T4, T5],
        arg5: TModule[T5, T6],
        arg6: TModule[T6, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg: "OrderedDict[str, TModule[InType, OutType]]",
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg: "OrderedDict[str, nn.Module]",
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        *args: nn.Module,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        ...

    def __init__(
        self,
        *args,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:  # type: ignore
        nn.Sequential.__init__(self, *args)
        TModule.__init__(self)

        self._unpack_tuple = unpack_tuple
        self._unpack_dict = unpack_dict

    def __call__(self, x: InType) -> OutType:
        return nn.Sequential.__call__(self, x)

    def forward(self, x: InType) -> OutType:
        for module in self:
            if self._unpack_tuple and isinstance(x, tuple):
                x = module(*x)
            elif self._unpack_dict and isinstance(x, dict):
                x = module(**x)
            else:
                x = module(x)
        return x  # type: ignore


def __test_typing() -> None:
    import torch
    from torch import Tensor

    class LayerA(TModule[Tensor, Tensor]):
        def forward(self, x: Tensor) -> Tensor:
            return x * x

    class LayerB(TModule[Tensor, int]):
        def forward(self, x: Tensor) -> int:
            return int(x.sum().item())

    class LayerC(TModule[int, Tensor]):
        def forward(self, x: int) -> Tensor:
            return torch.as_tensor(x)

    x = torch.rand(10)
    xa = LayerA()(x)
    xb = LayerB()(x)

    seq = TSequential(LayerA(), LayerA(), LayerB())
    xab = seq(x)

    seq = LayerA().compose(LayerA()).compose(LayerB())
    xab = seq(x)

    seq = LayerC().compose(LayerA())
    xc = seq(2)

    assert isinstance(xa, Tensor)
    assert isinstance(xb, int)
    assert isinstance(xab, int)
    assert isinstance(xc, Tensor)
