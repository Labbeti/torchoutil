#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Generic, OrderedDict, TypeVar, overload

from torch import nn

InType = TypeVar("InType", covariant=False, contravariant=True)
OutType = TypeVar("OutType", covariant=True, contravariant=False)
InterType1 = TypeVar("InterType1")
InterType2 = TypeVar("InterType2")
InterType3 = TypeVar("InterType3")
InterType4 = TypeVar("InterType4")
InterType5 = TypeVar("InterType5")
InterType6 = TypeVar("InterType6")


class TModule(nn.Module, Generic[InType, OutType]):
    """Typed version of torch.nn.Module. Can specify an input and output type."""

    def __call__(self, *args: InType, **kwargs: InType) -> OutType:
        return super().__call__(*args, **kwargs)

    def forward(self, *args: InType, **kwargs: InType) -> OutType:
        return super().forward(*args, **kwargs)


class TSequential(nn.Sequential, TModule, Generic[InType, OutType]):
    """Typed version of torch.nn.Sequential, designed to work with torchoutil.nn.TModules."""

    @overload
    def __init__(
        self,
        arg0: TModule[InType, OutType],
        /,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TModule[InType, InterType1],
        arg1: TModule[InterType1, OutType],
        /,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TModule[InType, InterType1],
        arg1: TModule[InterType1, InterType2],
        arg2: TModule[InterType2, OutType],
        /,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TModule[InType, InterType1],
        arg1: TModule[InterType1, InterType2],
        arg2: TModule[InterType2, InterType3],
        arg3: TModule[InterType3, OutType],
        /,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TModule[InType, InterType1],
        arg1: TModule[InterType1, InterType2],
        arg2: TModule[InterType2, InterType3],
        arg3: TModule[InterType3, InterType4],
        arg4: TModule[InterType4, OutType],
        /,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TModule[InType, InterType1],
        arg1: TModule[InterType1, InterType2],
        arg2: TModule[InterType2, InterType3],
        arg3: TModule[InterType3, InterType4],
        arg4: TModule[InterType4, InterType5],
        arg5: TModule[InterType5, OutType],
        /,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TModule[InType, InterType1],
        arg1: TModule[InterType1, InterType2],
        arg2: TModule[InterType2, InterType3],
        arg3: TModule[InterType3, InterType4],
        arg4: TModule[InterType4, InterType5],
        arg5: TModule[InterType5, InterType6],
        arg6: TModule[InterType6, OutType],
        /,
    ) -> None:
        ...

    @overload
    def __init__(self, *args: nn.Module) -> None:
        ...

    @overload
    def __init__(self, arg: "OrderedDict[str, nn.Module]") -> None:
        ...

    @overload
    def __init__(self, arg: "OrderedDict[str, TModule[InType, OutType]]") -> None:
        ...

    def __init__(self, *args) -> None:  # type: ignore
        nn.Sequential.__init__(self, *args)
        TModule.__init__(self)

    def __call__(self, *args: InType, **kwargs: InType) -> OutType:
        return nn.Sequential.__call__(self, *args, **kwargs)

    def forward(self, *args: InType, **kwargs: InType) -> OutType:
        return nn.Sequential.forward(self, *args, **kwargs)


def __test_typing() -> None:
    import torch
    from torch import Tensor

    class LayerA(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return x * x

    class LayerB(nn.Module):
        def forward(self, x: Tensor) -> int:
            return int(x.sum().item())

    x = torch.rand(10)
    xa = LayerA()(x)
    xb = LayerB()(x)

    seq = TSequential(LayerA(), LayerA(), LayerB())
    xab = seq(x)

    assert isinstance(xa, Tensor)
    assert isinstance(xb, int)
    assert isinstance(xab, int)
