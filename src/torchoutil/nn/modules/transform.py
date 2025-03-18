#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Tensor, nn

from torchoutil.core.make import DeviceLike, DTypeLike, GeneratorLike
from torchoutil.extras.numpy import np
from torchoutil.nn.functional.transform import (
    PadCropAlign,
    PadMode,
    PadValue,
    SqueezeMode,
    T_BuiltinScalar,
    as_tensor,
    flatten,
    identity,
    move_to_rec,
    pad_and_crop_dim,
    repeat_interleave_nd,
    resample_nearest_freqs,
    resample_nearest_rates,
    resample_nearest_steps,
    shuffled,
    squeeze,
    to_item,
    transform_drop,
    unsqueeze,
    view_as_complex,
    view_as_real,
)
from torchoutil.nn.modules.module import Module
from torchoutil.pyoutil.collections import dump_dict
from torchoutil.pyoutil.typing import BuiltinScalar, SizedIterable
from torchoutil.types._typing import ComplexFloatingTensor, ScalarLike, T_TensorOrArray

T = TypeVar("T")


class AsTensor(Module):
    """
    Module version of :func:`~to.as_tensor`.
    """

    def __init__(
        self,
        *,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, x: Any) -> Tensor:
        return as_tensor(x, dtype=self.dtype, device=self.device)

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                dtype=self.dtype,
                device=self.device,
            ),
            ignore_lst=(None,),
        )


class Flatten(Module):
    def __init__(self, start_dim: int = 0, end_dim: Optional[int] = None) -> None:
        """
        For more information, see :func:`~torchoutil.nn.functional.transform.flatten`.
        """
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    @overload
    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        ...

    @overload
    def forward(self, x: Union[np.ndarray, np.generic]) -> np.ndarray:
        ...

    @overload
    def forward(self, x: T_BuiltinScalar) -> List[T_BuiltinScalar]:
        ...

    @overload
    def forward(self, x: Iterable[T_BuiltinScalar]) -> List[T_BuiltinScalar]:  # type: ignore
        ...

    def forward(self, x: Any) -> Any:
        return flatten(
            x,
            start_dim=self.start_dim,
            end_dim=self.end_dim,
        )

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                start_dim=self.start_dim,
                end_dim=self.end_dim,
            )
        )


class Identity(Module):
    def __init__(self, *args, **kwargs) -> None:
        """Identity class placeholder.

        Unlike torch.nn.Identity which only supports Tensor typing, its type output is the same than its input type.
        """
        super().__init__()

    def forward(self, x: T) -> T:
        return identity(x)


class MoveToRec(Module):
    """
    Module version of :func:`~torchoutil.move_to_rec`.
    """

    def __init__(
        self,
        predicate: Optional[Callable[[Union[Tensor, nn.Module]], bool]] = None,
    ) -> None:
        super().__init__()
        self.predicate = predicate

    def forward(self, x: Any) -> Any:
        return move_to_rec(x, predicate=self.predicate)


class PadAndCropDim(Module):
    def __init__(
        self,
        target_length: int,
        align: PadCropAlign = "left",
        pad_value: PadValue = 0.0,
        dim: int = -1,
        mode: PadMode = "constant",
        generator: GeneratorLike = None,
    ) -> None:
        """
        For more information, see :func:`~torchoutil.nn.functional.transform.pad_and_crop_dim`.
        """
        super().__init__()
        self.target_length = target_length
        self.align: PadCropAlign = align
        self.pad_value = pad_value
        self.dim = dim
        self.mode: PadMode = mode
        self.generator: GeneratorLike = generator

    def forward(self, x: Tensor) -> Tensor:
        return pad_and_crop_dim(
            x,
            self.target_length,
            align=self.align,
            pad_value=self.pad_value,
            dim=self.dim,
            mode=self.mode,
            generator=self.generator,
        )

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                target_length=self.target_length,
                align=self.align,
                pad_value=self.pad_value,
                dim=self.dim,
                mode=self.mode,
            )
        )


class RepeatInterleaveNd(Module):
    """
    For more information, see :func:`~to.repeat_interleave_nd`.
    """

    def __init__(self, repeats: int, dim: int) -> None:
        super().__init__()
        self.repeats = repeats
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return repeat_interleave_nd(x, self.repeats, self.dim)

    def extra_repr(self) -> str:
        return dump_dict(dict(repeats=self.repeats, dim=self.dim))


class ResampleNearestRates(Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.transform.resample_nearest_rates`.
    """

    def __init__(
        self,
        rates: Union[float, Iterable[float]],
        dims: Union[int, Iterable[int]] = -1,
        round_fn: Callable[[Tensor], Tensor] = torch.floor,
    ) -> None:
        super().__init__()
        self.rates = rates
        self.dims = dims
        self.round_fn = round_fn

    def forward(self, x: Tensor) -> Tensor:
        return resample_nearest_rates(
            x,
            rates=self.rates,
            dims=self.dims,
            round_fn=self.round_fn,
        )

    def extra_repr(self) -> str:
        return dump_dict(dict(rates=self.rates, dims=self.dims))


class ResampleNearestFreqs(Module):
    def __init__(
        self,
        orig_freq: int,
        new_freq: int,
        dims: Union[int, Iterable[int]] = -1,
        round_fn: Callable[[Tensor], Tensor] = torch.floor,
    ) -> None:
        """
        For more information, see :func:`~torchoutil.nn.functional.transform.resample_nearest_freqs`.
        """
        super().__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.dims = dims
        self.round_fn = round_fn

    def forward(self, x: Tensor) -> Tensor:
        return resample_nearest_freqs(
            x,
            orig_freq=self.orig_freq,
            new_freq=self.new_freq,
            dims=self.dims,
            round_fn=self.round_fn,
        )

    def extra_repr(self) -> str:
        return dump_dict(
            dict(orig_freq=self.orig_freq, new_freq=self.new_freq, dims=self.dims)
        )


class ResampleNearestSteps(Module):
    def __init__(
        self,
        steps: Union[float, Iterable[float]],
        dims: Union[int, Iterable[int]] = -1,
        round_fn: Callable[[Tensor], Tensor] = torch.floor,
    ) -> None:
        """
        For more information, see :func:`~torchoutil.nn.functional.transform.resample_nearest_steps`.
        """
        super().__init__()
        self.steps = steps
        self.dims = dims
        self.round_fn = round_fn

    def forward(self, x: Tensor) -> Tensor:
        return resample_nearest_steps(
            x,
            steps=self.steps,
            dims=self.dims,
            round_fn=self.round_fn,
        )

    def extra_repr(self) -> str:
        return dump_dict(dict(steps=self.steps, dims=self.dims))


class Squeeze(Module):
    """
    Module version of :func:`~torchoutil.squeeze`.
    """

    def __init__(
        self,
        dim: Union[int, Iterable[int], None] = None,
        mode: SqueezeMode = "view_if_possible",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.mode: SqueezeMode = mode

    def forward(self, x: Tensor) -> Tensor:
        return squeeze(x, self.dim, self.mode)

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                dim=self.dim,
                mode=self.mode,
            ),
        )


class Shuffled(Module):
    def __init__(
        self,
        dims: Union[int, Iterable[int]],
        generator: GeneratorLike,
    ) -> None:
        """
        For more information, see :func:`~torchoutil.nn.functional.transform.shuffled`.
        """
        super().__init__()
        self.dims = dims
        self.generator: GeneratorLike = generator

    def forward(self, x: Tensor) -> Tensor:
        return shuffled(x, dims=self.dims, generator=self.generator)

    def extra_repr(self) -> str:
        return dump_dict(dict(dims=self.dims))


class ToItem(Module):
    """
    Module version of :func:`~torchoutil.to_item`.
    """

    def forward(
        self,
        x: Union[ScalarLike, Tensor, np.ndarray, SizedIterable],
    ) -> BuiltinScalar:
        return to_item(x)  # type: ignore


class TransformDrop(Module[T, T]):
    def __init__(
        self,
        transform: Callable[[T], T],
        p: float,
        generator: GeneratorLike = None,
    ) -> None:
        """
        For more information, see :func:`~torchoutil.nn.functional.transform.transform_drop`.
        """
        super().__init__()
        self.transform = transform
        self.p = p
        self.generator: GeneratorLike = generator

    def forward(self, x: T) -> T:
        return transform_drop(
            transform=self.transform,
            x=x,
            p=self.p,
            generator=self.generator,
        )

    def extra_repr(self) -> str:
        return dump_dict(dict(p=self.p))


class Unsqueeze(Module):
    """
    Module version of :func:`~torchoutil.unsqueeze`.
    """

    def __init__(
        self, dim: Union[int, Iterable[int]], mode: SqueezeMode = "view_if_possible"
    ) -> None:
        super().__init__()
        self.dim = dim
        self.mode: SqueezeMode = mode

    def forward(self, x: T_TensorOrArray) -> T_TensorOrArray:
        return unsqueeze(x, self.dim, self.mode)

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                dim=self.dim,
                mode=self.mode,
            ),
        )


class ViewAsReal(Module):
    """
    Module version of :func:`~torchoutil.to_item`.
    """

    def forward(
        self, x: Union[Tensor, np.ndarray, complex]
    ) -> Union[Tensor, np.ndarray, Tuple[float, float]]:
        return view_as_real(x)


class ViewAsComplex(Module):
    """
    Module version of :func:`~torchoutil.to_item`.
    """

    def forward(
        self, x: Union[Tensor, np.ndarray, Tuple[float, float]]
    ) -> Union[ComplexFloatingTensor, np.ndarray, complex]:
        return view_as_complex(x)


# Aliases
ToTensor = AsTensor
