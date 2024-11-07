#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Generator, Tensor, nn

from torchoutil.core.get import DeviceLike, DTypeLike
from torchoutil.extras.numpy import np
from torchoutil.nn.functional.transform import (
    PadCropAlign,
    PadMode,
    PadValue,
    T_BuiltinScalar,
    flatten,
    identity,
    pad_and_crop_dim,
    repeat_interleave_nd,
    resample_nearest_freqs,
    resample_nearest_rates,
    resample_nearest_steps,
    shuffled,
    to_tensor,
    transform_drop,
)
from torchoutil.pyoutil.collections import dump_dict

T = TypeVar("T")


class RepeatInterleaveNd(nn.Module):
    """
    For more information, see :func:`~torchoutil.nn.functional.transform.repeat_interleave_nd`.
    """

    def __init__(self, repeats: int, dim: int) -> None:
        super().__init__()
        self.repeats = repeats
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return repeat_interleave_nd(x, self.repeats, self.dim)

    def extra_repr(self) -> str:
        return dump_dict(dict(repeats=self.repeats, dim=self.dim))


class ResampleNearestRates(nn.Module):
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


class ResampleNearestFreqs(nn.Module):
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


class ResampleNearestSteps(nn.Module):
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


class TransformDrop(Generic[T], nn.Module):
    def __init__(
        self,
        transform: Callable[[T], T],
        p: float,
        generator: Union[int, Generator, None] = None,
    ) -> None:
        """
        For more information, see :func:`~torchoutil.nn.functional.transform.transform_drop`.
        """
        super().__init__()
        self.transform = transform
        self.p = p
        self.generator = generator

    def forward(self, x: T) -> T:
        return transform_drop(
            transform=self.transform,
            x=x,
            p=self.p,
            generator=self.generator,
        )

    def extra_repr(self) -> str:
        return dump_dict(dict(p=self.p))


class PadAndCropDim(nn.Module):
    def __init__(
        self,
        target_length: int,
        align: PadCropAlign = "left",
        pad_value: PadValue = 0.0,
        dim: int = -1,
        mode: PadMode = "constant",
        generator: Union[int, Generator, None] = None,
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
        self.generator = generator

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


class Shuffled(nn.Module):
    def __init__(
        self,
        dims: Union[int, Iterable[int]],
        generator: Union[int, Generator, None],
    ) -> None:
        """
        For more information, see :func:`~torchoutil.nn.functional.transform.shuffled`.
        """
        super().__init__()
        self.dims = dims
        self.generator = generator

    def forward(self, x: Tensor) -> Tensor:
        return shuffled(x, dims=self.dims, generator=self.generator)

    def extra_repr(self) -> str:
        return dump_dict(dict(dims=self.dims))


class Flatten(nn.Module):
    def __init__(self, start_dim: int = 0, end_dim: Optional[int] = None) -> None:
        """
        For more information, see :func:`~torchoutil.nn.functional.transform.flatten`.
        """
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    @overload
    def forward(self, x: Tensor) -> Tensor:
        ...

    @overload
    def forward(self, x: Union[np.ndarray, np.generic]) -> np.ndarray:
        ...

    @overload
    def forward(self, x: T_BuiltinScalar) -> List[T_BuiltinScalar]:
        ...

    @overload
    def forward(self, x: Iterable[T_BuiltinScalar]) -> List[T_BuiltinScalar]:
        ...

    @overload
    def forward(self, x: Any) -> List[Any]:
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


class Identity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        """Identity class placeholder.

        Unlike torch.nn.Identity which only supports Tensor typing, its type output is the same than its input type.
        """
        super().__init__()

    def forward(self, x: T) -> T:
        return identity(x)


class ToTensor(nn.Module):
    def __init__(
        self,
        *,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """
        For more information, see :func:`~torchoutil.nn.functional.transform.to_tensor`.
        """
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, x: Any) -> Tensor:
        return to_tensor(x, dtype=self.dtype, device=self.device)

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                dtype=self.dtype,
                device=self.device,
            ),
            ignore_lst=(None,),
        )
