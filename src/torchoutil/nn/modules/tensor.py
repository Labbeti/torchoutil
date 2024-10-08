#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module versions of tensor functions that do not already exists in PyTorch."""

from typing import Any, List, Optional, Sequence, Tuple, Union, overload

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.types import Device, Number

from torchoutil.pyoutil.collections import dump_dict
from torchoutil.types import BuiltinNumber
from torchoutil.utils import return_types


class Abs(nn.Module):
    """
    Module version of :func:`~torch.abs`.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.abs()


class Angle(nn.Module):
    """
    Module version of :func:`~torch.angle`.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.angle()


class AsTensor(nn.Module):
    """
    Module version of :func:`~torch.as_tensor`.
    """

    def __init__(
        self,
        *,
        device: Device = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, x: Any) -> Tensor:
        return torch.as_tensor(x, dtype=self.dtype, device=self.device)

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                dtype=self.dtype,
                device=self.device,
            ),
            ignore_lst=(None,),
        )


class Exp(nn.Module):
    """
    Module version of :func:`~torch.exp`.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.exp()


class Exp2(nn.Module):
    """
    Module version of :func:`~torch.exp2`.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.exp2()


class FFT(nn.Module):
    """
    Module version of :func:`~torch.fft.fft`.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.fft.fft(x)


class IFFT(nn.Module):
    """
    Module version of :func:`~torch.fft.ifft`.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.fft.ifft(x)


class Imag(nn.Module):
    """
    Module version of :func:`~torch.Tensor.imag`.
    """

    def __init__(self, *, return_zeros: bool = False) -> None:
        """Return the imaginary part of a complex tensor.

        Args:
            return_zeros:
                If True and the input is not a complex tensor, the module will return a tensor of same shape containing zeros.
                If False and the input is not a complex tensor, raises the default RuntimError of PyTorch.
        """
        super().__init__()
        self.return_zeros = return_zeros

    def forward(self, x: Tensor) -> Tensor:
        if self.return_zeros and not x.is_complex():
            return torch.zeros_like(x)
        else:
            return x.imag


class Interpolate(nn.Module):
    """
    Module version of :func:`~torch.nn.functional.interpolate`.
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, ...], None] = None,
        scale_factor: Union[float, Tuple[float, ...], None] = None,
        mode: str = "nearest",
        align_corners: Optional[bool] = None,
        recompute_scale_factor: Optional[bool] = None,
        antialias: bool = False,
    ) -> None:
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias

    def forward(self, x: Tensor) -> Tensor:
        return F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
            antialias=self.antialias,
        )


class Log(nn.Module):
    """
    Module version of :func:`~torch.log`.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.log()


class Log10(nn.Module):
    """
    Module version of :func:`~torch.log10`.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.log10()


class Log2(nn.Module):
    """
    Module version of :func:`~torch.log2`.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.log2()


class Max(nn.Module):
    """
    Module version of :func:`~torch.max`.
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        return_values: bool = True,
        return_indices: Optional[bool] = None,
        keepdim: bool = False,
    ) -> None:
        if return_indices is None:
            return_indices = dim is not None
        if not return_values and not return_indices:
            msg = f"Invalid combinaison of arguments {return_values=} and {return_indices=}. (at least one of them must be True)"
            raise ValueError(msg)
        if dim is None and keepdim:
            msg = f"Invalid combinaison of arguments {dim=} and {keepdim=}. (expected dim is not None or keepdim=False)"
            raise ValueError(msg)

        super().__init__()
        self.dim = dim
        self.return_values = return_values
        self.return_indices = return_indices
        self.keepdim = keepdim

    def forward(self, x: Tensor) -> Union[Tensor, return_types.max]:
        if self.dim is None:
            index = x.argmax()
            values_indices = return_types.max([x.flatten()[index], index])
        else:
            values_indices = x.max(dim=self.dim, keepdim=self.keepdim)

        if self.return_values and self.return_indices:
            return values_indices  # type: ignore
        elif self.return_values:
            return values_indices.values
        elif self.return_indices:
            return values_indices.indices
        else:
            msg = f"Invalid combinaison of arguments {self.return_values=} and {self.return_indices=}. (at least one of them must be True)"
            raise ValueError(msg)

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                dim=self.dim,
                return_values=self.return_values,
                return_indices=self.return_indices,
                keepdim=self.keepdim,
            ),
        )


class Mean(nn.Module):
    """
    Module version of :func:`~torch.mean`.
    """

    def __init__(self, dim: Optional[int] = None, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: Tensor) -> Tensor:
        if self.dim is None:
            return x.mean()
        else:
            return x.mean(dim=self.dim, keepdim=self.keepdim)

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                dim=self.dim,
                keepdim=self.keepdim,
            ),
            ignore_lst=(None,),
        )


class Min(nn.Module):
    """
    Module version of :func:`~torch.min`.
    """

    def __init__(
        self,
        dim: Optional[int] = None,
        return_values: bool = True,
        return_indices: Optional[bool] = None,
        keepdim: bool = False,
    ) -> None:
        if return_indices is None:
            return_indices = dim is not None
        if not return_values and not return_indices:
            msg = f"Invalid combinaison of arguments {return_values=} and {return_indices=}. (at least one of them must be True)"
            raise ValueError(msg)
        if dim is None and keepdim:
            msg = f"Invalid combinaison of arguments {dim=} and {keepdim=}. (expected dim is not None or keepdim=False)"
            raise ValueError(msg)

        super().__init__()
        self.dim = dim
        self.return_values = return_values
        self.return_indices = return_indices
        self.keepdim = keepdim

    def forward(self, x: Tensor) -> Union[Tensor, return_types.min]:
        if self.dim is None:
            index = x.argmin()
            values_indices = return_types.min([x.flatten()[index], index])
        else:
            values_indices = x.min(dim=self.dim, keepdim=self.keepdim)

        if self.return_values and self.return_indices:
            return values_indices  # type: ignore
        elif self.return_values:
            return values_indices.values
        elif self.return_indices:
            return values_indices.indices
        else:
            msg = f"Invalid combinaison of arguments {self.return_values=} and {self.return_indices=}. (at least one of them must be True)"
            raise ValueError(msg)

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                dim=self.dim,
                return_values=self.return_values,
                return_indices=self.return_indices,
                keepdim=self.keepdim,
            ),
        )


class Normalize(nn.Module):
    """
    Module version of :func:`~torch.nn.functional.normalize`.
    """

    def __init__(
        self,
        p: float = 2.0,
        dim: int = 1,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.p = p
        self.dim = dim
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, self.p, self.dim, self.eps)

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                p=self.p,
                dim=self.dim,
                eps=self.eps,
            )
        )


class Permute(nn.Module):
    """
    Module version of :func:`~torch.permute`.
    """

    def __init__(self, *args: int) -> None:
        super().__init__()
        self.dims = tuple(args)

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(self.dims)

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                dims=self.dims,
            ),
            fmt="{value}",
        )


class Pow(nn.Module):
    """
    Module version of :func:`~torch.Tensor.pow`.
    """

    def __init__(self, exponent: Union[Number, Tensor]) -> None:
        super().__init__()
        self.exponent = exponent

    def forward(self, x: Tensor) -> Tensor:
        return x.pow(self.exponent)

    def extra_repr(self) -> str:
        return dump_dict(dict(exponent=self.exponent))


class Real(nn.Module):
    """
    Module version of :func:`~torch.Tensor.real`.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.real


class Repeat(nn.Module):
    """
    Module version of :func:`~torch.repeat`.
    """

    def __init__(self, *repeats: int) -> None:
        super().__init__()
        self.repeats = repeats

    def forward(self, x: Tensor) -> Tensor:
        return x.repeat(self.repeats)

    def extra_repr(self) -> str:
        return dump_dict(dict(repeats=self.repeats))


class RepeatInterleave(nn.Module):
    """
    Module version of :func:`~torch.repeat_interleave`.
    """

    def __init__(
        self,
        repeats: Union[int, Tensor],
        dim: int,
        output_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.repeats = repeats
        self.dim = dim
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        return x.repeat_interleave(self.repeats, self.dim, output_size=self.output_size)

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                repeats=self.repeats,
                dim=self.dim,
                output_size=self.output_size,
            ),
            ignore_lst=(None,),
        )


class Reshape(nn.Module):
    """
    Module version of :func:`~torch.reshape`.
    """

    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(self.shape)

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                shape=self.shape,
            ),
        )


class Squeeze(nn.Module):
    """
    Module version of :func:`~torch.squeeze`.
    """

    def __init__(self, dim: Optional[int] = None, inplace: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if not self.inplace:
            if self.dim is None:
                return x.squeeze()
            else:
                return x.squeeze(self.dim)
        else:
            if self.dim is None:
                return x.squeeze_()
            else:
                return x.squeeze_(self.dim)

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                dim=self.dim,
                inplace=self.inplace,
            ),
        )


class TensorTo(nn.Module):
    """
    Module version of :func:`~torch.Tensor.to`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x: Tensor) -> Tensor:
        return x.to(**self.kwargs)

    def extra_repr(self) -> str:
        return dump_dict(dict(self.kwargs))


class ToItem(nn.Module):
    """
    Module version of :func:`~torch.Tensor.item`.
    """

    def forward(self, x: Tensor) -> BuiltinNumber:
        return x.item()


class ToList(nn.Module):
    """
    Module version of :func:`~torch.Tensor.tolist`.
    """

    def forward(self, x: Tensor) -> List:
        return x.tolist()


class Topk(nn.Module):
    """
    Module version of :func:`~torch.Tensor.topk`.
    """

    def __init__(
        self,
        k: int,
        dim: int = -1,
        largest: bool = True,
        sorted: bool = True,
        return_values: bool = True,
        return_indices: bool = True,
    ) -> None:
        if not return_values and not return_indices:
            msg = f"Invalid combinaison of arguments {return_values=} and {return_indices=}. (at least one of them must be True)"
            raise ValueError(msg)

        super().__init__()
        self.k = k
        self.dim = dim
        self.largest = largest
        self.sorted = sorted
        self.return_values = return_values
        self.return_indices = return_indices

    def forward(self, x: Tensor) -> Union[Tensor, return_types.topk]:
        values_indices = x.topk(
            k=self.k,
            dim=self.dim,
            largest=self.largest,
            sorted=self.sorted,
        )
        if self.return_values and self.return_indices:
            return values_indices
        elif self.return_values:
            return values_indices.values
        elif self.return_indices:
            return values_indices.indices
        else:
            msg = f"Invalid combinaison of arguments {self.return_values=} and {self.return_indices=}. (at least one of them must be True)"
            raise ValueError(msg)

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                k=self.k,
                dim=self.dim,
                largest=self.largest,
                sorted=self.sorted,
                return_values=self.return_values,
                return_indices=self.return_indices,
            ),
        )


class Transpose(nn.Module):
    """
    Module version of :func:`~torch.transpose`.
    """

    def __init__(self, dim0: int, dim1: int, copy: bool = False) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        self.copy = copy

    def forward(self, x: Tensor) -> Tensor:
        if self.copy:
            return torch.transpose_copy(x, self.dim0, self.dim1)
        else:
            return torch.transpose(x, self.dim0, self.dim1)

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                dim0=self.dim0,
                dim1=self.dim1,
            ),
            fmt="{value}",
        )


class Unsqueeze(nn.Module):
    """
    Module version of :func:`~torch.Tensor.unsqueeze`.
    """

    def __init__(self, dim: int, inplace: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if not self.inplace:
            return x.unsqueeze(self.dim)
        else:
            return x.unsqueeze_(self.dim)

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                dim=self.dim,
                inplace=self.inplace,
            ),
        )


class View(nn.Module):
    @overload
    def __init__(self, dtype: torch.dtype) -> None:
        ...

    @overload
    def __init__(self, size: Sequence[int]) -> None:
        ...

    @overload
    def __init__(self, *size: int) -> None:
        ...

    def __init__(self, *args) -> None:
        super().__init__()
        self.args = args

    def forward(self, x: Tensor) -> Tensor:
        return x.view(*self.args)
