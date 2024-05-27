#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module versions of tensor functions that do not already exists in PyTorch."""

from typing import List, Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.types import Device, Number

from torchoutil.nn.functional.get import get_device
from torchoutil.utils import return_types
from torchoutil.utils.collections import dump_dict


class Reshape(nn.Module):
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


class Mean(nn.Module):
    def __init__(self, dim: Union[int, None], keepdim: bool = False) -> None:
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
            ),
            ignore_lst=(None,),
        )


class Max(nn.Module):
    def __init__(
        self,
        dim: Optional[int],
        return_values: bool = True,
        return_indices: Optional[bool] = None,
        keepdim: bool = False,
    ) -> None:
        if return_indices is None:
            return_indices = dim is not None
        if not return_values and not return_indices:
            raise ValueError(
                f"Invalid combinaison of arguments {return_values=} and {return_indices=}. (at least one of them must be True)"
            )
        if dim is None and keepdim:
            raise ValueError(
                f"Invalid combinaison of arguments {dim=} and {keepdim=}. (expected dim is not None or keepdim=False)"
            )

        super().__init__()
        self.dim = dim
        self.return_values = return_values
        self.return_indices = return_indices
        self.keepdim = keepdim

    def forward(self, x: Tensor) -> Union[Tensor, return_types.max]:
        if self.dim is None:
            index = x.argmax()
            values_indices = return_types.max(x.flatten()[index], index)
        else:
            values_indices = x.max(dim=self.dim, keepdim=self.keepdim)

        if self.return_values and self.return_indices:
            return values_indices
        elif self.return_values:
            return values_indices.values
        elif self.return_indices:
            return values_indices.indices
        else:
            raise ValueError(
                f"Invalid combinaison of arguments {self.return_values=} and {self.return_indices=}. (at least one of them must be True)"
            )

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                dim=self.dim,
                return_values=self.return_values,
                return_indices=self.return_indices,
                keepdim=self.keepdim,
            ),
        )


class Min(nn.Module):
    def __init__(
        self,
        dim: Optional[int],
        return_values: bool = True,
        return_indices: Optional[bool] = None,
        keepdim: bool = False,
    ) -> None:
        if return_indices is None:
            return_indices = dim is not None
        if not return_values and not return_indices:
            raise ValueError(
                f"Invalid combinaison of arguments {return_values=} and {return_indices=}. (at least one of them must be True)"
            )
        if dim is None and keepdim:
            raise ValueError(
                f"Invalid combinaison of arguments {dim=} and {keepdim=}. (expected dim is not None or keepdim=False)"
            )

        super().__init__()
        self.dim = dim
        self.return_values = return_values
        self.return_indices = return_indices
        self.keepdim = keepdim

    def forward(self, x: Tensor) -> Union[Tensor, return_types.min]:
        if self.dim is None:
            index = x.argmin()
            values_indices = return_types.min(x.flatten()[index], index)
        else:
            values_indices = x.min(dim=self.dim, keepdim=self.keepdim)

        if self.return_values and self.return_indices:
            return values_indices
        elif self.return_values:
            return values_indices.values
        elif self.return_indices:
            return values_indices.indices
        else:
            raise ValueError(
                f"Invalid combinaison of arguments {self.return_values=} and {self.return_indices=}. (at least one of them must be True)"
            )

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


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(self.dim0, self.dim1)

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                dim0=self.dim0,
                dim1=self.dim1,
            ),
            fmt="{value}",
        )


class Squeeze(nn.Module):
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


class Unsqueeze(nn.Module):
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


class TensorTo(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x: Tensor) -> Tensor:
        return x.to(**self.kwargs)

    def extra_repr(self) -> str:
        return dump_dict(dict(self.kwargs))


class Permute(nn.Module):
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


class ToList(nn.Module):
    def forward(self, x: Tensor) -> List:
        return x.tolist()


class ToItem(nn.Module):
    def forward(self, x: Tensor) -> Number:
        return x.item()


class AsTensor(nn.Module):
    def __init__(
        self,
        *,
        device: Device = None,
        dtype: Union[torch.dtype, None] = None,
    ) -> None:
        device = get_device(device)
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, x: Union[List, Number, Tensor]) -> Tensor:
        return torch.as_tensor(x, dtype=self.dtype, device=self.device)  # type: ignore

    def extra_repr(self) -> str:
        return dump_dict(
            dict(
                device=self.device,
                dtype=self.dtype,
            ),
            ignore_lst=(None,),
        )


class Abs(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.abs()


class Angle(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.angle()


class Real(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.real


class Imag(nn.Module):
    def __init__(self, return_zeros: bool = False) -> None:
        """Return the imaginary part of a complex tensor.

        Args:
            return_zeros: If the input is not a complex tensor and return_zeros=True, the module will return a tensor containing zeros.
        """
        super().__init__()
        self.return_zeros = return_zeros

    def forward(self, x: Tensor) -> Tensor:
        if self.return_zeros and not x.is_complex():
            return torch.zeros_like(x)
        else:
            return x.imag


class Pow(nn.Module):
    def __init__(self, exponent: Union[Number, Tensor]) -> None:
        super().__init__()
        self.exponent = exponent

    def forward(self, x: Tensor) -> Tensor:
        return x.pow(self.exponent)

    def extra_repr(self) -> str:
        return dump_dict(dict(exponent=self.exponent))


class FFT(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.fft.fft(x)


class IFFT(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.fft.ifft(x)


class Log(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.log()


class Log10(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.log10()


class Log2(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.log2()


class Repeat(nn.Module):
    def __init__(self, *repeats: int) -> None:
        super().__init__()
        self.repeats = repeats

    def forward(self, x: Tensor) -> Tensor:
        return x.repeat(self.repeats)

    def extra_repr(self) -> str:
        return dump_dict(dict(repeats=self.repeats))


class RepeatInterleave(nn.Module):
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
