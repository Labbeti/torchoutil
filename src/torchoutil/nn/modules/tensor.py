#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module versions of tensor functions that do not already exists in PyTorch."""

import copy

from typing import List, Optional, Tuple, Union

import torch

from torch import nn, Tensor
from torch.nn import functional as F
from torch.types import Number

from torchoutil.nn.functional.get import get_device


class Reshape(nn.Module):
    def __init__(self, shape: Tuple[int, ...]) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(self.shape)


class Mean(nn.Module):
    def __init__(self, dim: Optional[int]) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        if self.dim is None:
            return x.mean()
        else:
            return x.mean(dim=self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class Max(nn.Module):
    def __init__(
        self,
        dim: Optional[int],
        return_values: bool = True,
        return_indices: bool = False,
    ) -> None:
        if not return_values and not return_indices:
            raise ValueError(
                f"Invalid combinaison of arguments {return_values=} and {return_indices=}. (at least one of them must be True)"
            )
        super().__init__()
        self.dim = dim
        self.return_values = return_values
        self.return_indices = return_indices

    def forward(self, x: Tensor) -> Union[Tensor, torch.return_types.max]:
        if self.dim is None:
            index = x.argmax()
            values_indices = torch.return_types.max(x[index], index)
        else:
            values_indices = x.max(dim=self.dim)

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
        return f"dim={self.dim}"


class Min(nn.Module):
    def __init__(
        self,
        dim: Optional[int],
        return_values: bool = True,
        return_indices: bool = False,
    ) -> None:
        if not return_values and not return_indices:
            raise ValueError(
                f"Invalid combinaison of arguments {return_values=} and {return_indices=}. (at least one of them must be True)"
            )
        super().__init__()
        self.dim = dim
        self.return_values = return_values
        self.return_indices = return_indices

    def forward(self, x: Tensor) -> Union[Tensor, torch.return_types.min]:
        if self.dim is None:
            index = x.argmin()
            values_indices = torch.return_types.min(x[index], index)
        else:
            values_indices = x.min(dim=self.dim)

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
        return f"dim={self.dim}"


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

    def forward(self, data: Tensor) -> Tensor:
        return F.normalize(data, self.p, self.dim, self.eps)


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(self.dim0, self.dim1)

    def extra_repr(self) -> str:
        return f"{self.dim0}, {self.dim1}"


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
        return f"dim={self.dim}, inplace={self.inplace}"


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
        return f"dim={self.dim}, inplace={self.inplace}"


class TensorTo(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        new_kwargs = copy.copy(self.kwargs)
        new_kwargs.update(kwargs)
        return x.to(**new_kwargs)

    def extra_repr(self) -> str:
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        return kwargs_str


class Permute(nn.Module):
    def __init__(self, *args: int) -> None:
        super().__init__()
        self._dims = tuple(args)

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(self._dims)

    def extra_repr(self) -> str:
        return ", ".join(map(str, self._dims))


class Div(nn.Module):
    def __init__(
        self,
        divisor: Union[float, Tensor],
        rounding_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.divisor = divisor
        self.rounding_mode = rounding_mode

    def forward(self, x: Tensor) -> Tensor:
        return x.div(self.divisor, rounding_mode=self.rounding_mode)


class ToList(nn.Module):
    def forward(self, x: Tensor) -> List:
        return x.tolist()


class AsTensor(nn.Module):
    def __init__(
        self,
        device: Union[str, torch.device, None] = None,
        dtype: Union[torch.dtype, None] = None,
    ) -> None:
        device = get_device(device)
        super().__init__()
        self.device = device
        self.dtype = dtype

    def forward(self, x: Union[List, Number]) -> Tensor:
        return torch.as_tensor(x, dtype=self.dtype, device=self.device)
