#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module used when numpy is not installed."""

from typing import Any


class generic:
    def __getattr__(self, name: str) -> Any:
        return self


class number(generic):
    ...


class bool_(generic):
    ...


class dtype:
    def __getattr__(self, name: str) -> Any:
        return self


class ndarray:
    def __getattr__(self, name: str) -> Any:
        return self

    def __getitem__(self, *args) -> Any:
        return self


def array(x: Any, *args, **kwargs):
    msg = "Cannot call function 'array' because optional dependancy 'numpy' is not installed. Please install it using 'pip install torchoutil[extras]'"
    raise NotImplementedError(msg)


def asarray(x: Any, *args, **kwargs):
    msg = "Cannot call function 'asarray' because optional dependancy 'numpy' is not installed. Please install it using 'pip install torchoutil[extras]'"
    raise NotImplementedError(msg)


def iscomplexobj(x: Any):
    msg = "Cannot call function 'iscomplexobj' because optional dependancy 'numpy' is not installed. Please install it using 'pip install torchoutil[extras]'"
    raise NotImplementedError(msg)


class complex64(dtype):
    ...


class complex128(dtype):
    ...


class complex256(dtype):
    ...


class float16(dtype):
    ...


class float32(dtype):
    ...


class float64(dtype):
    ...


class float128(dtype):
    ...


class floating(dtype):
    ...


class int16(dtype):
    ...


class int32(dtype):
    ...


class int64(dtype):
    ...


class uint8(dtype):
    ...
