#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module used when numpy is not installed."""

from typing import Any

from typing_extensions import Never


class generic:
    def __getattribute__(self, name: str) -> Any:
        return None


class number(generic):
    ...


class bool_(generic):
    ...


class dtype:
    def __getattribute__(self, name: str) -> Any:
        return None


class ndarray:
    def __getattribute__(self, name: str) -> Any:
        return None


def array(x: Any, *args, **kwargs) -> Never:
    raise NotImplementedError(
        "Cannot call function 'array'. Please install numpy first."
    )


def asarray(x: Any, *args, **kwargs) -> Never:
    raise NotImplementedError(
        "Cannot call function 'asarray'. Please install numpy first."
    )
