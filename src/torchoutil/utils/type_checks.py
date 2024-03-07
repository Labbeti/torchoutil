#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, Iterable, Mapping, Union

from typing_extensions import TypeGuard


def is_dict_str(x: Any) -> TypeGuard[Dict[str, Any]]:
    return isinstance(x, dict) and all(isinstance(key, str) for key in x.keys())


def is_iterable_int(x: Any) -> TypeGuard[Iterable[int]]:
    return isinstance(x, Iterable) and all(isinstance(xi, int) for xi in x)


def is_iterable_str(x: Any) -> TypeGuard[Iterable[str]]:
    return not isinstance(x, str) and (
        isinstance(x, Iterable) and all(isinstance(xi, str) for xi in x)
    )


def is_iterable_bytes_list(x: Any) -> TypeGuard[Iterable[Union[bytes, list]]]:
    return isinstance(x, Iterable) and all(isinstance(xi, (bytes, list)) for xi in x)


def is_iterable_iterable_int(x: Any) -> TypeGuard[Iterable[Iterable[int]]]:
    return (
        isinstance(x, Iterable)
        and all(isinstance(xi, Iterable) for xi in x)
        and all(isinstance(xij, int) for xi in x for xij in xi)
    )


def is_mapping_str_any(x: Any) -> TypeGuard[Mapping[str, Any]]:
    return isinstance(x, Mapping) and all(isinstance(key, str) for key in x.keys())
