#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, Optional, Union

_TRUE_VALUES = ("True", "t", "yes", "y", "1")
_FALSE_VALUES = ("False", "f", "no", "n", "0")
_NONE_VALUES = ("None", "null")


def str_to_bool(
    x: str,
    *,
    case_sensitive: bool = False,
    true_values: Union[str, Iterable[str]] = _TRUE_VALUES,
    false_values: Union[str, Iterable[str]] = _FALSE_VALUES,
) -> bool:
    """Convert string values to bool. Intended for argparse boolean arguments.

    - True values: 'True', 'T', 'yes', 'y', '1'.
    - False values: 'False', 'F', 'no', 'n', '0'.
    - Other raises ValueError.
    """
    if isinstance(true_values, str):
        true_values = [true_values]
    else:
        true_values = list(true_values)

    if isinstance(false_values, str):
        false_values = [false_values]
    else:
        false_values = list(false_values)

    if _str_in(x, true_values, case_sensitive):
        return True
    elif _str_in(x, false_values, case_sensitive):
        return False
    else:
        values = tuple(true_values + false_values)
        raise ValueError(f"Invalid argument '{x}'. (expected one of {values})")


def str_to_optional_int(
    x: str,
    *,
    case_sensitive: bool = False,
    none_values: Union[str, Iterable[str]] = _NONE_VALUES,
) -> Optional[int]:
    if isinstance(none_values, str):
        none_values = [none_values]
    else:
        none_values = list(none_values)

    if _str_in(x, _NONE_VALUES, case_sensitive):
        return None
    else:
        return int(x)


def str_to_optional_str(x: str, *, case_sensitive: bool = False) -> Optional[str]:
    if _str_in(x, _NONE_VALUES, case_sensitive):
        return None
    else:
        return x


def _str_in(x: str, values: Iterable[str], case_sensitive: bool) -> bool:
    if case_sensitive:
        return x in values
    else:
        return x.lower() in map(str.lower, values)
