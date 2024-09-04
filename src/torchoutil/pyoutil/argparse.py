#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, List, Optional, Union

DEFAULT_TRUE_VALUES = ("True", "t", "yes", "y", "1")
DEFAULT_FALSE_VALUES = ("False", "f", "no", "n", "0")
DEFAULT_NONE_VALUES = ("None", "null")


def str_to_bool(
    x: str,
    *,
    case_sensitive: bool = False,
    true_values: Union[str, Iterable[str]] = DEFAULT_TRUE_VALUES,
    false_values: Union[str, Iterable[str]] = DEFAULT_FALSE_VALUES,
) -> bool:
    """Convert string values to bool. Intended for argparse boolean arguments.

    - True values: 'True', 'T', 'yes', 'y', '1'.
    - False values: 'False', 'F', 'no', 'n', '0'.
    - Other raises ValueError.
    """
    true_values = _sanitize_values(true_values)
    if _str_in(x, true_values, case_sensitive):
        return True

    false_values = _sanitize_values(false_values)
    if _str_in(x, false_values, case_sensitive):
        return False

    values = tuple(true_values + false_values)
    raise ValueError(f"Invalid argument '{x}'. (expected one of {values})")


def str_to_optional_int(
    x: str,
    *,
    case_sensitive: bool = False,
    none_values: Union[str, Iterable[str]] = DEFAULT_NONE_VALUES,
) -> Optional[int]:
    none_values = _sanitize_values(none_values)
    if _str_in(x, none_values, case_sensitive):
        return None

    return int(x)


def str_to_optional_str(
    x: str,
    *,
    case_sensitive: bool = False,
    none_values: Union[str, Iterable[str]] = DEFAULT_NONE_VALUES,
) -> Optional[str]:
    none_values = _sanitize_values(none_values)
    if _str_in(x, none_values, case_sensitive):
        return None

    return x


def str_to_optional_bool(
    x: str,
    *,
    case_sensitive: bool = False,
    true_values: Union[str, Iterable[str]] = DEFAULT_TRUE_VALUES,
    false_values: Union[str, Iterable[str]] = DEFAULT_FALSE_VALUES,
    none_values: Union[str, Iterable[str]] = DEFAULT_NONE_VALUES,
) -> Optional[bool]:
    true_values = _sanitize_values(true_values)
    if _str_in(x, true_values, case_sensitive):
        return True

    false_values = _sanitize_values(false_values)
    if _str_in(x, false_values, case_sensitive):
        return False

    none_values = _sanitize_values(none_values)
    if _str_in(x, none_values, case_sensitive):
        return None

    values = tuple(true_values + false_values + none_values)
    raise ValueError(f"Invalid argument '{x}'. (expected one of {values})")


def _sanitize_values(values: Union[str, Iterable[str]]) -> List[str]:
    if isinstance(values, str):
        values = [values]
    else:
        values = list(values)
    return values


def _str_in(x: str, values: List[str], case_sensitive: bool) -> bool:
    if case_sensitive:
        return x in values
    else:
        return x.lower() in map(str.lower, values)
