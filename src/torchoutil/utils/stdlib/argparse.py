#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, Optional

_TRUE_VALUES = ("True", "t", "yes", "y", "1")
_FALSE_VALUES = ("False", "f", "no", "n", "0")
_NONE_VALUES = ("None", "null")


def str_to_bool(x: str, *, case_sensitive: bool = False) -> bool:
    """Convert string values to bool. Intended for argparse boolean arguments.

    - True values: 'True', 'T', 'yes', 'y', '1'.
    - False values: 'False', 'F', 'no', 'n', '0'
    - Other raises ValueError.
    """
    if _str_in(x, _TRUE_VALUES, case_sensitive):
        return True
    elif _str_in(x, _FALSE_VALUES, case_sensitive):
        return False
    else:
        values = _TRUE_VALUES + _FALSE_VALUES
        raise ValueError(f"Invalid argument '{x}'. (expected one of {values})")


def str_to_optional_int(x: str, *, case_sensitive: bool = False) -> Optional[int]:
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
