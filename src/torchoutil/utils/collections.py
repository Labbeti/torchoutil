#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

T = TypeVar("T")
U = TypeVar("U")


def list_dict_to_dict_list(
    lst: Sequence[Mapping[str, T]],
    default_val: U = None,
    error_on_missing_key: bool = False,
) -> Dict[str, List[Union[T, U]]]:
    """Convert a list of dicts to a dict of lists.

    Example 1
    ----------
    >>> lst = [{'a': 1, 'b': 2}, {'a': 4, 'b': 3, 'c': 5}]
    >>> output = list_dict_to_dict_list(lst, default_val=0)
    {'a': [1, 4], 'b': [2, 3], 'c': [0, 5]}
    """
    if len(lst) == 0:
        return {}

    if error_on_missing_key:
        keys = set(lst[0])
        for dic in lst:
            if keys != set(dic.keys()):
                raise ValueError(
                    f"Invalid dict keys for list_dict_to_dict_list. (found {keys} and {dic.keys()})"
                )

    keys = {}
    for dic in lst:
        keys.update(dict.fromkeys(dic.keys()))

    out = {
        key: [
            lst[i][key] if key in lst[i].keys() else default_val
            for i in range(len(lst))
        ]
        for key in keys
    }
    return out


def dump_dict(
    dic: Mapping[str, Any],
    /,
    join: str = ", ",
    fmt: str = "{name}={value}",
    ignore_none: bool = False,
) -> str:
    result = join.join(
        fmt.format(name=name, value=value)
        for name, value in dic.items()
        if not (value is None and ignore_none)
    )
    return result


def pass_filter(
    name: T,
    include: Optional[Iterable[T]] = None,
    exclude: Optional[Iterable[T]] = None,
) -> bool:
    """Returns True if name in include set and not in exclude set."""
    if include is not None and exclude is not None:
        return (name in include) and (name not in exclude)
    if include is not None:
        return name in include
    elif exclude is not None:
        return name not in exclude
    else:
        return True


def filter_iterable(
    it: Iterable[T],
    include: Optional[Iterable[T]] = None,
    exclude: Optional[Iterable[T]] = None,
) -> List[T]:
    return [item for item in it if pass_filter(item, include, exclude)]
