#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from torchoutil.utils.type_checks import is_mapping_str_any

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
    fmt: str = "{key}={value}",
    ignore_none: bool = False,
) -> str:
    """Dump dictionary to string."""
    result = join.join(
        fmt.format(key=key, value=value)
        for key, value in dic.items()
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


def all_eq(it: Iterable[T], eq_fn: Optional[Callable[[T, T], bool]] = None) -> bool:
    """Returns true if all elements in inputs are equal."""
    it = list(it)
    first = it[0]
    if eq_fn is None:
        return all(first == elt for elt in it)
    else:
        return all(eq_fn(first, elt) for elt in it)


def flat_dict_of_dict(
    nested_dic: Mapping[str, Any],
    sep: str = ".",
    flat_iterables: bool = False,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """Flat a nested dictionary.

    Example 1
    ---------
    ```
    >>> dic = {
    ...     "a": 1,
    ...     "b": {
    ...         "a": 2,
    ...         "b": 10,
    ...     },
    ... }
    >>> flat_dict(dic)
    ... {"a": 1, "b.a": 2, "b.b": 10}
    ```

    Example 2
    ---------
    ```
    >>> dic = {"a": ["hello", "world"], "b": 3}
    >>> flat_dict(dic, flat_iterables=True)
    ... {"a.0": "hello", "a.1": "world", "b": 3}
    ```

    Args:
        nested_dict: Nested mapping containing sub-mappings or iterables.
        sep: Separators between keys.
        flat_iterables: If True, flat iterable and use index as key.
        overwrite: If True, overwrite duplicated keys in output. Otherwise duplicated keys will raises a ValueError.
    """
    output = {}
    for k, v in nested_dic.items():
        if is_mapping_str_any(v):
            v = flat_dict_of_dict(v, sep, flat_iterables)
            v = {f"{k}{sep}{kv}": vv for kv, vv in v.items()}
            output.update(v)

        elif flat_iterables and isinstance(v, Iterable) and not isinstance(v, str):
            v = {f"{i}": vi for i, vi in enumerate(v)}
            v = flat_dict_of_dict(v, sep, flat_iterables)
            v = {f"{k}{sep}{kv}": vv for kv, vv in v.items()}
            output.update(v)

        elif overwrite or k not in output:
            output[k] = v

        else:
            raise ValueError(f"Ambiguous flatten dict with key '{k}'.")

    return output


def flat_list(lst: Iterable[Sequence[T]]) -> Tuple[List[T], List[int]]:
    """Return a flat version of the input list of sublists with each sublist size."""
    flatten_lst = [element for sublst in lst for element in sublst]
    sizes = [len(sents) for sents in lst]
    return flatten_lst, sizes
