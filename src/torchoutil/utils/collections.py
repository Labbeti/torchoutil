#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

from torchoutil.utils.type_checks import is_mapping_str

K = TypeVar("K")
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")

KEY_MODES = ("same", "intersect", "union")
KeyMode = Literal["intersect", "same", "union"]


def sorted_dict(
    x: Mapping[K, V],
    /,
    *,
    key: Optional[Callable[[K], Any]] = None,
    reverse: bool = False,
) -> Dict[K, V]:
    return {k: x[k] for k in sorted(x.keys(), key=key, reverse=reverse)}


@overload
def list_dict_to_dict_list(
    lst: Sequence[Mapping[K, V]],
    key_mode: Literal["intersect", "same"],
    default_val: Any = None,
) -> Dict[K, List[V]]:
    ...


@overload
def list_dict_to_dict_list(
    lst: Sequence[Mapping[K, V]],
    key_mode: Literal["union"] = "union",
    default_val: W = None,
) -> Dict[K, List[Union[V, W]]]:
    ...


def list_dict_to_dict_list(
    lst: Sequence[Mapping[K, V]],
    key_mode: KeyMode = "union",
    default_val: W = None,
) -> Dict[K, List[Union[V, W]]]:
    """Convert list of dicts to dict of lists.

    Args:
        lst: The list of dict to merge.
        key_mode: Can be "same" or "intersect".
            If "same", all the dictionaries must contains the same keys otherwise a ValueError will be raised.
            If "intersect", only the intersection of all keys will be used in output.
            If "union", the output dict will contains the union of all keys, and the missing value will use the argument default_val.
        default_val: Default value of an element when key_mode is "union". defaults to None.
    """
    if len(lst) <= 0:
        return {}

    keys = set(lst[0].keys())
    if key_mode == "same":
        invalids = [list(item.keys()) for item in lst[1:] if keys != set(item.keys())]
        if len(invalids):
            raise ValueError(
                f"Invalid keys with {key_mode=}. (with {keys=} and {invalids=})"
            )
    elif key_mode == "intersect":
        keys = intersect_lists([item.keys() for item in lst])
    elif key_mode == "union":
        keys = union_lists([item.keys() for item in lst])
    else:
        raise ValueError(
            f"Invalid argument key_mode={key_mode}. (expected one of {KEY_MODES})"
        )

    return {key: [item.get(key, default_val) for item in lst] for key in keys}


@overload
def dict_list_to_list_dict(
    dic: Mapping[T, Sequence[U]],
    key_mode: Literal["same", "intersect"],
    default_val: Any = None,
) -> List[Dict[T, U]]:
    ...


@overload
def dict_list_to_list_dict(
    dic: Mapping[T, Sequence[U]],
    key_mode: Literal["union"] = "union",
    default_val: W = None,
) -> List[Dict[T, Union[U, W]]]:
    ...


def dict_list_to_list_dict(
    dic: Mapping[T, Sequence[U]],
    key_mode: KeyMode = "union",
    default_val: W = None,
) -> List[Dict[T, Union[U, W]]]:
    """Convert dict of lists with same sizes to list of dicts.

    Example 1
    ----------
    ```
    >>> dic = {"a": [1, 2], "b": [3, 4]}
    >>> dict_list_to_list_dict(dic)
    ... [{"a": 1, "b": 3}, {"a": 2, "b": 4}]
    ```
    """
    if len(dic) == 0:
        return []

    lengths = [len(seq) for seq in dic.values()]
    if key_mode == "same":
        if not all_eq(lengths):
            raise ValueError("Invalid sequences for batch.")
        length = lengths[0]
    elif key_mode == "intersect":
        length = min(lengths)
    elif key_mode == "union":
        length = max(lengths)
    else:
        raise ValueError(
            f"Invalid argument key_mode={key_mode}. (expected one of {KEY_MODES})"
        )

    result = [
        {k: (v[i] if i < len(v) else default_val) for k, v in dic.items()}
        for i in range(length)
    ]
    return result


def intersect_lists(lst_of_lst: Sequence[Iterable[T]]) -> List[T]:
    """Performs intersection of elements in lists (like set intersection), but keep their original order."""
    if len(lst_of_lst) <= 0:
        return []
    out = list(dict.fromkeys(lst_of_lst[0]))
    for lst_i in lst_of_lst[1:]:
        out = [name for name in out if name in lst_i]
        if len(out) == 0:
            break
    return out


def union_lists(lst_of_lst: Iterable[Iterable[T]]) -> List[T]:
    """Performs union of elements in lists (like set union), but keep their original order."""
    out = {}
    for lst_i in lst_of_lst:
        out.update(dict.fromkeys(lst_i))
    out = list(out)
    return out


def dump_dict(
    dic: Mapping[str, T],
    /,
    join: str = ", ",
    fmt: str = "{key}={value}",
    ignore_lst: Iterable[T] = (),
) -> str:
    """Dump dictionary to string."""
    ignore_lst = dict.fromkeys(ignore_lst)
    result = join.join(
        fmt.format(key=key, value=value)
        for key, value in dic.items()
        if value not in ignore_lst
    )
    return result


def pass_filter(
    x: T,
    include: Optional[Iterable[T]] = None,
    exclude: Optional[Iterable[T]] = None,
) -> bool:
    """Returns True if name in include set and not in exclude set."""
    if include is not None and exclude is not None:
        return (x in include) and (x not in exclude)
    if include is not None:
        return x in include
    elif exclude is not None:
        return x not in exclude
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


def all_ne(it: Iterable[T], ne_fn: Optional[Callable[[T, T], bool]] = None) -> bool:
    """Returns true if all elements in inputs are differents."""
    it = list(it)
    if ne_fn is None:
        return all(
            it[i] != it[j] for i in range(len(it)) for j in range(i + 1, len(it))
        )
    else:
        return all(
            ne_fn(it[i], it[j]) for i in range(len(it)) for j in range(i + 1, len(it))
        )


def flat_dict_of_dict(
    nested_dic: Mapping[str, Any],
    *,
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
    >>> flat_dict_of_dict(dic)
    ... {"a": 1, "b.a": 2, "b.b": 10}
    ```

    Example 2
    ---------
    ```
    >>> dic = {"a": ["hello", "world"], "b": 3}
    >>> flat_dict_of_dict(dic, flat_iterables=True)
    ... {"a.0": "hello", "a.1": "world", "b": 3}
    ```

    Args:
        nested_dict: Nested mapping containing sub-mappings or iterables.
        sep: Separators between keys.
        flat_iterables: If True, flat iterable and use index as key.
        overwrite: If True, overwrite duplicated keys in output. Otherwise duplicated keys will raises a ValueError.
    """

    def _flat_dict_of_dict_impl(nested_dic: Mapping[str, Any]) -> Dict[str, Any]:
        output = {}
        for k, v in nested_dic.items():
            if is_mapping_str(v):
                v = _flat_dict_of_dict_impl(v)
                v = {f"{k}{sep}{kv}": vv for kv, vv in v.items()}
                output.update(v)

            elif flat_iterables and isinstance(v, Iterable) and not isinstance(v, str):
                v = {f"{i}": vi for i, vi in enumerate(v)}
                v = _flat_dict_of_dict_impl(v)
                v = {f"{k}{sep}{kv}": vv for kv, vv in v.items()}
                output.update(v)

            elif overwrite or k not in output:
                output[k] = v

            else:
                raise ValueError(
                    f"Ambiguous flatten dict with key '{k}'. (with value '{v}')"
                )
        return output

    return _flat_dict_of_dict_impl(nested_dic)


def unflat_dict_of_dict(dic: Mapping[str, Any], sep: str = ".") -> Dict[str, Any]:
    """Unflat a dictionary.

    Example 1
    ----------
    ```
    >>> dic = {
        "a.a": 1,
        "b.a": 2,
        "b.b": 3,
        "c": 4,
    }
    >>> unflat_dict_of_dict(dic)
    ... {"a": {"a": 1}, "b": {"a": 2, "b": 3}, "c": 4}
    ```
    """
    output = {}
    for k, v in dic.items():
        if sep not in k:
            output[k] = v
        else:
            idx = k.index(sep)
            k, kk = k[:idx], k[idx + 1 :]
            if k not in output:
                output[k] = {}
            elif not isinstance(output[k], Mapping):
                raise ValueError(
                    f"Invalid dict argument. (found keys {k} and {k}{sep}{kk})"
                )

            output[k][kk] = v

    output = {
        k: (unflat_dict_of_dict(v) if isinstance(v, Mapping) else v)
        for k, v in output.items()
    }
    return output


def flat_list_of_list(lst: Iterable[Sequence[T]]) -> Tuple[List[T], List[int]]:
    """Return a flat version of the input list of sublists with each sublist size."""
    flatten_lst = [elt for sublst in lst for elt in sublst]
    sizes = [len(sents) for sents in lst]
    return flatten_lst, sizes


def unflat_list_of_list(
    flatten_lst: Sequence[T],
    sizes: Iterable[int],
) -> List[List[T]]:
    """Unflat a list to a list of sublists of given sizes."""
    lst = []
    start = 0
    stop = 0
    for count in sizes:
        stop += count
        lst.append(flatten_lst[start:stop])
        start = stop
    return lst


@overload
def unzip(lst: Iterable[Tuple[T]]) -> Tuple[List[T]]:
    ...


@overload
def unzip(lst: Iterable[Tuple[T, U]]) -> Tuple[List[T], List[U]]:
    ...


@overload
def unzip(lst: Iterable[Tuple[T, U, V]]) -> Tuple[List[T], List[U], List[V]]:
    ...


@overload
def unzip(
    lst: Iterable[Tuple[T, U, V, W]]
) -> Tuple[List[T], List[U], List[V], List[W]]:
    ...


def unzip(lst):
    """Invert zip() function.

    .. code-block:: python
        :caption:  Example

        >>> lst1 = [1, 2, 3, 4]
        >>> lst2 = [5, 6, 7, 8]
        >>> lst_zipped = list(zip(lst1, lst2))
        >>> lst_zipped
        ... [(1, 5), (2, 6), (3, 7), (4, 8)]
        >>> unzip(lst_zipped)
        ... ([1, 2, 3, 4], [5, 6, 7, 8])
    """
    return tuple(map(list, zip(*lst)))


def prod(x: Iterable[T], /, start: T = 1) -> T:
    result = copy.copy(start)
    for xi in x:
        result = result * xi
    return result
