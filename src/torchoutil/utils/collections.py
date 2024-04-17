#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

from torchoutil.utils.type_checks import is_mapping_str_any

K = TypeVar("K")
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")

KEY_MODES = ("same", "intersect", "union")
KeyMode = Literal["intersect", "same", "union"]


@overload
def list_dict_to_dict_list(
    lst: Sequence[Mapping[K, V]],
    key_mode: Literal["intersect", "same"],
    default_val: W = None,
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
        if not all(keys == set(item.keys()) for item in lst[1:]):
            raise ValueError("Invalid keys for batch.")
    elif key_mode == "intersect":
        keys = intersect_lists([item.keys() for item in lst])
    elif key_mode == "union":
        keys = union_lists([item.keys() for item in lst])
    else:
        raise ValueError(
            f"Invalid argument key_mode={key_mode}. (expected one of {KEY_MODES})"
        )

    return {key: [item.get(key, default_val) for item in lst] for key in keys}


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
        out |= dict.fromkeys(lst_i)
    out = list(out)
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
