#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import operator
import sys
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
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

from typing_extensions import TypeIs

from torchoutil.pyoutil.typing import T_BuiltinScalar, is_builtin_scalar, is_mapping_str

K = TypeVar("K", covariant=True)
T = TypeVar("T", covariant=True)
U = TypeVar("U", covariant=True)
V = TypeVar("V", covariant=True)
W = TypeVar("W", covariant=True)
X = TypeVar("X", covariant=True)

KeyMode = Literal["intersect", "same", "union"]
KEY_MODES = ("same", "intersect", "union")


@overload
def list_dict_to_dict_list(
    lst: Sequence[Mapping[K, V]],
    key_mode: Literal["intersect", "same"],
    default_val: Any = None,
    *,
    default_val_fn: Any = None,
) -> Dict[K, List[V]]:
    ...


@overload
def list_dict_to_dict_list(
    lst: Sequence[Mapping[K, V]],
    key_mode: Literal["union"],
    default_val: Any = None,
    *,
    default_val_fn: Callable[[K], X],
) -> Dict[K, List[Union[V, X]]]:
    ...


@overload
def list_dict_to_dict_list(
    lst: Sequence[Mapping[K, V]],
    key_mode: Literal["union"],
    default_val: W = None,
    *,
    default_val_fn: None = None,
) -> Dict[K, List[Union[V, W]]]:
    ...


def list_dict_to_dict_list(
    lst: Sequence[Mapping[K, V]],
    key_mode: KeyMode,
    default_val: W = None,
    default_val_fn: Optional[Callable[[K], X]] = None,
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
        if len(invalids) > 0:
            msg = f"Invalid dict keys for conversion from List[dict] to Dict[list]. (with {key_mode=}, {keys=} and {invalids=})"
            raise ValueError(msg)

    elif key_mode == "intersect":
        keys = intersect_lists([item.keys() for item in lst])

    elif key_mode == "union":
        keys = union_lists(item.keys() for item in lst)

    else:
        msg = f"Invalid argument key_mode={key_mode}. (expected one of {KEY_MODES})"
        raise ValueError(msg)

    result = {
        key: [
            item.get(
                key, default_val_fn(key) if default_val_fn is not None else default_val
            )
            for item in lst
        ]
        for key in keys
    }
    return result


@overload
def dict_list_to_list_dict(
    dic: Mapping[T, Iterable[U]],
    key_mode: Literal["same", "intersect"],
    default_val: Any = None,
) -> List[Dict[T, U]]:
    ...


@overload
def dict_list_to_list_dict(
    dic: Mapping[T, Iterable[U]],
    key_mode: Literal["union"] = "union",
    default_val: W = None,
) -> List[Dict[T, Union[U, W]]]:
    ...


def dict_list_to_list_dict(
    dic: Mapping[T, Iterable[U]],
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

    Example 2
    ----------
    ```
    >>> dic = {"a": [1, 2, 3], "b": [4], "c": [5, 6]}
    >>> dict_list_to_list_dict(dic, key_mode="union", default=-1)
    ... [{"a": 1, "b": 4, "c": 5}, {"a": 2, "b": -1, "c": 6}, {"a": 3, "b": -1, "c": -1}]
    ```
    """
    if len(dic) == 0:
        return []

    dic = {k: list(v) if not isinstance(v, Sequence) else v for k, v in dic.items()}
    lengths = [len(seq) for seq in dic.values()]

    if key_mode == "same":
        if not all_eq(lengths):
            msg = f"Invalid sequences for batch. (found different lengths in sub-lists: {set(lengths)})"
            raise ValueError(msg)
        length = lengths[0]

    elif key_mode == "intersect":
        length = min(lengths)

    elif key_mode == "union":
        length = max(lengths)

    else:
        msg = f"Invalid argument key_mode={key_mode}. (expected one of {KEY_MODES})"
        raise ValueError(msg)

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
    """Custom dictionary of scalars to string function to customize representation.

    Example 1:
    ----------
    ```
    >>> d = {"a": 1, "b": 2}
    >>> dump_dict(d)
    ... 'a=1, b=2'
    ```
    """
    ignore_lst = dict.fromkeys(ignore_lst)
    result = join.join(
        fmt.format(key=key, value=value)
        for key, value in dic.items()
        if value not in ignore_lst
    )
    return result


@overload
def find(
    x: T,
    include: Iterable[V],
    *,
    match_fn: Callable[[V, T], bool] = operator.eq,
    order: Literal["right"] = "right",
    default: U = -1,
) -> Union[int, U]:
    ...


@overload
def find(
    x: T,
    include: Iterable[V],
    *,
    match_fn: Callable[[T, V], bool] = operator.eq,
    order: Literal["left"],
    default: U = -1,
) -> Union[int, U]:
    ...


def find(
    x: T,
    include: Iterable[T],
    *,
    match_fn: Callable[[T, T], bool] = operator.eq,
    order: Literal["left", "right"] = "right",
    default: U = -1,
) -> Union[int, U]:
    if order == "right":
        pass
    elif order == "left":

        def revert(f):
            def reverted_f(a, b):
                return f(b, a)

            return reverted_f

        match_fn = revert(match_fn)
    else:
        ORDER_VALUES = ("left", "right")
        raise ValueError(f"Invalid argument {order=}. (expected one of {ORDER_VALUES})")

    for i, include_i in enumerate(include):
        if match_fn(include_i, x):
            return i
    return default


def contained(
    x: T,
    include: Optional[Iterable[T]] = None,
    exclude: Optional[Iterable[T]] = None,
    *,
    match_fn: Callable[[T, T], bool] = operator.eq,
    order: Literal["left", "right"] = "right",
) -> bool:
    """Returns True if name in include set and not in exclude set."""
    if (
        include is not None
        and find(x, include, match_fn=match_fn, order=order, default=-1) == -1
    ):
        return False

    if (
        exclude is not None
        and find(x, exclude, match_fn=match_fn, order=order, default=-1) != -1
    ):
        return False

    return True


def filter_iterable(
    it: Iterable[T],
    include: Optional[Iterable[T]] = None,
    exclude: Optional[Iterable[T]] = None,
    *,
    match_fn: Callable[[T, T], bool] = operator.eq,
    order: Literal["left", "right"] = "right",
) -> List[T]:
    return [
        item
        for item in it
        if contained(
            item,
            include=include,
            exclude=exclude,
            match_fn=match_fn,
            order=order,
        )
    ]


def all_eq(it: Iterable[T], eq_fn: Optional[Callable[[T, T], bool]] = None) -> bool:
    """Returns true if all elements in iterable are equal.

    Note: This function returns True for iterable that contains 0 or 1 element.
    """
    it = list(it)
    first = it[0]
    if eq_fn is None:
        return all(first == elt for elt in it)
    else:
        return all(eq_fn(first, elt) for elt in it)


def all_ne(
    it: Iterable[T],
    ne_fn: Optional[Callable[[T, T], bool]] = None,
    use_set: bool = False,
) -> bool:
    """Returns true if all elements in iterable are differents.

    Note: This function returns True for iterable that contains 0 or 1 element.
    """
    if use_set and ne_fn is not None:
        raise ValueError(f"Cannot use arguments {use_set=} with {ne_fn=}.")

    it = list(it)
    if use_set:
        return len(it) == len(set(it))
    elif ne_fn is None:
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
        nested_dic: Nested mapping containing sub-mappings or iterables.
        sep: Separators between keys.
        flat_iterables: If True, flat iterable and use index as key.
        overwrite: If True, overwrite duplicated keys in output. Otherwise duplicated keys will raises a ValueError.
    """

    def _impl(nested_dic: Mapping[str, Any]) -> Dict[str, Any]:
        output = {}
        for k, v in nested_dic.items():
            if is_mapping_str(v):
                v = _impl(v)
                v = {f"{k}{sep}{kv}": vv for kv, vv in v.items()}
                output.update(v)

            elif flat_iterables and isinstance(v, Iterable) and not isinstance(v, str):
                v = {f"{i}": vi for i, vi in enumerate(v)}
                v = _impl(v)
                v = {f"{k}{sep}{kv}": vv for kv, vv in v.items()}
                output.update(v)

            elif overwrite or k not in output:
                output[k] = v

            else:
                msg = f"Ambiguous flatten dict with key '{k}'. (with value '{v}')"
                raise ValueError(msg)
        return output

    return _impl(nested_dic)


def unflat_dict_of_dict(dic: Mapping[str, Any], *, sep: str = ".") -> Dict[str, Any]:
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
                msg = f"Invalid dict argument. (found keys {k} and {k}{sep}{kk})"
                raise ValueError(msg)

            output[k][kk] = v

    output = {
        k: (unflat_dict_of_dict(v) if isinstance(v, Mapping) else v)
        for k, v in output.items()
    }
    return output


@overload
def flat_list_of_list(
    lst: Iterable[Sequence[T]],
    return_sizes: Literal[True] = True,
) -> Tuple[List[T], List[int]]:
    ...


@overload
def flat_list_of_list(
    lst: Iterable[Sequence[T]],
    return_sizes: Literal[False],
) -> List[T]:
    ...


def flat_list_of_list(
    lst: Iterable[Sequence[T]],
    return_sizes: bool = True,
) -> Union[Tuple[List[T], List[int]], List[T]]:
    """Return a flat version of the input list of sublists with each sublist size."""
    flatten_lst = [elt for sublst in lst for elt in sublst]
    sizes = [len(sents) for sents in lst]

    if return_sizes:
        return flatten_lst, sizes
    else:
        return flatten_lst


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
def unzip(lst: Iterable[Tuple[()]]) -> Tuple[()]:
    ...


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


@overload
def unzip(
    lst: Iterable[Tuple[T, U, V, W, X]]
) -> Tuple[List[T], List[U], List[V], List[W], List[X]]:
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
        ... [1, 2, 3, 4], [5, 6, 7, 8]
    """
    return tuple(map(list, zip(*lst)))


def prod(x: Iterable[T], /, start: T = 1) -> T:
    result = copy.copy(start)
    for xi in x:
        result = result * xi  # type: ignore
    return result


def sorted_dict(
    x: Mapping[K, V],
    /,
    *,
    key: Optional[Callable[[K], Any]] = None,
    reverse: bool = False,
) -> Dict[K, V]:
    return {k: x[k] for k in sorted(x.keys(), key=key, reverse=reverse)}  # type: ignore


@overload
def flatten(
    x: T_BuiltinScalar,
    start_dim: int = 0,
    end_dim: Optional[int] = None,
) -> List[T_BuiltinScalar]:
    ...


@overload
def flatten(
    x: Iterable[T_BuiltinScalar],
    start_dim: int = 0,
    end_dim: Optional[int] = None,
) -> List[T_BuiltinScalar]:
    ...


@overload
def flatten(
    x: Any,
    start_dim: int = 0,
    end_dim: Optional[int] = None,
    is_scalar_fn: Callable[[Any], TypeIs[T]] = is_builtin_scalar,
) -> List[Any]:
    ...


def flatten(
    x: Any,
    start_dim: int = 0,
    end_dim: Optional[int] = None,
    is_scalar_fn: Callable[[Any], TypeIs[T]] = is_builtin_scalar,
) -> List[Any]:
    if end_dim is None:
        end_dim = sys.maxsize
    if start_dim < 0:
        raise ValueError(f"Invalid argument {start_dim=}. (expected positive integer)")
    if end_dim < 0:
        raise ValueError(f"Invalid argument {end_dim=}. (expected positive integer)")
    if start_dim > end_dim:
        msg = f"Invalid arguments {start_dim=} and {end_dim=}. (expected start_dim <= end_dim)"
        raise ValueError(msg)

    def flatten_impl(x: Any, start_dim: int, end_dim: int) -> List[Any]:
        if is_scalar_fn(x):
            return [x]
        elif isinstance(x, Iterable):
            if start_dim > 0:
                return [flatten_impl(xi, start_dim - 1, end_dim - 1) for xi in x]
            elif end_dim > 0:
                return [
                    xij
                    for xi in x
                    for xij in flatten_impl(xi, start_dim - 1, end_dim - 1)
                ]
            else:
                return list(x)
        else:
            raise TypeError(f"Invalid argument type {type(x)=}.")

    return flatten_impl(x, start_dim, end_dim)


def recursive_generator(x: Any) -> Generator[Tuple[Any, int, int], None, None]:
    def recursive_generator_impl(
        x: Any,
        i: int,
        deep: int,
    ) -> Generator[Tuple[Any, int, int], None, None]:
        if isinstance(x, Iterable):
            for j, xj in enumerate(x):
                if xj == x:
                    yield xj, i, deep
                    return
                else:
                    yield from recursive_generator_impl(xj, j, deep + 1)
        else:
            yield x, i, deep
        return

    return recursive_generator_impl(x, 0, 0)


def is_sorted(
    x: Iterable[Any],
    *,
    reverse: bool = False,
    strict: bool = False,
) -> bool:
    it = iter(x)
    try:
        prev = next(it)
    except StopIteration:
        return True

    for xi in it:
        if not reverse and prev > xi:
            return False
        if reverse and prev < xi:
            return False
        if strict and prev == xi:
            return False
        prev = xi
    return True


def union_dicts(dicts: Iterable[Mapping[K, V]]) -> Dict[K, V]:
    result = {}
    for dic in dicts:
        result.update(dic)
    return result


def argmin(x: Iterable) -> int:
    min_index, _max_value = min(enumerate(x), key=lambda t: t[1])
    return min_index


def argmax(x: Iterable) -> int:
    max_index, _max_value = max(enumerate(x), key=lambda t: t[1])
    return max_index
