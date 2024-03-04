#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, TypeVar

import h5py

T = TypeVar("T")


# Force this encoding
HDF_ENCODING = "utf-8"
# Key suffix to store tensor shapes (because they are padded in hdf file)
SHAPE_SUFFIX = "_shape"
# Type for strings
HDF_STRING_DTYPE = h5py.string_dtype(HDF_ENCODING, None)
# Type for empty lists
HDF_VOID_DTYPE = h5py.opaque_dtype("V1")


def _tuple_to_dict(x: Tuple[T, ...]) -> Dict[str, T]:
    return dict(zip(map(str, range(len(x))), x))


def _dict_to_tuple(x: Dict[str, T]) -> Tuple[T, ...]:
    return tuple(x.values())
