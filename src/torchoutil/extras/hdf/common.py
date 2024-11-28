#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Literal, TypedDict

import h5py
import numpy as np

# Force this encoding
HDF_ENCODING = "utf-8"
# Key suffix to store tensor shapes (because they are padded in hdf file)
SHAPE_SUFFIX = "__shape"
# Type for strings
HDF_STRING_DTYPE: np.dtype = h5py.string_dtype(HDF_ENCODING, None)
# Type for empty lists
HDF_VOID_DTYPE: np.dtype = h5py.opaque_dtype("V1")


HDFItemType = Literal["dict", "tuple"]


class HDFDatasetAttributes(TypedDict):
    added_columns: List[str]
    creation_date: str
    encoding: str
    file_kwds: Dict[str, Any]
    global_hash_value: int
    info: Dict[str, Any]
    item_type: HDFItemType
    length: int
    load_as_complex: Dict[str, bool]
    pre_transform: str
    shape_suffix: str
    source_dataset: str
    src_np_dtypes: Dict[str, np.dtype]
    use_vlen_str: bool
    user_attrs: Any
    torchoutil_version: str


_DUMPED_JSON_KEYS = (
    "file_kwds",
    "info",
    "load_as_complex",
    "src_np_dtypes",
    "user_attrs",
)
_DEFAULTS_RAW_HDF_ATTRIBUTES = {
    "added_columns": [],
    "creation_date": "unknown",
    "encoding": HDF_ENCODING,
    "file_kwds": "{}",
    "global_hash_value": -1,
    "info": "{}",
    "item_type": "dict",
    "length": 0,
    "load_as_complex": "{}",
    "pre_transform": "unknown",
    "shape_suffix": SHAPE_SUFFIX,
    "source_dataset": "unknown",
    "src_np_dtypes": "{}",
    "store_complex_as_real": False,
    "store_str_as_vlen": False,
    "user_attrs": "None",
    "torchoutil_version": "unknown",
}
