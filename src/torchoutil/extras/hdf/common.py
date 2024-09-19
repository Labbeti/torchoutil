#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Literal, TypedDict

import h5py
import numpy as np

# Force this encoding
HDF_ENCODING = "utf-8"
# Key suffix to store tensor shapes (because they are padded in hdf file)
SHAPE_SUFFIX = "_shape"
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
    shape_suffix: str
    source_dataset: str
    src_np_dtypes: Dict[str, np.dtype]
    use_vlen_str: bool
    user_attrs: Dict[str, Any]
    version: str


DUMPED_JSON_KEYS = (
    "file_kwds",
    "info",
    "load_as_complex",
    "src_np_dtypes",
    "user_attrs",
)
DEFAULTS_HDF_ATTRIBUTES = {
    "added_columns": [],
    "creation_date": "unknown",
    "encoding": HDF_ENCODING,
    "file_kwds": {},
    "global_hash_value": -1,
    "info": {},
    "item_type": "dict",
    "length": 0,
    "load_as_complex": {},
    "shape_suffix": SHAPE_SUFFIX,
    "source_dataset": "unknown",
    "src_np_dtypes": {},
    "use_vlen_str": False,
    "user_attrs": {},
    "version": "unknown",
}
