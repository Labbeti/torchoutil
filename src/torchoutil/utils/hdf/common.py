#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Literal, TypedDict

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
    creation_date: str
    source_dataset: str
    length: int
    metadata: str
    encoding: str
    info: str
    global_hash_value: int
    item_type: HDFItemType
    added_columns: List[str]
    shape_suffix: str
    file_kwds: str
    user_attrs: str
    load_as_complex: str
    is_unicode: str
