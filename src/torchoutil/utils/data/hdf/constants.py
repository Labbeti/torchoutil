#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py

# Force this encoding
HDF_ENCODING = "utf-8"
# Key suffix to store tensor shapes (because they are padded in hdf file)
SHAPE_SUFFIX = "_shape"
# Type for strings
HDF_STRING_DTYPE = h5py.string_dtype(HDF_ENCODING, None)
# Type for empty lists
HDF_VOID_DTYPE = h5py.opaque_dtype("V1")
