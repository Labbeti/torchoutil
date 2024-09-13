#!/usr/bin/env python
# -*- coding: utf-8 -*-


from torchoutil.core.packaging import _H5PY_AVAILABLE, _NUMPY_AVAILABLE

if not _H5PY_AVAILABLE:
    msg = "Cannot import hdf objects because optional dependancy 'h5py' is not installed. Please install it using 'pip install torchoutil[extras]'"
    raise ImportError(msg)
if not _NUMPY_AVAILABLE:
    msg = "Cannot import hdf objects because optional dependancy 'numpy' is not installed. Please install it using 'pip install torchoutil[extras]'"
    raise ImportError(msg)

del _H5PY_AVAILABLE, _NUMPY_AVAILABLE


from .dataset import HDFDataset
from .pack import pack_to_hdf
