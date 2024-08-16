#!/usr/bin/env python
# -*- coding: utf-8 -*-


from torchoutil.utils.packaging import _H5PY_AVAILABLE, _NUMPY_AVAILABLE

if not _H5PY_AVAILABLE:
    raise ImportError(
        "Cannot import hdf objects because optional dependancy 'h5py' is not installed. Please install it using 'pip install torchoutil[extras]'"
    )
if not _NUMPY_AVAILABLE:
    raise ImportError(
        "Cannot import hdf objects because optional dependancy 'numpy' is not installed. Please install it using 'pip install torchoutil[extras]'"
    )

del _H5PY_AVAILABLE, _NUMPY_AVAILABLE


from torchoutil.utils.hdf.dataset import HDFDataset
from torchoutil.utils.hdf.pack import pack_to_hdf
