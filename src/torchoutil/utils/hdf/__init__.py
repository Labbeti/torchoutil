#!/usr/bin/env python
# -*- coding: utf-8 -*-


from torchoutil.utils.packaging import _H5PY_AVAILABLE

if not _H5PY_AVAILABLE:
    raise ImportError(
        "Optional dependancy 'h5py' is not installed. Please install it using 'pip install torchoutil[extras]'"
    )

del _H5PY_AVAILABLE


from torchoutil.utils.hdf.dataset import HDFDataset
from torchoutil.utils.hdf.pack import pack_to_hdf
