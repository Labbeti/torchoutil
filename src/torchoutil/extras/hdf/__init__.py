#!/usr/bin/env python
# -*- coding: utf-8 -*-


from torchoutil.core.packaging import _EXTRA_AVAILABLE

_MISSING_DEPS = {
    name: _EXTRA_AVAILABLE[name]
    for name in ("h5py", "numpy", "tqdm")
    if not _EXTRA_AVAILABLE[name]
}
if len(_MISSING_DEPS) > 0:
    if len(_MISSING_DEPS) == 1:
        deps_msg = f"dependancy '{next(iter(_MISSING_DEPS.keys()))}' is"
    else:
        deps_msg = (
            "dependancies " + ", ".join(f"'{k}'" for k in _MISSING_DEPS.keys()) + " are"
        )

    msg = f"Cannot import hdf objects because optional {deps_msg} not installed. Please install it using 'pip install torchoutil[extras]'"
    raise ImportError(msg)

del _MISSING_DEPS

from .dataset import HDFDataset
from .pack import pack_to_hdf
