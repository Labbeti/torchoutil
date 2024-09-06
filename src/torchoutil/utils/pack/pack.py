#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Tuple, TypeVar, Union

import torch
from torch import Tensor
from torch.utils.data.dataloader import DataLoader

import pyoutil as po
import torchoutil as to
from torchoutil import nn
from torchoutil.utils.data.dataloader import get_auto_num_cpus
from torchoutil.utils.data.dataset import (
    SizedDatasetLike,
    SizedIterableDatasetLike,
    TransformWrapper,
)
from torchoutil.utils.data.dtype import (
    merge_numpy_dtypes,
    numpy_dtype_to_fill_value,
    scan_shape_dtypes,
)
from torchoutil.utils.hdf.common import _tuple_to_dict
from torchoutil.utils.pack.common import ATTRS_FNAME, CONTENT_DNAME, ContentMode
from torchoutil.utils.pack.dataset import PackedDataset
from torchoutil.utils.packaging import _NUMPY_AVAILABLE, _TQDM_AVAILABLE
from torchoutil.utils.saving.common import to_builtin

if _TQDM_AVAILABLE:
    import tqdm
if _NUMPY_AVAILABLE:
    import numpy as np


T = TypeVar("T", covariant=True)
U = TypeVar("U", covariant=True)
T_DictOrTuple = TypeVar("T_DictOrTuple", tuple, dict, covariant=True)


ExistsMode = Literal["overwrite", "skip", "error"]
SaveMode = Literal["numpy"]
EXISTS_MODES = ("overwrite", "skip", "error")
SAVE_MODES = ("numpy",)


pylog = logging.getLogger(__name__)


@torch.inference_mode()
def pack_dataset(
    dataset: SizedDatasetLike[T],
    root: Union[str, Path],
    pre_transform: Optional[Callable[[T], U]] = None,
    *,
    batch_size: int = 32,
    num_workers: Union[int, Literal["auto"]] = "auto",
    exists: ExistsMode = "error",
    content_mode: ContentMode = "item",
    custom_file_fmt: Union[None, str, Callable[[int], str]] = None,
    save_fn: Callable[..., None] = torch.save,  # type: ignore
    subdir_size: Optional[int] = 100,
    transform_in_worker: bool = True,
    verbose: int = 0,
) -> PackedDataset[U, U]:
    """Pack a dataset to pickle files.

    Here is an example how files are stored on disk for a dataset containing 1000 items:

    .. code-block:: text
        :caption:  Dataset folder tree example

        {root}
        ├── attributes.json
        └── data
            ├── 0
            |   └── 100 pt files
            ├── 1
            |   └── 100 pt files
            ├── ...
            |   └── ...
            └── 9
                └── 100 pt files

    Args:
        dataset: Dataset-like to pack.
        root: Directory to store pickled data.
        pre_transform: Transform to apply to each item before saving. defaults to None.
        batch_size: Batch size used by the dataloader. defaults to 32.
        num_workers: Number of workers used by the dataloader. defaults to "auto".
        exists: Determine which action should be performed if the target HDF file already exists.
            "overwrite": Replace the target files then pack dataset.
            "skip": Skip this function and returns the packed dataset.
            "error": Raises a ValueError.
        content_mode: Specify how the data should be stored.
            If "item", each dataset item will be stored in a separate file.
            If "batch", each dataset item will be stored in a batch file of size batch_size.
        custom_file_fmt: Custom file format.
            If None, defaults to "{{i:0{num_digits}d}}.pt".
            defaults to None.
        save_fn: Custom save function to save an item or a batch. defaults to torch.save.
        subdir_size: Optional number of files per folder.
            Using None will disable subdir an put all files in data/ folder.
            defaults to 100.
        transform_in_worker: If True, appoly transform in parallel with workers' Dataloader.
        verbose: Verbose level during packing. Higher value means more messages. defaults to 0.
    """
    # Check inputs
    if not isinstance(dataset, SizedDatasetLike):
        raise TypeError(
            f"Cannot pack to hdf a non-sized-dataset '{dataset.__class__.__name__}'."
        )
    if len(dataset) == 0:
        raise ValueError("Cannot pack to hdf an empty dataset.")
    if transform_in_worker and isinstance(dataset, SizedIterableDatasetLike):
        raise NotImplementedError(
            "Cannot apply transform in worker with an iterable dataset kind."
        )

    root = Path(root).resolve()
    if root.exists() and not root.is_dir():
        raise RuntimeError(f"Item {root=} exists but it is not a file.")
    content_dpath = root.joinpath(CONTENT_DNAME)

    if num_workers == "auto":
        num_workers = get_auto_num_cpus()

    packed, root, content_dpath = _setup_args(root, exists)
    if packed is not None:
        return packed

    if transform_in_worker:
        wrapped = TransformWrapper(dataset, pre_transform)
        src_dataset_name = wrapped.unwrap().__class__.__name__
    else:
        wrapped = dataset
        src_dataset_name = dataset.__class__.__name__

    loader = DataLoader(
        wrapped,  # type: ignore
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=nn.Identity(),
        drop_last=False,
        pin_memory=False,
    )

    if content_mode == "item":
        num_files = len(wrapped)
    elif content_mode == "batch":
        num_files = math.ceil(len(wrapped) / batch_size)
    else:
        raise ValueError(f"Invalid argument {content_mode=}.")

    if custom_file_fmt is None:
        num_digits = math.ceil(math.log10(num_files))
        file_fmt = f"{{:0{num_digits}d}}.pt".format
    elif isinstance(custom_file_fmt, str):
        file_fmt = custom_file_fmt.format
    else:
        file_fmt = custom_file_fmt

    fnames = [file_fmt(i) for i in range(num_files)]

    if subdir_size is not None and len(fnames) > subdir_size:
        num_subdirs = math.ceil(len(fnames) / subdir_size)
        num_digits = math.ceil(math.log10(num_subdirs))
        dir_fmt = f"{{:0{num_digits}d}}".format
        sub_dnames = [dir_fmt(i) for i in range(num_subdirs)]
        fnames = [
            str(Path(sub_dnames[i // subdir_size]).joinpath(fname))
            for i, fname in enumerate(fnames)
        ]

    fpaths = [content_dpath.joinpath(fname) for fname in fnames]

    if custom_file_fmt is not None or subdir_size is not None:
        unique_parents = dict.fromkeys(fpath.parent for fpath in fpaths)
        for parent in unique_parents:
            parent.mkdir(parents=True, exist_ok=True)

    if _TQDM_AVAILABLE and verbose >= 1:
        loader = tqdm.tqdm(loader, desc=f"Packing {len(fnames)} items...")

    i = 0
    for batch_lst in loader:
        if pre_transform is not None and not transform_in_worker:
            batch_lst = [pre_transform(item) for item in batch_lst]

        if content_mode == "item":
            for item in batch_lst:
                fpath = fpaths[i]
                save_fn(item, fpath)
                i += 1

        elif content_mode == "batch":
            fpath = fpaths[i]
            save_fn(batch_lst, fpath)
            i += 1

        else:
            raise ValueError(f"Invalid argument {content_mode=}.")

    if hasattr(dataset, "info"):
        info = dataset.info  # type: ignore
        info = to_builtin(info)
    else:
        info = {}

    if hasattr(dataset, "attrs"):
        source_attrs = dataset.attrs  # type: ignore
        source_attrs = to_builtin(source_attrs)
    else:
        source_attrs = {}

    attributes = {
        "source_dataset": src_dataset_name,
        "length": len(wrapped),
        "creation_date": po.now_iso(),
        "batch_size": batch_size,
        "content_mode": content_mode,
        "content_dname": CONTENT_DNAME,
        "info": info,
        "source_attrs": source_attrs,
        "num_files": len(fnames),
        "files": fnames,
        "subdir_size": subdir_size,
    }

    attrs_fpath = root.joinpath(ATTRS_FNAME)
    with open(attrs_fpath, "w") as file:
        json.dump(attributes, file, indent="\t")

    packed = PackedDataset(root)
    return packed


def pack_dataset_per_column(
    dataset: SizedDatasetLike[T],
    root: Union[str, Path],
    pre_transform: Optional[Callable[[T], T_DictOrTuple]] = None,
    *,
    batch_size: int = 32,
    num_workers: Union[int, Literal["auto"]] = "auto",
    exists: ExistsMode = "error",
    verbose: int = 0,
) -> PackedDataset[T_DictOrTuple, T_DictOrTuple]:
    if not isinstance(dataset, SizedDatasetLike):
        raise TypeError(
            f"Cannot pack to hdf a non-sized-dataset '{dataset.__class__.__name__}'."
        )
    if len(dataset) == 0:
        raise ValueError("Cannot pack to hdf an empty dataset.")

    if not _NUMPY_AVAILABLE:
        msg = f"Function '{po.get_current_fn_name()}' cannot be called without numpy installed. Please install it with `pip install torchoutil[extras]`."
        raise RuntimeError(msg)

    packed, root, content_dpath = _setup_args(root, exists)
    if packed is not None:
        return packed

    if pre_transform is None:
        pre_transform = nn.Identity()
    if num_workers == "auto":
        num_workers = get_auto_num_cpus()

    src_dataset_name = dataset.__class__.__name__

    loader = DataLoader(
        dataset,  # type: ignore
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=nn.Identity(),
        drop_last=False,
        pin_memory=False,
    )

    it = iter(loader)
    item_0 = next(it)
    item_0 = pre_transform(item_0)

    dict_pre_transform: Callable[[T], Dict[str, Any]]

    if po.is_dict_str(item_0):
        item_type = "dict"
        dict_pre_transform = pre_transform  # type: ignore

    elif isinstance(item_0, tuple):
        item_type = "tuple"
        dict_pre_transform = po.compose(pre_transform, _tuple_to_dict)
        item_0 = _tuple_to_dict(item_0)

    else:
        msg = f"Invalid item type for {dataset.__class__.__name__}. (expected dict[str, Any] or tuple but found {type(item_0)})"
        raise ValueError(msg)

    infos_0 = {name: scan_shape_dtypes(value) for name, value in item_0.items()}
    max_shapes = {name: info.shape for name, info in infos_0.items()}
    dtypes = {name: info.numpy_dtype for name, info in infos_0.items()}

    del pre_transform, item_0, infos_0

    if _TQDM_AVAILABLE:
        it = tqdm.tqdm(it, total=len(loader), disable=verbose < 1)

    i = 0
    for batch in it:
        batch = [dict_pre_transform(item) for item in batch]
        for item in batch:
            for name, value in item.items():
                info = scan_shape_dtypes(value)

                if info.ndim != len(max_shapes[name]):
                    msg = f"Invalid argument shape at index {i} with key {name}. (found {info.ndim} ndim but expected {len(max_shapes[name])})"
                    raise ValueError(msg)

                max_shapes[name] = tuple(
                    max(info.shape[i], max_shapes[name][i]) for i in range(info.ndim)
                )
                dtypes[name] = merge_numpy_dtypes([info.numpy_dtype, dtypes[name]])
            i += 1

    fill_values = {
        name: numpy_dtype_to_fill_value(np_dtype) for name, np_dtype in dtypes.items()
    }

    data_dict = {
        name: np.full(
            (len(dataset),) + shape,
            fill_value=fill_values[name],
            dtype=dtypes[name],
        )
        for name, shape in max_shapes.items()
    }

    i = 0
    it = iter(loader)
    if _TQDM_AVAILABLE:
        it = tqdm.tqdm(it, total=len(loader), disable=verbose < 1)

    for batch in it:
        batch = [dict_pre_transform(item) for item in batch]
        for item in batch:
            for k, v in item.items():
                data_dict[k][i] = to.to_numpy(v)
            i += 1

    if hasattr(dataset, "info"):
        info = dataset.info  # type: ignore
        info = to_builtin(info)
    else:
        info = {}

    if hasattr(dataset, "attrs"):
        source_attrs = dataset.attrs  # type: ignore
        source_attrs = to_builtin(source_attrs)
    else:
        source_attrs = {}

    attributes = {
        "source_dataset": src_dataset_name,
        "batch_size": len(dataset),
        "item_type": item_type,
        "info": info,
        "source_attrs": source_attrs,
        "subdir_size": None,
    }
    return _pack_dataset_dict(
        data_dict=data_dict,  # type: ignore
        root=root,
        content_dpath=content_dpath,
        attributes=attributes,
    )


def pack_dataset_dict(
    data_dict: Dict[str, Union[np.ndarray, Tensor]],
    root: Union[str, Path],
    *,
    exists: ExistsMode = "error",
) -> PackedDataset:
    if not po.all_eq(map(len, data_dict.values())):
        raise ValueError

    packed, root, content_dpath = _setup_args(root, exists)
    if packed is not None:
        return packed

    attributes = {
        "source_dataset": data_dict.__class__.__name__,
        "batch_size": len(next(iter(data_dict.values()))),
        "item_type": "batch",
        "info": {},
        "source_attrs": {},
        "subdir_size": None,
    }
    return _pack_dataset_dict(
        data_dict=data_dict,
        root=root,
        content_dpath=content_dpath,
        attributes=attributes,
    )


def _pack_dataset_dict(
    data_dict: Dict[str, Union[np.ndarray, Tensor]],
    root: Path,
    content_dpath: Path,
    attributes: Dict[str, Any],
) -> PackedDataset:
    fnames = []
    for name, values in data_dict.items():
        fname = f"{name}.npy"
        fpath = content_dpath.joinpath(fname)
        values = to.to_numpy(values)
        np.save(fpath, values)
        fnames.append(fname)

    attributes = {
        "length": len(data_dict),
        "creation_date": po.now_iso(),
        "content_mode": "column",
        "content_dname": CONTENT_DNAME,
        "num_files": len(fnames),
        "files": fnames,
    } | attributes

    attrs_fpath = root.joinpath(ATTRS_FNAME)
    with open(attrs_fpath, "w") as file:
        json.dump(attributes, file, indent="\t")

    packed = PackedDataset(root)
    return packed


def _setup_args(
    root: Union[str, Path],
    exists: ExistsMode,
) -> Tuple[Optional[PackedDataset], Path, Path]:
    root = Path(root).resolve()
    if root.exists() and not root.is_dir():
        raise RuntimeError(f"Path {root=} exists but it is not a directory.")

    content_dpath = root.joinpath(CONTENT_DNAME)

    if not content_dpath.is_dir():
        pass
    elif exists == "error":
        msg = f"Cannot overwrite root data {str(content_dpath)}. Please remove it or use exists='overwrite' option."
        raise ValueError(msg)
    elif exists == "skip":
        return PackedDataset(root), root, content_dpath
    elif exists == "overwrite":
        shutil.rmtree(content_dpath)
    else:
        msg = f"Invalid argument {exists=}. (expected one of {EXISTS_MODES})"
        raise ValueError(msg)

    content_dpath.mkdir(parents=True, exist_ok=True)
    return None, root, content_dpath
