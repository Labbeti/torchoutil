#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import json
import logging
import math
import shutil
from pathlib import Path
from typing import Callable, List, Literal, Optional, TypeVar, Union

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from torchoutil.utils.data.dataloader import get_auto_num_cpus
from torchoutil.utils.data.dataset import SizedDatasetLike, TransformWrapper
from torchoutil.utils.pack.common import ATTRS_FNAME, CONTENT_DNAME, ContentMode
from torchoutil.utils.pack.dataset import PackedDataset
from torchoutil.utils.packaging import _TQDM_AVAILABLE
from torchoutil.utils.saving.common import to_builtin

if _TQDM_AVAILABLE:
    import tqdm


T = TypeVar("T")
U = TypeVar("U")


pylog = logging.getLogger(__name__)


@torch.inference_mode()
def pack_dataset(
    dataset: SizedDatasetLike[T],
    root: Union[str, Path],
    pre_transform: Optional[Callable[[T], U]] = None,
    batch_size: int = 32,
    num_workers: Union[int, Literal["auto"]] = "auto",
    overwrite: bool = False,
    content_mode: ContentMode = "item",
    custom_file_fmt: Union[None, str, Callable[[int], str]] = None,
    save_fn: Callable[[Union[U, List[U]], Path], None] = torch.save,
    subdir_size: Optional[int] = 100,
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
        overwrite: If True, overwrite all data in root directory. defaults to False.
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
        verbose: Verbose level during packing. Higher value means more messages. defaults to 0.
    """

    # Check inputs
    if not isinstance(dataset, SizedDatasetLike):
        raise TypeError(
            f"Cannot pack to hdf a non-sized-dataset '{dataset.__class__.__name__}'."
        )
    if len(dataset) == 0:
        raise ValueError("Cannot pack to hdf an empty dataset.")

    root = Path(root).resolve()
    if root.exists() and not root.is_dir():
        raise RuntimeError(f"Item {root=} exists but it is not a file.")

    if num_workers == "auto":
        num_workers = get_auto_num_cpus()

    content_dpath = root.joinpath(CONTENT_DNAME)

    if content_dpath.is_dir():
        if not overwrite:
            raise ValueError(
                f"Cannot overwrite root data {str(content_dpath)}. Please remove it or use overwrite=True option."
            )
        shutil.rmtree(content_dpath)

    content_dpath.mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.now()
    creation_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    wrapped = TransformWrapper(dataset, pre_transform)

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
        loader = tqdm.tqdm(loader)

    i = 0
    for batch_lst in loader:
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
        "source_dataset": wrapped.unwrap().__class__.__name__,
        "length": len(wrapped),
        "creation_date": creation_date,
        "batch_size": batch_size,
        "content_mode": content_mode,
        "content_dname": CONTENT_DNAME,
        "subdir_size": subdir_size,
        "info": info,
        "source_attrs": source_attrs,
        "num_files": len(fnames),
        "files": fnames,
    }

    attrs_fpath = root.joinpath(ATTRS_FNAME)
    with open(attrs_fpath, "w") as file:
        json.dump(attributes, file, indent="\t")

    pack = PackedDataset(root)
    return pack
