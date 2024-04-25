#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import json
import logging
import math
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable, List, Literal, Optional, TypeVar, Union

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from torchoutil.utils.data.dataloader import get_auto_num_cpus
from torchoutil.utils.data.dataset import SizedDatasetLike, TransformWrapper
from torchoutil.utils.pickle_dataset.common import (
    ATTRS_FNAME,
    CONTENT_DNAME,
    ContentMode,
)
from torchoutil.utils.pickle_dataset.dataset import PickleDataset
from torchoutil.utils.type_checks import is_mapping_str

T = TypeVar("T")
U = TypeVar("U")


pylog = logging.getLogger(__name__)


@torch.inference_mode()
def pack_to_pickle(
    dataset: SizedDatasetLike[T],
    root: Union[str, Path],
    pre_transform: Optional[Callable[[T], U]] = None,
    batch_size: int = 32,
    num_workers: Union[int, Literal["auto"]] = "auto",
    overwrite: bool = False,
    content_mode: ContentMode = "item",
    custom_file_fmt: Union[None, str, Callable[[int], str]] = None,
    save_fn: Callable[[Union[U, List[U]], Path], None] = torch.save,
) -> PickleDataset[U, U]:
    """Pack a dataset to pickle files.

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

    i = 0
    for batch_lst in loader:
        if content_mode == "item":
            for item in batch_lst:
                fname = fnames[i]
                fpath = content_dpath.joinpath(fname)
                if custom_file_fmt is not None:
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                save_fn(item, fpath)
                i += 1

        elif content_mode == "batch":
            fname = fnames[i]
            fpath = content_dpath.joinpath(fname)
            if custom_file_fmt is not None:
                fpath.parent.mkdir(parents=True, exist_ok=True)
            save_fn(batch_lst, fpath)
            i += 1

        else:
            raise ValueError(f"Invalid argument {content_mode=}.")

    if hasattr(dataset, "info"):
        info = dataset.info  # type: ignore
        if is_dataclass(info):
            info = asdict(info)
        elif is_mapping_str(info):
            info = dict(info.items())  # type: ignore
        else:
            info = {}
    else:
        info = {}

    if hasattr(dataset, "attrs"):
        source_attrs = dataset.attrs  # type: ignore
        if is_dataclass(source_attrs):
            info = asdict(source_attrs)
        elif is_mapping_str(source_attrs):
            source_attrs = dict(source_attrs.items())  # type: ignore
        else:
            pylog.warning(f"Ignore source attributes type {type(source_attrs)}.")
            source_attrs = {}
    else:
        source_attrs = {}

    attributes = {
        "source_dataset": wrapped.unwrap().__class__.__name__,
        "length": len(wrapped),
        "creation_date": creation_date,
        "batch_size": batch_size,
        "content_mode": content_mode,
        "content_dname": CONTENT_DNAME,
        "info": info,
        "source_attrs": source_attrs,
        "num_files": len(fnames),
        "files": fnames,
    }

    attrs_fpath = root.joinpath(ATTRS_FNAME)
    with open(attrs_fpath, "w") as file:
        json.dump(attributes, file, indent="\t")

    pickle_dataset = PickleDataset(root)
    return pickle_dataset
