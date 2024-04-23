#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import json
import math
import shutil
from pathlib import Path
from typing import Callable, List, Literal, Optional, TypeVar, Union

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from torchoutil.utils.data.dataloader import get_auto_num_cpus
from torchoutil.utils.data.dataset import SizedDatasetLike
from torchoutil.utils.pickle_dataset.common import (
    CONTENT_DNAME,
    INFO_FNAME,
    ContentMode,
)
from torchoutil.utils.pickle_dataset.dataset import PickleDataset

T = TypeVar("T")
U = TypeVar("U")


@torch.inference_mode()
def pack_to_pickle(
    dataset: SizedDatasetLike[T],
    root: Union[str, Path],
    pre_transform: Optional[Callable[[T], U]] = None,
    batch_size: int = 32,
    num_workers: Union[int, Literal["auto"]] = "auto",
    overwrite: bool = False,
    content_mode: ContentMode = "item",
    fmt: Optional[str] = None,
    save_fn: Callable[[Union[U, List[U]], Path], None] = torch.save,
) -> PickleDataset[U, U]:
    """Pack a dataset to pickle files.

    Args:
        dataset: Dataset-like to pack.
        root: Directory to store pickled data.
        pre_transform: Transform to apply to each item before saving. defaults to None.
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

    if pre_transform is None:
        pre_transform = nn.Identity()

    now = datetime.datetime.now()
    creation_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    loader = DataLoader(
        dataset,  # type: ignore
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=nn.Identity(),
        drop_last=False,
        pin_memory=False,
    )

    if content_mode == "item":
        num_files = len(dataset)
    elif content_mode == "batch":
        num_files = math.ceil(len(dataset) / batch_size)
    else:
        raise ValueError(f"Invalid argument {content_mode=}.")

    if fmt is None:
        num_digits = math.ceil(math.log10(num_files))
        fmt = f"{{i:0{num_digits}d}}.pt"

    fnames = [fmt.format(i=i) for i in range(num_files)]

    i = 0
    for batch_lst in loader:
        batch_lst = [pre_transform(item) for item in batch_lst]

        if content_mode == "item":
            for item in batch_lst:
                fname = fnames[i]
                path = content_dpath.joinpath(fname)
                save_fn(item, path)
                i += 1

        elif content_mode == "batch":
            fname = fnames[i]
            path = content_dpath.joinpath(fname)
            save_fn(batch_lst, path)
            i += 1

        else:
            raise ValueError(f"Invalid argument {content_mode=}.")

    info = {
        "source_dataset": dataset.__class__.__name__,
        "length": len(dataset),
        "creation_date": creation_date,
        "batch_size": batch_size,
        "content_mode": content_mode,
        "content_dname": CONTENT_DNAME,
        "num_files": len(fnames),
        "files": fnames,
    }

    info_fpath = root.joinpath(INFO_FNAME)
    with open(info_fpath, "w") as file:
        json.dump(info, file, indent="\t")

    dataset = PickleDataset(root)
    return dataset
