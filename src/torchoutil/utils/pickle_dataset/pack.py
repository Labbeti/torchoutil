#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import json
import math
import shutil
from pathlib import Path
from typing import Callable, Literal, Optional, TypeVar, Union

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from torchoutil.utils.data.dataloader import get_auto_num_cpus
from torchoutil.utils.data.dataset import SizedDatasetLike
from torchoutil.utils.pickle_dataset.common import CONTENT_DNAME, INFO_FNAME
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
) -> PickleDataset:
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

    num_digits = math.ceil(math.log10(len(dataset)))
    fmt = f"{{i:0{num_digits}d}}.pt"

    i = 0
    for batch_lst in loader:
        batch_lst = [pre_transform(item) for item in batch_lst]
        for item in batch_lst:
            fname = fmt.format(i=i)
            path = content_dpath.joinpath(fname)
            torch.save(item, path)
            i += 1

    attributes = {
        "source_dataset": dataset.__class__.__name__,
        "length": len(dataset),
        "content_dname": CONTENT_DNAME,
        "creation_date": creation_date,
        "creation_kwargs": {
            "batch_size": batch_size,
            "num_workers": num_workers,
        },
    }

    info_fpath = root.joinpath(INFO_FNAME)
    with open(info_fpath, "w") as file:
        json.dump(attributes, file, indent="\t")

    dataset = PickleDataset(root)
    return dataset
