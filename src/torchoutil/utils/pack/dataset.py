#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

import torch
from torch.utils.data.dataset import Dataset

from torchoutil.utils.pack.common import (
    ATTRS_FNAME,
    ContentMode,
    PackedDatasetAttributes,
)

pylog = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")


class PackedDataset(Generic[T, U], Dataset[U]):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable[[T], U]] = None,
        *,
        load_fn: Callable[..., Union[T, List[T]]] = torch.load,
        load_kwds: Optional[Dict[str, Any]] = None,
        use_cache: bool = False,
    ) -> None:
        """

        Args:
            root: Root directory containing the info.json and the data files.
            transform: Optional transform to apply to each item. defaults to None.
            load_fn: Load function to load an item or batch. defaults to torch.load.
            load_kwds: Keywords arguments passed to load_fn. defaults to None.
            use_cache: If True, cache each item or batch in memory. defaults to False.
        """
        root = Path(root)
        if load_kwds is None:
            load_kwds = {}

        super().__init__()
        self._root = root
        self._transform = transform
        self._load_fn = load_fn
        self._load_kwds = load_kwds
        self._use_cache = use_cache

        self._attrs = {}
        self._fpaths = []
        self._cache = None
        self._cache_idx = None
        self._reload_data()

    @property
    def attrs(self) -> PackedDatasetAttributes:
        return self._attrs  # type: ignore

    @property
    def content_mode(self) -> Optional[ContentMode]:
        return self._attrs.get("content_mode")

    @property
    def batch_size(self) -> int:
        return self._attrs["batch_size"]

    def __getitem__(self, idx: int) -> U:
        item = self._load_item(idx)
        if self._transform is not None:
            item = self._transform(item)
        return item  # type: ignore

    def __len__(self) -> int:
        return self._attrs["length"]

    def _load_item(self, idx: int) -> T:
        if self.content_mode == "item":
            batch_size = 1
        elif self.content_mode == "batch":
            batch_size = self.batch_size
        else:
            raise RuntimeError(
                f"Invalid PickleDataset state. (cannot load item with {self.content_mode=})"
            )

        target_cache_idx = idx // batch_size
        if self._cache is None or self._cache_idx != target_cache_idx:
            self._cache = None
            path = self._fpaths[target_cache_idx]
            item_or_batch = self._load_fn(path, **self._load_kwds)

            if self._use_cache:
                self._cache = item_or_batch
                self._cache_idx = target_cache_idx
        else:
            item_or_batch = self._cache

        if self.content_mode == "item":
            item: T = item_or_batch  # type: ignore
        else:
            batch: List[T] = item_or_batch  # type: ignore
            local_idx = idx % batch_size
            item = batch[local_idx]

        return item

    def _reload_data(self) -> None:
        attrs_fpath = self._root.joinpath(ATTRS_FNAME)

        if not attrs_fpath.is_file():
            raise FileNotFoundError(f"Cannot find attribute file '{str(attrs_fpath)}'.")

        with open(attrs_fpath, "r") as file:
            attrs = json.load(file)

        # Disable check for python <= 3.8 because __required_keys__ does not exists in this version
        if sys.version_info.major < 3 or (
            sys.version_info.major == 3 and sys.version_info.minor <= 8
        ):
            missing = []
        else:
            missing = list(
                set(PackedDatasetAttributes.__required_keys__).difference(attrs)
            )

        if len(missing) > 0:
            raise RuntimeError(
                f"Missing {len(missing)} keys in attribute file. (with {missing=})"
            )

        content_dname = attrs["content_dname"]
        content_dpath = self._root.joinpath(content_dname)

        fnames = attrs["files"]
        fpaths = [content_dpath.joinpath(fname) for fname in fnames]

        self._attrs = attrs
        self._fpaths = fpaths

    @classmethod
    def is_pickle_root(cls, root: Union[str, Path]) -> bool:
        try:
            PackedDataset(root)
            return True
        except (FileNotFoundError, RuntimeError):
            return False
