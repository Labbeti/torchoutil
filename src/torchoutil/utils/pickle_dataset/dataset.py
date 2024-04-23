#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.utils.data.dataset import Dataset

from torchoutil.utils.pickle_dataset.common import INFO_FNAME, ContentMode

pylog = logging.getLogger(__name__)


class PickleDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        *,
        map_location: Union[str, torch.device, None] = None,
    ) -> None:
        """

        Args:
            root: Root directory containing the info.json and the data files.
            map_location: Device to map items. defaults to None.
        """
        root = Path(root)

        super().__init__()
        self._root = root
        self._map_location = map_location

        self._info = {}
        self._fpaths = []
        self._cache = None
        self._cache_idx = None
        self._reload_data()

    @property
    def info(self) -> Dict[str, Any]:
        return self._info

    @property
    def content_mode(self) -> Optional[ContentMode]:
        return self._info.get("content_mode")

    @property
    def batch_size(self) -> int:
        return self._info["batch_size"]

    def __getitem__(self, idx: int) -> Any:
        return self._load_item(idx)

    def __len__(self) -> int:
        return self._info["length"]

    def _load_item(self, idx: int) -> Any:
        if self.content_mode == "item":
            path = self._fpaths[idx]
            data = torch.load(path, map_location=self._map_location)
            return data

        elif self.content_mode == "batch":
            target_cache_idx = idx // self.batch_size
            if self._cache is None or self._cache_idx != target_cache_idx:
                self._cache = None
                path = self._fpaths[target_cache_idx]
                data = torch.load(path, map_location=self._map_location)
                self._cache = data
                self._cache_idx = target_cache_idx
            else:
                data = self._cache

            local_idx = idx % self.batch_size
            return data[local_idx]

        else:
            raise RuntimeError(
                f"Invalid PickleDataset state. (cannot load item with {self.content_mode=})"
            )

    def _reload_data(self) -> None:
        info_fpath = self._root.joinpath(INFO_FNAME)

        if info_fpath.is_file():
            with open(info_fpath, "r") as file:
                info = json.load(file)
        else:
            info = {}

        content_dname = info["content_dname"]
        content_dpath = self._root.joinpath(content_dname)

        fnames = info["files"]
        fpaths = [content_dpath.joinpath(fname) for fname in fnames]

        self._info = info
        self._fpaths = fpaths
