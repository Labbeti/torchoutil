#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import json
import logging
import os
import os.path as osp
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Union

import torch
from torch.utils.data.dataset import Dataset

from torchoutil.utils.pickle_dataset.common import CONTENT_DNAME, INFO_FNAME

pylog = logging.getLogger(__name__)

CheckFileMode = Literal["raise", "exclude", "no_check"]


class PickleDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path],
        *,
        map_location: Union[str, torch.device, None] = None,
        check_files_mode: CheckFileMode = "raise",
        verbose: int = 0,
    ) -> None:
        root = Path(root)

        super().__init__()
        self._root = root
        self._map_location = map_location
        self._check_files_mode: CheckFileMode = check_files_mode
        self._verbose = verbose

        self._fpaths = []
        self._info = {}
        self._reload_data()

    @property
    def info(self) -> Dict[str, Any]:
        return self._info

    def __getitem__(self, idx: int) -> Any:
        path = self._fpaths[idx]
        data = torch.load(path, map_location=self._map_location)
        return data

    def __len__(self) -> int:
        return len(self._fpaths)

    def _reload_data(self) -> None:
        info_fpath = self._root.joinpath(INFO_FNAME)

        if info_fpath.is_file():
            with open(info_fpath, "r") as file:
                info = json.load(file)
        else:
            info = {}

        content_dname = info.get("content_dname", CONTENT_DNAME)
        content_dpath = self._root.joinpath(content_dname)

        fpaths = _search_fpaths(
            content_dpath,
            use_glob=False,
            check_files_mode="raise",
            verbose=self._verbose,
        )

        self._fpaths = fpaths
        self._info = info


def _search_fpaths(
    paths: Union[str, Path, Iterable[Union[str, Path]]],
    use_glob: bool = False,
    check_files_mode: CheckFileMode = "raise",
    verbose: int = 0,
) -> List[str]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    else:
        paths = list(paths)

    paths = [str(path) if isinstance(path, Path) else path for path in paths]

    if use_glob:
        fpaths = [match for path in paths for match in glob.glob(path)]
    else:
        fpaths = [path for path in paths if osp.isfile(path)]
        fpaths += [
            osp.join(path, fname)
            for path in paths
            if osp.isdir(path)
            for fname in os.listdir(path)
        ]

    if check_files_mode == "no_check":
        return fpaths

    invalids = [not osp.isfile(fpath) for fpath in fpaths]
    num_invalids = sum(invalids)
    if num_invalids == 0:
        return fpaths

    if check_files_mode == "raise":
        raise FileNotFoundError(
            f"Invalid argument {paths=}. (found {num_invalids}/{len(fpaths)} invalid files with {check_files_mode=})"
        )
    else:
        if verbose >= 2:
            pylog.debug(
                f"Excluding {num_invalids}/{len(fpaths)} files. (with {check_files_mode=})"
            )
        fpaths = [fpath for fpath, invalid_i in zip(fpaths, invalids) if not invalid_i]
        return fpaths
