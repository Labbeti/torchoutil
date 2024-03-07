#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import os
import os.path as osp

from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict, Union

import torch

from torch import Tensor

from torchoutil.nn.functional.get import get_device


pylog = logging.getLogger(__name__)


class CheckpointInfo(TypedDict):
    architecture: str
    url: str
    hash: str
    fname: str


class ModelCheckpointRegister:
    def __init__(
        self,
        infos: Dict[str, CheckpointInfo],
        state_dict_key: Optional[str],
        ckpt_parent_path: Union[str, Path, None] = None,
    ) -> None:
        """
        Args:
            infos: Mapping model_name to their checkpoint information, with architecture, download url, hash value and filename.
            state_dict_key: Optional key in the checkpoint data where state_dict is stored.
            ckpt_parent_path: Directory where checkpoints are saved. If None, defaults to `~/.cache/torch/hub/checkpoints`.
        """
        if ckpt_parent_path is None:
            ckpt_parent_path = _default_ckpt_parent_path()
        else:
            ckpt_parent_path = Path(ckpt_parent_path)

        super().__init__()
        self._infos = infos
        self._state_dict_key = state_dict_key
        self._ckpt_parent_path = ckpt_parent_path

    @property
    def infos(self) -> Dict[str, CheckpointInfo]:
        return self._infos

    @property
    def state_dict_key(self) -> Optional[str]:
        return self._state_dict_key

    @property
    def ckpt_parent_path(self) -> Path:
        return self._ckpt_parent_path.resolve()

    @property
    def model_names(self) -> List[str]:
        return list(self._infos.keys())

    def get_ckpt_path(self, model_name: str) -> Path:
        if model_name not in self.model_names:
            raise ValueError(
                f"Invalid argument {model_name=}. (expected one of {self.model_names})"
            )

        fname = self._infos[model_name]["fname"]
        fpath = self.ckpt_parent_path.joinpath(fname)
        return fpath

    def load_state_dict(
        self,
        model_name_or_path: Union[str, Path],
        device: Union[str, torch.device, None] = None,
        offline: bool = False,
        verbose: int = 0,
    ) -> Dict[str, Tensor]:
        """Load state_dict weights.

        Args:
            model_name_or_path: Model name (case sensitive) or path to checkpoint file.
            device: Device of checkpoint weights.
            offline: If False, the checkpoint from a model name will be automatically downloaded.
            verbose: Verbose level. defaults to 0.

        Returns:
            State dict of model weights.
        """
        device = get_device(device)
        model_name_or_path = str(model_name_or_path)

        if osp.isfile(model_name_or_path):
            model_path = model_name_or_path
        else:
            try:
                model_path = self.get_ckpt_path(model_name_or_path)
            except ValueError:
                raise ValueError(
                    f"Invalid argument {model_name_or_path=}. (expected a path to a checkpoint file or a model name in {self.model_names})"
                )

            if not osp.isfile(model_path):
                if offline:
                    raise FileNotFoundError(
                        f"Cannot find checkpoint model file in '{model_path}' with mode {offline=}."
                    )
                else:
                    self.download_ckpt(model_name_or_path, verbose=verbose)

        del model_name_or_path

        data = torch.load(model_path, map_location=device)

        if self._state_dict_key is None:
            state_dict = data
        else:
            state_dict = data[self._state_dict_key]

        if verbose >= 1:
            test_map = data.get("test_mAP", "unknown")
            pylog.info(
                f"Loading encoder weights from '{model_path}'... (with test_mAP={test_map})"
            )

        return state_dict

    def download_ckpt(
        self,
        model_name: str,
        force: bool = False,
        verbose: int = 0,
    ) -> Tuple[Path, bool]:
        """Download checkpoint file."""
        fpath = self.get_ckpt_path(model_name)
        exists = fpath.exists()

        if exists and not force:
            return fpath, False

        if exists and force:
            os.remove(fpath)

        os.makedirs(fpath.parent, exist_ok=True)

        url = self._infos[model_name]["url"]
        torch.hub.download_url_to_file(url, str(fpath), progress=verbose >= 1)

        return fpath, True

    def save(self, path: Union[str, Path]) -> None:
        """Save info to JSON file."""
        args = {
            "infos": self._infos,
            "state_dict_key": self._state_dict_key,
            "ckpt_parent_path": str(self._ckpt_parent_path),
        }
        with open(path, "r") as file:
            json.dump(args, file)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ModelCheckpointRegister":
        """Load register info from JSON file."""
        with open(path, "r") as file:
            args = json.load(file)
        return ModelCheckpointRegister(**args)


def _default_ckpt_parent_path() -> Path:
    """Default checkpoint path: `~/.cache/torch/hub/checkpoints`."""
    path = torch.hub.get_dir()
    path = Path(path)
    path = path.joinpath("checkpoints")
    return path
