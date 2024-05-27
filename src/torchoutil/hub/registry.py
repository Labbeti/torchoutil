#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import os
import os.path as osp
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

import torch
from torch import Tensor
from torch.types import Device
from typing_extensions import NotRequired

from torchoutil.hub.download import HashType, hash_file
from torchoutil.nn.functional.get import get_device

pylog = logging.getLogger(__name__)


T = TypeVar("T")


class RegistryEntry(TypedDict):
    url: str
    fname: str
    hash_value: NotRequired[str]
    hash_type: NotRequired[HashType]
    state_dict_key: NotRequired[str]
    architecture: NotRequired[str]


class RegistryHub(Generic[T]):
    def __init__(
        self,
        infos: Mapping[T, RegistryEntry],
        register_root: Union[str, Path, None] = None,
    ) -> None:
        """
        Args:
            infos: Maps model_name to their checkpoint information, with download url, filename, hash value, hash type and state_dict key.
            register_root: Directory where checkpoints are saved. If None, defaults to `~/.cache/torch/hub/checkpoints`.
        """
        infos = dict(infos.items())
        if register_root is None:
            register_root = get_default_register_root()
        else:
            register_root = Path(register_root)

        super().__init__()
        self._infos = infos
        self._ckpt_parent_path = register_root

    @property
    def infos(self) -> Dict[T, RegistryEntry]:
        return self._infos

    @property
    def register_root(self) -> Path:
        return self._ckpt_parent_path.resolve()

    @property
    def names(self) -> List[T]:
        return list(self._infos.keys())

    @property
    def paths(self) -> List[Path]:
        return [self.get_path(model_name) for model_name in self.names]

    def get_path(self, name: T) -> Path:
        if name not in self.names:
            raise ValueError(
                f"Invalid argument {name=}. (expected one of {self.names})"
            )

        fname = self._infos[name]["fname"]
        fpath = self.register_root.joinpath(fname)
        return fpath

    def load_state_dict(
        self,
        name_or_path: Union[T, str, Path],
        *,
        device: Device = None,
        offline: bool = False,
        load_fn: Callable = torch.load,
        verbose: int = 0,
    ) -> Dict[str, Tensor]:
        """Load state_dict weights.

        Args:
            model_name_or_path: Model name (case sensitive) or path to checkpoint file.
            device: Device of checkpoint weights.
            offline: If False, the checkpoint from a model name will be automatically downloaded.
            load_fn: Load function backend. defaults to torch.load.
            verbose: Verbose level. defaults to 0.

        Returns:
            Loaded file content.
        """
        device = get_device(device)

        if osp.isfile(name_or_path):
            path = name_or_path
            name = self._get_name(path)
        else:
            name = name_or_path
            try:
                path = self.get_path(name_or_path)
            except ValueError:
                raise ValueError(
                    f"Invalid argument {name_or_path=}. (expected a path to a checkpoint file or a model name in {self.names})"
                )

            if not osp.isfile(path):
                if offline:
                    raise FileNotFoundError(
                        f"Cannot find checkpoint model file in '{path}' for model '{name_or_path}' with mode {offline=}."
                    )
                else:
                    self.download_file(name_or_path, verbose=verbose)

        del name_or_path

        data = load_fn(path, map_location=device)

        info = self._infos.get(name, {})  # type: ignore
        state_dict_key = info.get("state_dict_key", None)

        if state_dict_key is None:
            state_dict = data
        else:
            state_dict = data[state_dict_key]

        if verbose >= 1:
            test_map = data.get("test_mAP", "unknown")
            pylog.info(
                f"Loading encoder weights from '{path}'... (with test_mAP={test_map})"
            )

        return state_dict

    def download_file(
        self,
        name: T,
        force: bool = False,
        check_hash: bool = True,
        verbose: int = 0,
    ) -> Tuple[Path, bool]:
        """Download checkpoint file."""
        model_path = self.get_path(name)
        exists = model_path.exists()

        if exists and not force:
            return model_path, False

        if exists and force:
            os.remove(model_path)

        os.makedirs(model_path.parent, exist_ok=True)

        url = self._infos[name]["url"]
        torch.hub.download_url_to_file(url, str(model_path), progress=verbose >= 1)

        if not check_hash:
            return model_path, True

        valid = self.is_valid_hash(name)

        if valid:
            return model_path, True
        else:
            raise ValueError(f"Invalid hash for file '{model_path}'.")

    def is_valid_hash(
        self,
        name: T,
    ) -> bool:
        info = self.infos[name]
        if "hash_type" not in info or "hash_value" not in info:
            pylog.warning(
                f"Cannot check hash for {name}. (cannot find any expected hash value or type)"
            )
            return True

        hash_type = info["hash_type"]
        expected_hash_value = info["hash_value"]

        model_path = self.get_path(name)
        hash_value = hash_file(model_path, hash_type)
        valid = hash_value == expected_hash_value
        return valid

    def save(self, path: Union[str, Path]) -> None:
        """Save info to JSON file."""
        args = {
            "infos": self._infos,
            "register_root": str(self._ckpt_parent_path),
        }
        with open(path, "r") as file:
            json.dump(args, file)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "RegistryHub":
        """Load register info from JSON file."""
        with open(path, "r") as file:
            args = json.load(file)
        return RegistryHub(**args)

    def _get_name(self, path: Union[str, Path]) -> Optional[T]:
        path_to_name = {
            path_i.resolve().expanduser(): name_i
            for path_i, name_i in zip(self.paths, self.names)
        }
        path = Path(path).resolve().expanduser()
        if path in path_to_name:
            name = path_to_name[path]
        else:
            name = None
        return name


def get_default_register_root() -> Path:
    """Default register root path is `~/.cache/torch/hub/checkpoints`, which is based on `torch.hub.get_dir`."""
    path = torch.hub.get_dir()
    path = Path(path)
    path = path.joinpath("checkpoints")
    return path
