#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import pickle
import sys
from io import BufferedReader
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

from typing_extensions import override

from torchoutil.pyoutil import is_iterable_str, open_close_wrap
from torchoutil.types.tensor_typing import Tensor1D
from torchoutil.utils.data import DatasetSlicer
from torchoutil.utils.pack.common import (
    ATTRS_FNAME,
    ContentMode,
    ItemType,
    PackedDatasetAttributes,
    _dict_to_tuple,
)

T = TypeVar("T")
U = TypeVar("U")
LoadFn = Callable[[BufferedReader], T]

pylog = logging.getLogger(__name__)


class PackedDataset(Generic[T, U], DatasetSlicer[U]):
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable[[T], U]] = None,
        *,
        load_fn: LoadFn[Union[T, List[T]]] = pickle.load,
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

        super().__init__()
        self._root = root
        self._transform = transform
        self._load_fn = load_fn

        self._attrs = {}
        self._fpaths: list[Path] = []
        self._column_to_fname: Dict[str, str] = {}
        self._reload_data(root, load_fn)

    @property
    def attrs(self) -> PackedDatasetAttributes:
        return self._attrs  # type: ignore

    @property
    def content_mode(self) -> Optional[ContentMode]:
        return self._attrs.get("content_mode")

    @property
    def item_type(self) -> ItemType:
        return self._attrs.get("item_type", "raw")

    @property
    def batch_size(self) -> int:
        return self._attrs["batch_size"]

    @override
    def __len__(self) -> int:
        return self._attrs["length"]

    def to_dict(self) -> Dict[str, Sequence]:
        if self.item_type != "dict":
            msg = f"Cannot convert non-dict dataset to dict. (found {self.item_type=})"
            raise ValueError(msg)

        if self.content_mode == "column":
            return self._load_item_from_columns(slice(None), None)
        else:
            return self[:]  # type: ignore

    @override
    def get_item(self, idx: int) -> Union[T, U]:
        if self.content_mode == "column":
            item = self._load_item_from_columns(idx)
            if self._transform is not None:
                item = self._transform(item)  # type: ignore
            return item
        elif self.content_mode == "item":
            batch_size = 1
        elif self.content_mode == "batch":
            batch_size = self.batch_size
        else:
            msg = f"Invalid PickleDataset state. (cannot load item with {self.content_mode=})"
            raise RuntimeError(msg)

        target_idx = idx // batch_size
        path = self._fpaths[target_idx]
        with open(path, "rb") as file:
            item_or_batch = self._load_fn(file)

        if self.content_mode == "item":
            item: T = item_or_batch  # type: ignore
        else:
            batch: List[T] = item_or_batch  # type: ignore
            local_idx = idx % batch_size
            item = batch[local_idx]

        if self._transform is not None:
            item = self._transform(item)  # type: ignore
        return item  # type: ignore

    @override
    def get_items_indices(
        self,
        indices: Union[Iterable[int], Tensor1D],
        *args,
    ) -> List[U]:
        if self.content_mode == "column":
            return self._load_item_from_columns(indices)
        else:
            return super().get_items_indices(indices, *args)

    @override
    def get_items_mask(
        self,
        mask: Union[Iterable[bool], Tensor1D],
        *args,
    ) -> List[U]:
        if self.content_mode == "column":
            return self._load_item_from_columns(mask)
        else:
            return super().get_items_mask(mask, *args)

    @override
    def get_items_slice(
        self,
        slice_: slice,
        *args,
    ) -> List[U]:
        if self.content_mode == "column":
            return self._load_item_from_columns(slice_)
        else:
            return super().get_items_slice(slice_, *args)

    def _load_item_from_columns(
        self,
        idx: Union[int, Iterable[int], Iterable[bool], Tensor1D, slice],
        column: Union[str, Iterable[str], None] = None,
    ) -> Any:
        if isinstance(column, str):
            columns = [column]
        elif is_iterable_str(column):
            columns = column
        elif column is None:
            columns = self._column_to_fname.keys()

        fnames = dict.fromkeys(self._column_to_fname[column_i] for column_i in columns)
        fpaths = [
            fpath for fname in fnames for fpath in self._fpaths if fpath.name == fname
        ]
        loaded = {fpath.name: open_close_wrap(self._load_fn, fpath) for fpath in fpaths}
        fname_to_column = {
            fname: column for column, fname in self._column_to_fname.items()
        }

        item = {fname_to_column[fname]: v[idx] for fname, v in loaded.items()}  # type: ignore
        if self.item_type == "tuple":
            item = _dict_to_tuple(item)  # type: ignore
        return item

    def _reload_data(
        self,
        root: Path,
        load_fn: LoadFn[Union[T, List[T]]],
    ) -> None:
        attrs_fpath = root.joinpath(ATTRS_FNAME)

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
            msg = f"Missing {len(missing)} keys in attribute file. (with {missing=} in {attrs_fpath=})"
            raise RuntimeError(msg)

        content_dname = attrs["content_dname"]
        content_dpath = root.joinpath(content_dname)

        fnames = attrs["files"]
        fpaths = [content_dpath.joinpath(fname) for fname in fnames]
        column_to_fname = attrs.get("column_to_fname", {})

        self._root = root
        self._load_fn = load_fn
        self._attrs = attrs
        self._fpaths = fpaths
        self._column_to_fname = column_to_fname

    @classmethod
    def is_pickle_root(cls, root: Union[str, Path]) -> bool:
        msg = "Call classmethod `is_pickle_root` is deprecated. Please use `is_packed_root` instead."
        pylog.warning(msg)
        return cls.is_packed_root(root)

    @classmethod
    def is_packed_root(cls, root: Union[str, Path]) -> bool:
        try:
            PackedDataset(root)
            return True
        except (FileNotFoundError, RuntimeError):
            return False
