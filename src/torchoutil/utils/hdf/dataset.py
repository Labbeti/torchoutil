#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import os
import os.path as osp
import pickle
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import h5py
import numpy as np
import torch
from h5py import Dataset as HDFRawDataset
from torch import Tensor
from typing_extensions import TypeGuard

import torchoutil as to
from torchoutil.nn.functional.indices import get_inverse_perm
from torchoutil.pyoutil.collections import all_eq
from torchoutil.pyoutil.typing import (
    is_iterable_bytes_or_list,
    is_iterable_int,
    is_iterable_str,
)
from torchoutil.utils.data import DatasetSlicer
from torchoutil.utils.hdf.common import (
    HDF_ENCODING,
    HDF_STRING_DTYPE,
    SHAPE_SUFFIX,
    HDFDatasetAttributes,
    HDFItemType,
)
from torchoutil.utils.pack.common import _dict_to_tuple

T = TypeVar("T")
U = TypeVar("U")

IndexLike = Union[int, Iterable[int], Tensor, slice, None]
ColumnLike = Union[str, Iterable[str], None]

pylog = logging.getLogger(__name__)


class HDFDataset(Generic[T, U], DatasetSlicer[U]):
    def __init__(
        self,
        hdf_fpath: Union[str, Path],
        transform: Optional[Callable[[T], U]] = None,
        keep_padding: Iterable[str] = (),
        return_added_columns: bool = False,
        open_hdf: bool = True,
        auto_open: bool = False,
        cast: Literal[
            "to_torch_or_builtin", "to_torch_or_numpy", "to_builtin", "none"
        ] = "to_torch_or_builtin",
        file_kwds: Optional[Dict[str, Any]] = None,
    ) -> None:
        """HDFDataset to read an packed hdf file.

        In HDF, all tensors are padded internally then cropped on-the-fly.
        Several options allows to extract non-padded tensors or return the internal shape of each column.

        Args:
            hdf_fpath: The path to the HDF file.
            transforms: The transform to apply values. default to None.
            keep_padding: Keys to keep padding values. defaults to ().
            return_added_columns: Returns the columns added by pack_to_hdf(.) function.
            open_hdf: If True, open the HDF file at start. defaults to True.
            auto_open: If True and open_hdf=False, it will open HDF file only when __getitem__ or __len__ is called. defaults to False.
            numpy_to_torch: If True, converts numpy array to PyTorch tensors. defaults to True.
            file_kwds: Options given to h5py file object. defaults to None.
        """
        hdf_fpath = Path(hdf_fpath).resolve().expanduser()
        if not hdf_fpath.is_file():
            names = os.listdir(osp.dirname(hdf_fpath))
            names = [name for name in names if name.endswith(".hdf")]
            names = list(sorted(names))
            names_str = "\n - ".join(names)
            msg = f"Cannot find HDF file in path {hdf_fpath=}."
            if len(names) > 0:
                msg += f" Possible HDF files are:\n - {names_str}"
            raise FileNotFoundError(msg)

        keep_padding = list(keep_padding)
        if file_kwds is None:
            file_kwds = {}

        super().__init__(
            add_indices_support=False,
            add_mask_support=False,
            add_slice_support=False,
        )
        self._hdf_fpath = hdf_fpath
        self._transform = transform
        self._keep_padding = keep_padding
        self._return_added_columns = return_added_columns
        self._auto_open = auto_open
        self._cast = cast
        self._file_kwds = file_kwds

        self._hdf_file: Any = None

        if open_hdf:
            self.open()

    # Properties
    @property
    def added_columns(self) -> List[str]:
        """Return the list of columns added by pack_to_hdf function."""
        return list(self._hdf_file.attrs.get("added_columns", []))

    @property
    def all_columns(self) -> List[str]:
        """The name of all columns of the dataset."""
        return list(self.get_hdf_keys())

    @cached_property
    def attrs(self) -> HDFDatasetAttributes:
        attrs = dict(self._hdf_file.attrs)
        return attrs  # type: ignore

    @property
    def column_names(self) -> List[str]:
        """The name of each column of the dataset."""
        column_names = self.all_columns
        column_names = [
            name
            for name in column_names
            if self._return_added_columns or name not in self.added_columns
        ]
        return column_names

    @cached_property
    def info(self) -> Dict[str, Any]:
        """Return the global dataset info."""
        return json.loads(self._hdf_file.attrs.get("info", "{}"))

    @property
    def item_type(self) -> HDFItemType:
        """Return the global dataset info."""
        return self._hdf_file.attrs.get("item_type", "dict")

    @property
    def keep_padding(self) -> List[str]:
        return self._keep_padding

    @keep_padding.setter
    def keep_padding(self, new_value: Iterable[str]) -> None:
        self._keep_padding = list(new_value)

    @property
    def metadata(self) -> str:
        return self.user_attrs

    @property
    def num_columns(self) -> int:
        return len(self.column_names)

    @property
    def num_rows(self) -> int:
        return len(self)

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the Clotho dataset."""
        return len(self), len(self.column_names)

    @property
    def shape_suffix(self) -> str:
        """Return the tensor shape suffix in column names."""
        return self._hdf_file.attrs.get("shape_suffix", SHAPE_SUFFIX)

    @property
    def transform(self) -> Optional[Callable[[T], U]]:
        return self._transform

    @cached_property
    def user_attrs(self) -> Any:
        if "user_attrs" in self.attrs:
            return json.loads(self._hdf_file.attrs.get("user_attrs", "null"))
        else:
            return self._hdf_file.attrs.get("metadata", "")

    @property
    def _encoding(self) -> str:
        return self.attrs.get("encoding", HDF_ENCODING)

    @cached_property
    def _is_unicode(self) -> Dict[str, bool]:
        return json.loads(self._hdf_file.attrs.get("is_unicode", "{}"))

    @cached_property
    def _load_as_complex(self) -> Dict[str, bool]:
        return json.loads(self._hdf_file.attrs.get("load_as_complex", "{}"))

    # Public methods
    @overload
    def get_item(self, index: int) -> T:
        ...

    @overload
    def get_item(self, index: Union[Iterable[int], slice, None], column: str) -> List:
        ...

    @overload
    def get_item(self, index: Union[Iterable[int], slice, None]) -> Dict[str, List]:
        ...

    @overload
    def get_item(
        self,
        index: Union[Iterable[int], slice, None],
        column: Union[Iterable[str], None],
    ) -> Dict[str, List]:
        ...

    @overload
    def get_item(self, index: Any, column: Any) -> Any:
        ...

    def get_item(
        self,
        index: IndexLike = None,
        column: ColumnLike = None,
        raw: bool = False,
    ) -> Any:
        if self.is_closed():
            if self._auto_open:
                self.open()
            else:
                msg = f"Cannot get_raw value with closed HDF file. ({self._hdf_file is not None=} and {bool(self._hdf_file)=})"
                raise RuntimeError(msg)

        if index is None:
            index = slice(None)
        elif isinstance(index, Tensor):
            index = index.tolist()
        if column is None:
            column = self.column_names

        if is_iterable_str(column, accept_str=False):
            return {column_i: self.get_item(index, column_i) for column_i in column}

        if column not in self.all_columns:
            msg = f"Invalid argument {column=}. (expected one of {tuple(self.all_columns)})"
            raise ValueError(msg)

        if isinstance(index, slice) or is_iterable_int(index):
            is_mult = True
        elif isinstance(index, int):
            if not (-len(self) <= index < len(self)):
                msg = f"Invalid argument {index=}. (expected int in range [{-len(self)}, {len(self)-1}])"
                raise IndexError(msg)
            is_mult = False
        else:
            raise TypeError(f"Invalid argument type {type(index)=}.")

        hdf_value = self._raw_get_item(index, column)
        if raw:
            return hdf_value

        if is_mult:
            hdf_values = hdf_value
        else:
            hdf_values = hdf_value[None]
        del hdf_value

        shape_name = f"{column}{self.shape_suffix}"
        must_remove_padding = (
            shape_name in self._hdf_file.keys() and column not in self._keep_padding
        )
        hdf_ds: HDFRawDataset = self._hdf_file[column]
        hdf_dtype: np.dtype = hdf_ds.dtype

        if must_remove_padding:
            shapes = self._raw_get_item(index, shape_name)
            if not is_mult:
                shapes = shapes[None]
            slices_lst = [
                tuple(slice(shape_i) for shape_i in shape) for shape in shapes
            ]
        else:
            slices_lst = [None] * int(hdf_ds.shape[0])

        outputs = []

        if self._is_unicode.get(column, False):
            hdf_values = np.char.decode(hdf_values, encoding=self._encoding)
        elif hdf_dtype == HDF_STRING_DTYPE:  # old supports vlen_str
            hdf_values = _decode_rec(hdf_values, encoding=self._encoding)

        for hdf_value, slices in zip(hdf_values, slices_lst, strict=True):
            # Remove the padding part
            if slices is not None:
                hdf_value = hdf_value[slices]

            if self._cast == "none":
                continue

            elif self._cast == "to_torch_or_builtin":
                if hdf_dtype.kind not in ("V", "S", "O"):
                    hdf_value = to.numpy_to_tensor(hdf_value)
                else:
                    hdf_value = hdf_value.tolist()

            elif self._cast == "to_torch_or_numpy":
                if hdf_dtype.kind not in ("V", "S", "O"):
                    hdf_value = to.numpy_to_tensor(hdf_value)

            elif self._cast == "to_builtin":
                hdf_value = hdf_value.tolist()

            outputs.append(hdf_value)

        if not is_mult:
            outputs = outputs[0]
        return outputs

    def close(self, ignore_if_closed: bool = False) -> None:
        if not self.is_open():
            if ignore_if_closed:
                return None
            else:
                raise RuntimeError("Cannot close the HDF file twice.")

        self._hdf_file.close()
        self._hdf_file = None
        self._clear_caches()

    def get_attrs(self) -> Dict[str, Any]:
        return self._hdf_file.attrs

    def get_hdf_fpath(self) -> Path:
        return self._hdf_fpath

    def get_hdf_keys(self) -> Tuple[str, ...]:
        if self.is_open():
            return tuple(self._hdf_file.keys())
        else:
            raise RuntimeError("Cannot get keys from a closed HDF file.")

    def get_column_shape(self, column_name: str) -> Tuple[int, ...]:
        if not self.is_open():
            msg = f"Cannot get max_shape with a closed HDF file. ({self._hdf_file is not None=} and {bool(self._hdf_file)=})"
            raise RuntimeError(msg)
        return tuple(self._hdf_file[column_name].shape)

    def is_closed(self) -> bool:
        return not self.is_open()

    def is_open(self) -> bool:
        return self._hdf_file is not None and bool(self._hdf_file)

    def open(self, ignore_if_opened: bool = False) -> None:
        if self.is_open():
            if ignore_if_opened:
                return None
            else:
                raise RuntimeError("Cannot open the HDF file twice.")

        self._clear_caches()
        self._hdf_file = h5py.File(self._hdf_fpath, "r", **self._file_kwds)
        self._sanity_check()

    # Magic methods
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, HDFDataset) and pickle.dumps(self) == pickle.dumps(__o)

    def __exit__(self) -> None:
        if self.is_open():
            self.close()

    @overload
    def __getitem__(self, index: int) -> U:
        ...

    @overload
    def __getitem__(
        self,
        index: Union[Iterable[int], slice, None],
    ) -> Dict[str, list]:
        ...

    @overload
    def __getitem__(self, index: Any) -> Any:
        ...

    def __getitem__(
        self,
        index: Union[IndexLike, Tuple[IndexLike, ColumnLike]],
    ) -> Any:
        if (
            isinstance(index, tuple)
            and len(index) == 2
            and _is_index(index[0])
            and _is_column(index[1])
        ):
            index, column = index
        else:
            column = None

        item = self.get_item(index, column)  # type: ignore

        if isinstance(index, int) and (
            column is None
            or (
                isinstance(column, Iterable)
                and not isinstance(column, str)
                and set(column) == set(self.column_names)
            )
        ):
            if self.item_type == "tuple":
                item = _dict_to_tuple(item)
            if self._transform is not None:
                item = self._transform(item)  # type: ignore
        return item

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "hdf_fpath": self._hdf_fpath,
            "transform": self._transform,
            "keep_padding": self._keep_padding,
            "return_added_columns": self._return_added_columns,
            "auto_open": self._auto_open,
            "cast": self._cast,
            "file_kwds": self._file_kwds,
            "is_open": self.is_open(),
        }

    def __hash__(self) -> int:
        hash_value = 0
        if self.is_open():
            hash_value += self._hdf_file.attrs["global_hash_value"]
        if self._transform is not None:
            hash_value += hash(self._transform)
        hash_value += sum(map(hash, self._keep_padding))
        return hash_value

    def __len__(self) -> int:
        auto_close = False
        if self.is_closed():
            if self._auto_open:
                self.open()
                auto_close = True
            else:
                msg = f"Cannot length of a closed HDF file. ({self._hdf_file is not None=} and {bool(self._hdf_file)=})"
                raise RuntimeError(msg)

        if "length" in self._hdf_file.attrs:
            length = self._hdf_file.attrs["length"]
        elif len(self._hdf_file) > 0:
            dataset: HDFRawDataset = next(iter(self._hdf_file.values()))
            length = len(dataset)
        else:
            length = 0

        if auto_close:
            self.close()
        return length

    def __repr__(self) -> str:
        repr_hparams = {"file": osp.basename(self._hdf_fpath), "size": len(self)}
        repr_ = ", ".join(f"{k}={v}" for k, v in repr_hparams.items())
        return f"HDFDataset({repr_})"

    def __setstate__(self, data: Dict[str, Any]) -> None:
        is_init = hasattr(self, "_hdf_fpath") and hasattr(self, "_hdf_file")
        files_are_different = is_init and self._hdf_fpath != data["hdf_fpath"]
        is_open = is_init and self.is_open()

        if is_init and files_are_different and is_open:
            self.close()

        self._hdf_fpath = data["hdf_fpath"]
        self._transform = data["transform"]
        self._keep_padding = data["keep_padding"]
        self._return_added_columns = data["return_added_columns"]
        self._auto_open = data["auto_open"]
        self._cast = data["cast"]
        self._file_kwds = data["file_kwds"]

        self._hdf_file = None

        if not is_init or (files_are_different and is_open):
            self.open()

    # Private methods
    def _raw_get_item(
        self,
        index: Union[int, Iterable[int], slice],
        column: str,
    ) -> np.ndarray:
        if isinstance(index, Iterable):
            sorted_idxs, local_idxs = torch.as_tensor(index).sort(dim=-1)
            sorted_idxs = sorted_idxs.numpy()
            hdf_value: Any = self._hdf_file[column][sorted_idxs]
            inv_local_idxs = get_inverse_perm(local_idxs).numpy()
            # TODO: rm
            # hdf_value = [hdf_value[local_idx] for local_idx in inv_local_idxs]
            hdf_value = hdf_value[inv_local_idxs]
            if self._load_as_complex.get(column, False):
                hdf_value = [to.view_as_complex(value) for value in hdf_value]
        else:
            hdf_value: Any = self._hdf_file[column][index]
            if self._load_as_complex.get(column, False):
                hdf_value = to.view_as_complex(hdf_value)

            hdf_value = np.array(hdf_value)
        return hdf_value

    def _sanity_check(self) -> None:
        lens = [dset.shape[0] for dset in self._hdf_file.values()]
        if not all_eq(lens) or lens[0] != len(self):
            pylog.error(
                f"Incorrect length stored in HDF file. (found {lens=} and {len(self)=})"
            )

        if not hasattr(self, "__orig_class__"):
            return None

        t_type = self.__orig_class__.__args__[0]  # type: ignore
        if t_type is not Any and (
            (issubclass(t_type, dict) and self.item_type != "dict")
            or (issubclass(t_type, tuple) and self.item_type != "tuple")
        ):
            raise TypeError(
                f"Invalid HDFDataset typing. (found specified type '{t_type.__name__}' but the internal dataset contains type '{self.item_type}')"
            )

    def _clear_caches(self) -> None:
        if hasattr(self, "info"):
            del self.info
        if hasattr(self, "user_attrs"):
            del self.user_attrs
        if hasattr(self, "_is_unicode"):
            del self._is_unicode
        if hasattr(self, "_load_as_complex"):
            del self._load_as_complex


def _decode_rec(
    encoded: Union[bytes, np.ndarray, Iterable],
    encoding: str,
    to_builtin: bool = False,
) -> Union[str, np.ndarray, list]:
    """Decode bytes to str with the specified encoding. Works recursively on list of bytes, list of list of bytes, etc."""
    if isinstance(encoded, bytes):
        return encoded.decode(encoding=encoding)
    elif isinstance(encoded, np.ndarray) and encoded.dtype.kind in ("S",):
        decoded = np.char.decode(encoded, encoding=encoding)
        if to_builtin:
            decoded = decoded.tolist()
        return decoded
    elif is_iterable_bytes_or_list(encoded):
        return [_decode_rec(elt, encoding, to_builtin=to_builtin) for elt in encoded]
    else:
        msg = f"Invalid argument type {type(encoded)}. (expected bytes, np array or Iterable)"
        raise TypeError(msg)


def _is_index(index: Any) -> TypeGuard[IndexLike]:
    return (
        isinstance(index, int)
        or is_iterable_int(index)
        or isinstance(index, slice)
        or index is None
        or (isinstance(index, Tensor) and not index.is_floating_point())
    )


def _is_column(column: Any) -> TypeGuard[ColumnLike]:
    return is_iterable_str(column, accept_str=True) or column is None
