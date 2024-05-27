#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import os
import os.path as osp
import pickle
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
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
from torch.utils.data.dataset import Dataset
from typing_extensions import TypeGuard

from torchoutil.nn.functional.indices import get_inverse_perm
from torchoutil.utils.collections import all_eq
from torchoutil.utils.hdf.common import (
    HDF_ENCODING,
    HDF_STRING_DTYPE,
    HDF_VOID_DTYPE,
    SHAPE_SUFFIX,
    HDFDatasetAttributes,
    _dict_to_tuple,
)
from torchoutil.utils.type_checks import (
    is_iterable_bytes_list,
    is_iterable_int,
    is_iterable_str,
)

pylog = logging.getLogger(__name__)


T = TypeVar("T")
U = TypeVar("U")


IndexType = Union[int, Iterable[int], Tensor, slice, None]
ColumnType = Union[str, Iterable[str], None]


def _is_index(index: Any) -> TypeGuard[IndexType]:
    return (
        isinstance(index, int)
        or is_iterable_int(index)
        or isinstance(index, slice)
        or index is None
        or (isinstance(index, Tensor) and not index.is_floating_point())
    )


def _is_column(column: Any) -> TypeGuard[ColumnType]:
    return is_iterable_str(column, accept_str=True) or column is None


class HDFDataset(Generic[T, U], Dataset[U]):
    def __init__(
        self,
        hdf_fpath: Union[str, Path],
        transform: Optional[Callable[[T], U]] = None,
        keep_padding: Iterable[str] = (),
        return_added_columns: bool = False,
        open_hdf: bool = True,
        auto_open: bool = False,
        numpy_to_torch: bool = True,
        file_kwargs: Optional[Dict[str, Any]] = None,
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
            auto_open: If True, open HDF dataset only when __getitem__ or __len__ is called. defaults to False.
            numpy_to_torch: If True, converts numpy array to PyTorch tensors. defaults to True.
            file_kwargs: Options given to h5py file object. defaults to None.
        """
        hdf_fpath = Path(hdf_fpath).resolve()
        if not hdf_fpath.is_file():
            names = os.listdir(osp.dirname(hdf_fpath))
            names = [name for name in names if name.endswith(".hdf")]
            names = list(sorted(names))
            names_str = "\n - ".join(names)
            raise FileNotFoundError(
                f"Cannot find HDF file in path {hdf_fpath=}. Possible HDF files are:\n - {names_str}"
            )
        keep_padding = list(keep_padding)
        if file_kwargs is None:
            file_kwargs = {}

        super().__init__()
        self._hdf_fpath = hdf_fpath
        self._transform = transform
        self._keep_padding = keep_padding
        self._return_added_columns = return_added_columns
        self._auto_open = auto_open
        self._numpy_to_torch = numpy_to_torch
        self._file_kwargs = file_kwargs

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

    @property
    def attrs(self) -> HDFDatasetAttributes:
        return dict(self._hdf_file.attrs)  # type: ignore

    @property
    def metadata(self) -> str:
        return self.attrs.get("metadata", "")

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

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the Clotho dataset."""
        return len(self), len(self.column_names)

    @property
    def shape_suffix(self) -> str:
        """Return the tensor shape suffix in column names."""
        return self._hdf_file.attrs.get("shape_suffix", SHAPE_SUFFIX)

    @property
    def info(self) -> Dict[str, Any]:
        """Return the global dataset info."""
        return json.loads(self._hdf_file.attrs.get("info", "{}"))

    @property
    def item_type(self) -> str:
        """Return the global dataset info."""
        return str(self._hdf_file.attrs.get("item_type", "dict"))

    @property
    def num_columns(self) -> int:
        return len(self.column_names)

    @property
    def num_rows(self) -> int:
        return len(self)

    @property
    def transform(self) -> Optional[Callable[[T], U]]:
        return self._transform

    # Public methods
    @overload
    def at(self, index: int) -> T:
        ...

    @overload
    def at(self, index: Union[Iterable[int], slice, None], column: str) -> List:
        ...

    @overload
    def at(self, index: Union[Iterable[int], slice, None]) -> Dict[str, List]:
        ...

    @overload
    def at(
        self,
        index: Union[Iterable[int], slice, None],
        column: Union[Iterable[str], None],
    ) -> Dict[str, List]:
        ...

    @overload
    def at(self, index: Any, column: Any) -> Any:
        ...

    def at(
        self,
        index: IndexType = None,
        column: ColumnType = None,
        raw: bool = False,
    ) -> Any:
        if self.is_closed():
            if self._auto_open:
                self.open()
            else:
                raise RuntimeError(
                    f"Cannot get_raw value with closed HDF file. ({self._hdf_file is not None=} and {bool(self._hdf_file)=})"
                )

        if index is None:
            index = slice(None)
        elif isinstance(index, Tensor):
            index = index.tolist()
        if column is None:
            column = self.column_names

        if is_iterable_str(column, accept_str=False):
            return {column_i: self.at(index, column_i) for column_i in column}

        if column not in self.all_columns:
            raise ValueError(
                f"Invalid argument {column=}. (expected one of {tuple(self.all_columns)})"
            )

        if isinstance(index, slice) or is_iterable_int(index):
            is_mult = True
        elif isinstance(index, int):
            if not (-len(self) <= index < len(self)):
                raise IndexError(
                    f"Invalid argument {index=}. (expected int in range [{-len(self)}, {len(self)-1}])"
                )
            is_mult = False
        else:
            raise TypeError(f"Invalid argument type {type(index)=}.")

        hdf_value = self._raw_at(index, column)
        if raw:
            return hdf_value

        if is_mult:
            hdf_values = hdf_value
        else:
            hdf_values = [hdf_value]
        del hdf_value

        shape_name = f"{column}{self.shape_suffix}"
        must_remove_padding = (
            shape_name in self._hdf_file.keys() and column not in self._keep_padding
        )
        hdf_ds: HDFRawDataset = self._hdf_file[column]
        hdf_dtype = hdf_ds.dtype

        if must_remove_padding:
            shapes = self._raw_at(index, shape_name)
            if not is_mult:
                shapes = [shapes]
            slices_lst = [
                tuple(slice(shape_i) for shape_i in shape) for shape in shapes
            ]
        else:
            slices_lst = [None] * int(hdf_ds.shape[0])

        outputs = []

        for hdf_value, slices in zip(hdf_values, slices_lst):
            # Remove the padding part
            if slices is not None:
                hdf_value = hdf_value[slices]

            # Decode all bytes to string
            if hdf_dtype == HDF_STRING_DTYPE:
                hdf_value = _decode_rec(hdf_value, HDF_ENCODING)
            # Convert numpy.array to torch.Tensor
            elif isinstance(hdf_value, np.ndarray):
                if not self._numpy_to_torch:
                    pass
                elif hdf_dtype != HDF_VOID_DTYPE:
                    hdf_value = torch.from_numpy(hdf_value)
                else:
                    hdf_value = hdf_value.tolist()
            # Convert numpy scalars
            elif np.isscalar(hdf_value) and hasattr(hdf_value, "item"):
                hdf_value = hdf_value.item()  # type: ignore

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

        self._hdf_file = h5py.File(self._hdf_fpath, "r", **self._file_kwargs)
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
        index: Union[IndexType, Tuple[IndexType, ColumnType]],
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

        item = self.at(index, column)  # type: ignore

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
            "numpy_to_torch": self._numpy_to_torch,
            "file_kwargs": self._file_kwargs,
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
        self._numpy_to_torch = data["numpy_to_torch"]
        self._file_kwargs = data["file_kwargs"]

        self._hdf_file = None

        if not is_init or (files_are_different and is_open):
            self.open()

    # Private methods
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

    def _raw_at(self, index: Union[int, Iterable[int], slice], column: str) -> Any:
        if isinstance(index, Iterable):
            sorted_idxs, local_idxs = torch.as_tensor(index).sort(dim=-1)
            sorted_idxs = sorted_idxs.numpy()
            hdf_value: Any = self._hdf_file[column][sorted_idxs]
            inv_local_idxs = get_inverse_perm(local_idxs)
            hdf_value = [hdf_value[local_idx] for local_idx in inv_local_idxs]
        else:
            hdf_value: Any = self._hdf_file[column][index]
        return hdf_value


def _decode_rec(value: Union[bytes, Iterable], encoding: str) -> Union[str, list]:
    """Decode bytes to str with the specified encoding. Works recursively on list of bytes, list of list of bytes, etc."""
    if isinstance(value, bytes):
        return value.decode(encoding=encoding)
    elif is_iterable_bytes_list(value):
        return [_decode_rec(elt, encoding) for elt in value]
    else:
        raise TypeError(
            f"Invalid argument type {type(value)}. (expected bytes or Iterable)"
        )
