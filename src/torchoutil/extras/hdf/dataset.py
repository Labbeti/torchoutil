#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import json
import logging
import os
import os.path as osp
import pickle
from functools import cached_property
from json import JSONDecodeError
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
from h5py import Dataset as HDFRawDataset
from torch import Tensor
from typing_extensions import override

import torchoutil as to
from torchoutil.extras.hdf.common import (
    _DEFAULTS_RAW_HDF_ATTRIBUTES,
    _DUMPED_JSON_KEYS,
    HDFDatasetAttributes,
    HDFItemType,
)
from torchoutil.extras.numpy.scan_info import numpy_dtype_to_torch_dtype
from torchoutil.nn.functional.indices import get_inverse_perm
from torchoutil.pyoutil.collections import all_eq
from torchoutil.pyoutil.difflib import find_closest_in_list
from torchoutil.pyoutil.inspect import get_current_fn_name
from torchoutil.pyoutil.typing import is_iterable_bytes_or_list, is_iterable_str
from torchoutil.types._typing import ScalarLike
from torchoutil.types.guards import is_scalar_like
from torchoutil.utils.data import DatasetSlicer
from torchoutil.utils.pack.common import _dict_to_tuple
from torchoutil.utils.saving import to_builtin

T = TypeVar("T", covariant=True)
U = TypeVar("U", covariant=False)

IndexLike = Union[int, Iterable[int], Tensor, slice, None]
ColumnLike = Union[str, Iterable[str], None]
CastMode = Literal[
    "to_torch_or_builtin",
    "to_torch_or_numpy",
    "to_builtin",
    "to_numpy_src",
    "to_torch_src",
    "none",
]
CAST_MODES = (
    "to_torch_or_builtin",
    "to_torch_or_numpy",
    "to_builtin",
    "to_numpy_src",
    "to_torch_src",
    "none",
)

pylog = logging.getLogger(__name__)


class HDFDataset(Generic[T, U], DatasetSlicer[U]):
    def __init__(
        self,
        hdf_fpath: Union[str, Path],
        transform: Optional[Callable[[T], U]] = None,
        keep_padding: Iterable[str] = (),
        return_added_columns: bool = False,
        open_hdf: bool = True,
        cast: CastMode = "none",
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
        self._cast: CastMode = cast
        self._file_kwds = file_kwds

        self._hdf_file: Any = None

        if open_hdf:
            self.open()

    # Public properties
    @property
    def added_columns(self) -> List[str]:
        """Return the list of columns added by pack_to_hdf function."""
        return self.attrs["added_columns"]

    @property
    def all_columns(self) -> List[str]:
        """The name of all columns of the dataset."""
        return list(self.get_hdf_keys())

    @cached_property
    def attrs(self) -> HDFDatasetAttributes:
        attrs = copy.copy(_DEFAULTS_RAW_HDF_ATTRIBUTES)
        attrs.update(self._hdf_file.attrs)

        for name in _DUMPED_JSON_KEYS:
            try:
                attrs[name] = json.loads(attrs[name])
            except JSONDecodeError:
                pylog.error(f"Cannot load JSON data {attrs[name]=} from {name=}.")

        attrs["added_columns"] = list(attrs["added_columns"])
        attrs["src_np_dtypes"] = {
            k: np.dtype(v) for k, v in attrs["src_np_dtypes"].items()
        }
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

    @property
    def info(self) -> Dict[str, Any]:
        """Return the global dataset info."""
        return self.attrs["info"]

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
    def transform(self) -> Optional[Callable[[T], U]]:
        return self._transform

    @property
    def user_attrs(self) -> Any:
        return self.attrs["user_attrs"]

    # Private properties to define internal behaviours
    @property
    def _encoding(self) -> str:
        return self.attrs["encoding"]

    @property
    def _shape_suffix(self) -> str:
        """Return the tensor shape suffix in column names."""
        return self.attrs["shape_suffix"]

    @property
    def _src_np_dtypes(self) -> Dict[str, np.dtype]:
        return self.attrs["src_np_dtypes"]

    @cached_property
    def _src_is_unicode(self) -> Dict[str, bool]:
        is_unicode = {name: dt.kind == "U" for name, dt in self._src_np_dtypes.items()}
        return is_unicode

    @property
    def _load_as_complex(self) -> Dict[str, bool]:
        return self.attrs["load_as_complex"]

    # Public methods
    @overload
    def get_item(
        self,
        index: int,
        column: None = None,
    ) -> T:
        ...

    @overload
    def get_item(
        self,
        index: Union[Iterable[int], slice, None],
        column: str,
    ) -> List:
        ...

    @overload
    def get_item(
        self,
        index: Union[Iterable[int], slice, None],
        column: Union[List[str], None] = None,
    ) -> Dict[str, List]:
        ...

    @overload
    def get_item(
        self,
        index: Any,
        column: Any,
        raw: bool = False,
    ) -> Any:
        ...

    @override
    def get_item(
        self,
        index: IndexLike,
        column: ColumnLike = None,
        raw: bool = False,
    ) -> Any:
        if self.is_closed():
            msg = f"Cannot get_raw value with closed HDF file. ({self._hdf_file is not None=} and {bool(self._hdf_file)=})"
            raise RuntimeError(msg)

        if index is None:
            index = slice(None)
        elif is_scalar_like(index):
            index = to.to_item(index)  # type: ignore
        elif isinstance(index, Iterable):
            index = to.to_numpy(index)
        elif isinstance(index, (int, slice)):
            pass
        else:
            raise TypeError(f"Invalid argument type {type(index)=}.")

        if column is None:
            column = self.column_names

        if is_iterable_str(column, accept_str=False):
            result_dict = {
                column_i: self.get_item(index, column_i) for column_i in column
            }
            if self.item_type == "tuple":
                result = _dict_to_tuple(result_dict)
            else:
                result = result_dict

            if (
                isinstance(index, int)
                and self._transform is not None
                and set(column) == set(self.column_names)
            ):
                result = self._transform(result)  # type: ignore
            return result  # type: ignore

        if column not in self.all_columns:
            closest = find_closest_in_list(column, self.all_columns)  # type: ignore
            msg = f"Invalid argument {column=}. (did you mean '{closest}'? Expected one of {tuple(self.all_columns)})"
            raise ValueError(msg)

        if isinstance(index, slice) or (
            isinstance(index, np.ndarray)
            and index.ndim == 1
            and index.dtype.kind in ("b", "i")
        ):
            is_mult = True

        elif isinstance(index, int):
            if not (-len(self) <= index < len(self)):
                msg = f"Invalid argument {index=}. (expected int in range [{-len(self)}, {len(self)-1}])"
                raise IndexError(msg)
            is_mult = False
        else:
            raise TypeError(f"Invalid argument type {type(index)=}.")

        hdf_value = self._get_raw_item(index, column)
        if raw:
            return hdf_value

        if is_mult:
            hdf_values = hdf_value
        else:
            hdf_values = hdf_value[None]
        del hdf_value

        shape_column = f"{column}{self._shape_suffix}"
        must_remove_padding = (
            shape_column in self._hdf_file.keys() and column not in self._keep_padding
        )
        hdf_ds: HDFRawDataset = self._hdf_file[column]
        hdf_dtype: np.dtype = hdf_ds.dtype

        if must_remove_padding:
            shapes = self._get_raw_item(index, shape_column)
            if not is_mult:
                shapes = shapes[None]
            slices_lst = [
                tuple(slice(shape_i) for shape_i in shape) for shape in shapes
            ]
        else:
            slices_lst = [None] * int(hdf_ds.shape[0])

        if (
            self._src_is_unicode.get(column, False)
            or h5py.check_vlen_dtype(hdf_dtype) is str
        ):
            hdf_values = _decode_bytes(hdf_values, encoding=self._encoding)

        if must_remove_padding:
            outputs = []
            for hdf_value, slices in zip(hdf_values, slices_lst):
                # Remove the padding part
                if slices is not None:
                    assert isinstance(hdf_value, np.ndarray) and not isinstance(
                        hdf_value, str
                    )
                    hdf_value = hdf_value[slices]
                hdf_value = self._cast_values(hdf_value, column, hdf_dtype)
                outputs.append(hdf_value)
        else:
            outputs = self._cast_values(hdf_values, column, hdf_dtype)
        del hdf_values

        if not is_mult:
            outputs = outputs[0]
        return outputs

    def close(self, ignore_if_closed: bool = False, remove_file: bool = False) -> None:
        if self.is_closed() and not ignore_if_closed:
            raise RuntimeError("Cannot close the HDF file twice.")

        if not ignore_if_closed:
            self._hdf_file.close()
        if remove_file:
            os.remove(self._hdf_fpath)

        self._hdf_file = None
        self._clear_caches()

    def get_attrs(self) -> HDFDatasetAttributes:
        return self.attrs

    def get_hdf_fpath(self) -> Path:
        return self._hdf_fpath

    def get_hdf_keys(self) -> Tuple[str, ...]:
        if self.is_closed():
            raise RuntimeError("Cannot get keys from a closed HDF file.")
        return tuple(self._hdf_file.keys())

    def get_column_shape(self, column_name: str) -> Tuple[int, ...]:
        if self.is_closed():
            msg = f"Cannot get_column_shape with a closed HDF file. ({self._hdf_file is None=} or {not bool(self._hdf_file)=})"
            raise RuntimeError(msg)
        return tuple(self._hdf_file[column_name].shape)

    def get_columns_shapes(self) -> Dict[str, Tuple[int, ...]]:
        if self.is_closed():
            msg = f"Cannot get_columns_shapes with a closed HDF file. ({self._hdf_file is None=} or {not bool(self._hdf_file)=})"
            raise RuntimeError(msg)

        return {
            column_name: tuple(self._hdf_file[column_name].shape)
            for column_name in self.column_names
        }

    def get_column_dtype(self, column_name: str) -> np.dtype:
        if self.is_closed():
            msg = f"Cannot get dtype with a closed HDF file. ({self._hdf_file is None=} or {not bool(self._hdf_file)=})"
            raise RuntimeError(msg)
        return self._hdf_file[column_name].dtype

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

    def __enter__(self) -> "HDFDataset":
        return self

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

    def __getitem__(  # type: ignore
        self,
        index: Union[IndexLike, Tuple[IndexLike, ColumnLike]],
    ) -> Any:
        return super().__getitem__(index)  # type: ignore

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "hdf_fpath": self._hdf_fpath,
            "transform": self._transform,
            "keep_padding": self._keep_padding,
            "return_added_columns": self._return_added_columns,
            "cast": self._cast,
            "file_kwds": self._file_kwds,
            "is_open": self.is_open(),
        }

    def __hash__(self) -> int:
        hash_value = 0
        if self.is_open():
            hash_value += self.attrs["global_hash_value"]
        if self._transform is not None:
            hash_value += hash(self._transform)
        hash_value += sum(map(hash, self._keep_padding))
        return hash_value

    def __len__(self) -> int:
        if self.is_closed():
            msg = f"Cannot length of a closed HDF file. ({self._hdf_file is not None=} and {bool(self._hdf_file)=})"
            raise RuntimeError(msg)

        if "length" in self._hdf_file.attrs:
            length = self._hdf_file.attrs["length"]
        elif len(self._hdf_file) > 0:
            hdf_dsets: List[HDFRawDataset] = list(self._hdf_file.values())
            hdf_dsets_lens = [len(ds) for ds in hdf_dsets]
            if not all_eq(hdf_dsets_lens):
                msg = f"Found an different number of lengths in hdf sub-datasets. (found {set(hdf_dsets_lens)})"
                raise ValueError(msg)
            length = hdf_dsets_lens[0]
        else:
            length = 0

        return length

    def __repr__(self) -> str:
        repr_hparams = {"file": osp.basename(self._hdf_fpath), "shape": self.shape}
        repr_ = ", ".join(f"{k}={v}" for k, v in repr_hparams.items())
        return f"{HDFDataset.__name__}({repr_})"

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
        self._cast = data["cast"]
        self._file_kwds = data["file_kwds"]

        self._hdf_file = None

        if not is_init or (files_are_different and is_open):
            self.open()

    # Private methods
    def _get_raw_item(
        self,
        index: Union[int, slice, np.ndarray],
        column: str,
    ) -> np.ndarray:
        if isinstance(index, (int, slice)) or (
            isinstance(index, np.ndarray)
            and index.ndim == 1
            and index.dtype.kind == "b"
        ):
            hdf_value: Any = self._hdf_file[column][index]
            if self._load_as_complex.get(column, False):
                hdf_value = to.view_as_complex(hdf_value)
            hdf_value = np.array(hdf_value)

        elif (
            isinstance(index, np.ndarray)
            and index.ndim == 1
            and index.dtype.kind == "i"
        ):
            # Note: slicing with indices required strict sorted list of indices, so we sort + remove duplicates before loading
            local_idxs = np.argsort(index, axis=-1)
            sorted_idxs = index[local_idxs]
            uniq, counts = np.unique(sorted_idxs, return_counts=True)

            hdf_value: Any = self._hdf_file[column][uniq]

            hdf_value = np.repeat(hdf_value, counts, axis=0)
            inv_local_idxs = get_inverse_perm(to.numpy_to_tensor(local_idxs)).numpy()
            hdf_value = hdf_value[inv_local_idxs]

            if self._load_as_complex.get(column, False):
                hdf_value = [to.view_as_complex(value) for value in hdf_value]

        else:
            raise TypeError(f"Invalid argument type {type(index)=}.")

        return hdf_value

    def _sanity_check(self) -> None:
        lens = [dset.shape[0] for dset in self._hdf_file.values()]
        if not all_eq(lens) or lens[0] != len(self):
            msg = (
                f"Incorrect length stored in HDF file. (found {lens=} and {len(self)=})"
            )
            pylog.error(msg)

        if not hasattr(self, "__orig_class__"):
            return None

        t_type = self.__orig_class__.__args__[0]  # type: ignore
        if t_type is not Any and (
            (issubclass(t_type, dict) and self.item_type != "dict")
            or (issubclass(t_type, tuple) and self.item_type != "tuple")
        ):
            msg = f"Invalid HDFDataset typing. (found specified type '{t_type.__name__}' but the internal dataset contains type '{self.item_type}')"
            raise TypeError(msg)

    def _clear_caches(self) -> None:
        if hasattr(self, "attrs"):
            del self.attrs
        if hasattr(self, "_is_unicode"):
            del self._src_is_unicode

    def _cast_values(
        self,
        hdf_values: Union[ScalarLike, np.ndarray, List],
        column: str,
        hdf_dtype: np.dtype,
    ) -> Any:
        if self._cast == "none":
            return hdf_values

        elif self._cast == "to_torch_or_builtin":
            valid = to.shape(hdf_values, return_valid=True).valid
            if valid and hdf_dtype.kind not in ("V", "S", "O"):
                result = to.to_tensor(hdf_values)
            elif isinstance(hdf_values, np.ndarray):
                result = hdf_values.tolist()
            else:
                result = to_builtin(hdf_values)

        elif self._cast == "to_torch_or_numpy":
            valid = to.shape(hdf_values, return_valid=True).valid
            if valid and hdf_dtype.kind not in ("V", "S", "O"):
                result = to.to_tensor(hdf_values)
            else:
                result = np.array(hdf_values)

        elif self._cast == "to_builtin":
            if isinstance(hdf_values, np.ndarray):
                result = hdf_values.tolist()
            else:
                result = to_builtin(hdf_values)

        elif self._cast == "to_numpy_src":
            assert isinstance(hdf_values, np.ndarray), f"{type(hdf_values)=}"
            valid = to.shape(hdf_values, return_valid=True).valid
            src_np_dtypes = self.attrs["src_np_dtypes"]
            target_np_dtype = src_np_dtypes.get(column, hdf_values.dtype)

            if isinstance(hdf_values, np.ndarray):
                result = hdf_values.astype(target_np_dtype)
            elif valid:
                result = np.array(hdf_values, dtype=target_np_dtype)
            else:
                result = hdf_values

        elif self._cast == "to_torch_src":
            assert isinstance(hdf_values, np.ndarray), f"{type(hdf_values)=}"
            valid = to.shape(hdf_values, return_valid=True).valid
            src_np_dtypes = self.attrs["src_np_dtypes"]
            target_np_dtype = src_np_dtypes.get(column, hdf_values.dtype)
            target_pt_dtype = numpy_dtype_to_torch_dtype(target_np_dtype, invalid=None)

            if isinstance(hdf_values, np.ndarray):
                hdf_values_view = hdf_values.view(target_np_dtype)
                result = to.numpy_to_tensor(hdf_values_view)
            elif valid:
                result = to.to_tensor(hdf_values, dtype=target_pt_dtype)
            else:
                result = hdf_values

        else:
            msg = f"Invalid argument {self._cast=}. (expected one of {CAST_MODES})"
            raise ValueError(msg)

        return result


def _decode_bytes(
    encoded: Union[bytes, np.ndarray, Iterable],
    encoding: str,
) -> Union[str, np.ndarray, list]:
    """Decode bytes to str with the specified encoding. Works recursively on list of bytes, list of list of bytes, etc."""
    if isinstance(encoded, (bytes, bytearray)):
        return encoded.decode(encoding=encoding)

    elif isinstance(encoded, np.ndarray):
        if encoded.dtype.kind == "S":
            return np.char.decode(encoded, encoding=encoding)
        elif encoded.dtype.kind == "O" and encoded.ndim > 0:
            return [
                _decode_bytes(encoded_i, encoding=encoding) for encoded_i in encoded
            ]
        else:
            return _decode_bytes(encoded.item(), encoding=encoding)

    elif is_iterable_bytes_or_list(encoded):
        return [_decode_bytes(elt, encoding) for elt in encoded]

    else:
        msg = f"Invalid argument type {type(encoded)} for {get_current_fn_name()}. (expected bytes, bytes ndarray or Iterable)"
        raise TypeError(msg)
