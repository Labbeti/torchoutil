#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import h5py
import numpy as np
import torch
import tqdm
from h5py import Dataset as HDFRawDataset
from torch.utils.data.dataloader import DataLoader

import pyoutil as po
import torchoutil as to
from torchoutil import nn
from torchoutil.extras.hdf.common import (
    _DUMPED_JSON_KEYS,
    HDF_ENCODING,
    HDF_STRING_DTYPE,
    HDF_VOID_DTYPE,
    SHAPE_SUFFIX,
    HDFItemType,
)
from torchoutil.extras.hdf.dataset import HDFDataset
from torchoutil.extras.numpy import (
    merge_numpy_dtypes,
    numpy_is_complex_dtype,
    scan_shape_dtypes,
)
from torchoutil.pyoutil.collections import all_eq
from torchoutil.pyoutil.datetime import now_iso
from torchoutil.pyoutil.functools import Compose
from torchoutil.pyoutil.typing import is_dataclass_instance, is_dict_str
from torchoutil.types import BuiltinScalar
from torchoutil.utils.data.dataloader import get_auto_num_cpus
from torchoutil.utils.data.dataset import IterableDataset, SizedDatasetLike
from torchoutil.utils.pack.common import EXISTS_MODES, ExistsMode, _tuple_to_dict
from torchoutil.utils.saving.common import to_builtin

T = TypeVar("T", covariant=True)
T_DictOrTuple = TypeVar("T_DictOrTuple", tuple, dict, covariant=True)

HDFDType = Union[np.dtype, Literal["b", "i", "u", "f", "c"], type]


pylog = logging.getLogger(__name__)


@torch.inference_mode()
def pack_to_hdf(
    dataset: SizedDatasetLike[T],
    hdf_fpath: Union[str, Path],
    pre_transform: Optional[Callable[[T], T_DictOrTuple]] = None,
    *,
    exists: ExistsMode = "error",
    verbose: int = 0,
    batch_size: int = 32,
    num_workers: Union[int, Literal["auto"]] = "auto",
    shape_suffix: str = SHAPE_SUFFIX,
    store_str_as_vlen: bool = False,
    file_kwds: Optional[Dict[str, Any]] = None,
    ds_kwds: Optional[Dict[str, Any]] = None,
    user_attrs: Any = None,
    skip_scan: bool = False,
) -> HDFDataset[T_DictOrTuple, T_DictOrTuple]:
    """Pack a dataset to HDF file.

    Args:
        dataset: The sized dataset to pack. Must be sized and all items must be of dict type.
            The key of each dictionaries are strings and values can be int, float, str, Tensor, non-empty List[int], non-empty List[float], non-empty List[str].
            If values are tensors or lists, the number of dimensions must be the same for all items in the dataset.
        hdf_fpath: The path to the HDF file.
        pre_transform: The optional transform to apply to audio returned by the dataset BEFORE storing it in HDF file.
            Can be used for deterministic transforms like Resample, LogMelSpectrogram, etc. defaults to None.
        exists: Determine which action should be performed if the target HDF file already exists.
            "overwrite": Replace the target file then pack dataset.
            "skip": Skip this function and returns the packed dataset.
            "error": Raises a ValueError.
        verbose: Verbose level. defaults to 0.
        batch_size: The batch size of the dataloader. defaults to 32.
        num_workers: The number of workers of the dataloader.
            If "auto", it will be set to `len(os.sched_getaffinity(0))`. defaults to "auto".
        shape_suffix: Shape column suffix in HDF file. defaults to "_shape".
        store_str_as_vlen: If True, store strings as variable length string dtype. defaults to False.
        file_kwds: Options given to h5py file object. defaults to None.
        ds_kwds: Keywords arguments passed to the returned HDFDataset instance if the target file already exists and if exists == "skip".
        user_attrs: Additional metadata to add to the hdf file. It must be convertible to JSON with `json.dumps`. defaults to None.
        skip_scan: If True, the input dataset will be considered as fully homogeneous, which means that all columns values contains the same shape and dtype, which will be inferred from the first batch.
            It is meant to skip the first step which scans each dataset item once and speed up packing to HDF file.
            defaults to False.

    Returns:
        hdf_dataset: The target HDF dataset object.
    """
    if not isinstance(dataset, SizedDatasetLike):
        msg = f"Cannot pack to hdf a non-sized-dataset '{dataset.__class__.__name__}'."
        raise TypeError(msg)
    if len(dataset) == 0:
        msg = f"Cannot pack to hdf an empty dataset. (found {len(dataset)=})"
        raise ValueError(msg)

    hdf_fpath = Path(hdf_fpath).resolve()
    if hdf_fpath.exists() and not hdf_fpath.is_file():
        msg = f"Item {hdf_fpath=} exists but it is not a file."
        raise RuntimeError(msg)

    if ds_kwds is None:
        ds_kwds = {}

    if not hdf_fpath.is_file() or exists == "overwrite":
        pass
    elif exists == "skip":
        return HDFDataset(hdf_fpath, **ds_kwds)
    elif exists == "error":
        msg = f"Cannot overwrite file {hdf_fpath}. Please remove it or use exists='overwrite' or exists='skip' option."
        raise ValueError(msg)
    else:
        msg = f"Invalid argument {exists=}. (expected one of {EXISTS_MODES})"
        raise ValueError(msg)

    if file_kwds is None:
        file_kwds = {}

    if num_workers == "auto":
        num_workers = get_auto_num_cpus()
        if verbose >= 2:
            pylog.debug(f"Found num_workers=='auto', set to {num_workers}.")

    if verbose >= 2:
        pylog.debug(f"Start packing data into HDF file '{hdf_fpath}'...")

    # Step 1: First pass to the dataset to build static HDF dataset shapes (much faster for read the resulting file)
    encoding = HDF_ENCODING
    (
        dict_pre_transform,
        item_type,
        max_shapes,
        hdf_dtypes,
        all_eq_shapes,
        src_np_dtypes,
    ) = _scan_dataset(
        dataset=dataset,
        pre_transform=pre_transform,
        batch_size=batch_size,
        num_workers=num_workers,
        verbose=verbose,
        store_str_as_vlen=store_str_as_vlen,
        encoding=encoding,
        skip_scan=skip_scan,
    )

    total = sum(po.prod(shape) for shape in max_shapes.values())
    max_shapes_ratios = {
        attr_name: po.prod(shape) / total for attr_name, shape in max_shapes.items()
    }

    # For debugging purposes
    data = {
        "item_type": item_type,
        "max_shapes": max_shapes,
        "hdf_dtypes": hdf_dtypes,
        "all_eq_shapes": all_eq_shapes,
        "src_np_dtypes": src_np_dtypes,
    }
    data = to_builtin(data)

    with NamedTemporaryFile(
        "w",
        prefix="HDF_scan_results_",
        suffix=".json",
        delete=False,
    ) as file:
        json.dump(data, file)
        scan_results_fpath = Path(file.name)

    creation_date = now_iso()

    with h5py.File(hdf_fpath, "w", **file_kwds) as hdf_file:
        # Step 2: Build hdf datasets in file
        hdf_dsets: Dict[str, HDFRawDataset] = {}

        # Create sub-datasets for main data
        for attr_name, shape in max_shapes.items():
            hdf_dtype = hdf_dtypes.get(attr_name)

            kwargs: Dict[str, Any] = {}
            fill_value = hdf_dtype_to_fill_value(hdf_dtype)
            if fill_value is not None:
                kwargs["fillvalue"] = fill_value

            hdf_ds_shape = (len(dataset),) + shape
            try:
                hdf_dsets[attr_name] = hdf_file.create_dataset(
                    attr_name, hdf_ds_shape, hdf_dtype, **kwargs
                )
            except ValueError as err:
                msg = f"Cannot create hdf dataset {attr_name=} of shape '{hdf_ds_shape}' with dtype '{hdf_dtype}' and {kwargs=}."
                pylog.error(msg)
                raise err

        if verbose >= 2:
            num_scalars = sum(len(hdf_ds.shape) == 1 for hdf_ds in hdf_dsets.values())
            ratio = num_scalars / total
            msg = f"{num_scalars}/{len(hdf_dsets)} column dsets contains a single dim. ({ratio*100:.3f}%)"
            pylog.debug(msg)

            if num_scalars < len(hdf_dsets):
                msg = "Others multidims column dsets are:"
                pylog.debug(msg)

            for attr_name, hdf_ds in hdf_dsets.items():
                if len(hdf_ds.shape) == 1:
                    continue
                ratio = max_shapes_ratios[attr_name]
                msg = f"HDF column dset multidim '{attr_name}' has been built. (with shape={hdf_ds.shape}, nelement_per_item={po.prod(hdf_ds.shape[1:])} ({ratio*100:.3f}%), dtype={hdf_ds.dtype})"
                pylog.debug(msg)

        added_columns: List[str] = []

        # Create sub-datasets for shape data
        for attr_name, shape in max_shapes.items():
            if len(shape) == 0 or all_eq_shapes[attr_name]:
                continue

            shape_name = f"{attr_name}{shape_suffix}"
            raw_dset_shape = len(dataset), len(shape)

            if shape_name not in hdf_dsets:
                pass
            elif hdf_dsets[shape_name].shape == raw_dset_shape:
                continue
            else:
                msg = f"Column {shape_name} already exists in source dataset with a different shape. (found shape={hdf_dsets[shape_name].shape} but expected shape is {raw_dset_shape})"
                raise RuntimeError(msg)

            hdf_dsets[shape_name] = hdf_file.create_dataset(
                shape_name,
                raw_dset_shape,
                np.int32,
                fillvalue=-1,
            )
            added_columns.append(shape_name)

        # Fill sub-datasets with a second pass through the whole dataset
        i = 0
        global_hash_value = 0

        loader = DataLoader(
            dataset,  # type: ignore
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=nn.Identity(),
            drop_last=False,
            pin_memory=False,
        )

        for _batch_idx, batch in enumerate(
            tqdm.tqdm(
                loader,
                desc="Pack data into HDF...",
                disable=verbose <= 0,
            )
        ):
            batch = [dict_pre_transform(item) for item in batch]

            for item in batch:
                for attr_name, value in item.items():
                    hdf_dset = hdf_dsets[attr_name]
                    shape = to.shape(value)

                    # Check every shape
                    if len(shape) != hdf_dset.ndim - 1:
                        msg = f"Invalid number of dimension in audio (expected {len(shape)}, found {hdf_dset.ndim - 1=})."
                        raise ValueError(msg)

                    # Check dataset size
                    if any(
                        shape_i > dset_shape_i
                        for shape_i, dset_shape_i in zip(shape, hdf_dset.shape[1:])
                    ):
                        msg = f"Resize hdf_dset {attr_name} of shape {tuple(hdf_dset.shape[1:])} with new {shape=}."
                        pylog.error(msg)
                        msg = "INTERNAL ERROR: Cannot resize dataset when pre-computing shapes."
                        raise RuntimeError(msg)

                    # Note: "hdf_dset[slices]" is a generic version of "hdf_dset[i, :shape_0, :shape_1]"
                    slices = (i,) + tuple(slice(shape_i) for shape_i in shape)

                    try:
                        hdf_dset[slices] = value
                    except (TypeError, ValueError, OSError) as err:
                        msg = f"Cannot set data {value} of shape {shape} into {hdf_dset.shape=} ({attr_name=}, {i=}, {slices=}, {value.dtype=} {hdf_dset.dtype=})"
                        pylog.error(msg)
                        raise err

                    # Store original shape if needed
                    shape_name = f"{attr_name}{shape_suffix}"
                    if shape_name in hdf_dsets.keys():
                        hdf_shapes_dset = hdf_dsets[shape_name]
                        hdf_shapes_dset[i] = shape

                    global_hash_value += to.checksum(value)

                i += 1

        # note: HDF cannot save too large int values with too many bits
        global_hash_value = global_hash_value % (2**31)

        if not hasattr(dataset, "info"):
            info = {}
        else:
            info = dataset.info  # type: ignore
            if is_dataclass_instance(info):
                info = asdict(info)
            elif isinstance(info, Mapping):
                info = dict(info.items())  # type: ignore
            else:
                info = {}

        src_np_dtypes_dumped = {
            name: str(merge_numpy_dtypes(np_dtypes, empty=HDF_VOID_DTYPE))
            for name, np_dtypes in src_np_dtypes.items()
        }
        attributes = {
            "added_columns": added_columns,
            "creation_date": creation_date,
            "encoding": encoding,
            "file_kwds": file_kwds,
            "global_hash_value": global_hash_value,
            "info": to_builtin(info),
            "item_type": item_type,
            "length": len(dataset),
            "load_as_complex": {},  # for backward compatibility only
            "pre_transform": pre_transform.__class__.__name__,
            "shape_suffix": shape_suffix,
            "source_dataset": dataset.__class__.__name__,
            "src_np_dtypes": src_np_dtypes_dumped,
            "store_complex_as_real": False,  # for backward compatibility only
            "store_str_as_vlen": store_str_as_vlen,
            "user_attrs": to_builtin(user_attrs),
            "torchoutil_version": str(to.__version__),
        }
        for name in _DUMPED_JSON_KEYS:
            attributes[name] = json.dumps(attributes[name])

        if verbose >= 2:
            dumped_attributes = json.dumps(attributes, indent="\t")
            pylog.debug(f"Saving attributes in HDF file:\n{dumped_attributes}")

        attrs_errors: List[TypeError] = []
        for attr_name, attr_val in attributes.items():
            try:
                hdf_file.attrs[attr_name] = attr_val
            except TypeError as err:
                msg = f"Cannot store attribute {attr_name=} with value {attr_val=} in HDF."
                pylog.error(msg)
                attrs_errors.append(err)

    # Raises attributes errors after closing HDF file
    for err in attrs_errors:
        raise err

    if verbose >= 2:
        pylog.debug(f"Data has been packed into HDF file '{hdf_fpath}'.")

    if scan_results_fpath.is_file():
        os.remove(scan_results_fpath)

    hdf_dataset = HDFDataset(hdf_fpath, **ds_kwds)
    return hdf_dataset


def hdf_dtype_to_fill_value(hdf_dtype: Optional[HDFDType]) -> BuiltinScalar:
    if isinstance(hdf_dtype, np.dtype):
        hdf_dtype = hdf_dtype.type

    if hdf_dtype == "b" or hdf_dtype == np.bool_:
        return False
    elif hdf_dtype in ("i", "u") or (
        isinstance(hdf_dtype, type) and issubclass(hdf_dtype, np.integer)
    ):
        return 0
    elif hdf_dtype == "f" or (
        isinstance(hdf_dtype, type) and issubclass(hdf_dtype, np.floating)
    ):
        return 0.0
    elif (
        hdf_dtype == "c"
        or (
            isinstance(hdf_dtype, type)
            and (
                hdf_dtype in (np.void, np.object_, np.bytes_, np.str_)
                or issubclass(hdf_dtype, np.complexfloating)
            )
        )
        or (isinstance(hdf_dtype, np.dtype) and numpy_is_complex_dtype(hdf_dtype))
    ):
        return None
    else:
        msg = f"Unsupported type {hdf_dtype=}."
        raise ValueError(msg)


def numpy_dtype_to_hdf_dtype(
    dtype: Optional[np.dtype],
    *,
    encoding: str = HDF_ENCODING,
) -> np.dtype:
    if dtype is None:
        return HDF_VOID_DTYPE
    elif isinstance(dtype, np.dtype) and dtype.kind == "U":
        return h5py.string_dtype(encoding, None)
    else:
        return dtype


def hdf_dtype_to_numpy_dtype(hdf_dtype: HDFDType) -> np.dtype:
    if isinstance(hdf_dtype, np.dtype):
        return hdf_dtype
    if hdf_dtype == HDF_VOID_DTYPE:
        return np.dtype("V")
    if hdf_dtype == HDF_STRING_DTYPE:
        return np.dtype("<U")
    if hdf_dtype == "f":
        return np.dtype("float32")
    if hdf_dtype == "i":
        return np.dtype("int32")
    if hdf_dtype == "b":
        return np.dtype("int8")
    if hdf_dtype == "c":
        return np.dtype("|S1")

    raise ValueError(f"Unsupported dtype {hdf_dtype=} for numpy dtype.")


def _scan_dataset(
    dataset: SizedDatasetLike[T],
    pre_transform: Optional[Callable[[T], T_DictOrTuple]],
    batch_size: int,
    num_workers: int,
    store_str_as_vlen: bool,
    verbose: int,
    encoding: str,
    skip_scan: bool,
) -> Tuple[
    Callable[[T], Dict[str, Any]],
    HDFItemType,
    Dict[str, Tuple[int, ...]],
    Dict[str, HDFDType],
    Dict[str, bool],
    Dict[str, Set[np.dtype]],
]:
    if pre_transform is None:
        pre_transform = nn.Identity()

    if isinstance(dataset, IterableDataset):
        item_0 = next(iter(dataset))
    else:
        item_0 = dataset[0]

    def encode_array(x: np.ndarray) -> Any:
        if x.dtype.kind == "U":
            x = np.char.encode(x, encoding=encoding)
        if x.dtype.kind == "S":
            x = x.tolist()
        return x

    def encode_dict_array(x: Dict[str, np.ndarray]) -> Dict[str, Any]:
        return {k: encode_array(to.to_numpy(v)) for k, v in x.items()}  # type: ignore

    to_dict_fn: Callable[[T], Dict[str, Any]]

    if is_dict_str(item_0):
        item_type = "dict"
        to_dict_fn = to.identity  # type: ignore
    elif isinstance(item_0, tuple):
        item_type = "tuple"
        to_dict_fn = _tuple_to_dict  # type: ignore
    else:
        msg = f"Invalid item type for {dataset.__class__.__name__}. (expected Dict[str, Any] or tuple but found {type(item_0)})"
        raise ValueError(msg)
    del item_0

    encode_dict_fn = to.identity if store_str_as_vlen else encode_dict_array

    dict_pre_transform: Callable[[T], Dict[str, Any]] = Compose(
        pre_transform,
        to_dict_fn,
        encode_dict_fn,
    )

    loader = DataLoader(
        dataset,  # type: ignore
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=nn.Identity(),
        drop_last=False,
        pin_memory=False,
    )

    infos_dict: Dict[str, Set[Tuple[Tuple[int, ...], np.dtype]]] = {}
    src_np_dtypes: Dict[str, Set[np.dtype]] = {}

    for batch in tqdm.tqdm(
        loader,
        desc="Pre compute shapes...",
        disable=verbose <= 0 or skip_scan,
    ):
        batch = [pre_transform(item) for item in batch]
        batch = [to_dict_fn(item) for item in batch]

        for item in batch:
            for attr_name, value in item.items():
                info = scan_shape_dtypes(value, empty_np=HDF_VOID_DTYPE)
                shape = info.shape
                np_dtype = info.numpy_dtype
                kind = np_dtype.kind

                if attr_name in src_np_dtypes:
                    src_np_dtypes[attr_name].add(np_dtype)  # type: ignore
                else:
                    src_np_dtypes[attr_name] = {np_dtype}  # type: ignore

                value = to.to_numpy(value)
                if kind == "U" and not store_str_as_vlen:
                    value = encode_array(value)  # type: ignore
                    # update shape and np_dtype after encoding
                    info = scan_shape_dtypes(value, empty_np=HDF_VOID_DTYPE)
                    shape = info.shape
                    np_dtype = info.numpy_dtype

                if attr_name in infos_dict:
                    infos_dict[attr_name].add((shape, np_dtype))  # type: ignore
                else:
                    infos_dict[attr_name] = {(shape, np_dtype)}  # type: ignore

            if skip_scan:
                break

    max_shapes: Dict[str, Tuple[int, ...]] = {}
    hdf_dtypes: Dict[str, HDFDType] = {}
    all_eq_shapes: Dict[str, bool] = {}

    for attr_name, info in infos_dict.items():
        shapes = [shape for shape, _ in info]
        ndims = list(map(len, shapes))
        if not all_eq(ndims):
            ndims_set = tuple(set(ndims))
            indices = [ndims.index(ndim) for ndim in ndims_set]
            msg = f"Invalid ndim for attribute {attr_name}. (found multiple ndims: {ndims_set} at {indices=})"
            raise ValueError(msg)

        np_dtypes = [np_dtype for _, np_dtype in info]
        np_dtype = merge_numpy_dtypes(np_dtypes, empty=HDF_VOID_DTYPE)
        hdf_dtype = numpy_dtype_to_hdf_dtype(np_dtype, encoding=encoding)

        all_eq_shapes[attr_name] = all_eq(shapes)
        max_shapes[attr_name] = tuple(map(max, zip(*shapes)))
        hdf_dtypes[attr_name] = hdf_dtype

    del infos_dict

    if verbose >= 2:
        pylog.debug(f"Found max_shapes:\n{max_shapes}")
        pylog.debug(f"Found hdf_dtypes:\n{hdf_dtypes}")
        pylog.debug(f"Found all_eq_shapes:\n{all_eq_shapes}")
        pylog.debug(f"Found src_np_dtypes:\n{src_np_dtypes}")

    return (
        dict_pre_transform,
        item_type,
        max_shapes,
        hdf_dtypes,
        all_eq_shapes,
        src_np_dtypes,
    )


def bytearray_to_bytes(x: Any) -> Any:
    if isinstance(x, (str, bytes)):
        return x
    if isinstance(x, bytearray):
        return bytes(x)
    elif isinstance(x, Mapping):
        return {bytearray_to_bytes(k): bytearray_to_bytes(v) for k, v in x.items()}
    elif isinstance(x, Iterable):
        return type(x)(bytearray_to_bytes(xi) for xi in x)
    else:
        return x
