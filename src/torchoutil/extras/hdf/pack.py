#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
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

import torchoutil as to
from torchoutil import nn
from torchoutil.extras.hdf.common import (
    HDF_ENCODING,
    HDF_STRING_DTYPE,
    HDF_VOID_DTYPE,
    SHAPE_SUFFIX,
    HDFItemType,
)
from torchoutil.extras.hdf.dataset import HDFDataset
from torchoutil.extras.numpy.scan_info import merge_numpy_dtypes, scan_shape_dtypes
from torchoutil.nn.functional.checksum import checksum
from torchoutil.pyoutil.collections import all_eq
from torchoutil.pyoutil.functools import Compose
from torchoutil.pyoutil.logging import warn_once
from torchoutil.pyoutil.typing import is_dataclass_instance, is_dict_str
from torchoutil.types import BuiltinScalar
from torchoutil.utils.data.dataloader import get_auto_num_cpus
from torchoutil.utils.data.dataset import IterableDataset, SizedDatasetLike
from torchoutil.utils.pack.common import EXISTS_MODES, ExistsMode, _tuple_to_dict
from torchoutil.utils.saving import to_builtin

T = TypeVar("T", covariant=True)
T_DictOrTuple = TypeVar("T_DictOrTuple", tuple, dict, covariant=True)

HDFDType = Union[Literal["b", "i", "u", "f", "c"], np.dtype]


_SUPPORTED_HDF_DTYPES = (
    "b",  # bool
    "i",  # int
    "u",  # uint
    "f",  # float
    "c",  # complex
    HDF_STRING_DTYPE,
    HDF_VOID_DTYPE,
)


pylog = logging.getLogger(__name__)


@torch.inference_mode()
def pack_to_hdf(
    dataset: SizedDatasetLike[T],
    hdf_fpath: Union[str, Path],
    pre_transform: Optional[Callable[[T], T_DictOrTuple]] = None,
    exists: ExistsMode = "error",
    verbose: int = 0,
    batch_size: int = 32,
    num_workers: Union[int, Literal["auto"]] = "auto",
    shape_suffix: str = SHAPE_SUFFIX,
    store_complex_as_real: bool = True,
    use_vlen_str: bool = False,
    file_kwds: Optional[Dict[str, Any]] = None,
    ds_kwds: Optional[Dict[str, Any]] = None,
    user_attrs: Any = None,
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
        open_hdf: If True, opens the output HDF dataset. defaults to True.
        file_kwds: Options given to h5py file object. defaults to None.
        user_attrs: Additional metadata to add to the hdf file. It must be JSON convertible. defaults to None.

    Returns:
        hdf_dataset: The target HDF dataset object.
    """
    # Check inputs
    if not isinstance(dataset, SizedDatasetLike):
        msg = f"Cannot pack to hdf a non-sized-dataset '{dataset.__class__.__name__}'."
        raise TypeError(msg)
    if len(dataset) == 0:
        msg = f"Cannot pack to hdf an empty dataset. (found {len(dataset)=})"
        raise ValueError(msg)

    hdf_fpath = Path(hdf_fpath).resolve()
    if hdf_fpath.exists() and not hdf_fpath.is_file():
        raise RuntimeError(f"Item {hdf_fpath=} exists but it is not a file.")

    if ds_kwds is None:
        ds_kwds = {}

    if not hdf_fpath.is_file() or exists == "overwrite":
        pass
    elif exists == "error":
        msg = f"Cannot overwrite file {hdf_fpath}. Please remove it or use exists='overwrite' or exists='skip' option."
        raise ValueError(msg)
    elif exists == "skip":
        return HDFDataset(hdf_fpath, **ds_kwds)
    else:
        msg = f"Invalid argument {exists=}. (expected one of {EXISTS_MODES})"
        raise ValueError(msg)

    if file_kwds is None:
        file_kwds = {}

    if num_workers == "auto":
        num_workers = get_auto_num_cpus()
        if verbose >= 2:
            pylog.debug(f"Found num_workers=='auto', set to {num_workers}.")

    if pre_transform is None:
        pre_transform = nn.Identity()

    if verbose >= 2:
        pylog.debug(f"Start packing data into HDF file '{hdf_fpath}'...")

    # Step 1: First pass to the dataset to build static HDF dataset shapes (much faster for read the resulting file)
    (
        dict_pre_transform,
        item_type,
        max_shapes,
        hdf_dtypes,
        all_eq_shapes,
        load_as_complex,
        src_np_dtypes,
    ) = _scan_dataset(
        dataset,
        pre_transform,
        batch_size,
        num_workers,
        store_complex_as_real,
        verbose,
        use_vlen_str=use_vlen_str,
    )

    now = datetime.datetime.now()
    creation_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    added_columns = []

    with h5py.File(
        hdf_fpath,
        "w",
        **file_kwds,
    ) as hdf_file:
        # Step 2: Build hdf datasets in file
        hdf_dsets: Dict[str, HDFRawDataset] = {}

        # Create sub-datasets for main data
        for attr_name, shape in max_shapes.items():
            hdf_dtype = hdf_dtypes.get(attr_name)

            kwargs: Dict[str, Any] = {}
            fill_value = hdf_dtype_to_fill_value(hdf_dtype)
            if fill_value is not None:
                kwargs["fillvalue"] = fill_value

            hdf_dsets[attr_name] = hdf_file.create_dataset(
                attr_name,
                (len(dataset),) + shape,
                hdf_dtype,
                **kwargs,
            )

        if verbose >= 2:
            num_scalars = sum(len(hdf_ds.shape) == 1 for hdf_ds in hdf_dsets.values())
            msg = f"{num_scalars}/{len(hdf_dsets)} column dsets contains a single dim."
            pylog.debug(msg)

            for attr_name, hdf_ds in hdf_dsets.items():
                if len(hdf_ds.shape) == 1:
                    continue
                msg = f"HDF column dset multidim '{attr_name}' with (shape={hdf_ds.shape}, dtype={hdf_ds.dtype}) has been built."
                pylog.debug(msg)

        # Create sub-datasets for shape data
        for attr_name, shape in max_shapes.items():
            if len(shape) == 0 or all_eq_shapes[attr_name]:
                continue

            shape_name = f"{attr_name}{shape_suffix}"
            raw_dset_shape = (len(dataset), len(shape))

            if shape_name in hdf_dsets:
                if hdf_dsets[shape_name].shape != raw_dset_shape:
                    msg = f"Column {shape_name} already exists in source dataset with a different shape. (found shape={hdf_dsets[shape_name].shape} but expected shape is {raw_dset_shape})"
                    raise RuntimeError(msg)
                else:
                    continue

            added_columns.append(shape_name)
            hdf_dsets[shape_name] = hdf_file.create_dataset(
                shape_name, raw_dset_shape, "i", fillvalue=-1
            )

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

        for batch in tqdm.tqdm(
            loader,
            desc="Pack data into HDF...",
            disable=verbose <= 0,
        ):
            batch = [dict_pre_transform(item) for item in batch]

            for item in batch:
                for attr_name, value in item.items():
                    hdf_dset = hdf_dsets[attr_name]

                    if load_as_complex.get(attr_name, False):
                        value = to.view_as_real(value)

                    shape = to.shape(value)

                    # Check every shape
                    if len(shape) != hdf_dset.ndim - 1:
                        msg = f"Invalid number of dimension in audio (expected {len(shape)}, found {len(shape)})."
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
                        # TODO: rm debug
                        breakpoint()
                        msg = f"Cannot set data {value} into {hdf_dset.shape=} ({attr_name=}, {i=}, {slices=}, {value.dtype=} {hdf_dset.dtype=})"
                        pylog.error(msg)
                        raise err

                    # Store original shape if needed
                    shape_name = f"{attr_name}{shape_suffix}"
                    if shape_name in hdf_dsets.keys():
                        hdf_shapes_dset = hdf_dsets[shape_name]
                        hdf_shapes_dset[i] = shape

                    global_hash_value += checksum(value)

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
            name: str(merge_numpy_dtypes(np_dtypes))
            for name, np_dtypes in src_np_dtypes.items()
        }
        attributes = {
            "creation_date": creation_date,
            "source_dataset": dataset.__class__.__name__,
            "length": len(dataset),
            "encoding": HDF_ENCODING,
            "info": json.dumps(info),
            "global_hash_value": global_hash_value,
            "item_type": item_type,
            "added_columns": added_columns,
            "shape_suffix": shape_suffix,
            "file_kwds": json.dumps(file_kwds),
            "load_as_complex": json.dumps(load_as_complex),
            "version": str(to.__version__),
            "user_attrs": json.dumps(to_builtin(user_attrs)),
            "src_np_dtypes": json.dumps(src_np_dtypes_dumped),
        }
        if verbose >= 2:
            dumped_attributes = json.dumps(attributes, indent="\t")
            pylog.debug(f"Saving attributes in HDF file:\n{dumped_attributes}")

        for attr_name, attr_val in attributes.items():
            try:
                hdf_file.attrs[attr_name] = attr_val
            except TypeError as err:
                msg = f"Cannot store attribute {attr_name=} with value {attr_val=} in HDF."
                pylog.error(msg)
                raise err

    if verbose >= 2:
        pylog.debug(f"Data has been packed into HDF file '{hdf_fpath}'.")

    hdf_dataset = HDFDataset(hdf_fpath, **ds_kwds)
    return hdf_dataset


def _scan_dataset(
    dataset: SizedDatasetLike[T],
    pre_transform: Callable[[T], T_DictOrTuple],
    batch_size: int,
    num_workers: int,
    store_complex_as_real: bool,
    verbose: int,
    use_vlen_str: bool = False,
) -> Tuple[
    Callable[[T], Dict[str, Any]],
    HDFItemType,
    Dict[str, Tuple[int, ...]],
    Dict[str, HDFDType],
    Dict[str, bool],
    Dict[str, bool],
    Dict[str, Set[np.dtype]],
]:
    if isinstance(dataset, IterableDataset):
        item_0 = next(iter(dataset))
    else:
        item_0 = dataset[0]

    def encode_array(x: np.ndarray) -> Any:
        if x.dtype.kind == "U":
            x = np.char.encode(x, encoding=HDF_ENCODING)
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

    encode_dict_fn = to.identity if use_vlen_str else encode_dict_array
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
        disable=verbose <= 0,
    ):
        batch = [pre_transform(item) for item in batch]
        batch = [to_dict_fn(item) for item in batch]  # type: ignore

        for item in batch:
            for attr_name, value in item.items():
                info = scan_shape_dtypes(value)
                shape = info.shape
                np_dtype = info.numpy_dtype
                kind = np_dtype.kind

                if attr_name in src_np_dtypes:
                    src_np_dtypes[attr_name].add(np_dtype)  # type: ignore
                else:
                    src_np_dtypes[attr_name] = {np_dtype}  # type: ignore

                value = to.to_numpy(value)
                if kind == "U" and not use_vlen_str:
                    value = encode_array(value)  # type: ignore
                    # update shape and np_dtype after encoding
                    info = scan_shape_dtypes(value)
                    shape = info.shape
                    np_dtype = info.numpy_dtype

                if attr_name in infos_dict:
                    infos_dict[attr_name].add((shape, np_dtype))  # type: ignore
                else:
                    infos_dict[attr_name] = {(shape, np_dtype)}  # type: ignore

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
        np_dtype = merge_numpy_dtypes(np_dtypes)
        hdf_dtype = numpy_dtype_to_hdf_dtype(np_dtype)  # type: ignore

        if (
            verbose >= 2
            and np_dtype
            not in (
                None,
                str,
                bool,
                np.int32,
                np.int8,
                np.float32,
            )
            and np_dtype.kind != "S"
            and (not use_vlen_str and np_dtype.kind == "U")
        ):
            expected_np_dtype = hdf_dtype_to_numpy_dtype(hdf_dtype)
            msg = f"Found input dtype {np_dtype} which is not compatible with HDF. It will be cast to {expected_np_dtype}. (with {hdf_dtype=})"
            warn_once(msg, __name__)

        all_eq_shapes[attr_name] = all_eq(shapes)
        max_shapes[attr_name] = tuple(map(max, zip(*shapes)))
        hdf_dtypes[attr_name] = hdf_dtype

    del infos_dict

    invalid = {
        name: hdf_dtype
        for name, hdf_dtype in hdf_dtypes.items()
        if hdf_dtype is None
        or (
            hdf_dtype not in _SUPPORTED_HDF_DTYPES
            and not (isinstance(hdf_dtype, np.dtype) and hdf_dtype.kind == "S")
        )
    }
    if len(invalid) > 0:
        msg = f"Invalid hdf dtype found in item. (found {len(invalid)}/{len(hdf_dtypes)} unsupported dtypes with {invalid=}, but expected dtypes in {_SUPPORTED_HDF_DTYPES})"
        raise ValueError(msg)

    load_as_complex: Dict[str, bool] = {}
    if store_complex_as_real:
        for attr_name in list(hdf_dtypes.keys()):
            if hdf_dtypes[attr_name] != "c":
                continue
            load_as_complex[attr_name] = True
            hdf_dtypes[attr_name] = "f"
            max_shapes[attr_name] = max_shapes[attr_name] + (2,)

    if verbose >= 2:
        pylog.debug(f"Found max_shapes:\n{max_shapes}")
        pylog.debug(f"Found hdf_dtypes:\n{hdf_dtypes}")
        pylog.debug(f"Found all_eq_shapes:\n{all_eq_shapes}")

    return (
        dict_pre_transform,
        item_type,
        max_shapes,
        hdf_dtypes,
        all_eq_shapes,
        load_as_complex,
        src_np_dtypes,
    )


def hdf_dtype_to_fill_value(hdf_dtype: Optional[HDFDType]) -> BuiltinScalar:
    if hdf_dtype == "b":
        return False
    elif hdf_dtype in ("i", "u"):
        return 0
    elif hdf_dtype == "f":
        return 0.0
    elif hdf_dtype == "c" or (
        isinstance(hdf_dtype, np.dtype) and hdf_dtype.kind in ("V", "O", "S")
    ):
        return None
    else:
        msg = (
            f"Unsupported type {hdf_dtype=}. (expected one of {_SUPPORTED_HDF_DTYPES})"
        )
        raise ValueError(msg)


def numpy_dtype_to_hdf_dtype(dtype: Optional[np.dtype]) -> HDFDType:
    if dtype is None:
        return HDF_VOID_DTYPE

    kind = dtype.kind

    if kind == "u":  # uint stored as int
        return "i"
    elif kind == "S":
        return h5py.string_dtype(HDF_ENCODING, dtype.itemsize)
    elif kind == "U":  # unicode string
        return HDF_STRING_DTYPE
    elif kind == "V":
        return HDF_VOID_DTYPE
    elif kind in ("b", "i", "f", "c"):
        return kind
    else:
        raise ValueError(f"Unsupported dtype {kind=} for HDF dtype.")


def hdf_dtype_to_numpy_dtype(hdf_dtype: HDFDType) -> np.dtype:
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
