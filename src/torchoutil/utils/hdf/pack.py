#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import json
import logging
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import h5py
import numpy as np
import torch
import tqdm
from h5py import Dataset as HDFRawDataset
from torch import Tensor, nn
from torch.utils.data.dataloader import DataLoader

import torchoutil as to
from torchoutil.nn.functional.checksum import checksum
from torchoutil.pyoutil.functools import Compose
from torchoutil.pyoutil.typing import is_dataclass_instance, is_dict_str
from torchoutil.types import BuiltinScalar
from torchoutil.utils.data.dataloader import get_auto_num_cpus
from torchoutil.utils.data.dataset import IterableDataset, SizedDatasetLike
from torchoutil.utils.data.dtype import scan_shape_dtypes
from torchoutil.utils.hdf.common import (
    HDF_ENCODING,
    HDF_STRING_DTYPE,
    HDF_VOID_DTYPE,
    SHAPE_SUFFIX,
    ItemType,
    _tuple_to_dict,
)
from torchoutil.utils.hdf.dataset import HDFDataset

T = TypeVar("T", covariant=True)
T_DictOrTuple = TypeVar("T_DictOrTuple", tuple, dict, covariant=True)

HDFDType = Union[
    Literal["b", "i", "u", "f", "c"], np.dtypes.BytesDType, np.dtypes.VoidDType
]


_SUPPORTED_HDF_DTYPES = (
    "b",  # bool
    "i",  # int
    "u",  # uint
    "f",  # float
    "c",  # complex
    HDF_STRING_DTYPE,
    HDF_VOID_DTYPE,
)


class DatasetKind(Enum):
    MAP = 0
    ITERABLE = 1


pylog = logging.getLogger(__name__)


@torch.inference_mode()
def pack_to_hdf(
    dataset: SizedDatasetLike[T],
    hdf_fpath: Union[str, Path],
    pre_transform: Optional[Callable[[T], T_DictOrTuple]] = None,
    exists: Literal["overwrite", "skip", "error"] = "error",
    metadata: str = "",
    verbose: int = 0,
    batch_size: int = 32,
    num_workers: Union[int, Literal["auto"]] = "auto",
    shape_suffix: str = SHAPE_SUFFIX,
    open_hdf: bool = True,
    file_kwargs: Optional[Dict[str, Any]] = None,
    store_complex_as_real: bool = True,
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
        metadata: Additional metadata string to add to the hdf file. defaults to ''.
        verbose: Verbose level. defaults to 0.
        batch_size: The batch size of the dataloader. defaults to 32.
        num_workers: The number of workers of the dataloader.
            If "auto", it will be set to `len(os.sched_getaffinity(0))`. defaults to "auto".
        shape_suffix: Shape column suffix in HDF file. defaults to "_shape".
        open_hdf: If True, opens the output HDF dataset. defaults to True.
        file_kwargs: Options given to h5py file object. defaults to None.

    Returns:
        hdf_dataset: The target HDF dataset object.
    """
    # Check inputs
    if not isinstance(dataset, SizedDatasetLike):
        raise TypeError(
            f"Cannot pack to hdf a non-sized-dataset '{dataset.__class__.__name__}'."
        )
    if len(dataset) == 0:
        msg = f"Cannot pack to hdf an empty dataset. (found {len(dataset)=})"
        raise ValueError(msg)

    hdf_fpath = Path(hdf_fpath).resolve()
    if hdf_fpath.exists() and not hdf_fpath.is_file():
        raise RuntimeError(f"Item {hdf_fpath=} exists but it is not a file.")

    if not hdf_fpath.is_file():
        pass
    elif exists == "overwrite":
        pass
    elif exists == "error":
        msg = f"Cannot overwrite file {hdf_fpath}. Please remove it or use exists='overwrite' or exists='skip' option."
        raise ValueError(msg)
    elif exists == "skip":
        return HDFDataset(hdf_fpath, open_hdf=open_hdf)
    else:
        EXISTS_VALUES = ("error", "skip", "overwrite")
        msg = f"Invalid argument {exists=}. (expected one of {EXISTS_VALUES})"
        raise ValueError(msg)

    if file_kwargs is None:
        file_kwargs = {}

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
    ) = _scan_dataset(
        dataset,
        pre_transform,
        batch_size,
        num_workers,
        store_complex_as_real,
        verbose,
    )

    now = datetime.datetime.now()
    creation_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    added_columns = []

    with h5py.File(
        hdf_fpath,
        "w",
        **file_kwargs,
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

            if verbose >= 2:
                msg = f"Build hdf dset '{attr_name}' with shape={(len(dataset),) + shape}."
                pylog.debug(msg)

            hdf_dsets[attr_name] = hdf_file.create_dataset(
                attr_name,
                (len(dataset),) + shape,
                hdf_dtype,
                **kwargs,
            )

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

                    # info = _scan_shape_dtype(value)
                    info = scan_shape_dtypes(value)
                    shape = info.shape
                    hdf_dtype = numpy_dtype_to_hdf_dtype(info.numpy_dtype)

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

                    if isinstance(value, Tensor) and value.is_cuda:  # type: ignore
                        value = value.cpu()  # type: ignore

                    # If the value is a sequence but not an array or tensor
                    if hdf_dtype in ("i", "f", "c") and not isinstance(
                        value, (Tensor, np.ndarray)
                    ):
                        value = np.array(value)

                    # Note: "hdf_dset[slices]" is a generic version of "hdf_dset[i, :shape_0, :shape_1]"
                    slices = (i,) + tuple(slice(shape_i) for shape_i in shape)
                    try:
                        hdf_dset[slices] = value
                    except (TypeError, ValueError) as err:
                        msg = f"Cannot set data {value} into {hdf_dset[slices].shape} ({attr_name=}, {i=}, {slices=})"
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

        attributes = {
            "creation_date": creation_date,
            "source_dataset": dataset.__class__.__name__,
            "length": len(dataset),
            "metadata": str(metadata),
            "encoding": HDF_ENCODING,
            "info": json.dumps(info),
            "global_hash_value": global_hash_value,
            "item_type": item_type,
            "added_columns": added_columns,
            "shape_suffix": shape_suffix,
            "file_kwargs": json.dumps(file_kwargs),
            "load_as_complex": json.dumps(load_as_complex),
            "version": str(to.__version__),
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
        pylog.debug(f"Data into has been packed into HDF file '{hdf_fpath}'.")

    hdf_dataset = HDFDataset(hdf_fpath, open_hdf=open_hdf, return_added_columns=False)
    return hdf_dataset


def _scan_dataset(
    dataset: SizedDatasetLike[T],
    pre_transform: Callable[[T], T_DictOrTuple],
    batch_size: int,
    num_workers: int,
    store_complex_as_real: bool,
    verbose: int,
) -> Tuple[
    Callable[[T], Dict[str, Any]],
    ItemType,
    Dict[str, Tuple[int, ...]],
    Dict[str, HDFDType],
    Dict[str, bool],
    Dict[str, bool],
]:
    shapes_0 = {}
    hdf_dtypes_0 = {}
    if isinstance(dataset, IterableDataset):
        item_0 = next(iter(dataset))
    else:
        item_0 = dataset[0]
    item_0 = pre_transform(item_0)

    dict_pre_transform: Callable[[T], Dict[str, Any]]

    if is_dict_str(item_0):
        item_type = "dict"
        dict_pre_transform = pre_transform  # type: ignore
        item_0_dict = item_0

    elif isinstance(item_0, tuple):
        item_type = "tuple"
        dict_pre_transform = Compose(pre_transform, _tuple_to_dict)
        item_0_dict = _tuple_to_dict(item_0)

    else:
        raise ValueError(
            f"Invalid item type for {dataset.__class__.__name__}. (expected dict[str, Any] or tuple but found {type(item_0)})"
        )
    del pre_transform, item_0

    for attr_name, value in item_0_dict.items():
        # info = _scan_shape_dtype(value)
        info = scan_shape_dtypes(value)
        shapes_0[attr_name] = info.shape
        hdf_dtypes_0[attr_name] = numpy_dtype_to_hdf_dtype(info.numpy_dtype)

    max_shapes: Dict[str, Tuple[int, ...]] = shapes_0
    hdf_dtypes: Dict[str, HDFDType] = hdf_dtypes_0
    all_eq_shapes: Dict[str, bool] = {
        attr_name: True for attr_name in item_0_dict.keys()
    }

    invalid = {
        name: hdf_dtype
        for name, hdf_dtype in hdf_dtypes.items()
        if hdf_dtype not in _SUPPORTED_HDF_DTYPES
    }
    if len(invalid) > 0:
        msg = f"Invalid hdf dtype found in item. (found {len(invalid)}/{len(hdf_dtypes)} unsupported dtypes with {invalid=}, but expected dtypes in {_SUPPORTED_HDF_DTYPES})"
        raise ValueError(msg)

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
        desc="Pre compute shapes...",
        disable=verbose <= 0,
    ):
        batch = [dict_pre_transform(item) for item in batch]
        for item in batch:
            for attr_name, value in item.items():
                # info = _scan_shape_dtype(value)
                info = scan_shape_dtypes(value)
                shape = info.shape
                hdf_dtype = numpy_dtype_to_hdf_dtype(info.numpy_dtype)

                all_eq_shapes[attr_name] &= max_shapes[attr_name] == shape
                max_shapes[attr_name] = tuple(
                    map(max, zip(max_shapes[attr_name], shape))
                )
                if hdf_dtypes[attr_name] == hdf_dtype or hdf_dtype == HDF_VOID_DTYPE:
                    # Note: HDF_VOID_DTYPE is compatible
                    pass
                elif hdf_dtypes[attr_name] == HDF_VOID_DTYPE:
                    # Note: if the element 0 was void dtype, override with more specific dtype
                    hdf_dtypes[attr_name] = hdf_dtype
                else:
                    raise ValueError(
                        f"Found different hdf_dtype. (with {hdf_dtypes[attr_name]=} != {hdf_dtype=} and {attr_name=} with {value=})"
                    )

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
    )


def hdf_dtype_to_fill_value(hdf_dtype: Optional[HDFDType]) -> BuiltinScalar:
    if hdf_dtype == "b":
        return False
    elif hdf_dtype in ("i", "u"):
        return 0
    elif hdf_dtype == "f":
        return 0.0
    elif hdf_dtype in ("c", HDF_STRING_DTYPE, HDF_VOID_DTYPE):
        return None
    else:
        msg = (
            f"Unsupported type {hdf_dtype=}. (expected one of {_SUPPORTED_HDF_DTYPES})"
        )
        raise ValueError(msg)


def numpy_dtype_to_hdf_dtype(
    dtype: Optional[np.dtype],
) -> HDFDType:
    if dtype is None:
        return HDF_VOID_DTYPE

    kind = dtype.kind
    if kind == "u":  # uint stored as int
        return "i"
    elif kind == "U":  # unicode string
        return HDF_STRING_DTYPE
    elif kind in ("b", "i", "u", "f", "c"):
        return kind
    else:
        raise ValueError(f"Unsupported dtype {kind=} for HDF dtype.")
