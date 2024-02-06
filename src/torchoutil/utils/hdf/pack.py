#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import json
import logging
import zlib
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
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

from torchoutil.utils.collections import all_eq
from torchoutil.utils.data.dataloader import get_auto_num_cpus
from torchoutil.utils.data.dataset import SizedDatasetLike
from torchoutil.utils.hdf.common import (
    HDF_ENCODING,
    HDF_STRING_DTYPE,
    HDF_VOID_DTYPE,
    SHAPE_SUFFIX,
    _tuple_to_dict,
)
from torchoutil.utils.hdf.dataset import HDFDataset
from torchoutil.utils.type_checks import is_dict_str

logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U", bound=Union[int, float, str, Tensor, list])
V = TypeVar("V", bound=Union[int, float, str, Tensor, list])


@torch.inference_mode()
def pack_to_hdf(
    dataset: SizedDatasetLike[T],
    hdf_fpath: Union[str, Path],
    pre_transform: Optional[Callable[[T], U]] = None,
    pre_batch_transform: Optional[
        Callable[[List[Dict[str, U]]], List[Dict[str, V]]]
    ] = None,
    overwrite: bool = False,
    metadata: str = "",
    verbose: int = 0,
    batch_size: int = 32,
    num_workers: Union[int, Literal["auto"]] = "auto",
) -> HDFDataset[V, V]:
    """Pack a dataset to HDF file.

    Args:
        dataset: The sized dataset to pack. Must be sized and all items must be of dict type.
            The key of each dictionaries are strings and values can be int, float, str, Tensor, non-empty List[int], non-empty List[float], non-empty List[str].
            If values are tensors or lists, the number of dimensions must be the same for all items in the dataset.
        hdf_fpath: The path to the HDF file.
        pre_transform: The optional transform to apply to audio returned by the dataset BEFORE storing it in HDF file.
            Can be used for deterministic transforms like Resample, LogMelSpectrogram, etc. defaults to None.
        overwrite: If True, the file hdf_fpath can be overwritten. defaults to False.
        metadata: Additional metadata string to add to the hdf file. defaults to ''.
        verbose: Verbose level. defaults to 0.
        batch_size: The batch size of the dataloader. defaults to 32.
        num_workers: The number of workers of the dataloader.
            If "auto", it will be set to `len(os.sched_getaffinity(0))`. defaults to "auto".
    """
    # Check inputs
    if not isinstance(dataset, SizedDatasetLike):
        raise TypeError(
            f"Cannot pack to hdf a non-sized-dataset '{dataset.__class__.__name__}'."
        )
    if len(dataset) == 0:
        raise ValueError("Cannot pack to hdf a empty dataset.")

    hdf_fpath = Path(hdf_fpath)
    if hdf_fpath.exists() and not hdf_fpath.is_file():
        raise RuntimeError(f"Item {hdf_fpath=} exists but it is not a file.")
    if not overwrite and hdf_fpath.is_file():
        raise ValueError(
            f"Cannot overwrite file {hdf_fpath}. Please remove it or use overwrite=True option."
        )

    if num_workers == "auto":
        num_workers = get_auto_num_cpus()
        if verbose >= 2:
            logger.debug(f"Found num_workers=='auto', set to {num_workers}.")

    if pre_transform is None:
        pre_transform = nn.Identity()
    if pre_batch_transform is None:
        pre_batch_transform = nn.Identity()

    if verbose >= 2:
        logger.debug(f"Start packing data into HDF file '{hdf_fpath}'...")

    # Step 1: Init max_shapes and hdf_dtypes with the first item
    shapes_0 = {}
    hdf_dtypes_0 = {}
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
        shape, hdf_dtype = _get_shape_and_dtype(value)
        shapes_0[attr_name] = shape
        hdf_dtypes_0[attr_name] = hdf_dtype

    max_shapes: Dict[str, Tuple[int, ...]] = shapes_0
    hdf_dtypes: Dict[str, str] = hdf_dtypes_0

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
        batch = pre_batch_transform(batch)

        for item in batch:
            for attr_name, value in item.items():
                shape, hdf_dtype = _get_shape_and_dtype(value)
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

    if verbose >= 2:
        logger.debug(f"Found max_shapes:\n{max_shapes}")
        logger.debug(f"Found hdf_dtypes:\n{hdf_dtypes}")

    now = datetime.datetime.now()
    creation_date = now.strftime("%Y-%m-%d_%H-%M-%S")

    if hasattr(dataset, "info"):
        info = dataset.info  # type: ignore
        if is_dataclass(info):
            info = asdict(info)
        elif isinstance(info, Mapping):
            info = dict(info.items())  # type: ignore
        else:
            info = {}
    else:
        info = {}

    with h5py.File(hdf_fpath, "w") as hdf_file:
        # Step 2: Build hdf datasets in file
        hdf_dsets = {}
        for attr_name, shape in max_shapes.items():
            hdf_dtype = hdf_dtypes.get(attr_name)

            kwargs: Dict[str, Any] = {}
            if hdf_dtype == "i":
                kwargs["fillvalue"] = 0
            elif hdf_dtype == "f":
                kwargs["fillvalue"] = 0.0
            elif hdf_dtype in (HDF_STRING_DTYPE, HDF_VOID_DTYPE):
                pass
            else:
                raise ValueError(
                    f"Unknown value {hdf_dtype=}. (with {attr_name=} and {attr_name in hdf_dtypes=})"
                )

            if verbose >= 2:
                logger.debug(
                    f"Build hdf dset '{attr_name}' with shape={(len(dataset),) + shape}."
                )

            hdf_dsets[attr_name] = hdf_file.create_dataset(
                attr_name,
                (len(dataset),) + shape,
                hdf_dtype,
                **kwargs,
            )

            if len(shape) > 0:
                shape_name = f"{attr_name}{SHAPE_SUFFIX}"
                hdf_dsets[shape_name] = hdf_file.create_dataset(
                    shape_name, (len(dataset), len(shape)), "i", fillvalue=-1
                )

        # Fill hdf datasets with a second pass through the whole dataset
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
                    hdf_dset: HDFRawDataset = hdf_dsets[attr_name]
                    shape, hdf_dtype = _get_shape_and_dtype(value)

                    # Check every shape
                    if len(shape) != hdf_dset.ndim - 1:
                        raise ValueError(
                            f"Invalid number of dimension in audio (expected {len(shape)}, found {len(shape)})."
                        )

                    # Check dataset size
                    if any(
                        shape_i > dset_shape_i
                        for shape_i, dset_shape_i in zip(shape, hdf_dset.shape[1:])
                    ):
                        logger.error(
                            f"Resize hdf_dset {attr_name} of shape {tuple(hdf_dset.shape[1:])} with new {shape=}."
                        )
                        raise RuntimeError(
                            "INTERNAL ERROR: Cannot resize dataset when pre-computing shapes."
                        )

                    if isinstance(value, Tensor) and value.is_cuda:  # type: ignore
                        value = value.cpu()  # type: ignore

                    # If the value is a sequence but not an array or tensor
                    if hdf_dtype in ("i", "f") and not isinstance(
                        value, (Tensor, np.ndarray)
                    ):
                        value = np.array(value)

                    # Note: "dset_audios[slices]" is a generic version of "dset_audios[i, :shape_0, :shape_1]"
                    slices = (i,) + tuple(slice(shape_i) for shape_i in shape)
                    try:
                        hdf_dset[slices] = value
                    except TypeError as err:
                        logger.error(
                            f"Cannot set data {value} into {hdf_dset[slices].shape} ({attr_name=}, {i=}, {slices=})"
                        )
                        raise err

                    # Store original shape if needed
                    shape_name = f"{attr_name}{SHAPE_SUFFIX}"
                    if shape_name in hdf_dsets.keys():
                        hdf_shapes_dset = hdf_dsets[shape_name]
                        hdf_shapes_dset[i] = shape

                    global_hash_value += _checksum(value)

                i += 1

        # note: HDF cannot save too large int values with too many bits
        global_hash_value = global_hash_value % (2**31)

        attributes = {
            "creation_date": creation_date,
            "source_dataset": dataset.__class__.__name__,
            "length": len(dataset),
            "metadata": str(metadata),
            "encoding": HDF_ENCODING,
            "info": json.dumps(info),
            "global_hash_value": global_hash_value,
            "item_type": item_type,
        }
        if verbose >= 2:
            dumped_attributes = json.dumps(attributes, indent="\t")
            logger.debug(f"Saving attributes in HDF file:\n{dumped_attributes}")

        for attr_name, attr_val in attributes.items():
            try:
                hdf_file.attrs[attr_name] = attr_val
            except TypeError as err:
                logger.error(
                    f"Cannot store attribute {attr_name=} with value {attr_val=} in HDF."
                )
                raise err

    if verbose >= 2:
        logger.debug(f"Data into has been packed into HDF file '{hdf_fpath}'.")

    hdf_dataset = HDFDataset(hdf_fpath, open_hdf=True)
    return hdf_dataset


class Compose:
    def __init__(self, *fns: Callable) -> None:
        super().__init__()
        self.fns = fns

    def __call__(self, x: Any) -> Any:
        for fn in self.fns:
            x = fn(x)
        return x


def _checksum(
    value: Any,
) -> int:
    if isinstance(value, bytes):
        return zlib.adler32(value)
    elif isinstance(value, (np.ndarray, Tensor)):
        return int(value.sum().item())
    elif isinstance(value, (int, float)):
        return int(value)
    elif isinstance(value, str):
        return _checksum(value.encode())
    elif isinstance(value, (list, tuple)):
        return sum(map(_checksum, value))
    else:
        raise TypeError(f"Invalid argument type {value.__class__.__name__}.")


def _get_shape_and_dtype(
    value: Union[int, float, str, Tensor, list]
) -> Tuple[Tuple[int, ...], str]:
    """Returns the shape and the hdf_dtype for an input."""
    if isinstance(value, int):
        shape = ()
        hdf_dtype = "i"

    elif isinstance(value, float):
        shape = ()
        hdf_dtype = "f"

    elif isinstance(value, str):
        shape = ()
        hdf_dtype = HDF_STRING_DTYPE

    elif isinstance(value, Tensor):
        shape = tuple(value.shape)
        if value.is_floating_point():
            hdf_dtype = "f"
        else:
            hdf_dtype = "i"

    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            shape = (0,)
            hdf_dtype = HDF_VOID_DTYPE
        else:
            sub_shapes_and_dtypes = list(map(_get_shape_and_dtype, value))
            sub_shapes = [shape for shape, _ in sub_shapes_and_dtypes]
            sub_dtypes = [dtype for _, dtype in sub_shapes_and_dtypes]
            sub_dims = list(map(len, sub_shapes))

            if not all_eq(sub_dims):
                raise TypeError(
                    f"Unsupported list of heterogeneous shapes lengths. (found {sub_dims=})"
                )
            if not all_eq(sub_dtypes):
                raise TypeError(
                    f"Unsupported list of heterogeneous types. (found {sub_dtypes=})"
                )
            # Check for avoid ragged array like [["a", "b"], ["c"], ["d", "e"]]
            if not all_eq(sub_shapes):
                raise TypeError(
                    f"Unsupported list of heterogeneous shapes. (found {sub_shapes=} for {value=})"
                )

            max_subshape = tuple(
                max(shape[i] for shape in sub_shapes) for i in range(len(sub_shapes[0]))
            )
            shape = (len(value),) + max_subshape
            hdf_dtype = sub_dtypes[0]
    else:
        raise TypeError(
            f"Unsupported type {value.__class__.__name__} in function get_shape_and_dtype."
        )

    return shape, hdf_dtype