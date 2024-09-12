#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, TypeVar

T = TypeVar("T", covariant=True)

ContentMode = Literal["item", "batch", "column"]
ItemType = Literal["dict", "tuple", "raw"]
ExistsMode = Literal["overwrite", "skip", "error"]


EXISTS_MODES = ("overwrite", "skip", "error")
CONTENT_DNAME = "data"
ATTRS_FNAME = "attributes.json"


class PackedDatasetAttributes(TypedDict):
    batch_size: int
    content_dname: str
    content_mode: ContentMode
    creation_date: str
    files: List[str]
    info: Dict[str, Any]
    length: int
    num_files: int
    source_attrs: Dict[str, Any]
    source_dataset: str
    subdir_size: Optional[int]
    item_type: ItemType


def _tuple_to_dict(x: Tuple[T, ...]) -> Dict[str, T]:
    return dict(zip(map(str, range(len(x))), x))


def _dict_to_tuple(x: Dict[str, T]) -> Tuple[T, ...]:
    return tuple(x.values())
