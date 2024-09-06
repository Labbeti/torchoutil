#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Literal, Optional, TypedDict

CONTENT_DNAME = "data"
ATTRS_FNAME = "attributes.json"

ContentMode = Literal["item", "batch"]


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
