#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Literal, Optional, TypedDict

CONTENT_DNAME = "data"
ATTRS_FNAME = "attributes.json"

ContentMode = Literal["item", "batch"]


class PackedDatasetAttributes(TypedDict):
    source_dataset: str
    length: int
    creation_date: str
    batch_size: int
    content_mode: ContentMode
    content_dname: str
    subdir_size: Optional[int]
    info: Dict[str, Any]
    source_attrs: "PackedDatasetAttributes"
    num_files: int
    files: List[str]
