#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Literal, TypedDict

CONTENT_DNAME = "data"
ATTRS_FNAME = "attributes.json"

ContentMode = Literal["item", "batch"]


class PickleAttributes(TypedDict):
    source_dataset: str
    length: int
    creation_date: str
    batch_size: int
    content_mode: ContentMode
    content_dname: str
    info: Dict[str, Any]
    source_attrs: "PickleAttributes"
    num_files: int
    files: List[str]
