#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Any, Optional, Union


def to_json(
    data: Any,
    fpath: Union[str, Path, None] = None,
    *,
    overwrite: bool = True,
    make_parents: bool = True,
    # JSON dump kwargs
    indent: Optional[int] = 4,
    ensure_ascii: bool = False,
    **json_dump_kwds,
) -> str:
    """Dump content to JSON format."""
    if fpath is not None:
        fpath = Path(fpath).resolve().expanduser()
        if not overwrite and fpath.exists():
            raise FileExistsError(f"File {fpath} already exists.")
        elif make_parents:
            fpath.parent.mkdir(parents=True, exist_ok=True)

    content = json.dumps(
        data,
        indent=indent,
        ensure_ascii=ensure_ascii,
        **json_dump_kwds,
    )
    if fpath is not None:
        fpath.write_text(content)
    return content


def load_json(fpath: Union[str, Path], **json_load_kwds) -> Any:
    fpath = Path(fpath)
    content = fpath.read_text()
    data = json.loads(content, **json_load_kwds)
    return data
