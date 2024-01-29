#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path as osp
from typing import Any, Dict, List

from torchoutil.utils.packaging import _TENSORBOARD_AVAILABLE

pylog = logging.getLogger(__name__)


EVENT_FILE_PREFIX = "events.out.tfevents."
DT_FLOAT = 1
DT_STRING = 7


if _TENSORBOARD_AVAILABLE:
    from tensorboard.backend.event_processing.event_file_loader import (  # type: ignore
        EventFileLoader,
    )

    def load_event_file(
        fpath: str,
        cast_float_and_str: bool = True,
        ignore_underscore_tags: bool = True,
        verbose: int = 0,
    ) -> List[Dict[str, Any]]:
        if not osp.isfile(fpath):
            raise FileNotFoundError(f"Invalid argument {fpath=}. (not a file)")

        event_file_loader = EventFileLoader(fpath)
        raw_data = []

        for event in event_file_loader.Load():
            wall_time: float = event.wall_time  # type: ignore
            event_values: list = event.summary.value  # type: ignore
            step: int = event.step  # type: ignore

            for event_value in event_values:
                tag = event_value.tag
                dtype = event_value.tensor.dtype
                string_val = event_value.tensor.string_val
                float_val = event_value.tensor.float_val

                data_i = {
                    "wall_time": wall_time,
                    "step": step,
                    "tag": tag,
                    "dtype": dtype,
                    "string_val": string_val,
                    "float_val": float_val,
                }
                raw_data.append(data_i)

        data = []
        for data_i in raw_data:
            tag: str = data_i["tag"]
            dtype: Any = data_i["dtype"]

            if ignore_underscore_tags and tag.startswith("_"):
                if verbose >= 2:
                    pylog.debug(
                        f'Skip value with tag "{tag}" which begins by an underscore.'
                    )
                continue

            if cast_float_and_str:
                if dtype == DT_FLOAT:
                    float_val: List[float] = data_i["float_val"]
                    assert len(float_val) == 1
                    value = float_val[0]
                    dtype = "float"

                elif dtype == DT_STRING:
                    string_val: str = data_i["string_val"]
                    value = string_val[3:-2]
                    dtype = "str"
                    tag = tag.split("/")[0]

                else:
                    raise RuntimeError(f"Unknown value {dtype=}.")

                del data_i["string_val"]
                del data_i["float_val"]

                data_i["tag"] = tag
                data_i["dtype"] = dtype
                data_i["value"] = value

            data.append(data_i)

        return data
