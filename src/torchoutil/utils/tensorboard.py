#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import logging
import os.path as osp
from pathlib import Path
from typing import Any, Dict, Iterable, List, TypedDict, Union

from typing_extensions import NotRequired

from torchoutil.utils.packaging import _TENSORBOARD_AVAILABLE

if not _TENSORBOARD_AVAILABLE:
    raise ImportError(
        "Optional dependancy 'tensorboard' is not installed. Please install it using 'pip install torchoutil[extras]'"
    )

from tensorboard.backend.event_processing.event_file_loader import (  # type: ignore
    EventFileLoader,
)

pylog = logging.getLogger(__name__)


_EVENT_FILE_PREFIX = "events.out.tfevents."
_DT_FLOAT = 1
_DT_STRING = 7
_DTYPES = (_DT_FLOAT, _DT_STRING)


class TensorboardEvent(TypedDict):
    wall_time: float
    step: int
    tag: str
    dtype: str
    value: Union[str, float]
    string_val: NotRequired[str]
    float_val: NotRequired[List[float]]


def load_event_file(
    fpath: Union[str, Path],
    cast_float_and_str: bool = True,
    ignore_underscore_tags: bool = True,
    verbose: int = 0,
) -> List[TensorboardEvent]:
    """
    Args:
        fpath: File path to a tensorboard event file.
        cast_float_and_str: Cast string to floats and store result in 'value' field. defaults to True.
        ignore_underscore_tags: Ignore event when tag starts with an underscore. defaults to True.
        verbose: Verbose level. Higher value means more log messages. defaults to 0.
    """
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

        if dtype == _DT_FLOAT:
            float_val: List[float] = data_i["float_val"]
            dtype = "float"

            if cast_float_and_str:
                assert len(float_val) == 1
                value = float_val[0]
                del data_i["string_val"]
                del data_i["float_val"]
            else:
                value = float_val

        elif dtype == _DT_STRING:
            string_val: str = data_i["string_val"]
            dtype = "str"

            if cast_float_and_str:
                value = string_val[3:-2]
                tag = tag.split("/")[0]
                del data_i["string_val"]
                del data_i["float_val"]
            else:
                value = string_val

        else:
            raise RuntimeError(f"Unknown value {dtype=}. (expected one of {_DTYPES})")

        data_i["tag"] = tag
        data_i["dtype"] = dtype
        data_i["value"] = value

        data.append(data_i)

    return data


def load_event_files(
    paths_or_patterns: Union[str, Path, Iterable[Union[str, Path]]],
    cast_float_and_str: bool = True,
    ignore_underscore_tags: bool = True,
    verbose: int = 0,
) -> Dict[str, List[TensorboardEvent]]:
    """
    Args:
        paths_or_patterns: Path or glob patterns to multiple files.
        cast_float_and_str: Cast string to floats and store result in 'value' field. defaults to True.
        ignore_underscore_tags: Ignore event when tag starts with an underscore. defaults to True.
        verbose: Verbose level. Higher value means more log messages. defaults to 0.
    """
    if isinstance(paths_or_patterns, (str, Path)):
        paths_or_patterns = [str(paths_or_patterns)]
    else:
        paths_or_patterns = [
            str(path_or_pattern) for path_or_pattern in paths_or_patterns
        ]

    paths = [
        path
        for path_or_pattern in paths_or_patterns
        for path in glob.iglob(path_or_pattern)
    ]
    all_events = {}
    for path in paths:
        events = load_event_file(
            path,
            cast_float_and_str,
            ignore_underscore_tags,
            verbose,
        )
        all_events[path] = events

    return all_events


def get_duration(
    fpath: Union[str, Path],
    verbose: int = 0,
) -> float:
    """Return time elapsed between first and last log in a tensorboard event file."""
    events = load_event_file(fpath, cast_float_and_str=True, verbose=verbose)
    wall_times = [event["wall_time"] for event in events]
    duration = max(wall_times) - min(wall_times)
    return duration
