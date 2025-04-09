#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os

from torchoutil.core.packaging import _COLORLOG_AVAILABLE

if not _COLORLOG_AVAILABLE:
    msg = "Cannot import colorlog objects because optional dependency 'colorlog' is not installed. Please install it using 'pip install torchoutil[extras]'"
    raise ImportError(msg)

from colorlog import ColoredFormatter  # type: ignore

pylog = logging.getLogger(__name__)


LOG_COLORS = {
    "DEBUG": "purple",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}


def get_colored_formatter(slurm_rank: bool = False) -> ColoredFormatter:
    if not _COLORLOG_AVAILABLE:
        msg = "Cannot call function get_colored_formatter because optional dependency 'colorlog' is not installed. Please install it using 'pip install torchoutil[extras]'"
        raise RuntimeError(msg)

    if slurm_rank:
        rank = os.getenv("SLURM_PROCID", "0")
        rank_fmt = f"[%(purple)sRANK{rank}%(reset)s]"
    else:
        rank_fmt = ""

    fmt = (
        rank_fmt
        + "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s"
    )
    formatter = ColoredFormatter(fmt=fmt, log_colors=LOG_COLORS)
    return formatter
