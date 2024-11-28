#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os

from torchoutil.core.packaging import _COLORLOG_AVAILABLE

if not _COLORLOG_AVAILABLE:
    msg = "Cannot import colorlog objects because optional dependancy 'colorlog' is not installed. Please install it using 'pip install torchoutil[extras]'"
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


def get_colored_formatter() -> ColoredFormatter:
    if not _COLORLOG_AVAILABLE:
        msg = "Cannot call function get_colored_formatter because optional dependancy 'colorlog' is not installed. Please install it using 'pip install torchoutil[extras]'"
        raise RuntimeError(msg)

    rank = os.getenv("SLURM_PROCID", 0)
    fmt = f"[%(purple)sRANK{rank}%(reset)s][%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s"
    formatter = ColoredFormatter(fmt=fmt, log_colors=LOG_COLORS)
    return formatter
