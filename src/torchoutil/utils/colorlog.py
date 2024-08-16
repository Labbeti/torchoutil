#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os

from torchoutil.utils.packaging import _COLORLOG_AVAILABLE

if not _COLORLOG_AVAILABLE:
    raise ImportError(
        "Optional dependancy 'colorlog' is not installed. Please install it using 'pip install torchoutil[extras]'"
    )

from colorlog import ColoredFormatter

pylog = logging.getLogger(__name__)


def get_colored_formatter() -> ColoredFormatter:
    if not _COLORLOG_AVAILABLE:
        raise RuntimeError(
            "Cannot call function get_colored_formatter() without colorlog installed. Please use `pip install colorlog` to install it."
        )

    rank = os.getenv("SLURM_PROCID", 0)
    fmt = f"[%(purple)sRANK{rank}%(reset)s][%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s"
    log_colors = {
        "DEBUG": "purple",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    }
    formatter = ColoredFormatter(fmt=fmt, log_colors=log_colors)
    return formatter
