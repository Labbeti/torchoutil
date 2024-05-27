#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import sys
from functools import cache
from logging import FileHandler, Formatter, Logger, StreamHandler
from pathlib import Path
from types import ModuleType
from typing import IO, List, Literal, Optional, Sequence, Union

from torchoutil.utils.packaging import _COLORLOG_AVAILABLE

pylog = logging.getLogger(__name__)

DEFAULT_FMT = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
PackageOrLogger = Union[str, ModuleType, None, Logger]


if _COLORLOG_AVAILABLE:
    from colorlog import ColoredFormatter


@cache
def warn_once(msg: str, logger: PackageOrLogger) -> None:
    pylog = _get_loggers(logger)[0]
    pylog.warning(msg)


def setup_logging_verbose(
    package_or_logger: Union[
        PackageOrLogger,
        Sequence[PackageOrLogger],
    ],
    verbose: Optional[int],
    fmt: Union[str, None, Formatter] = DEFAULT_FMT,
    stream: Union[IO[str], Literal["auto"]] = "auto",
) -> None:
    if verbose is None:
        level = None
    else:
        level = _verbose_to_logging_level(verbose)
    return setup_logging_level(package_or_logger, level=level, fmt=fmt, stream=stream)


def setup_logging_level(
    package_or_logger: Union[
        PackageOrLogger,
        Sequence[PackageOrLogger],
    ],
    level: Optional[int],
    fmt: Union[str, None, Formatter] = DEFAULT_FMT,
    stream: Union[IO[str], Literal["auto"]] = "auto",
) -> None:
    logger_lst = _get_loggers(package_or_logger)
    if isinstance(fmt, str):
        fmt = Formatter(fmt)
    if stream == "auto":
        if running_on_interpreter():
            stream = sys.stdout
        else:
            stream = sys.stderr

    for logger in logger_lst:
        found = False

        for handler in logger.handlers:
            if isinstance(handler, StreamHandler) and handler.stream is stream:
                handler.setFormatter(fmt)
                found = True
                break

        if not found:
            handler = StreamHandler(stream)
            handler.setFormatter(fmt)
            logger.addHandler(handler)

        if level is not None:
            logger.setLevel(level)


def _get_loggers(
    package_or_logger: Union[
        PackageOrLogger,
        Sequence[PackageOrLogger],
    ],
) -> List[Logger]:
    if package_or_logger is None or isinstance(
        package_or_logger, (str, Logger, ModuleType)
    ):
        package_or_logger_lst = [package_or_logger]
    else:
        package_or_logger_lst = list(package_or_logger)

    name_or_logger_lst = [
        pkg.__name__ if isinstance(pkg, ModuleType) else pkg
        for pkg in package_or_logger_lst
    ]
    logger_lst = [
        logging.getLogger(pkg_i) if not isinstance(pkg_i, Logger) else pkg_i
        for pkg_i in name_or_logger_lst
    ]
    return logger_lst


def _verbose_to_logging_level(verbose: int) -> int:
    if verbose < 0:
        level = logging.ERROR
    elif verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG
    return level


class CustomFileHandler(FileHandler):
    """FileHandler that build intermediate directories.

    Used for export hydra logs to a file contained in a folder that does not exists yet at the start of the program.
    """

    def __init__(
        self,
        filename: Union[str, Path],
        mode: str = "a",
        encoding: Optional[str] = None,
        delay: bool = True,
        errors: Optional[str] = None,
    ) -> None:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(filename, mode, encoding, delay, errors)


def running_on_interpreter() -> bool:
    return get_ipython_name() is None


def running_on_terminal() -> bool:
    return get_ipython_name() == "TerminalInteractiveShell"


def running_on_notebook() -> bool:
    return get_ipython_name() == "ZMQInteractiveShell"


def get_ipython_name() -> (
    Optional[Literal["TerminalInteractiveShell", "ZMQInteractiveShell"]]
):
    try:
        return get_ipython().__class__.__name__
    except NameError:
        return None


def get_colored_formatter() -> Formatter:
    if not _COLORLOG_AVAILABLE:
        raise RuntimeError(
            "Cannot call function get_colored_fmt() without colorlog installed. Please use `pip install torchoutil[extras]` to install it."
        )

    rank = os.getenv("SLURM_PROCID", 0)
    fmt = f"[%(purple)sRANK{rank}%(reset)s][%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s"
    formatter = ColoredFormatter(
        fmt=fmt,
        log_colors={
            "DEBUG": "purple",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
    return formatter
