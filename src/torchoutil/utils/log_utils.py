#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys
from functools import cache
from logging import Formatter, Logger, StreamHandler
from types import ModuleType
from typing import List, Sequence, Union

pylog = logging.getLogger(__name__)

DEFAULT_FMT = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"


@cache
def warn_once(msg: str, logger: Union[Logger, ModuleType, None]) -> None:
    if logger is None:
        pylog = logging.root
    elif isinstance(logger, ModuleType):
        pylog: Logger = logger.root
    else:
        pylog = logger

    pylog.warning(msg)


def setup_logging_verbose(
    package_or_logger: Union[
        str,
        ModuleType,
        None,
        Logger,
        Sequence[Union[str, ModuleType, None]],
        Sequence[Logger],
    ],
    verbose: int,
    fmt: Union[str, None, Formatter] = DEFAULT_FMT,
) -> None:
    level = _verbose_to_logging_level(verbose)
    return setup_logging_level(package_or_logger, level=level, fmt=fmt)


def setup_logging_level(
    package_or_logger: Union[
        str,
        ModuleType,
        None,
        Logger,
        Sequence[Union[str, ModuleType, None]],
        Sequence[Logger],
    ],
    level: int,
    fmt: Union[str, None, Formatter] = DEFAULT_FMT,
) -> None:
    logger_lst = _get_loggers(package_or_logger)
    if isinstance(fmt, str):
        fmt = Formatter(fmt)

    for logger in logger_lst:
        found = False

        for handler in logger.handlers:
            if isinstance(handler, StreamHandler) and handler.stream is sys.stdout:
                handler.setFormatter(fmt)
                found = True
                break

        if not found:
            handler = StreamHandler(sys.stdout)
            handler.setFormatter(fmt)
            logger.addHandler(handler)

        logger.setLevel(level)


def _get_loggers(
    package_or_logger: Union[
        str,
        ModuleType,
        None,
        Logger,
        Sequence[Union[str, ModuleType, None]],
        Sequence[Logger],
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
