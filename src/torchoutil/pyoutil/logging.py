#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import logging
import sys
from functools import lru_cache
from logging import FileHandler, Formatter, Logger, StreamHandler
from pathlib import Path
from types import ModuleType
from typing import IO, List, Literal, Optional, Sequence, TypeVar, Union

from .importlib import reload_submodules

T = TypeVar("T", covariant=True)

PackageOrLogger = Union[str, ModuleType, None, Logger, Literal["__parent_file__"]]
PackageOrLoggerList = Union[PackageOrLogger, Sequence[PackageOrLogger]]

_PARENT_FILE_KEY = "__parent_file__"
DEFAULT_FMT = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
VERBOSE_DEBUG = 2
VERBOSE_INFO = 1
VERBOSE_WARNING = 0
VERBOSE_ERROR = -1

pylog = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def warn_once(
    msg: str,
    logger: PackageOrLoggerList = _PARENT_FILE_KEY,
    *,
    level: int = logging.WARNING,
) -> None:
    loggers = _get_loggers(logger)
    for logger in loggers:
        logger.log(level, msg)


def setup_logging_verbose(
    package_or_logger: PackageOrLoggerList = None,
    verbose: Optional[int] = VERBOSE_INFO,
    *,
    fmt: Union[str, None, Formatter] = DEFAULT_FMT,
    stream: Union[IO[str], Literal["auto"]] = "auto",
    set_fmt: bool = True,
    capture_warnings: bool = True,
    autoreload: bool = True,
) -> None:
    """Helper function to customize logging messages using verbose_level.

    Note: Higher verbose values means more debug messages.
    """
    if verbose is None:
        level = None
    else:
        level = _verbose_to_logging_level(verbose)

    return setup_logging_level(
        package_or_logger,
        level=level,
        fmt=fmt,
        stream=stream,
        set_fmt=set_fmt,
        capture_warnings=capture_warnings,
        autoreload=autoreload,
    )


def setup_logging_level(
    package_or_logger: PackageOrLoggerList = None,
    level: Optional[int] = logging.INFO,
    *,
    fmt: Union[str, None, Formatter] = DEFAULT_FMT,
    stream: Union[IO[str], Literal["auto"]] = "auto",
    set_fmt: bool = True,
    capture_warnings: bool = True,
    autoreload: bool = True,
) -> None:
    """Helper function to customize logging messages using logging.level.

    Note: Lower level values means more debug messages.
    """
    logging.captureWarnings(capture_warnings)

    logger_lst = _get_loggers(package_or_logger)
    if isinstance(fmt, str):
        fmt = Formatter(fmt)

    if stream == "auto":
        if running_on_interpreter():
            stream = sys.stdout
        else:
            stream = sys.stderr

    for logger in logger_lst:
        if set_fmt:
            found = False

            for handler in logger.handlers:
                if isinstance(handler, StreamHandler) and handler.stream is stream:
                    handler.setFormatter(fmt)
                    found = True
                    break

            if not found:
                handler = StreamHandler(stream)  # type: ignore
                handler.setFormatter(fmt)
                logger.addHandler(handler)

        if level is not None:
            logger.setLevel(level)

    if autoreload:
        for logger in logger_lst:
            if logger.name not in sys.modules:
                continue
            reload_submodules(sys.modules[logger.name])


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
        return get_ipython().__class__.__name__  # type: ignore
    except NameError:
        return None


def get_current_file_logger(
    *,
    parent_deep: int = 1,
    default: T = logging.root,
) -> Union[Logger, T]:
    """Returns the logger of the caller file. If this cannot be found, returns the root logger."""
    try:
        frame = inspect.currentframe()
        for _ in range(parent_deep):
            frame = frame.f_back  # type: ignore
        parent_name = frame.f_globals["__name__"]  # type: ignore
        return logging.getLogger(parent_name)
    except (AttributeError, KeyError):
        return default


def get_null_logger() -> Logger:
    logger = logging.getLogger("null_logger")
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL + 1)
    return logger


def _get_loggers(pkg_name_log_arg: PackageOrLoggerList) -> List[Logger]:
    if pkg_name_log_arg is None or isinstance(
        pkg_name_log_arg, (str, Logger, ModuleType)
    ):
        pkg_name_log_lst = [pkg_name_log_arg]
    else:
        pkg_name_log_lst = list(pkg_name_log_arg)

    loggers: List[Logger] = []
    for pkg_name_log in pkg_name_log_lst:
        if isinstance(pkg_name_log, ModuleType):
            logger = logging.getLogger(pkg_name_log.__name__)

        elif pkg_name_log == _PARENT_FILE_KEY:
            logger = get_current_file_logger(parent_deep=2)

        elif isinstance(pkg_name_log, (type(None), str)):
            logger = logging.getLogger(pkg_name_log)

        else:
            logger = pkg_name_log

        loggers.append(logger)

    return loggers


class MkdirFileHandler(FileHandler):
    """FileHandler that build intermediate directories to filename.

    Used for export hydra logs to a file contained in a folder that does not exists yet at the start of the program.
    """

    def __init__(
        self,
        filename: Union[str, Path],
        mode: str = "a",
        encoding: Optional[str] = None,
        delay: bool = True,
        errors: Optional[str] = None,
        mkdir_parents: bool = True,
        mkdir_exist_ok: bool = True,
    ) -> None:
        filename = Path(filename)
        filename.parent.mkdir(parents=mkdir_parents, exist_ok=mkdir_exist_ok)

        super().__init__(filename, mode, encoding, delay, errors)


def _verbose_to_logging_level(verbose: int) -> int:
    if verbose <= VERBOSE_ERROR:
        level = logging.ERROR
    elif verbose == VERBOSE_WARNING:
        level = logging.WARNING
    elif verbose == VERBOSE_INFO:
        level = logging.INFO
    else:
        level = logging.DEBUG
    return level
