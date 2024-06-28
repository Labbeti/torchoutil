#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, Union

from torchoutil.hub.download import safe_rmdir
from torchoutil.utils.stdlib.argparse import str_to_bool
from torchoutil.utils.stdlib.os import tree_iter

pylog = logging.getLogger(__name__)


def print_tree(root: Union[str, Path], ignore: Iterable[str]) -> None:
    for line in tree_iter(root, ignore=ignore):
        print(f"{line}")


def main_tree() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "root",
        type=str,
        help="Root directory path.",
    )
    parser.add_argument(
        "--ignore",
        type=str,
        default=(),
        nargs="*",
        help="Ignored patterns files.",
    )
    args = parser.parse_args()
    print_tree(root=args.root, ignore=args.ignore)


def print_safe_rmdir(
    root: Union[str, Path],
    *,
    rm_root: bool = True,
    error_on_non_empty_dir: bool = True,
    followlinks: bool = False,
    verbose: int = 0,
) -> None:
    deleted = safe_rmdir(
        root=root,
        rm_root=rm_root,
        error_on_non_empty_dir=error_on_non_empty_dir,
        followlinks=followlinks,
        verbose=verbose,
    )
    if verbose >= 1:
        print(f"{len(deleted)} directories has been deleted.")


def main_safe_rmdir() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "root",
        type=str,
        help="Root directory path.",
    )
    parser.add_argument(
        "--rm_root",
        type=str_to_bool,
        default=True,
        help="If True, remove the root directory too if it is empty at the end. defaults to True.",
    )
    parser.add_argument(
        "--error_on_non_empty_dir",
        type=str_to_bool,
        default=True,
        help="If True, raises a RuntimeError if a subdirectory contains at least 1 file. Otherwise it will ignore non-empty directories. defaults to True.",
    )
    parser.add_argument(
        "--followlinks",
        type=str_to_bool,
        default=False,
        help="Indicates whether or not symbolic links shound be followed. defaults to False.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Verbose level. defaults to 0.",
    )
    args = parser.parse_args()
    print_safe_rmdir(
        root=args.root,
        rm_root=args.rm_root,
        error_on_non_empty_dir=args.error_on_non_empty_dir,
        followlinks=args.followlinks,
        verbose=args.verbose,
    )