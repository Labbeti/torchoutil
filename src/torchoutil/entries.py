#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, Union

from torchoutil.pyoutil.argparse import str_to_bool
from torchoutil.pyoutil.os import safe_rmdir, tree_iter
from torchoutil.pyoutil.re import PatternLike

pylog = logging.getLogger(__name__)


def print_tree(
    root: Union[str, Path],
    *,
    exclude: Union[PatternLike, Iterable[PatternLike]] = (),
    max_depth: int = sys.maxsize,
    followlinks: bool = False,
) -> None:
    num_dirs = 0
    num_files = 0
    for line in tree_iter(
        root=root,
        exclude=exclude,
        max_depth=max_depth,
        followlinks=followlinks,
    ):
        print(f"{line}")

        if line.endswith("/"):
            num_dirs += 1
        else:
            num_files += 1

    print(f"\n{num_dirs} directories, {num_files} files")


def main_tree() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "root",
        type=str,
        help="Root directory path.",
        default=".",
        nargs="?",  # for optional positional argument
    )
    parser.add_argument(
        "--exclude",
        type=str,
        help="Exclude file patterns.",
        default=(),
        nargs="*",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        help="Max directory tree depth.",
        default=sys.maxsize,
    )
    parser.add_argument(
        "--followlinks",
        type=str_to_bool,
        help="Indicates whether or not symbolic links shound be followed. defaults to True.",
        default=True,
    )
    args = parser.parse_args()

    print_tree(
        root=args.root,
        exclude=args.exclude,
        max_depth=args.max_depth,
        followlinks=args.followlinks,
    )


def print_safe_rmdir(
    root: Union[str, Path],
    *,
    rm_root: bool = True,
    error_on_non_empty_dir: bool = True,
    followlinks: bool = False,
    dry_run: bool = False,
    verbose: int = 0,
) -> None:
    deleted, reviewed = safe_rmdir(
        root=root,
        rm_root=rm_root,
        error_on_non_empty_dir=error_on_non_empty_dir,
        followlinks=followlinks,
        dry_run=dry_run,
        verbose=verbose,
    )
    if dry_run:
        print(
            f"Dry run mode enabled. Here is the list of directories to delete ({len(deleted)}/{len(reviewed)}):"
        )
        for path in deleted:
            print(f" - {path}")

    elif verbose >= 1:
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
        "--dry_run",
        type=str_to_bool,
        default=False,
        help="If True, does not remove any directory and just output the list of directories which could be deleted. defaults to False.",
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
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
