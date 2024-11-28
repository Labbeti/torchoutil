#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import os.path as osp
import sys
from pathlib import Path
from re import Pattern
from typing import Any, Generator, Iterable, List, Tuple, Union

from torchoutil.pyoutil.logging import warn_once
from torchoutil.pyoutil.re import PatternLike, compile_patterns, match_patterns

pylog = logging.getLogger(__name__)


def get_num_cpus_available() -> int:
    """Returns the number of CPUs available for the current process on Linux-based platforms.

    On Windows and MAC OS, this will just return the number of logical CPUs on this machine.
    If the number of CPUs cannot be detected, returns 0.
    """
    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        msg = "Cannot detect number of CPUs available for the current process. This function will just returns the number of CPUs."
        warn_once(msg, __name__)

        num_cpus = os.cpu_count()
        if num_cpus is None:
            num_cpus = 0
    return num_cpus


def safe_rmdir(
    root: Union[str, Path],
    *,
    rm_root: bool = True,
    error_on_non_empty_dir: bool = True,
    followlinks: bool = False,
    dry_run: bool = False,
    verbose: int = 0,
) -> Tuple[List[str], List[str]]:
    """Remove all empty sub-directories.

    Args:
        root: Root directory path.
        rm_root: If True, remove the root directory too if it is empty at the end. defaults to True.
        error_on_non_empty_dir: If True, raises a RuntimeError if a subdirectory contains at least 1 file. Otherwise it will ignore non-empty directories. defaults to True.
        followlinks: Indicates whether or not symbolic links shound be followed. defaults to False.
        dry_run: If True, does not remove any directory and just output the list of directories which could be deleted. defaults to False.
        verbose: Verbose level. defaults to 0.

    Returns:
        A tuple containing the list of directories paths deleted and the list of directories paths reviewed.
    """
    root = str(root)
    if not osp.isdir(root):
        msg = f"Target root directory does not exists. (with {root=})"
        raise FileNotFoundError(msg)

    to_delete = {}
    reviewed = []
    walker = os.walk(root, topdown=False, followlinks=followlinks)

    for dpath, dnames, fnames in walker:
        reviewed.append(dpath)

        if not rm_root and dpath == root:
            continue

        elif len(fnames) == 0 and (
            all(osp.join(dpath, dname) in to_delete for dname in dnames)
        ):
            to_delete[dpath] = None

        elif error_on_non_empty_dir:
            raise RuntimeError(f"Cannot remove non-empty directory '{dpath}'.")
        elif verbose >= 2:
            pylog.debug(f"Ignoring non-empty directory '{dpath}'...")

    if not dry_run:
        for dpath in to_delete:
            os.rmdir(dpath)

    return list(to_delete), reviewed


def tree_iter(
    root: Union[str, Path],
    *,
    exclude: Union[PatternLike, Iterable[PatternLike]] = (),
    space: str = "    ",
    branch: str = "│   ",
    tee: str = "├── ",
    last: str = "└── ",
    max_depth: int = sys.maxsize,
    followlinks: bool = False,
) -> Generator[str, Any, None]:
    """A recursive generator, given a directory Path object will yield a visual tree structure line by line with each line prefixed by the same characters

    Based on: https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Invalid argument path '{root}'. (not a directory)")

    if not followlinks and root.is_symlink():
        yield from ()
        return

    exclude = compile_patterns(exclude)
    if match_patterns(str(root), exclude):
        yield from ()
        return

    yield root.resolve().name + "/"

    if max_depth <= 0:
        return

    yield from _tree_impl(
        root,
        exclude=exclude,
        prefix="",
        space=space,
        branch=branch,
        tee=tee,
        last=last,
        depth=1,
        max_depth=max_depth,
        followlinks=followlinks,
    )


def _tree_impl(
    root: Path,
    exclude: List[Pattern],
    prefix: str,
    space: str,
    branch: str,
    tee: str,
    last: str,
    depth: int,
    max_depth: int,
    followlinks: bool,
) -> Generator[str, Any, None]:
    paths = root.iterdir()
    try:
        paths = [
            path
            for path in paths
            if (followlinks or not path.is_symlink())
            and not match_patterns(str(path), exclude)
        ]
    except PermissionError:
        paths = []

    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(paths) - 1) + [last]

    for pointer, path in zip(pointers, paths):
        is_dir = path.is_dir()
        suffix = "/" if is_dir else ""
        yield prefix + pointer + path.name + suffix

        if is_dir and depth <= max_depth:
            extension = branch if pointer == tee else space
            # i.e. space because last, └── , above so no more |
            yield from _tree_impl(
                path,
                exclude=exclude,
                prefix=prefix + extension,
                space=space,
                branch=branch,
                tee=tee,
                last=last,
                depth=depth + 1,
                max_depth=max_depth,
                followlinks=followlinks,
            )
