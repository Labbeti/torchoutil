#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import os.path as osp
import sys
from pathlib import Path
from re import Pattern
from typing import Any, Generator, Iterable, List, Union

from pyoutil.re import PatternLike, compile_patterns, pass_patterns

pylog = logging.getLogger(__name__)


def safe_rmdir(
    root: Union[str, Path],
    *,
    rm_root: bool = True,
    error_on_non_empty_dir: bool = True,
    followlinks: bool = False,
    verbose: int = 0,
) -> List[str]:
    """Remove all empty sub-directories.

    Args:
        root: Root directory path.
        rm_root: If True, remove the root directory too if it is empty at the end. defaults to True.
        error_on_non_empty_dir: If True, raises a RuntimeError if a subdirectory contains at least 1 file. Otherwise it will ignore non-empty directories. defaults to True.
        followlinks: Indicates whether or not symbolic links shound be followed. defaults to False.
        verbose: Verbose level. defaults to 0.

    Returns:
        The list of directories paths deleted.
    """
    root = str(root)
    if not osp.isdir(root):
        raise FileNotFoundError(
            f"Target root directory does not exists. (with {root=})"
        )

    to_delete = set()
    walker = os.walk(root, topdown=False, followlinks=followlinks)

    for dpath, dnames, fnames in walker:
        if not rm_root and dpath == root:
            continue
        elif len(fnames) == 0 and (
            all(osp.join(dpath, dname) in to_delete for dname in dnames)
        ):
            to_delete.add(dpath)
        elif error_on_non_empty_dir:
            raise RuntimeError(f"Cannot remove non-empty directory '{dpath}'.")
        elif verbose >= 2:
            pylog.debug(f"Ignoring non-empty directory '{dpath}'...")

    for dpath in to_delete:
        os.rmdir(dpath)

    return list(to_delete)


def tree_iter(
    root: Union[str, Path],
    *,
    ignore: Union[PatternLike, Iterable[PatternLike]] = (),
    space: str = "    ",
    branch: str = "│   ",
    tee: str = "├── ",
    last: str = "└── ",
    max_depth: int = sys.maxsize,
) -> Generator[str, Any, None]:
    """A recursive generator, given a directory Path object will yield a visual tree structure line by line with each line prefixed by the same characters

    Based on: https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Invalid argument path '{root}'. (not a directory)")

    ignore = compile_patterns(ignore)
    if pass_patterns(str(root), ignore):
        yield from ()

    yield root.resolve().name + "/"

    if max_depth <= 0:
        return

    yield from _tree_impl(
        root,
        ignore=ignore,
        prefix="",
        space=space,
        branch=branch,
        tee=tee,
        last=last,
        depth=1,
        max_depth=max_depth,
    )


def _tree_impl(
    root: Path,
    ignore: List[Pattern],
    prefix: str,
    space: str,
    branch: str,
    tee: str,
    last: str,
    depth: int,
    max_depth: int,
) -> Generator[str, Any, None]:
    paths = root.iterdir()
    paths = [path for path in paths if not pass_patterns(str(path), ignore)]

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
                ignore=ignore,
                prefix=prefix + extension,
                space=space,
                branch=branch,
                tee=tee,
                last=last,
                depth=depth + 1,
                max_depth=max_depth,
            )