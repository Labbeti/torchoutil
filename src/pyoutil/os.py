#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
from re import Pattern
from typing import Any, Generator, Iterable, List, Union

from pyoutil.re import PatternLike, compile_patterns, pass_patterns


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
