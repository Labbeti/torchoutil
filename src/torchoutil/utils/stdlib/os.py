#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from re import Pattern
from typing import Any, Generator, Iterable, List, Union

from torchoutil.utils.stdlib.re import PatternLike, compile_patterns, pass_patterns


def tree_iter(
    dpath: Union[str, Path],
    *,
    ignore: Union[PatternLike, Iterable[PatternLike]] = (),
    recurse: bool = True,
    space: str = "    ",
    branch: str = "│   ",
    tee: str = "├── ",
    last: str = "└── ",
) -> Generator[str, Any, None]:
    """A recursive generator, given a directory Path object will yield a visual tree structure line by line with each line prefixed by the same characters

    BASED ON https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
    """
    dpath = Path(dpath)
    ignore = compile_patterns(ignore)
    if pass_patterns(str(dpath), ignore):
        yield from ()

    yield dpath.name
    yield from _tree_impl(
        dpath,
        ignore=ignore,
        recurse=recurse,
        prefix="",
        space=space,
        branch=branch,
        tee=tee,
        last=last,
    )


def _tree_impl(
    dpath: Path,
    ignore: List[Pattern],
    recurse: bool,
    prefix: str,
    space: str,
    branch: str,
    tee: str,
    last: str,
) -> Generator[str, Any, None]:
    paths = dpath.iterdir()
    paths = [path for path in paths if not pass_patterns(str(path), ignore)]

    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(paths) - 1) + [last]

    for pointer, path in zip(pointers, paths):
        yield prefix + pointer + path.name

        if recurse and path.is_dir():
            extension = branch if pointer == tee else space
            # i.e. space because last, └── , above so no more |
            yield from _tree_impl(
                path,
                ignore=ignore,
                recurse=recurse,
                prefix=prefix + extension,
                space=space,
                branch=branch,
                tee=tee,
                last=last,
            )
