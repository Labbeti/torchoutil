#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

from torchoutil.utils.stdlib.os import tree_iter


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--ignore", default=())
    args = parser.parse_args()

    for line in tree_iter(args.path, ignore=args.ignore):
        print(f"{line}")


if __name__ == "__main__":
    main()
