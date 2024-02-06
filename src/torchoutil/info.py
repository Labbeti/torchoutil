#!/usr/bin/env python
# -*- coding: utf-8 -*-

import platform
import sys
from pathlib import Path
from typing import Dict

import torch

import torchoutil
from torchoutil.utils.collections import dump_dict


def get_package_repository_path() -> str:
    """Return the absolute path where the source code of this package is installed."""
    return str(Path(__file__).parent.parent.parent)


def get_install_info() -> Dict[str, str]:
    return {
        "torchoutil": torchoutil.__version__,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "os": platform.system(),
        "architecture": platform.architecture()[0],
        "torch": str(torch.__version__),
        "package_path": get_package_repository_path(),
    }


def print_install_info() -> None:
    """Show main packages versions."""
    install_info = get_install_info()
    dumped = dump_dict(install_info, join="\n", fmt="{key}: {value}")
    print(dumped)


if __name__ == "__main__":
    print_install_info()
