#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
from subprocess import CalledProcessError
from typing import TypeVar, Union

T = TypeVar("T")


def get_githash_short(*, default: T = "unknown") -> Union[str, T]:
    cmd = ["git", "rev-parse", "--short", "HEAD"]
    try:
        return subprocess.check_output(cmd).decode().strip()
    except (CalledProcessError, PermissionError):
        return default


def get_githash_full(*, default: T = "unknown") -> Union[str, T]:
    cmd = ["git", "rev-parse", "HEAD"]
    try:
        return subprocess.check_output(cmd).decode().strip()
    except (CalledProcessError, PermissionError):
        return default
