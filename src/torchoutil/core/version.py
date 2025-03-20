#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess


def get_githash_short() -> str:
    cmd = ["git", "rev-parse", "--short", "HEAD"]
    return subprocess.check_output(cmd).decode().strip()


def get_githash_full() -> str:
    cmd = ["git", "rev-parse", "HEAD"]
    return subprocess.check_output(cmd).decode().strip()
