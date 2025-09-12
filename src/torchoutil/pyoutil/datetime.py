#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime

from .warnings import deprecated_alias

ISO8601_DAY_FORMAT = r"%Y-%m-%d"
ISO8601_HOUR_FORMAT_DOUBLE_DOT = r"%Y-%m-%dT%H:%M:%S"


def now_iso8601() -> str:
    return now_iso(ISO8601_HOUR_FORMAT_DOUBLE_DOT)


def now(fmt: str = ISO8601_HOUR_FORMAT_DOUBLE_DOT) -> str:
    return datetime.datetime.now().strftime(fmt)


@deprecated_alias(now)
def now_iso(*args, **kwargs):
    ...
