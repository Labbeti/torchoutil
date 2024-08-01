#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime

ISO8601_DAY_FORMAT = r"%Y-%m-%d"
ISO8601_HOUR_FORMAT = r"%Y-%m-%dT%H:%M:%S"


def now_iso(fmt: str = ISO8601_HOUR_FORMAT) -> str:
    return datetime.datetime.now().strftime(fmt)
