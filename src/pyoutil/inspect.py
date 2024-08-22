#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect


def get_current_fn_name() -> str:
    return inspect.currentframe().f_back.f_code.co_name
