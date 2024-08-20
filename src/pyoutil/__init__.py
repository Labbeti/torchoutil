#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .argparse import (
    str_to_bool,
    str_to_optional_bool,
    str_to_optional_int,
    str_to_optional_str,
)
from .collections import (
    KeyMode,
    all_eq,
    all_ne,
    dict_list_to_list_dict,
    dump_dict,
    filter_iterable,
    flat_dict_of_dict,
    flat_list_of_list,
    flatten,
    get_key_fn,
    intersect_lists,
    list_dict_to_dict_list,
    pass_filter,
    prod,
    sort_with_patterns,
    sorted_dict,
    unflat_dict_of_dict,
    unflat_list_of_list,
    union_lists,
    unzip,
)
from .csv import load_csv, to_csv
from .dataclasses import get_defaults_values
from .datetime import now_iso
from .functools import identity
from .hashlib import hash_file
from .importlib import package_is_available, reimport_modules
from .logging import (
    VERBOSE_DEBUG,
    VERBOSE_ERROR,
    VERBOSE_INFO,
    VERBOSE_WARNING,
    CustomFileHandler,
    get_ipython_name,
    running_on_interpreter,
    running_on_notebook,
    running_on_terminal,
    setup_logging_level,
    setup_logging_verbose,
    warn_once,
)
from .math import clamp, clip
from .os import safe_rmdir, tree_iter
from .re import compile_patterns, find_pattern, pass_patterns
from .typing import *
