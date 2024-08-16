#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .argparse import str_to_bool, str_to_optional_int, str_to_optional_str
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
from .dataclasses import get_defaults_values
from .datetime import now_iso
from .functools import identity
from .importlib import reimport_modules
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
from .os import tree_iter
from .re import compile_patterns, find_pattern, pass_patterns
from .typing import (
    BuiltinCollection,
    BuiltinNumber,
    DataclassInstance,
    NamedTupleInstance,
    SizedIterable,
    is_builtin_number,
    is_dataclass_instance,
    is_dict_str,
    is_iterable_bool,
    is_iterable_bytes_or_list,
    is_iterable_float,
    is_iterable_int,
    is_iterable_iterable_int,
    is_iterable_str,
    is_list_bool,
    is_list_float,
    is_list_int,
    is_list_list_str,
    is_list_number,
    is_list_str,
    is_mapping_str,
    is_namedtuple_instance,
    is_sequence_bool,
    is_sequence_int,
    is_sequence_str,
    is_tuple_str,
)
