#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .abc import Singleton
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
    argmax,
    argmin,
    contained,
    dict_list_to_list_dict,
    dump_dict,
    filter_iterable,
    flat_dict_of_dict,
    flat_list_of_list,
    flatten,
    intersect_lists,
    list_dict_to_dict_list,
    prod,
    recursive_generator,
    sorted_dict,
    unflat_dict_of_dict,
    unflat_list_of_list,
    union_dicts,
    union_lists,
    unzip,
)
from .csv import load_csv, to_csv
from .dataclasses import get_defaults_values
from .datetime import now_iso
from .difflib import find_closest_in_list, sequence_matcher_ratio
from .enum import StrEnum
from .functools import Compose, compose, identity
from .hashlib import hash_file
from .importlib import (
    package_is_available,
    reload_globals_modules,
    reload_submodules,
    search_imported_submodules,
)
from .inspect import get_current_fn_name, get_fullname
from .io import open_close_wrap
from .json import load_json, to_json
from .logging import (
    VERBOSE_DEBUG,
    VERBOSE_ERROR,
    VERBOSE_INFO,
    VERBOSE_WARNING,
    MkdirFileHandler,
    get_current_file_logger,
    get_ipython_name,
    get_null_logger,
    running_on_interpreter,
    running_on_notebook,
    running_on_terminal,
    setup_logging_level,
    setup_logging_verbose,
    warn_once,
)
from .math import clamp, clip
from .os import get_num_cpus_available, safe_rmdir, tree_iter
from .re import (
    PatternLike,
    PatternListLike,
    compile_patterns,
    find_patterns,
    get_key_fn,
    match_patterns,
    sort_with_patterns,
)
from .typing import *
