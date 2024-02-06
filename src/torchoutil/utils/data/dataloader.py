#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


def get_auto_num_cpus() -> int:
    """Returns the number of CPUs available for several Linux-based platforms.

    Useful for setting num_workers argument in DataLoaders.
    """
    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cpus = os.cpu_count()
        if num_cpus is None:
            num_cpus = 0
    return num_cpus
