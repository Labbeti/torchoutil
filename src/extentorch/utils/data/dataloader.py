#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


def get_auto_num_cpus() -> int:
    """Returns the number of CPUs available, useful for num_workers attribute used in DataLoaders."""
    return len(os.sched_getaffinity(0))
