#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from torchoutil.pyoutil.os import get_num_cpus_available


def get_auto_num_cpus() -> int:
    """Returns the number of CPUs available for the process on Linux-based platforms.
    Useful for setting num_workers argument in DataLoaders.

    On Windows and MAC OS, this will just return the number of CPUs on this machine.
    If the number of CPUs cannot be detected, returns 0.

    Alias of `pyoutil.os.get_num_cpus`.
    """
    return get_num_cpus_available()


def get_auto_num_gpus() -> int:
    """Returns the number of GPUs available to the current process."""
    return torch.cuda.device_count()
