#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Collection of functions and modules to help development in PyTorch.
"""

__name__ = "torchoutil"
__author__ = "Étienne Labbé (Labbeti)"
__author_email__ = "labbeti.pub@gmail.com"
__license__ = "MIT"
__maintainer__ = "Étienne Labbé (Labbeti)"
__status__ = "Development"
__version__ = "0.3.2"


# Redundant imports to avoid to import PyTorch modules
from . import hub as hub
from . import nn as nn
from . import optim as optim
from . import utils as utils
from .nn.functional import *
