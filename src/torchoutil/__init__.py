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


from .nn.functional import *

# Redundant imports to avoid to import PyTorch modules
from . import hub as hub  # isort:skip
from . import nn as nn  # isort:skip
from . import optim as optim  # isort:skip
from . import utils as utils  # isort:skip
