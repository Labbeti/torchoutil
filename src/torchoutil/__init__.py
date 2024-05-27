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
__version__ = "0.4.0"


from torch import *

# Re-import for language servers
from . import hub as hub
from . import nn as nn
from . import optim as optim
from . import utils as utils
from .nn.functional import *
from .utils.type_checks import (
    is_numpy_scalar,
    is_python_scalar,
    is_scalar,
    is_torch_scalar,
)
