#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import string


def randstr(size: int = 10, letters: str = string.ascii_letters) -> str:
    """Returns a randomly generated string."""
    return "".join(random.choice(letters) for _ in range(size))
