#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random
import unittest
from unittest import TestCase

from torchoutil.pyoutil.math import nextafter


class TestPyoutilMath(TestCase):
    def test_nextafter_compat(self) -> None:
        examples = [
            math.nan,
            math.inf,
            +math.inf,
            +0.0,
            -0.0,
            1,
            random.random(),
            -random.random(),
        ]

        for example in examples:
            for towards in examples:
                result_1 = nextafter(example, towards)
                result_2 = math.nextafter(example, towards)
                assert (math.isnan(result_1) and math.isnan(result_2)) or (
                    result_1 == result_2
                )


if __name__ == "__main__":
    unittest.main()
