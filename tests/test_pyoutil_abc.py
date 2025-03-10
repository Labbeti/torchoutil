#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from torchoutil.pyoutil.abc import Singleton


class Singleton1(metaclass=Singleton):
    ...


class Singleton2(metaclass=Singleton):
    def __init__(self) -> None:
        super().__init__()
        self.num = 1
        self.attr = Singleton1()


class TestPyoutilABC(TestCase):
    def test_singleton(self) -> None:
        a1 = Singleton1()
        b1 = Singleton1()

        assert id(a1) == id(b1)

        a2 = Singleton2()
        b2 = Singleton2()

        assert id(a2) == id(b2)
        assert id(a1) != id(a2)
        assert id(a1) == id(a2.attr)
        assert id(a1) == id(b2.attr)


if __name__ == "__main__":
    unittest.main()
