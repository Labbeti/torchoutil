#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import yaml
from yaml.constructor import ConstructorError

from torchoutil.core.packaging import _YAML_AVAILABLE
from torchoutil.extras.yaml import FullLoader, IgnoreTagLoader, SafeLoader


class TestYaml(TestCase):
    def test_yaml_load_examples(self) -> None:
        if not _YAML_AVAILABLE:
            return None

        dumped = "a: !!python/tuple\n- 1\n- 2"

        result = yaml.load(dumped, Loader=IgnoreTagLoader)
        expected = {"a": [1, 2]}
        assert result == expected

        result = yaml.load(dumped, Loader=FullLoader)
        expected = {"a": (1, 2)}
        assert result == expected

        with self.assertRaises(ConstructorError):
            yaml.load(dumped, Loader=SafeLoader)


if __name__ == "__main__":
    unittest.main()
