#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

import torchoutil as to
from torchoutil.core.semver import Version


class TestVersion(TestCase):
    def test_versions(self) -> None:
        v1 = Version("1.0.0")
        assert v1.to_dict() == {"major": 1, "minor": 0, "patch": 0}

        v2 = Version("1.1.0")
        assert v2.to_dict() == {"major": 1, "minor": 1, "patch": 0}

        assert v1 != v2
        assert v1 < v2
        assert not (v1 > v2)

        v3 = Version("1.2.0")
        assert v3.to_dict() == {"major": 1, "minor": 2, "patch": 0}

        v4 = Version(minor=3, patch=10, major=2)
        assert v4.to_str() == "2.3.10"

        v5 = Version((2, 3, 10))
        assert v5.to_str() == "2.3.10"
        assert v4 == v5

        v6 = Version((3, 4, 5))
        v7 = Version(3, 4, 5)
        assert v6 == v7
        assert v6.to_str() == "3.4.5"

        v10 = Version("1.2.0+test")
        assert v10.to_dict() == {
            "major": 1,
            "minor": 2,
            "patch": 0,
            "buildmetadata": "test",
        }

        v11 = Version("1.2.3-pre.2+build.4")
        assert v11.to_dict() == {
            "major": 1,
            "minor": 2,
            "patch": 3,
            "prerelease": ["pre", 2],
            "buildmetadata": ["build", 4],
        }

        # Check if versions can be parsed
        Version(to.__version__)
        Version(torch.__version__)

        v12 = Version(1, 2)
        assert v12.to_tuple() == (1, 2, 0)

    def test_parse_invalid(self) -> None:
        with self.assertRaises(ValueError):
            Version()

        with self.assertRaises(ValueError):
            Version("")

        with self.assertRaises(ValueError):
            Version(".")

        with self.assertRaises(ValueError):
            Version("1.")

        with self.assertRaises(ValueError):
            Version("1.2")

        with self.assertRaises(ValueError):
            Version("1..0")

        with self.assertRaises(ValueError):
            Version("1.None")

        with self.assertRaises(ValueError):
            Version("1.2.X")

        with self.assertRaises(ValueError):
            Version("1.02.X")

    def test_semver(self) -> None:
        assert Version("1.0.0") < Version("2.0.0") < Version("2.1.0") < Version("2.1.1")
        assert Version("1.0.0-alpha") < Version("1.0.0")
        assert (
            Version("0.9.0")
            < Version("1.0.0-alpha")
            < Version("1.0.0-alpha.1")
            < Version("1.0.0-alpha.beta")
            < Version("1.0.0-beta")
            < Version("1.0.0-beta.2")
            < Version("1.0.0-beta.11")
            < Version("1.0.0-rc.1")
            < Version("1.0.0")
            < Version("2.0.0")
            < Version("2.0.1")
            < Version("2.1.0")
        )

    def test_parse(self) -> None:
        tests = [
            "1.0.0-alpha",
            "1.0.0-alpha.1",
            "1.0.0-0.3.7",
            "1.0.0-x.7.z.92",
            "1.0.0-x-y-z.--",
            "1.0.0-alpha+001",
            "1.0.0+20130313144700",
            "1.0.0-beta+exp.sha.5114f85",
            "1.0.0+21AF26D3----117B344092BD",
        ]
        for version_str in tests:
            _ = Version(version_str)


if __name__ == "__main__":
    unittest.main()
