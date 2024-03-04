#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import numpy as np
from torchvision.datasets import CIFAR10

from torchoutil.nn import IndicesToOneHot, ToList, ToNumpy, TSequential
from torchoutil.utils.hdf import pack_to_hdf


class TestHDF(TestCase):
    def test_cifar10_pack_to_hdf(self) -> None:
        dataset = CIFAR10(
            "/tmp",
            train=False,
            transform=ToNumpy(),
            target_transform=TSequential(IndicesToOneHot(10), ToList()),
            download=True,
        )

        path = "/tmp/cifar10.hdf"
        hdf_dataset = pack_to_hdf(dataset, path, overwrite=True)

        idx = 0
        image0, label0 = dataset[idx]
        image1, label1 = hdf_dataset[idx]

        self.assertTrue(np.equal(image0, image1).bool().all())
        self.assertEqual(label0, label1)


if __name__ == "__main__":
    unittest.main()
