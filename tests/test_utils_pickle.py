#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision.datasets import CIFAR10

from torchoutil.nn import IndicesToOneHot, ToList, ToNumpy, TSequential
from torchoutil.utils.pickle_dataset import pack_to_pickle


class TestHDF(TestCase):
    def test_cifar10_pack_to_pickle(self) -> None:
        dataset = CIFAR10(
            "/tmp",
            train=False,
            transform=ToNumpy(),
            target_transform=TSequential(IndicesToOneHot(10), ToList()),
            download=True,
        )
        dataset = Subset(
            dataset,
            torch.randint(0, len(dataset), (max(len(dataset) // 10, 1),)).tolist(),
        )

        path = "/tmp/test_cifar10"
        pkl_dataset = pack_to_pickle(dataset, path, overwrite=True)

        assert len(dataset) == len(pkl_dataset)

        idx = 0
        image0, label0 = dataset[idx]
        image1, label1 = pkl_dataset[idx]

        # breakpoint()

        assert len(dataset) == len(pkl_dataset)
        assert label0 == label1
        assert np.equal(image0, image1).all()


if __name__ == "__main__":
    unittest.main()
