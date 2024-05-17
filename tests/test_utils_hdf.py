#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest
from unittest import TestCase

import numpy as np
import torch
from torch import Tensor
from torch.utils.data.dataset import Subset
from torchvision.datasets import CIFAR10

from torchoutil.nn import ESequential, IndexToOnehot, ToList, ToNumpy
from torchoutil.utils.hdf import pack_to_hdf


class TestHDF(TestCase):
    def test_cifar10_pack_to_hdf(self) -> None:
        dataset = CIFAR10(
            "/tmp",
            train=False,
            transform=ToNumpy(),
            target_transform=ESequential(IndexToOnehot(10), ToList()),
            download=True,
        )
        dataset = Subset(
            dataset,
            torch.randint(0, len(dataset), (max(len(dataset) // 10, 1),)).tolist(),
        )

        path = "/tmp/test_cifar10.hdf"
        hdf_dataset = pack_to_hdf(dataset, path, overwrite=True)

        idx = 0
        image0, label0 = dataset[idx]
        image1, label1 = hdf_dataset[idx]

        assert len(dataset) == len(hdf_dataset)
        assert np.equal(image0, image1).bool().all()
        assert np.equal(label0, label1).bool().all()

        hdf_dataset.close()
        os.remove(path)

    def test_shape_column(self) -> None:
        data = [torch.rand(10, 2), torch.rand(10, 5), torch.rand(10, 3)]
        data_shape = [(10, 1), (10, 2), (10, 1)]

        dataset = [
            {"data": data[i], "data_shape": data_shape[i]} for i in range(len(data))
        ]

        path = "/tmp/test_shape.hdf"
        hdf_dataset = pack_to_hdf(dataset, path, overwrite=True)

        assert len(hdf_dataset.added_columns) == 0

        for i, item in enumerate(iter(hdf_dataset)):
            assert set(item.keys()) == {"data", "data_shape"}

            data_i = item["data"]
            assert isinstance(data_i, Tensor)
            assert data_i.shape == data_shape[i]

        hdf_dataset.close()
        os.remove(path)


if __name__ == "__main__":
    unittest.main()
