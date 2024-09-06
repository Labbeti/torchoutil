#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import unittest
from unittest import TestCase

import numpy as np
import torch
from torch import Tensor
from torch.utils.data.dataset import Subset
from torchvision.datasets import CIFAR10

from torchoutil.nn import ESequential, IndexToOnehot, ToList, ToNumpy
from torchoutil.pyoutil import dict_list_to_list_dict
from torchoutil.utils.hdf import HDFDataset, pack_to_hdf


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
        hdf_dataset = pack_to_hdf(dataset, path, exists="overwrite")

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
        hdf_dataset = pack_to_hdf(dataset, path, exists="overwrite")

        assert len(hdf_dataset.added_columns) == 0

        for i, item in enumerate(iter(hdf_dataset)):
            assert set(item.keys()) == {"data", "data_shape"}

            data_i = item["data"]
            assert isinstance(data_i, Tensor)
            assert data_i.shape == data_shape[i]

        hdf_dataset.close()
        os.remove(path)

    def test_special_floating_dtypes(self) -> None:
        ds_dict = {
            "f16": torch.rand(10, 2, 3, dtype=torch.float16),
            "c64": torch.rand(10, 2, 3, dtype=torch.complex64),
            "bool": torch.rand(10, 1) > 0.5,
        }
        ds_list = dict_list_to_list_dict(ds_dict, "same")
        keys = set(ds_dict.keys())

        path = "/tmp/test_complex.hdf"
        hdf_dataset = pack_to_hdf(ds_list, path, exists="overwrite")

        assert len(hdf_dataset.added_columns) == 0

        for i, item in enumerate(iter(hdf_dataset)):
            assert set(item.keys()) == keys
            assert set(ds_list[i].keys()) == keys

            for k in keys:
                assert torch.equal(ds_list[i][k], item[k])

        hdf_dataset.close()
        os.remove(path)

    def test_slice(self) -> None:
        ds_dict = {
            "a": list(range(10, 20)),
            "b": torch.rand(10, 2, 3),
        }
        ds_list = dict_list_to_list_dict(ds_dict, "same")

        path = "/tmp/test_slice.hdf"
        pack_to_hdf(ds_list, path, exists="overwrite", open_hdf=False)

        hdf_dataset = HDFDataset(path, numpy_to_torch=False)

        assert hdf_dataset[:, "a"] == ds_dict["a"]
        assert (hdf_dataset[:, "b"] == ds_dict["b"].numpy()).all()

    def test_string(self) -> None:
        ds_dict = {
            "i": list(range(10)),
            "s": ["".join(map(str, range(random.randint(10, 100)))) for _ in range(10)],
        }
        ds_list = dict_list_to_list_dict(ds_dict, "same")

        path = "/tmp/test_string.hdf"
        hdf_dataset = pack_to_hdf(ds_list, path, exists="overwrite")

        assert hdf_dataset[:, "i"] == ds_dict["i"]
        assert hdf_dataset[:, "s"] == ds_dict["s"]


if __name__ == "__main__":
    unittest.main()
