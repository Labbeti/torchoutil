#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tempfile
import time
import random
import unittest
from pathlib import Path
from unittest import TestCase

import numpy as np
import torch
from torch import Tensor
from torch.utils.data.dataset import Subset
from torchvision.datasets import CIFAR10

from torchoutil.extras.hdf import HDFDataset, pack_to_hdf
from torchoutil.nn import ESequential, IndexToOnehot, ToList, ToNumpy
from torchoutil.pyoutil import dict_list_to_list_dict


class TestHDF(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tmpdir = Path(os.getenv("TORCHOUTIL_TMPDIR", tempfile.gettempdir())).joinpath(
            "torchoutil_tests"
        )
        tmpdir.mkdir(parents=True, exist_ok=True)
        cls.tmpdir = tmpdir

    def test_cifar10_pack_to_hdf(self) -> None:
        cls = self.__class__
        tmpdir = cls.tmpdir

        dataset = CIFAR10(
            tmpdir,
            train=False,
            transform=ToNumpy(),
            target_transform=ESequential(IndexToOnehot(10), ToList()),
            download=True,
        )
        indices = torch.randint(0, len(dataset), (max(len(dataset) // 10, 1),)).tolist()
        dataset = Subset(dataset, indices)

        path = tmpdir.joinpath("test_cifar10.hdf")
        hdf_dataset = pack_to_hdf(dataset, path, exists="overwrite")

        idx = 0
        image0, label0 = dataset[idx]
        image1, label1 = hdf_dataset[idx]

        assert len(dataset) == len(hdf_dataset)
        assert np.equal(image0, image1).bool().all()
        assert np.equal(label0, label1).bool().all()

        hdf_dataset.close(remove_file=True)

    def test_shape_column(self) -> None:
        cls = self.__class__
        tmpdir = cls.tmpdir

        data = [torch.rand(10, 2), torch.rand(10, 5), torch.rand(10, 3)]
        data_shape = [(10, 1), (10, 2), (10, 1)]
        ds_list = [
            {"data": data[i], "data_shape": data_shape[i]} for i in range(len(data))
        ]

        path = tmpdir.joinpath("test_shape.hdf")
        hdf_dataset = pack_to_hdf(ds_list, path, exists="overwrite")

        assert len(hdf_dataset.added_columns) == 0
        assert len(hdf_dataset) == len(ds_list)

        for i, item in enumerate(iter(hdf_dataset)):
            assert set(item.keys()) == {"data", "data_shape"}

            data_i = item["data"]
            assert isinstance(data_i, Tensor)
            assert data_i.shape == data_shape[i]

        hdf_dataset.close()
        os.remove(path)

    def test_special_floating_dtypes(self) -> None:
        cls = self.__class__
        tmpdir = cls.tmpdir

        ds_dict = {
            "f16": torch.rand(10, 2, 3, dtype=torch.float16),
            "c64": torch.rand(10, 2, 3, dtype=torch.complex64),
            "bool": torch.rand(10, 1) > 0.5,
        }
        ds_list = dict_list_to_list_dict(ds_dict, "same")
        keys = set(ds_dict.keys())

        path = tmpdir.joinpath("test_complex.hdf")
        hdf_dataset = pack_to_hdf(ds_list, path, exists="overwrite")

        assert len(hdf_dataset.added_columns) == 0
        assert len(hdf_dataset) == len(ds_list)

        for i, item in enumerate(iter(hdf_dataset)):
            assert set(item.keys()) == keys
            assert set(ds_list[i].keys()) == keys

            for k in keys:
                assert torch.equal(
                    ds_list[i][k], item[k]
                ), f"{i=}; {k=}; {ds_list[i][k]=}; {item[k]=}"

        hdf_dataset.close()
        os.remove(path)

    def test_slice(self) -> None:
        cls = self.__class__
        tmpdir = cls.tmpdir

        ds_dict = {
            "a": list(range(10, 20)),
            "b": torch.rand(10, 2, 3),
        }
        ds_list = dict_list_to_list_dict(ds_dict, "same")

        path = tmpdir.joinpath("test_slice.hdf")
        pack_to_hdf(ds_list, path, exists="overwrite", ds_kwds=dict(open_hdf=False))

        hdf_dataset = HDFDataset(path, cast="to_builtin")

        assert len(hdf_dataset) == len(ds_list)
        assert hdf_dataset[:, "a"] == ds_dict["a"]
        assert hdf_dataset[:, "b"] == ds_dict["b"].tolist()

        indices = torch.randperm(len(hdf_dataset))

        assert hdf_dataset[indices, "a"] == [ds_dict["a"][idx] for idx in indices]
        assert (hdf_dataset[indices, "b"] == ds_dict["b"][indices].numpy()).all()

    def test_string(self) -> None:
        cls = self.__class__
        tmpdir = cls.tmpdir

        ds_dict = {
            "int": list(range(10)),
            "string": [
                "".join(map(str, range(random.randint(10, 100)))) for _ in range(10)
            ],
            "list_string": [
                [],
                ["aa", "bbb"],
                [],
                ["cccc"],
                [""],
                ["dd", "", "e", "ff", "gggggg"],
                ["éééé", "û", "é"],
            ]
            + [[]] * 3,
            "empty_lists": [[]] * 10,
            "bytes": [b""] * 6 + [b"a2", b"dnqzu1dhqz", b"0djqizjdz", b"du12qzd"],
        }
        ds_list = dict_list_to_list_dict(ds_dict, "same")

        path = tmpdir.joinpath("test_string.hdf")
        hdf_dataset = pack_to_hdf(
            ds_list,
            path,
            exists="overwrite",
            ds_kwds=dict(cast="to_builtin"),
        )

        assert hdf_dataset._is_unicode == {
            "int": False,
            "string": True,
            "list_string": True,
            "empty_lists": False,
            "bytes": False,
        }

        idx = torch.randint(0, len(hdf_dataset), ()).item()
        col = random.choice(list(ds_dict.keys()))
        assert hdf_dataset[idx, col] == ds_dict[col][idx], f"{idx=}; {col=}"

        assert len(hdf_dataset) == len(ds_list)
        for k in ds_dict.keys():
            assert (
                hdf_dataset[:, k] == ds_dict[k]
            ), f"{k=}, {hdf_dataset[:, k]=} != {ds_dict[k]=}"

    def test_string_comp(self) -> None:
        num_data = 10000
        max_string_len = 100
        max_sublst = 5
        seed = 42

        gen = torch.Generator().manual_seed(seed)
        # note: vlen_str does not support chr(0)
        strings = [
            [
                "".join(
                    map(
                        chr,
                        torch.randint(
                            1, 256, (max_string_len,), generator=gen
                        ).tolist(),
                    )
                )
                for _ in range(1, max_sublst + 1)
            ]
            for _ in range(num_data)
        ]
        ds_dict = {"strings": strings}
        ds_list = dict_list_to_list_dict(ds_dict, key_mode="same")

        ds_vlen = pack_to_hdf(
            ds_list,
            self.tmpdir.joinpath("test_string_comp_vlen.hdf"),
            exists="overwrite",
            use_vlen_str=True,
            verbose=2,
        )
        ds_bytes = pack_to_hdf(
            ds_list,
            self.tmpdir.joinpath("test_string_comp_bytes.hdf"),
            exists="overwrite",
            use_vlen_str=False,
            verbose=2,
        )

        n_try = 10
        duration_ds_vlen_lst = []
        duration_ds_bytes_lst = []

        for _ in range(n_try):
            start = time.perf_counter()
            ds_bytes_data = ds_bytes[:]
            duration_ds_bytes = time.perf_counter() - start

            start = time.perf_counter()
            ds_vlen_data = ds_vlen[:]
            duration_ds_vlen = time.perf_counter() - start

            duration_ds_bytes_lst.append(duration_ds_bytes)
            duration_ds_vlen_lst.append(duration_ds_vlen)
            assert ds_bytes_data == ds_vlen_data

        assert np.median(duration_ds_bytes_lst) < np.median(duration_ds_vlen_lst)

        ds_vlen.close(remove_file=True)
        ds_bytes.close(remove_file=True)


if __name__ == "__main__":
    unittest.main()
