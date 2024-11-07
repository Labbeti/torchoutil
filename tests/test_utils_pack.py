#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import TestCase

import torch
from torch import Generator
from torch.utils.data.dataset import Subset
from torchvision.datasets import CIFAR10

from torchoutil.extras.numpy import np
from torchoutil.nn import ESequential, IndexToOnehot, ToList, ToNumpy
from torchoutil.utils.pack.pack import pack_dataset, pack_dataset_to_columns


class TestPackCIFAR10(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tmpdir = Path(os.getenv("TORCHOUTIL_TMPDIR", tempfile.gettempdir())).joinpath(
            "torchoutil_tests"
        )
        dataset = CIFAR10(
            tmpdir,
            train=False,
            transform=ToNumpy(),
            target_transform=ESequential(IndexToOnehot(10), ToList()),
            download=True,
        )
        cls.dataset = dataset
        cls.tmpdir = tmpdir

    def test_cifar10_pack_per_item(self) -> None:
        cls = self.__class__
        dataset = cls.dataset
        tmpdir = cls.tmpdir

        seed = int(torch.randint(0, 10000, ()).item())
        generator = Generator().manual_seed(seed)
        dataset = Subset(
            dataset,
            torch.randint(
                0,
                len(dataset),
                (max(len(dataset) // 10, 1),),
                generator=generator,
            ).tolist(),
        )

        path = tmpdir.joinpath("test_cifar10")
        pkl_dataset = pack_dataset(
            dataset,
            path,
            exists="overwrite",
        )

        assert len(dataset) == len(pkl_dataset)

        idx = 0
        image0, label0 = dataset[idx]
        image1, label1 = pkl_dataset[idx]

        assert len(dataset) == len(pkl_dataset)
        assert label0 == label1, f"{label0=}, {label1=}"
        assert np.equal(image0, image1).all()

    def test_cifar10_pack_per_batch(self) -> None:
        cls = self.__class__
        dataset = cls.dataset
        tmpdir = cls.tmpdir

        seed = int(torch.randint(0, 10000, ()).item())
        generator = Generator().manual_seed(seed)
        dataset = Subset(
            dataset,
            torch.randint(
                0,
                len(dataset),
                (max(len(dataset) // 10, 1),),
                generator=generator,
            ).tolist(),
        )

        path = tmpdir.joinpath("test_cifar10_batch")
        pkl_dataset = pack_dataset(
            dataset,
            path,
            exists="overwrite",
            content_mode="batch",
        )

        assert len(dataset) == len(pkl_dataset)

        idx = 0
        image0, label0 = dataset[idx]
        image1, label1 = pkl_dataset[idx]

        assert len(dataset) == len(pkl_dataset)
        assert label0 == label1, f"{label0=}, {label1=}"
        assert np.equal(image0, image1).all()

    def test_cifar10_pack_subdir(self) -> None:
        cls = self.__class__
        dataset = cls.dataset
        tmpdir = cls.tmpdir

        seed = int(torch.randint(0, 10000, ()).item())
        generator = Generator().manual_seed(seed)
        dataset = Subset(
            dataset,
            torch.randint(
                0,
                len(dataset),
                (max(len(dataset) // 50, 1),),
                generator=generator,
            ).tolist(),
        )

        path = tmpdir.joinpath("test_cifar10_subdir")
        pkl_dataset = pack_dataset(
            dataset,
            path,
            exists="overwrite",
            content_mode="item",
            subdir_size=100,
        )

        assert len(dataset) == len(pkl_dataset)

        idx = 0
        image0, label0 = dataset[idx]
        image1, label1 = pkl_dataset[idx]

        assert len(dataset) == len(pkl_dataset)
        assert label0 == label1, f"{label0=}, {label1=}"
        assert np.equal(image0, image1).all()

    def test_cifar10_pack_columns(self) -> None:
        cls = self.__class__
        dataset = cls.dataset
        tmpdir = cls.tmpdir

        path = tmpdir.joinpath("test_cifar10_columns")
        packed = pack_dataset_to_columns(
            dataset,
            path,
            exists="overwrite",
            save_fn="numpy",
            ds_kwds=dict(load_fn="numpy"),
        )

        assert len(dataset) == len(packed)
        for i in range(len(dataset)):
            sample_i, label_i = dataset[i]
            psample_i, plabel_i = packed[i]

            assert np.all(sample_i == psample_i)
            assert np.all(label_i == plabel_i)


class TestPackSpeechCommands(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tmpdir = Path(os.getenv("TORCHOUTIL_TMPDIR", tempfile.gettempdir())).joinpath(
            "torchoutil_tests"
        )
        tmpdir.mkdir(parents=True, exist_ok=True)
        cls.tmpdir = tmpdir

    def test_example_1(self) -> None:
        from torch import nn
        from torchaudio.datasets import SPEECHCOMMANDS
        from torchaudio.transforms import Spectrogram

        from torchoutil.utils.pack import pack_dataset

        speech_commands_root = self.tmpdir.joinpath("speech_commands")
        packed_root = self.tmpdir.joinpath("packed_speech_commands")

        os.makedirs(speech_commands_root, exist_ok=True)
        os.makedirs(packed_root, exist_ok=True)

        dataset = SPEECHCOMMANDS(
            speech_commands_root,
            download=True,
            subset="validation",
        )
        indices = torch.randperm(len(dataset))[: len(dataset) // 10].tolist()
        dataset = Subset(dataset, indices)

        class MyTransform(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.spectrogram_extractor = Spectrogram()

            def forward(self, item):
                waveform = item[0]
                spectrogram = self.spectrogram_extractor(waveform)
                return (spectrogram,) + item[1:]

        transform = MyTransform()
        pack_dataset(
            dataset,
            packed_root,
            transform,
            exists="overwrite",
            num_workers=0,
        )

        # Read from pickle
        from torchoutil.utils.pack import PackedDataset

        packed_root = self.tmpdir.joinpath("packed_speech_commands")
        pack = PackedDataset(packed_root)
        pack[0]  # first transformed item

        # Tests
        indices = torch.randperm(len(dataset))[:10].tolist()

        assert len(dataset) == len(pack)

        for idx in indices:
            item_1 = transform(dataset[idx])
            item_2 = pack[idx]

            assert isinstance(item_1, tuple)
            assert isinstance(item_2, tuple)
            assert len(item_1) == len(item_2)
            assert torch.equal(item_1[0], item_2[0])

            for i in range(1, len(item_1)):
                assert item_1[i] == item_2[i]

        num_steps = 5
        durations_1 = []
        durations_2 = []

        for i in range(num_steps):
            start_1 = time.perf_counter()
            for idx in indices:
                transform(dataset[idx])
            end_1 = time.perf_counter()

            start_2 = time.perf_counter()
            for idx in indices:
                pack[idx]
            end_2 = time.perf_counter()

            durations_1.append(end_1 - start_1)
            durations_2.append(end_2 - start_2)

        duration_1 = sum(durations_1)
        duration_2 = sum(durations_2)
        assert duration_1 > duration_2, f"Found {duration_1} <= {duration_2}"


if __name__ == "__main__":
    unittest.main()
