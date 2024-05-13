#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import unittest
from unittest import TestCase

import numpy as np
import torch
from torch import Generator
from torch.utils.data.dataset import Subset
from torchvision.datasets import CIFAR10

from torchoutil.nn import IndexToOnehot, ToList, ToNumpy, TSequential
from torchoutil.utils.pickle_dataset import pack_to_pickle


class TestCIFAR10ToPickle(TestCase):
    def test_cifar10_pack_to_pickle(self) -> None:
        dataset = CIFAR10(
            "/tmp",
            train=False,
            transform=ToNumpy(),
            target_transform=TSequential(IndexToOnehot(10), ToList()),
            download=True,
        )

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

        path = "/tmp/test_cifar10"
        pkl_dataset = pack_to_pickle(dataset, path, overwrite=True)

        assert len(dataset) == len(pkl_dataset)

        idx = 0
        image0, label0 = dataset[idx]
        image1, label1 = pkl_dataset[idx]

        assert len(dataset) == len(pkl_dataset)
        assert label0 == label1, f"{label0=}, {label1=}"
        assert np.equal(image0, image1).all()

    def test_cifar10_pack_to_pickle_batch(self) -> None:
        dataset = CIFAR10(
            "/tmp",
            train=False,
            transform=ToNumpy(),
            target_transform=TSequential(IndexToOnehot(10), ToList()),
            download=True,
        )

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

        path = "/tmp/test_cifar10_batch"
        pkl_dataset = pack_to_pickle(
            dataset, path, overwrite=True, content_mode="batch"
        )

        assert len(dataset) == len(pkl_dataset)

        idx = 0
        image0, label0 = dataset[idx]
        image1, label1 = pkl_dataset[idx]

        assert len(dataset) == len(pkl_dataset)
        assert label0 == label1, f"{label0=}, {label1=}"
        assert np.equal(image0, image1).all()

    def test_cifar10_pack_to_pickle_subdir(self) -> None:
        dataset = CIFAR10(
            "/tmp",
            train=False,
            transform=ToNumpy(),
            target_transform=TSequential(IndexToOnehot(10), ToList()),
            download=True,
        )

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

        path = "/tmp/test_cifar10_subdir"
        pkl_dataset = pack_to_pickle(
            dataset,
            path,
            overwrite=True,
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


class TestSpeechCommandsToPickle(TestCase):
    def test_example_1(self) -> None:
        # Pack to pickle
        from torch import nn
        from torchaudio.datasets import SPEECHCOMMANDS
        from torchaudio.transforms import Spectrogram

        from torchoutil.utils.pickle_dataset import pack_to_pickle

        speech_commands_root = "/tmp/speech_commands"
        pickle_root = "/tmp/pickled_speech_commands"

        os.makedirs(speech_commands_root, exist_ok=True)
        os.makedirs(pickle_root, exist_ok=True)

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
        pack_to_pickle(dataset, pickle_root, transform, overwrite=True, num_workers=0)

        # Read from pickle
        from torchoutil.utils.pickle_dataset import PickleDataset

        pickle_root = "/tmp/pickled_speech_commands"
        pickle_dataset = PickleDataset(pickle_root)
        pickle_dataset[0]  # first transformed item

        # Tests
        indices = torch.randperm(len(dataset))[:10].tolist()

        assert len(dataset) == len(pickle_dataset)

        for idx in indices:
            item_1 = transform(dataset[idx])
            item_2 = pickle_dataset[idx]

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
                pickle_dataset[idx]
            end_2 = time.perf_counter()

            durations_1.append(end_1 - start_1)
            durations_2.append(end_2 - start_2)

        duration_1 = sum(durations_1)
        duration_2 = sum(durations_2)
        assert duration_1 > duration_2, f"Found {duration_1} <= {duration_2}"


if __name__ == "__main__":
    unittest.main()
