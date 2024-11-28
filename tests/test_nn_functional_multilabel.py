#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchoutil.core.get import get_device
from torchoutil.nn.functional.multilabel import (
    indices_to_multihot,
    indices_to_multinames,
    multihot_to_indices,
    multihot_to_multinames,
    multinames_to_indices,
    multinames_to_multihot,
    probs_to_indices,
    probs_to_multihot,
    probs_to_multinames,
)


class TestMultilabel(TestCase):
    def test_indices_to_multihot_1(self) -> None:
        indices = [[1, 2], [0], [], [3]]
        num_classes = 4
        expected_multihot = torch.as_tensor(
            [[0, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.bool
        )

        result = indices_to_multihot(indices, num_classes=num_classes)

        assert torch.equal(result, expected_multihot)

    def test_probs_to_indices_1(self) -> None:
        probs = torch.as_tensor([[1.0, 0.4, 0.1, 0.9]])
        expected_indices = [[0, 3]]

        result = probs_to_indices(probs, threshold=0.5)

        assert result == expected_indices

    def test_probs_to_multihot_1(self) -> None:
        probs = torch.as_tensor([[1.0, 0.4, 0.1, 0.9]])
        expected_multihot = torch.as_tensor([[True, False, False, True]])

        result = probs_to_multihot(probs, threshold=0.5)

        assert torch.equal(result, expected_multihot)

    def test_probs_to_multinames_1(self) -> None:
        probs = torch.as_tensor([[1.0, 0.4, 0.1, 0.9]])
        expected_multinames = [["0", "3"]]

        num_classes = probs.shape[-1]
        idx_to_name = dict(zip(range(num_classes), map(str, range(num_classes))))
        result = probs_to_multinames(probs, threshold=0.5, idx_to_name=idx_to_name)

        assert result == expected_multinames

    def test_convert_multihot(self) -> None:
        num_samples = int(torch.randint(1, 20, ()).item())
        num_classes = int(torch.randint(1, 20, ()).item())
        threshold = torch.rand(())
        idx_to_name = dict(zip(range(num_classes), map(str, range(num_classes))))

        probs_1 = torch.rand(num_samples, num_classes)
        multihot_1 = probs_1.ge(threshold)

        indices_1 = multihot_to_indices(multihot_1)
        multinames_1 = indices_to_multinames(indices_1, idx_to_name)
        multihot_2 = multinames_to_multihot(multinames_1, idx_to_name)

        multinames_2 = multihot_to_multinames(multihot_2, idx_to_name)
        indices_2 = multinames_to_indices(multinames_2, idx_to_name)

        assert torch.equal(multihot_1, multihot_2), f"{multihot_1=} ; {multihot_2=}"
        assert multinames_1 == multinames_2
        assert indices_1 == indices_2

    def test_ints_to_multihots(self) -> None:
        device = get_device()
        ints = torch.as_tensor([[0, 1, 1]], device=device)
        num_classes = 5
        multihots = indices_to_multihot(ints, num_classes, dtype=torch.int)
        expected = torch.as_tensor([[1, 1, 0, 0, 0]], device=device)

        assert multihots.shape == expected.shape
        assert multihots.eq(expected).all(), f"{multihots=}"

    def test_convert_and_reconvert(self) -> None:
        device = get_device()
        multihots = torch.as_tensor(
            [
                [1, 1, 1],
                [0, 0, 0],
                [1, 1, 0],
                [0, 0, 1],
            ],
            device=device,
        )
        expected_ints = [[0, 1, 2], [], [0, 1], [2]]

        num_classes = multihots.shape[1]
        ints = multihot_to_indices(multihots)
        new_multihots = indices_to_multihot(
            ints, num_classes, dtype=torch.int, device=device
        )

        assert ints == expected_ints
        assert multihots.shape == new_multihots.shape
        assert multihots.eq(new_multihots).all()

    def test_empty_case_1(self) -> None:
        num_samples = 0
        num_classes = 5

        multihot = torch.rand(num_samples, num_classes).ge(0.5)
        with self.assertRaises(ValueError):
            multihot_to_indices(multihot)

        indices = torch.empty(0, 5)
        with self.assertRaises(ValueError):
            indices_to_multihot(indices, num_classes)

    def test_empty_case_2(self) -> None:
        num_samples = 5
        num_classes = 0
        expected = [[], [], [], [], []]

        multihot = torch.rand(num_samples, num_classes).ge(0.5)

        result_1 = multihot_to_indices(multihot)
        assert result_1 == expected

        result_2 = indices_to_multihot(expected, num_classes)
        assert torch.equal(result_2, multihot)

    def test_empty_case_3(self) -> None:
        num_steps = 3
        num_samples = 0
        num_classes = 5

        multihot = torch.rand(num_steps, num_samples, num_classes).ge(0.5)

        with self.assertRaises(ValueError):
            multihot_to_indices(multihot)

        indices = [torch.empty(0, 5) for _ in range(num_steps)]
        with self.assertRaises(ValueError):
            indices_to_multihot(indices, num_classes)

    def test_probs_to_multihot_dim(self) -> None:
        probs = torch.rand(16, 10, 5)
        threshold = torch.full((10,), 0.3)

        multihot = probs_to_multihot(probs, threshold, dim=1)
        assert multihot.shape == probs.shape

    def test_probs_to_indices_dim(self) -> None:
        # 2x5
        probs = torch.as_tensor(
            [
                [0.6, 0.9, 0.1, 0.0, 1.0],
                [0.0, 0.1, 0.9, 0.0, 1.0],
            ]
        )
        indices_0 = probs_to_indices(probs, 0.5, dim=0)
        assert indices_0 == [[0], [0], [1], [], [0, 1]]

        indices_1 = probs_to_indices(probs, 0.5, dim=1)
        assert indices_1 == [[0, 1, 4], [2, 4]]

        # duplicate probs for 3D test with 2x2x5
        probs = torch.stack([probs, probs])

        indices_0 = probs_to_indices(probs, 0.5, dim=0)
        assert indices_0 == [
            [[0, 1], []],
            [[0, 1], []],
            [[], [0, 1]],
            [[], []],
            [[0, 1], [0, 1]],
        ]

        indices_1 = probs_to_indices(probs, 0.5, dim=1)
        assert indices_1 == [[[0], [0], [1], [], [0, 1]], [[0], [0], [1], [], [0, 1]]]

        indices_2 = probs_to_indices(probs, 0.5, dim=2)
        assert indices_2 == [[[0, 1, 4], [2, 4]], [[0, 1, 4], [2, 4]]]


if __name__ == "__main__":
    unittest.main()
