#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchoutil.nn.functional.segments import extract_segments, segments_to_list


class TestSegments(TestCase):
    def test_example_1(self) -> None:
        x = torch.as_tensor([0, 1, 1, 0, 0, 1, 1, 1, 1, 0]).bool()
        segments = extract_segments(x)
        starts, ends = segments

        assert torch.equal(starts, torch.as_tensor([1, 5]))
        assert torch.equal(ends, torch.as_tensor([3, 9]))

        segments_lst = segments_to_list(segments)
        assert segments_lst == [(1, 3), (5, 9)]

    def test_example_2(self) -> None:
        x = torch.as_tensor([[1, 1, 1, 0], [1, 0, 0, 1]]).bool()
        segments = extract_segments(x)
        indices, starts, ends = segments

        assert torch.equal(indices, torch.as_tensor([0, 1, 1]))
        assert torch.equal(starts, torch.as_tensor([0, 0, 3]))
        assert torch.equal(ends, torch.as_tensor([3, 1, 4]))

        segments_lst = segments_to_list(segments)
        assert segments_lst == [[(0, 3)], [(0, 1), (3, 4)]]


if __name__ == "__main__":
    unittest.main()
