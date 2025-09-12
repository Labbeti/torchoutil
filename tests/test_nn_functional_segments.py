#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

import torch

from torchoutil.nn.functional.segments import (
    activity_to_segments,
    segments_list_to_activity,
    segments_to_segments_list,
)


class TestSegments(TestCase):
    def test_example_1(self) -> None:
        x = torch.as_tensor([0, 1, 1, 0, 0, 1, 1, 1, 1, 0]).bool()
        segments = activity_to_segments(x)
        starts, ends = segments

        assert torch.equal(starts, torch.as_tensor([1, 5]))
        assert torch.equal(ends, torch.as_tensor([3, 9]))

        segments_lst = segments_to_segments_list(segments)
        assert segments_lst == [(1, 3), (5, 9)]

        activity = segments_list_to_activity(segments_lst, x.shape[-1])
        assert torch.equal(x, activity)

    def test_example_2(self) -> None:
        x = torch.as_tensor([[1, 1, 1, 0], [1, 0, 0, 1]]).bool()
        segments = activity_to_segments(x)
        indices, starts, ends = segments

        assert torch.equal(indices, torch.as_tensor([0, 1, 1]))
        assert torch.equal(starts, torch.as_tensor([0, 0, 3]))
        assert torch.equal(ends, torch.as_tensor([3, 1, 4]))

        segments_lst = segments_to_segments_list(segments)
        assert segments_lst == [[(0, 3)], [(0, 1), (3, 4)]]

        activity = segments_list_to_activity(segments_lst, x.shape[-1])
        assert torch.equal(x, activity)

    def test_example_3(self) -> None:
        activity = segments_list_to_activity([], 5)
        expected = torch.full((5,), False)
        assert torch.equal(activity, expected)

    def test_example_4(self) -> None:
        segments = torch.as_tensor([[0, 4, 8], [2, 6, 10]]).T

        maxsize = None
        expected = torch.as_tensor([1, 1, 0, 0, 1, 1, 0, 0, 1, 1]).bool()
        assert torch.equal(segments_list_to_activity(segments, maxsize), expected)

    def test_example_5(self) -> None:
        segments = torch.as_tensor([[0, 4, 8], [2, 6, 10]]).T

        maxsize = 13
        expected = torch.as_tensor([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0]).bool()
        assert torch.equal(segments_list_to_activity(segments, maxsize), expected)

    def test_example_6(self) -> None:
        segments = torch.as_tensor([[[600, 1322]], [[6, 1248]], [[43, 497]]])

        maxsize = 1460
        expected_shape = (len(segments), maxsize)
        assert segments_list_to_activity(segments, maxsize).shape == expected_shape


if __name__ == "__main__":
    unittest.main()
