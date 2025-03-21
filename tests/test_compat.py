#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import unittest
from typing import Any, Callable, List, Tuple
from unittest import TestCase

import torch
from torch import Tensor

import torchoutil
from torchoutil.pyoutil.inspect import get_fullname
from torchoutil.pyoutil.typing import SizedIterable, isinstance_guard


class TestFunctionsCompat(TestCase):
    def test_all_results(self) -> None:
        src_modules = [
            torch,
            torch.Tensor,
            torch.nn.functional,  # type: ignore
            torch.fft,
        ]
        all_base_fn_names = {
            src_module: dict.fromkeys(
                name
                for name in dir(src_module)
                if callable(getattr(src_module, name))
                and "_VariableFunctionsClass"
                not in get_fullname(getattr(src_module, name))
            )
            for src_module in src_modules
        }

        tested_fns = set()
        missed_fns = set()

        num_modules = 0
        targets: List[Tuple[Callable, Any, str]] = []

        for fn_name in dir(torchoutil):
            tgt_fn = getattr(torchoutil, fn_name)
            if not inspect.isfunction(tgt_fn) and not inspect.ismethod(tgt_fn):
                continue

            num_modules += 1

            fn_targets: List[Tuple[Callable, Any, str]] = []
            for src_module, fn_names in all_base_fn_names.items():
                if fn_name not in fn_names:
                    continue
                src_fn = getattr(src_module, fn_name)
                target = tgt_fn, src_fn, fn_name
                fn_targets.append(target)

            if len(fn_targets) == 0:
                missed_fns.add(fn_name)
            else:
                tested_fns.add(fn_name)

            targets += fn_targets

        args_lst = [torch.rand((10,))]
        for tgt_fn, src_fn, fn_name in targets:
            for args in args_lst:
                try:
                    result = tgt_fn(*args)
                except Exception as err:
                    result = err

                try:
                    expected = src_fn(*args)
                except Exception as err:
                    expected = err

                if isinstance(result, Exception) and isinstance(expected, Exception):
                    pass
                elif isinstance(result, Tensor) and isinstance(expected, Tensor):
                    assert torch.equal(result, expected)
                elif isinstance_guard(
                    result, SizedIterable[Tensor]
                ) and isinstance_guard(expected, SizedIterable[Tensor]):
                    assert len(result) == len(expected)
                    for result_i, expected_i in zip(result, expected):
                        assert torch.equal(result_i, expected_i)
                else:
                    msg = f"Invalid results for '{fn_name}': {type(result)=} != {type(expected)=} from {get_fullname(tgt_fn)} and {get_fullname(src_fn)}, with {args}."
                    assert result == expected, msg

        print(
            f"Total base coverage hit: {len(tested_fns)}/{len(tested_fns)+len(missed_fns)}"
        )
        print(f"Hit_: {tested_fns}")
        print(f"Miss: {missed_fns}")


if __name__ == "__main__":
    unittest.main()
