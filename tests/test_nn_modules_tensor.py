#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import unittest
from typing import Any, List, Tuple, Type
from unittest import TestCase

import torch
from torch import nn

from torchoutil.nn.functional import deep_equal
from torchoutil.nn.modules import tensor as tensor_module
from torchoutil.pyoutil.inspect import get_fullname


def module_name_to_fn_name(x: str) -> str:
    if x.isupper() or x.islower():
        return x.lower()

    x = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", x)
    x = x.lower()
    return x


class TestModuleCompat(TestCase):
    def test_all_results(self) -> None:
        base_modules = [
            torch,
            torch.Tensor,
            torch.nn.functional,  # type: ignore
            torch.fft,
        ]
        all_fn_names = {
            base_module: dict.fromkeys(
                name
                for name in dir(base_module)
                if callable(getattr(base_module, name))
            )
            for base_module in base_modules
        }

        tested_modules = set()
        missed_modules = set()

        num_modules = 0
        targets: List[Tuple[Type[nn.Module], Any, str]] = []

        for name in dir(tensor_module):
            module_cls = getattr(tensor_module, name)
            if (
                not isinstance(module_cls, type)
                or not issubclass(module_cls, nn.Module)
                or module_cls is nn.Module
            ):
                continue

            num_modules += 1
            fn_name = module_name_to_fn_name(name)

            module_targets: List[Tuple[Type[nn.Module], Any, str]] = []
            for base_module, fn_names in all_fn_names.items():
                if fn_name not in fn_names:
                    continue
                target = module_cls, base_module, fn_name
                module_targets.append(target)

            if len(module_targets) == 0:
                missed_modules.add(name)
            else:
                tested_modules.add(name)

            targets += module_targets

        x = torch.rand((10,))
        for module_cls, base_module, fn_name in targets:
            try:
                module = module_cls()
                result = module(x)
            except Exception as err:
                result = err

            fn = getattr(base_module, fn_name)
            try:
                expected = fn(x)
            except Exception as err:
                expected = err

            if isinstance(result, Exception) and isinstance(expected, Exception):
                pass
            else:
                msg = f"{result=} ; {expected=} from {get_fullname(module_cls)} and {get_fullname(fn)}"
                assert deep_equal(result, expected), msg

        print(
            f"Total base coverage hit: {len(tested_modules)}/{len(tested_modules) + len(missed_modules)}"
        )
        print(f"Hit_: {tested_modules}")
        print(f"Miss: {missed_modules}")


if __name__ == "__main__":
    unittest.main()
