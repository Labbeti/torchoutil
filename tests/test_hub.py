#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from torch import Tensor

from torchoutil.hub.registry import RegistryHub


class TestRegistryHub(TestCase):
    def test_cnext_register(self) -> None:
        register = RegistryHub(
            infos={
                "cnext_bl_70": {
                    "architecture": "ConvNeXt",
                    "url": "https://zenodo.org/record/8020843/files/convnext_tiny_465mAP_BL_AC_70kit.pth?download=1",
                    "hash_value": "0688ae503f5893be0b6b71cb92f8b428",
                    "hash_type": "md5",
                    "fname": "convnext_tiny_465mAP_BL_AC_70kit.pth",
                    "state_dict_key": "model",
                },
            },
        )

        model_name = "cnext_bl_70"
        register.download_file(model_name, force=False)
        state_dict = register.load_state_dict(model_name, offline=True, device="cpu")

        assert isinstance(state_dict, dict)
        assert all(isinstance(k, str) for k in state_dict.keys())
        assert all(isinstance(v, Tensor) for v in state_dict.values())


if __name__ == "__main__":
    unittest.main()
