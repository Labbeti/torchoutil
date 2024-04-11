#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from torch import Tensor

from torchoutil.utils.ckpt import ModelCheckpointRegister


class TestRegister(TestCase):
    def test_cnext_register(self) -> None:
        register = ModelCheckpointRegister(
            infos={
                "cnext_bl_70": {
                    "architecture": "ConvNeXt",
                    "url": "https://zenodo.org/record/8020843/files/convnext_tiny_465mAP_BL_AC_70kit.pth?download=1",
                    "hash": "0688ae503f5893be0b6b71cb92f8b428",
                    "fname": "convnext_tiny_465mAP_BL_AC_70kit.pth",
                },
            },
            state_dict_key="model",
        )

        model_name = "cnext_bl_70"
        register.download_ckpt(model_name, force=True)
        state_dict = register.load_state_dict(model_name, offline=True, device="cpu")

        assert isinstance(state_dict, dict)
        assert all(isinstance(k, str) for k in state_dict.keys())
        assert all(isinstance(v, Tensor) for v in state_dict.values())


if __name__ == "__main__":
    unittest.main()
