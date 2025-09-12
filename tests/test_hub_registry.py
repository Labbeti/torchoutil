#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase

from torch import Tensor

from torchoutil.hub.registry import RegistryHub


class TestRegistryHub(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tmpdir = Path(os.getenv("TORCHOUTIL_TMPDIR", tempfile.gettempdir())).joinpath(
            "torchoutil_tests"
        )
        tmpdir.mkdir(parents=True, exist_ok=True)
        cls.tmpdir = tmpdir

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
            register_root=self.tmpdir,
        )

        model_name = "cnext_bl_70"
        register.download_file(model_name, force=False)
        state_dict = register.load_state_dict(
            model_name,
            offline=True,
            load_kwds=dict(map_location="cpu"),
        )

        assert isinstance(state_dict, dict)
        assert all(isinstance(k, str) for k in state_dict.keys())
        assert all(isinstance(v, Tensor) for v in state_dict.values())


if __name__ == "__main__":
    unittest.main()
