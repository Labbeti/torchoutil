#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Generic, TypeVar

from torch import nn


OutType = TypeVar("OutType")


class ExtendedModule(nn.Module, Generic[OutType]):
    def __call__(self, *args, **kwargs) -> OutType:
        return super().__call__(*args, **kwargs)

    def forward(self, *args, **kwargs) -> OutType:
        return super().forward(*args, **kwargs)
