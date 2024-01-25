#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, ClassVar, Dict, Generic, TypeVar

from torch import nn


OutType = TypeVar("OutType")


class ExtendedModule(nn.Module, Generic[OutType]):
    JOIN_STR: ClassVar[str] = ", "
    REPR_PARAM_FORMAT: ClassVar[str] = "{name}={value}"

    def __call__(self, *args, **kwargs) -> OutType:
        return super().__call__(*args, **kwargs)

    def forward(self, *args, **kwargs) -> OutType:
        return super().forward(*args, **kwargs)

    def get_repr_params(self) -> Dict[str, Any]:
        return {}

    def extra_repr(self) -> str:
        extra = ExtendedModule.JOIN_STR.join(
            ExtendedModule.REPR_PARAM_FORMAT.format(name=name, value=value)
            for name, value in self.get_repr_params()
        )
        return extra
