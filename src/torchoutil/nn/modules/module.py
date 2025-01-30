#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Generic

from torchoutil.nn.functional.checksum import checksum_module
from torchoutil.nn.functional.others import count_parameters

from torchoutil.nn.modules._mixins import (
    InType,
    OutType,
    TypedModule,
    ConfigModule,
    ProxyDeviceModule,
    DeviceDetectMode,
    _DEFAULT_DEVICE_DETECT_MODE,
)


class EModule(
    Generic[InType, OutType],
    ConfigModule,
    TypedModule[InType, OutType],
    ProxyDeviceModule,
):
    """Enriched torch.nn.Module with proxy device, forward typing and automatic configuration detection from attributes.

    The default behaviour is the same than PyTorch Module class.
    """

    def __init__(
        self,
        *,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None:
        """
        Args:
            strict_load: If True, Module config will be compared during load_state_dict(...) method call and raises a ValueError. defaults to False.
            config_to_extra_repr: If True, add config to extra repr. defaults to False.
            device_detect_mode: Enable automatic detection of the module device. defaults to "first_param".
        """
        # ConfigModule must be first
        ConfigModule.__init__(
            self,
            strict_load=strict_load,
            config_to_extra_repr=config_to_extra_repr,
        )
        TypedModule.__init__(self)
        ProxyDeviceModule.__init__(
            self,
            device_detect_mode=device_detect_mode,
        )

    def count_parameters(
        self,
        *,
        recurse: bool = True,
        only_trainable: bool = False,
        buffers: bool = False,
    ) -> int:
        """Returns the number of parameters in this module."""
        return count_parameters(
            self,
            recurse=recurse,
            only_trainable=only_trainable,
            buffers=buffers,
        )

    def checksum(
        self,
        *,
        only_trainable: bool = False,
        with_names: bool = False,
        buffers: bool = False,
        training: bool = False,
    ) -> int:
        return checksum_module(
            self,
            only_trainable=only_trainable,
            with_names=with_names,
            buffers=buffers,
            training=training,
        )
