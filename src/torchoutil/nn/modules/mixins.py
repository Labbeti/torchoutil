#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import re
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Literal,
    Optional,
    OrderedDict,
    Type,
    TypeVar,
    Union,
    get_args,
    overload,
)

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from torchoutil.nn.functional.others import count_parameters
from torchoutil.utils.type_checks import is_dict_str, is_mapping_str

InType = TypeVar("InType", covariant=False, contravariant=False)
OutType = TypeVar("OutType", covariant=True, contravariant=False)
OutType2 = TypeVar("OutType2", covariant=True, contravariant=False)


DEVICE_DETECT_MODES = ("proxy", "first_param", "none")
DeviceDetectMode = Literal["proxy", "first_param", "none"]


class ProxyDeviceModuleMixin(nn.Module):
    def __init__(
        self,
        *,
        device_detect_mode: DeviceDetectMode = "proxy",
    ) -> None:
        if device_detect_mode not in DEVICE_DETECT_MODES:
            raise ValueError(
                f"Invalid argument {device_detect_mode=}. (expected one of {DEVICE_DETECT_MODES})"
            )

        super().__init__()
        self.__device_detect_mode = device_detect_mode
        self.register_buffer("__proxy", torch.empty((0,)), persistent=False)
        self.__proxy: Tensor

    @property
    def device_detect_mode(self) -> str:
        return self.__device_detect_mode

    @property
    def device(self) -> Optional[torch.device]:
        """Returns the Module device according to device_detect_mode property."""
        if self.__device_detect_mode == "proxy":
            return self.__proxy.device
        elif self.__device_detect_mode == "first_param":
            try:
                device0 = next(iter(self.devices(params=True, buffers=False)))
                return device0
            except StopIteration:
                return None
        else:
            return None

    def devices(
        self,
        *,
        params: bool = True,
        buffers: bool = True,
        recurse: bool = True,
    ) -> Iterator[torch.device]:
        """Returns an iterator over all unique devices in module."""
        its: List[Union[Tensor, Parameter]] = []
        if params:
            its.append(self.parameters(recurse=recurse))
        if buffers:
            its.append(self.buffers(recurse=recurse))

        devices = {}
        for it in its:
            for param_or_buffer in it:
                device = param_or_buffer.device
                if device not in devices:
                    yield device
                devices[param_or_buffer.device] = None


class ConfigModuleMixin(nn.Module):
    _CONFIG_EXCLUDE = tuple(f".*{k}" for k in nn.Module().__dict__.keys()) + ("_.*",)
    _CONFIG_TYPES = (int, str, bool, float)

    def __init__(self) -> None:
        object.__setattr__(self, f"_{ConfigModuleMixin.__name__}__config", {})
        super().__init__()
        self.__config: Dict[str, Any]

    def add_module(self, name: str, module: nn.Module | None) -> None:
        self.__update_config(name, module)
        return super().add_module(name, module)

    def __setattr__(self, name: str, value: Any) -> None:
        self.__update_config(name, value)
        return super().__setattr__(name, value)

    def __delattr__(self, name) -> None:
        self.__config.pop(name, None)
        return super().__delattr__(name)

    def __update_config(self, name: str, value: Any) -> None:
        prefix = f"{name}."
        if self._is_config_value(name, value):
            subconfig = {name: value}
            prefix = ""
        elif isinstance(value, ConfigModuleMixin):
            subconfig = value.config
        elif hasattr(value, "_hparams") and is_mapping_str(value):
            subconfig = dict(value._hparams.items())  # type: ignore
        elif hasattr(value, "__dict__"):
            subconfig = value.__dict__
        else:
            subconfig = {}

        subconfig = {f"{prefix}{k}": v for k, v in subconfig.items()}
        subconfig = {k: v for k, v in subconfig.items() if self._is_config_value(k, v)}
        object.__setattr__(
            self, f"_{ConfigModuleMixin.__name__}__config", self.__config | subconfig
        )

    @classmethod
    def _is_config_value(cls, name: str, value: Any) -> bool:
        return isinstance(value, cls._CONFIG_TYPES) and all(
            re.match(exclude_i, name) is None for exclude_i in cls._CONFIG_EXCLUDE
        )

    @property
    def config(self) -> Dict[str, Any]:
        return self.__config


class TypedModuleMixin(Generic[InType, OutType], nn.Module):
    """Typed version of torch.nn.Module. Can specify an input and output type."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def in_type(self) -> Type[InType]:
        return get_args(self.__orig_bases__[0])[0]  # type: ignore

    @property
    def out_type(self) -> Type[OutType]:
        return get_args(self.__orig_bases__[0])[1]  # type: ignore

    def __call__(self, *args: InType, **kwargs: InType) -> OutType:
        return super().__call__(*args, **kwargs)

    def forward(self, *args: InType, **kwargs: InType) -> OutType:
        return super().forward(*args, **kwargs)

    @overload
    def compose(
        self,
        other: "TypedModuleMixin[Any, OutType2]",
    ) -> "TypedSequentialMixin[InType, OutType2]":
        ...

    @overload
    def compose(self, other: nn.Module) -> "TypedSequentialMixin[InType, Any]":
        ...

    def compose(self, other) -> "TypedSequentialMixin[InType, Any]":
        return TypedSequentialMixin(self, other)


class TypedSequentialMixin(
    Generic[InType, OutType],
    TypedModuleMixin[InType, OutType],
    nn.Sequential,
):
    """Typed version of torch.nn.Sequential, designed to work with torchoutil.nn.TModules."""

    @overload
    def __init__(
        self,
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TypedModuleMixin[InType, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TypedModuleMixin[InType, Any],
        arg1: TypedModuleMixin[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TypedModuleMixin[InType, Any],
        arg1: TypedModuleMixin[Any, Any],
        arg2: TypedModuleMixin[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TypedModuleMixin[InType, Any],
        arg1: TypedModuleMixin[Any, Any],
        arg2: TypedModuleMixin[Any, Any],
        arg3: TypedModuleMixin[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TypedModuleMixin[InType, Any],
        arg1: TypedModuleMixin[Any, Any],
        arg2: TypedModuleMixin[Any, Any],
        arg3: TypedModuleMixin[Any, Any],
        arg4: TypedModuleMixin[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TypedModuleMixin[InType, Any],
        arg1: TypedModuleMixin[Any, Any],
        arg2: TypedModuleMixin[Any, Any],
        arg3: TypedModuleMixin[Any, Any],
        arg4: TypedModuleMixin[Any, Any],
        arg5: TypedModuleMixin[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TypedModuleMixin[InType, Any],
        arg1: TypedModuleMixin[Any, Any],
        arg2: TypedModuleMixin[Any, Any],
        arg3: TypedModuleMixin[Any, Any],
        arg4: TypedModuleMixin[Any, Any],
        arg5: TypedModuleMixin[Any, Any],
        arg6: TypedModuleMixin[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg: "OrderedDict[str, TypedModuleMixin[InType, OutType]]",
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg: "OrderedDict[str, nn.Module]",
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        *args: nn.Module,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        ...

    def __init__(
        self,
        *args,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        TypedModuleMixin.__init__(self)
        nn.Sequential.__init__(self, *args)

        self.__unpack_tuple = unpack_tuple
        self.__unpack_dict = unpack_dict

    @property
    def unpack_tuple(self) -> bool:
        return self.__unpack_tuple

    @property
    def unpack_dict(self) -> bool:
        return self.__unpack_dict

    def __call__(self, x: InType) -> OutType:
        return nn.Sequential.__call__(self, x)

    def forward(self, x: InType) -> OutType:
        for module in self:
            if self.__unpack_tuple and isinstance(x, tuple):
                x = module(*x)
            elif self.__unpack_dict and is_dict_str(x):
                x = module(**x)
            else:
                x = module(x)
        return x  # type: ignore

    def tolist(self) -> List[nn.Module]:
        return list(self._modules.values())

    def todict(self) -> Dict[str, nn.Module]:
        return copy.copy(self._modules)


class EModule(
    Generic[InType, OutType],
    ProxyDeviceModuleMixin,
    ConfigModuleMixin,
    TypedModuleMixin[InType, OutType],
):
    """Enriched torch.nn.Module with proxy device, forward typing and automatic configuration detection from attributes.

    The default behaviour is the same than PyTorch Module class.
    """

    def __init__(
        self,
        *,
        device_detect_mode: DeviceDetectMode = "proxy",
    ) -> None:
        ProxyDeviceModuleMixin.__init__(self, device_detect_mode=device_detect_mode)
        ConfigModuleMixin.__init__(self)
        TypedModuleMixin.__init__(self)

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


class ESequential(
    Generic[InType, OutType],
    EModule[InType, OutType],
    TypedSequentialMixin[InType, OutType],
):
    """Enriched torch.nn.Sequential with proxy device, forward typing and automatic configuration detection from attributes.

    The default behaviour is the same than PyTorch Sequential class.
    """

    @overload
    def __init__(
        self,
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        device_detect_mode: DeviceDetectMode = "proxy",
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TypedModuleMixin[InType, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        device_detect_mode: DeviceDetectMode = "proxy",
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TypedModuleMixin[InType, Any],
        arg1: TypedModuleMixin[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        device_detect_mode: DeviceDetectMode = "proxy",
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TypedModuleMixin[InType, Any],
        arg1: TypedModuleMixin[Any, Any],
        arg2: TypedModuleMixin[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        device_detect_mode: DeviceDetectMode = "proxy",
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TypedModuleMixin[InType, Any],
        arg1: TypedModuleMixin[Any, Any],
        arg2: TypedModuleMixin[Any, Any],
        arg3: TypedModuleMixin[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        device_detect_mode: DeviceDetectMode = "proxy",
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TypedModuleMixin[InType, Any],
        arg1: TypedModuleMixin[Any, Any],
        arg2: TypedModuleMixin[Any, Any],
        arg3: TypedModuleMixin[Any, Any],
        arg4: TypedModuleMixin[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        device_detect_mode: DeviceDetectMode = "proxy",
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TypedModuleMixin[InType, Any],
        arg1: TypedModuleMixin[Any, Any],
        arg2: TypedModuleMixin[Any, Any],
        arg3: TypedModuleMixin[Any, Any],
        arg4: TypedModuleMixin[Any, Any],
        arg5: TypedModuleMixin[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        device_detect_mode: DeviceDetectMode = "proxy",
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TypedModuleMixin[InType, Any],
        arg1: TypedModuleMixin[Any, Any],
        arg2: TypedModuleMixin[Any, Any],
        arg3: TypedModuleMixin[Any, Any],
        arg4: TypedModuleMixin[Any, Any],
        arg5: TypedModuleMixin[Any, Any],
        arg6: TypedModuleMixin[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        device_detect_mode: DeviceDetectMode = "proxy",
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg: "OrderedDict[str, TypedModuleMixin[InType, OutType]]",
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        device_detect_mode: DeviceDetectMode = "proxy",
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg: "OrderedDict[str, nn.Module]",
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        device_detect_mode: DeviceDetectMode = "proxy",
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        *args: nn.Module,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        device_detect_mode: DeviceDetectMode = "proxy",
    ) -> None:
        ...

    def __init__(
        self,
        *args,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        device_detect_mode: DeviceDetectMode = "proxy",
    ) -> None:
        EModule.__init__(self, device_detect_mode=device_detect_mode)
        TypedSequentialMixin.__init__(
            self,
            *args,
            unpack_tuple=unpack_tuple,
            unpack_dict=unpack_dict,
        )


def __test_typing_1() -> None:
    import torch
    from torch import Tensor

    class LayerA(TypedModuleMixin[Tensor, Tensor]):
        def forward(self, x: Tensor) -> Tensor:
            return x * x

    class LayerB(TypedModuleMixin[Tensor, int]):
        def forward(self, x: Tensor) -> int:
            return int(x.sum().item())

    class LayerC(TypedModuleMixin[int, Tensor]):
        def forward(self, x: int) -> Tensor:
            return torch.as_tensor(x)

    x = torch.rand(10)
    xa = LayerA()(x)
    xb = LayerB()(x)

    seq = ESequential(LayerA(), LayerA(), LayerB())
    xab = seq(x)

    seq = LayerA().compose(LayerA()).compose(LayerB())
    xab = seq(x)

    seq = LayerC().compose(LayerA())
    xc = seq(2)

    assert isinstance(xa, Tensor)
    assert isinstance(xb, int)
    assert isinstance(xab, int)
    assert isinstance(xc, Tensor)

    class LayerD(TypedModuleMixin[Tensor, int]):
        def forward(self, x: Tensor) -> int:
            return int(x.item())

    class LayerE(TypedModuleMixin[bool, str]):
        def forward(self, x: bool) -> str:
            return str(x)

    seq = ESequential(LayerD(), LayerE())
    y = seq(torch.rand())

    assert isinstance(y, str)
