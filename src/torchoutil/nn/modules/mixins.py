#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import logging
import re
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    OrderedDict,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from pyoutil.collections import dump_dict
from pyoutil.typing import is_dict_str, is_mapping_str
from torchoutil.nn.functional.checksum import checksum_module
from torchoutil.nn.functional.others import count_parameters

T = TypeVar("T", covariant=True)
InType = TypeVar("InType", covariant=False, contravariant=True)
OutType = TypeVar("OutType", covariant=True, contravariant=False)
OutType2 = TypeVar("OutType2", covariant=True, contravariant=False)
OutType3 = TypeVar("OutType3", covariant=False, contravariant=False)

DeviceDetectMode = Literal["proxy", "first_param", "none"]
DEVICE_DETECT_MODES = ("proxy", "first_param", "none")
_DEFAULT_DEVICE_DETECT_MODE = "first_param"


pylog = logging.getLogger(__name__)


__all__ = [
    "DeviceDetectMode",
    "EModule",
    "ESequential",
    "EModuleList",
    "EModuleDict",
]


class SupportsTypedForward(Protocol[InType, OutType]):
    def __call__(self, *args, **kwargs):
        ...

    def forward(self, x: InType) -> OutType:
        ...


TypedModuleLike = Union[
    SupportsTypedForward[InType, OutType],
    "TypedModule[InType, OutType]",
]


class ProxyDeviceModule(nn.Module):
    def __init__(
        self,
        *,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None:
        if device_detect_mode not in DEVICE_DETECT_MODES:
            raise ValueError(
                f"Invalid argument {device_detect_mode=}. (expected one of {DEVICE_DETECT_MODES})"
            )

        super().__init__()
        self.__device_detect_mode: DeviceDetectMode = device_detect_mode
        if device_detect_mode == "proxy":
            self.register_buffer("__proxy", torch.empty((0,)), persistent=False)

    @property
    def device_detect_mode(self) -> DeviceDetectMode:
        return self.__device_detect_mode

    def get_device(self) -> Optional[torch.device]:
        """Returns the Module device according to device_detect_mode property."""
        if self.__device_detect_mode == "proxy":
            return self._buffers["__proxy"].device  # type: ignore
        elif self.__device_detect_mode == "first_param":
            try:
                device0 = next(self._get_devices_iterator(params=True, buffers=True))
                return device0
            except StopIteration:
                return None
        else:
            return None

    def get_devices(
        self,
        *,
        params: bool = True,
        buffers: bool = True,
        recurse: bool = True,
        output_type: Callable[[Iterator[torch.device]], T] = list,
    ) -> T:
        return output_type(
            self._get_devices_iterator(
                params=params,
                buffers=buffers,
                recurse=recurse,
            )
        )

    def _get_devices_iterator(
        self,
        *,
        params: bool = True,
        buffers: bool = True,
        recurse: bool = True,
    ) -> Iterator[torch.device]:
        """Returns an iterator over all unique devices in module."""
        its: List[Iterator[Union[Tensor, Parameter]]] = []
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


class ConfigModule(nn.Module):
    _CONFIG_TYPES: ClassVar[Tuple[type, ...]] = (int, str, bool, float)
    _CONFIG_EXCLUDE: ClassVar[Tuple[str, ...]] = ("^_.*",) + tuple(
        f".*{k}$" for k in nn.Module().__dict__.keys()
    )

    def __init__(
        self,
        *,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
    ) -> None:
        attrs = {
            "config": {},
            "strict_load": strict_load,
            "config_to_extra_repr": config_to_extra_repr,
        }
        for name, value in attrs.items():
            object.__setattr__(self, f"_{ConfigModule.__name__}__{name}", value)

        super().__init__()
        self.__config: Dict[str, Any]
        self.__strict_load: bool
        self.__config_to_extra_repr: bool

    @property
    def config(self) -> Dict[str, Any]:
        return self.__config

    def __setattr__(self, name: str, value: Any) -> None:
        self.__update_config(name, value)
        return super().__setattr__(name, value)

    def __delattr__(self, name) -> None:
        self.__config.pop(name, None)
        return super().__delattr__(name)

    def extra_repr(self) -> str:
        if not self.__config_to_extra_repr:
            return super().extra_repr()
        else:
            return dump_dict(self.config)

    def add_module(self, name: str, module: Union[nn.Module, None]) -> None:
        self.__update_config(name, module)
        return super().add_module(name, module)

    def get_extra_state(self) -> Any:
        state = {"config": self.__config}
        return state

    def set_extra_state(self, state: Any) -> None:
        if not self.__strict_load:
            return None

        in_config = state["config"]
        if self.config == in_config:
            return None

        if is_dict_str(in_config) and is_dict_str(self.config):
            MISSING = "<missing>"
            union = set(in_config.keys()).union(self.config.keys())
            msgs = []
            for key in union:
                v1 = in_config.get(key, MISSING)
                v2 = in_config.get(key, MISSING)
                if v1 != v2:
                    msgs.append(f"{v1} != {v2}")
            msg = (
                "Invalid loaded config with current one. Invalid keys are:\n"
                + "\n\t".join(msgs)
            )
        else:
            msg = f"Invalid loaded config {in_config} with current one {self.config}."

        raise ValueError(msg)

    def __update_config(self, name: str, value: Any) -> None:
        prefix = f"{name}."
        if self._is_config_value(name, value):
            subconfig = {name: value}
            prefix = ""
        elif isinstance(value, ConfigModule):
            subconfig = value.config
        elif hasattr(value, "_hparams") and is_mapping_str(value._hparams):
            subconfig = dict(value._hparams.items())  # type: ignore
        elif hasattr(value, "__dict__"):
            subconfig = value.__dict__
        else:
            subconfig = {}

        subconfig = {f"{prefix}{k}": v for k, v in subconfig.items()}
        subconfig = {k: v for k, v in subconfig.items() if self._is_config_value(k, v)}
        self.__config.update(subconfig)

    @classmethod
    def _is_config_value(cls, name: str, value: Any) -> bool:
        return isinstance(value, cls._CONFIG_TYPES) and all(
            re.match(exclude_i, name) is None for exclude_i in cls._CONFIG_EXCLUDE
        )


class TypedModule(Generic[InType, OutType], nn.Module):
    """Typed version of torch.nn.Module. Can specify an input and output type."""

    def __call__(self, *args: InType, **kwargs: InType) -> OutType:
        return super().__call__(*args, **kwargs)

    def forward(self, *args: InType, **kwargs: InType) -> OutType:
        return super().forward(*args, **kwargs)

    @overload
    def chain(
        self,
        *others: TypedModuleLike[Any, OutType],
    ) -> "ESequential[InType, OutType]":
        ...

    @overload
    def chain(self, *others: nn.Module) -> "ESequential[InType, Any]":
        ...

    def chain(self, *others):
        return ESequential(self, *others)

    def __or__(
        self,
        other: TypedModuleLike[Any, OutType],
    ) -> "ESequential[InType, OutType]":
        return self.chain(other)

    def __ror__(
        self,
        other: TypedModuleLike[InType, Any],
    ) -> "ESequential[InType, OutType]":
        return ESequential(other, self)


class TypedSequential(
    Generic[InType, OutType],
    TypedModule[InType, OutType],
    nn.Sequential,
):
    def __init__(
        self,
        *args,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
    ) -> None:
        TypedModule.__init__(self)
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
    ) -> int:
        return checksum_module(self, only_trainable=only_trainable)


class ESequential(
    Generic[InType, OutType],
    EModule[InType, OutType],
    TypedSequential[InType, OutType],
):
    """Enriched torch.nn.Sequential with proxy device, forward typing and automatic configuration detection from attributes.

    Designed to work with `torchoutil.nn.EModule` instances.
    The default behaviour is the same than PyTorch Sequential class.
    """

    @overload
    def __init__(
        self,
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TypedModuleLike[InType, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
        unpack_dict: bool = False,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TypedModuleLike[InType, Any],
        arg1: TypedModuleLike[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TypedModuleLike[InType, Any],
        arg1: TypedModuleLike[Any, Any],
        arg2: TypedModuleLike[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TypedModuleLike[InType, Any],
        arg1: TypedModuleLike[Any, Any],
        arg2: TypedModuleLike[Any, Any],
        arg3: TypedModuleLike[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TypedModuleLike[InType, Any],
        arg1: TypedModuleLike[Any, Any],
        arg2: TypedModuleLike[Any, Any],
        arg3: TypedModuleLike[Any, Any],
        arg4: TypedModuleLike[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TypedModuleLike[InType, Any],
        arg1: TypedModuleLike[Any, Any],
        arg2: TypedModuleLike[Any, Any],
        arg3: TypedModuleLike[Any, Any],
        arg4: TypedModuleLike[Any, Any],
        arg5: TypedModuleLike[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg0: TypedModuleLike[InType, Any],
        arg1: TypedModuleLike[Any, Any],
        arg2: TypedModuleLike[Any, Any],
        arg3: TypedModuleLike[Any, Any],
        arg4: TypedModuleLike[Any, Any],
        arg5: TypedModuleLike[Any, Any],
        arg6: TypedModuleLike[Any, OutType],
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        arg: "OrderedDict[str, TypedModuleLike[InType, OutType]]",
        /,
        *,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
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
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        *args: nn.Module,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None:
        ...

    def __init__(
        self,
        *args,
        unpack_tuple: bool = False,
        unpack_dict: bool = False,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None:
        """
        Args:
            unpack_tuple: If True, the outputs of a module that returns a tuple at position i will be unpacked for positional arguments for the next module at position i+1. defaults to False.
            unpack_tuple: If True, the outputs of a module that returns a dict at position i will be unpacked for keywords arguments for the next module at position i+1. defaults to False.
            strict_load: If True, Module config will be compared during load_state_dict(...) method call and raises a ValueError. defaults to False.
            config_to_extra_repr: If True, add config to extra repr. defaults to False.
            device_detect_mode: Enable automatic detection of the module device. defaults to "first_param".
        """
        EModule.__init__(
            self,
            strict_load=strict_load,
            config_to_extra_repr=config_to_extra_repr,
            device_detect_mode=device_detect_mode,
        )
        TypedSequential.__init__(
            self,
            *args,
            unpack_tuple=unpack_tuple,
            unpack_dict=unpack_dict,
        )


class EModuleList(
    Generic[InType, OutType3],
    EModule[InType, List[OutType3]],
    nn.ModuleList,
):
    """Enriched torch.nn.ModuleList with proxy device, forward typing and automatic configuration detection from attributes.

    Designed to work with `torchoutil.nn.EModule` instances.
    The default behaviour is the same than PyTorch ModuleList class, except for the forward call which returns a list containing the output of each model called separately.
    """

    @overload
    def __init__(
        self,
        modules: Optional[Iterable[TypedModuleLike[InType, OutType3]]] = None,
        *,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        modules: Optional[Iterable[nn.Module]] = None,
        *,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None:
        ...

    def __init__(
        self,
        modules=None,
        *,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None:
        EModule.__init__(
            self,
            strict_load=strict_load,
            config_to_extra_repr=config_to_extra_repr,
            device_detect_mode=device_detect_mode,
        )
        nn.ModuleList.__init__(self, modules)

    def forward(self, *args: InType, **kwargs: InType) -> List[OutType3]:
        return [module(*args, **kwargs) for module in self]


class EModuleDict(
    Generic[InType, OutType3],
    EModule[InType, Dict[str, OutType3]],
    nn.ModuleDict,
):
    """Enriched torch.nn.ModuleDict with proxy device, forward typing and automatic configuration detection from attributes.

    Designed to work with `torchoutil.nn.EModule` instances.
    The default behaviour is the same than PyTorch ModuleDict class, except for the forward call which returns a dict containing the output of each model called separately.
    """

    @overload
    def __init__(
        self,
        modules: Optional[Mapping[str, TypedModuleLike[InType, OutType3]]] = None,
        *,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        modules: Optional[Mapping[str, nn.Module]] = None,
        *,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None:
        ...

    def __init__(
        self,
        modules=None,
        *,
        strict_load: bool = False,
        config_to_extra_repr: bool = False,
        device_detect_mode: DeviceDetectMode = _DEFAULT_DEVICE_DETECT_MODE,
    ) -> None:
        EModule.__init__(
            self,
            strict_load=strict_load,
            config_to_extra_repr=config_to_extra_repr,
            device_detect_mode=device_detect_mode,
        )
        nn.ModuleDict.__init__(self, modules)

    def forward(self, *args: InType, **kwargs: InType) -> Dict[str, OutType3]:
        return {name: module(*args, **kwargs) for name, module in self.items()}


def __test_typing_1() -> None:
    import torch
    from torch import Tensor

    class LayerA(TypedModule[Tensor, Tensor]):
        def forward(self, x: Tensor) -> Tensor:
            return x * x

    class LayerB(TypedModule[Tensor, int]):
        def forward(self, x: Tensor) -> int:
            return int(x.sum().item())

    class LayerC(TypedModule[int, Tensor]):
        def forward(self, x: int) -> Tensor:
            return torch.as_tensor(x)

    x = torch.rand(10)
    xa = LayerA()(x)
    xb = LayerB()(x)

    seq = ESequential(LayerA(), LayerA(), LayerB())
    xab = seq(x)

    seq = LayerA() | LayerA() | LayerB()
    xab = seq(x)

    seq = LayerC().chain(LayerA())
    xc = seq(2)

    assert isinstance(xa, Tensor)
    assert isinstance(xb, int)
    assert isinstance(xab, int)
    assert isinstance(xc, Tensor)

    class LayerD(nn.Module):
        def forward(self, x: Tensor) -> int:
            return int(x.item())

    class LayerE(nn.Module):
        def forward(self, x: bool) -> str:
            return str(x)

    seq = ESequential(LayerD(), LayerE())
    y = seq(torch.rand())

    assert isinstance(y, str)

    class LayerF(TypedModule[bool, str]):
        def forward(self, x):
            return str(x)

    seq = ESequential(LayerF())
    y = seq(True)
