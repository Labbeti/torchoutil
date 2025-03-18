#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, ClassVar, Dict, Type


class Singleton(type):
    """Singleton metaclass.

    To use it, just inherit from metaclass:
    ```
    >>> class MyClass(metaclass=Singleton):
    >>>     pass
    >>> a1 = MyClass()
    >>> a2 = MyClass()
    >>> # a1 and a2 are exactly the same instance, i.e. id(a1) == id(a2)
    ```
    """

    _instances: ClassVar[Dict[Type, Any]] = {}

    def __call__(cls, *args, **kwargs) -> Any:
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        else:
            instance = cls._instances[cls]
        return instance
