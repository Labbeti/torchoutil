#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import ClassVar, Dict, Type


class Singleton(type):
    """Singleton metaclass.

    To use it, just inherit from metaclass:
    ```
    >>> class MyClass(metaclass=Singleton):
    >>>     pass
    >>> a1 = MyClass()
    >>> a2 = MyClass()
    >>> # a1 and a2 are the same object, i.e. id(a1) == id(a2)
    ```
    """

    _instances: ClassVar[Dict[Type, object]] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        instance = cls._instances[cls]
        return instance
