#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum
from typing import Type, TypeVar

TStrEnum = TypeVar("TStrEnum", bound="StrEnum")


class StrEnum(str, Enum):
    @classmethod
    def from_str(
        cls: Type[TStrEnum],
        value: str,
        case_sensitive: bool = False,
    ) -> TStrEnum:
        members = cls.__members__.keys()
        for member in members:
            if member == value or (
                not case_sensitive and member.lower() == value.lower()
            ):
                return cls[member]

        raise ValueError(
            f"Invalid argument {value=}. (expected one of {tuple(members)})"
        )

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        other = other.value if isinstance(other, Enum) else str(other)
        return self.value.lower() == other.lower()  # type: ignore

    def __hash__(self) -> int:
        return hash(self.value.lower())
