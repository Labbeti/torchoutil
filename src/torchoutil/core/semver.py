#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple, TypedDict, Union, overload

from typing_extensions import NotRequired, TypeIs

from torchoutil.pyoutil.typing import NoneType, is_dict_str_optional_int

# Pattern of https://semver.org/
_VERSION_PATTERN = r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
_VERSION_FORMAT = r"{major}.{minor}.{patch}"
_VERSION_KEYS = ("major", "minor", "patch", "prerelease", "buildmetadata")


PreRelease = Union[int, str, None, list]
BuildMetadata = Union[int, str, None, list]


class VersionDict(TypedDict):
    major: int
    minor: int
    patch: int
    prerelease: NotRequired[PreRelease]
    buildmetadata: NotRequired[BuildMetadata]


VersionTupleLike = Union[
    Tuple[int, int, int],
    Tuple[int, int, int, PreRelease],
    Tuple[int, int, int, PreRelease, BuildMetadata],
]


@dataclass(init=False, eq=False)
class Version:
    """Version utility class following Semantic Versioning (SemVer) spec.

    Version format is: MAJOR.MINOR.PATCH[-PRERELEASE][+BUILDMETADATA]
    """

    major: int
    minor: int
    patch: int
    prerelease: Union[int, str, None, list]
    buildmetadata: Union[int, str, None, list]

    @overload
    def __init__(
        self,
        version_str: str,
        /,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        version_dict: Dict[str, Union[int, str, None, list]],
        /,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        version_tuple: VersionTupleLike,
        /,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        major: int,
        minor: int,
        patch: int,
        prerelease: Union[int, str, None, list] = None,
        buildmetadata: Union[int, str, None, list] = None,
    ) -> None:
        ...

    def __init__(self, *args, **kwargs) -> None:
        # Version str
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], str):
            version_str = args[0]
            version_dict = _parse_version_str(version_str)

        # Version dict
        elif len(args) == 1 and len(kwargs) == 0 and is_dict_str_optional_int(args[0]):
            version_dict = args[0]

        # Version tuple
        elif len(args) == 1 and len(kwargs) == 0 and _is_version_tuple(args[0]):
            version_tuple = args[0]
            version_dict = dict(zip(_VERSION_KEYS, version_tuple))

        # Version args/kwargs
        else:
            version_dict = dict(zip(_VERSION_KEYS, args))
            intersection = set(version_dict.keys()).intersection(kwargs.keys())
            if len(intersection) > 0:
                msg = f"Duplicated argument(s) {tuple(intersection)}. (with {args=} and {kwargs=})"
                raise ValueError(msg)
            version_dict.update(kwargs)  # type: ignore

        major = version_dict["major"]
        minor = version_dict["minor"]
        patch = version_dict["patch"]
        prerelease = version_dict.get("prerelease", None)
        buildmetadata = version_dict.get("buildmetadata", None)

        self.major = major  # type: ignore
        self.minor = minor  # type: ignore
        self.patch = patch  # type: ignore
        self.prerelease = prerelease  # type: ignore
        self.buildmetadata = buildmetadata  # type: ignore

    @classmethod
    def python(cls) -> "Version":
        return Version(
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
        )

    @classmethod
    def from_dict(cls, version_dict: Dict[str, int]) -> "Version":
        return Version(**version_dict)

    @classmethod
    def from_str(cls, version_str: str) -> "Version":
        version_dict = _parse_version_str(version_str)
        return Version(**version_dict)

    @classmethod
    def from_tuple(
        cls,
        version_tuple: VersionTupleLike,
    ) -> "Version":
        return Version(*version_tuple)

    def to_dict(self, exclude_none: bool = True) -> VersionDict:
        version_dict = asdict(self)
        if exclude_none:
            version_dict = {k: v for k, v in version_dict.items() if v is not None}
        return version_dict  # type: ignore

    def to_str(self) -> str:
        kwds = dict(
            major=self.major,
            minor=self.minor,
            patch=self.patch,
        )
        version_str = _VERSION_FORMAT.format(**kwds)
        if self.prerelease is not None:
            version_str = f"{version_str}-{self.prerelease}"
        if self.buildmetadata is not None:
            version_str = f"{version_str}+{self.buildmetadata}"

        return version_str

    def to_tuple(
        self,
        exclude_none: bool = True,
    ) -> VersionTupleLike:
        version_tuple = tuple(self.to_dict().values())
        if exclude_none:
            version_tuple = tuple(v for v in version_tuple if v is not None)
        return version_tuple  # type: ignore

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (dict, tuple, str)):
            other = Version(other)
        elif not isinstance(other, Version):
            return False

        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and self.prerelease == other.prerelease
            and self.buildmetadata == other.buildmetadata
        )

    def __lt__(self, other: "Version") -> bool:
        self_tuple = self.to_tuple()
        other_tuple = other.to_tuple()

        for self_v, other_v in zip(self_tuple, other_tuple):
            if self_v == other_v:
                continue

            if isinstance(self_v, (int, str, NoneType)):
                self_v = [self_v]
            if isinstance(other_v, (int, str, NoneType)):
                other_v = [other_v]

            minlen = min(len(self_v), len(other_v))
            if len(self_v) != len(other_v) and self_v[:minlen] == other_v[:minlen]:
                return len(self_v) < len(other_v)

            for self_vi, other_vi in zip(self_v, other_v):
                if self_vi == other_vi:
                    continue
                if isinstance(self_vi, int) and isinstance(other_vi, int):
                    return self_vi < other_vi
                if isinstance(self_vi, int) and isinstance(other_vi, str):
                    return True
                if isinstance(self_vi, str) and isinstance(other_vi, int):
                    return False
                if isinstance(self_vi, str) and isinstance(other_vi, str):
                    return self_vi < other_vi

                raise TypeError(f"Invalid attribute type {self_vi=} and {other_vi=}.")

        return True

    def __le__(self, other: "Version") -> bool:
        return self == other or self < other

    def __gt__(self, other: "Version") -> bool:
        return self != other and not (self < other)

    def __ge__(self, other: "Version") -> bool:
        return not (self < other)


def _parse_version_str(version_str: str) -> VersionDict:
    version_match = re.match(_VERSION_PATTERN, version_str)
    if version_match is None:
        msg = f"Invalid argument {version_str=}. (not a version)"
        raise ValueError(msg)

    version_dict = version_match.groupdict()
    result = {}
    for k, v in version_dict.items():
        if isinstance(v, str) and "." in v:
            v = v.split(".")
        else:
            v = [v]

        v = [int(vi) if isinstance(vi, str) and vi.isdigit() else vi for vi in v]
        if len(v) == 1:
            v = v[0]
        result[k] = v
    return result  # type: ignore


def _is_version_tuple(
    x: Any,
) -> TypeIs[VersionTupleLike]:
    if not isinstance(x, tuple):
        return False
    if len(x) not in (1, 2, 3):
        return False

    if not isinstance(x[0], int):
        return False
    elif len(x) == 1:
        return True

    if not isinstance(x[1], int):
        return False
    elif len(x) == 2:
        return True

    return isinstance(x[2], int)
