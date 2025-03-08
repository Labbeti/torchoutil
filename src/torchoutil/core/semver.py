#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
from dataclasses import asdict, dataclass
from typing import Iterable, List, Mapping, Tuple, TypedDict, Union, overload

from typing_extensions import NotRequired

from torchoutil.pyoutil.typing import NoneType, isinstance_guard

# Pattern of https://semver.org/
_VERSION_PATTERN = r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
_VERSION_FORMAT = r"{major}.{minor}.{patch}"
_VERSION_KEYS = ("major", "minor", "patch", "prerelease", "buildmetadata")


PreRelease = Union[int, str, None, List[Union[int, str]]]
BuildMetadata = Union[int, str, None, List[Union[int, str]]]


class VersionDict(TypedDict):
    major: int
    minor: NotRequired[int]
    patch: NotRequired[int]
    prerelease: NotRequired[PreRelease]
    buildmetadata: NotRequired[BuildMetadata]


VersionTuple = Union[
    Tuple[int, int, int],
    Tuple[int, int, int, PreRelease],
    Tuple[int, int, int, PreRelease, BuildMetadata],
]

VersionDictLike = Mapping[str, Union[int, PreRelease, BuildMetadata]]
VersionTupleLike = Iterable[Union[int, PreRelease, BuildMetadata]]


@dataclass(init=False, eq=False)
class Version:
    """Version utility class following Semantic Versioning (SemVer) spec.

    Version format is: MAJOR.MINOR.PATCH[-PRERELEASE][+BUILDMETADATA]
    """

    major: int
    minor: int
    patch: int
    prerelease: PreRelease
    buildmetadata: BuildMetadata

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
        version_dict: VersionDictLike,
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
        minor: int = 0,
        patch: int = 0,
        prerelease: Union[int, str, None, list] = None,
        buildmetadata: Union[int, str, None, list] = None,
    ) -> None:
        ...

    def __init__(self, *args, **kwargs) -> None:
        has_1_pos_arg = len(args) == 1 and len(kwargs) == 0
        # Version str
        if has_1_pos_arg and isinstance(args[0], str):
            version_str = args[0]
            version_dict = _parse_version_str(version_str)

        # Version dict
        elif has_1_pos_arg and isinstance_guard(args[0], VersionDictLike):
            version_dict = args[0]

        # Version tuple
        elif has_1_pos_arg and isinstance_guard(args[0], VersionTupleLike):
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

            invalid = tuple(set(version_dict.keys()).difference(_VERSION_KEYS))
            if len(invalid) > 0:
                msg = f"Invalid arguments {kwargs=}. (invalid keys: {invalid})"
                raise ValueError(msg)

        if not isinstance_guard(version_dict, VersionDict):
            msg = f"Invalid argument {args=} and {kwargs=}. (invalid argument types, expected (int, int, int, {PreRelease}, {BuildMetadata}))"
            raise ValueError(msg)

        major = version_dict["major"]
        minor = version_dict.get("minor", 0)
        patch = version_dict.get("patch", 0)
        prerelease = version_dict.get("prerelease", None)
        buildmetadata = version_dict.get("buildmetadata", None)

        self.major = major  # type: ignore
        self.minor = minor  # type: ignore
        self.patch = patch  # type: ignore
        self.prerelease = prerelease  # type: ignore
        self.buildmetadata = buildmetadata  # type: ignore

    @classmethod
    def python(cls) -> "Version":
        """Create an instance of Version with Python version."""
        return Version(
            major=sys.version_info.major,
            minor=sys.version_info.minor,
            patch=sys.version_info.micro,
            buildmetadata=sys.version_info.releaselevel,
        )

    @classmethod
    def from_dict(cls, version_dict: VersionDictLike) -> "Version":
        return Version(**version_dict)

    @classmethod
    def from_str(cls, version_str: str) -> "Version":
        version_dict = _parse_version_str(version_str)
        return Version(**version_dict)

    @classmethod
    def from_tuple(cls, version_tuple: VersionTupleLike) -> "Version":
        return Version(*version_tuple)

    def next_major(self) -> "Version":
        return Version(self.major + 1, 0, 0)

    def next_minor(self) -> "Version":
        return Version(self.major, self.minor + 1, 0)

    def next_patch(self) -> "Version":
        return Version(self.major, self.minor, self.patch + 1)

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
    ) -> VersionTuple:
        version_tuple = tuple(self.to_dict().values())
        if exclude_none:
            version_tuple = tuple(v for v in version_tuple if v is not None)
        return version_tuple  # type: ignore

    def __str__(self) -> str:
        return self.to_str()

    def __eq__(
        self, other: Union["Version", str, VersionDictLike, VersionTupleLike]
    ) -> bool:
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
