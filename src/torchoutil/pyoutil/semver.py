#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import sys
from dataclasses import asdict, dataclass
from typing import Any, Iterable, List, Mapping, Tuple, TypedDict, Union, overload

from typing_extensions import NotRequired, TypeAlias

from .typing import NoneType, isinstance_guard

# Pattern of https://semver.org/
_VERSION_PATTERN = r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+(?P<buildmetadata>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
_VERSION_FORMAT = r"{major}.{minor}.{patch}"
_VERSION_KEYS = ("major", "minor", "patch", "prerelease", "buildmetadata")


PreRelease: TypeAlias = Union[int, str, None, List[Union[int, str]]]
BuildMetadata: TypeAlias = Union[int, str, None, List[Union[int, str]]]


class VersionDict(TypedDict):
    major: int
    minor: NotRequired[int]
    patch: NotRequired[int]
    prerelease: NotRequired[PreRelease]
    buildmetadata: NotRequired[BuildMetadata]


VersionTuple: TypeAlias = Union[
    Tuple[int, int, int],
    Tuple[int, int, int, PreRelease],
    Tuple[int, int, int, PreRelease, BuildMetadata],
]

VersionDictLike: TypeAlias = Mapping[str, Union[int, PreRelease, BuildMetadata]]
VersionTupleLike: TypeAlias = Iterable[Union[int, PreRelease, BuildMetadata]]
VersionLike: TypeAlias = Union["Version", str, VersionDictLike, VersionTupleLike]


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
        version: "Version",
        /,
    ) -> None:
        ...

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
        minor: int,
        patch: int,
        prerelease: PreRelease = None,
        buildmetadata: BuildMetadata = None,
    ) -> None:
        ...

    def __init__(self, *args, **kwargs) -> None:
        has_1_pos_arg = len(args) == 1 and len(kwargs) == 0
        # Version
        if has_1_pos_arg and isinstance(args[0], Version):
            version = args[0]
            version_dict = version.to_dict(exclude_none=False)

        # Version str
        elif has_1_pos_arg and isinstance(args[0], str):
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
            intersection = tuple(set(version_dict.keys()).intersection(kwargs.keys()))
            if len(intersection) > 0:
                msg = f"Got multiple values for argument(s) {intersection}. (with {args=} and {kwargs=})"
                raise TypeError(msg)
            version_dict.update(kwargs)  # type: ignore

            invalid = tuple(set(version_dict.keys()).difference(_VERSION_KEYS))
            if len(invalid) > 0:
                msg = f"Got an unexpected arguments {invalid=}. (with {args=} and {kwargs=})"
                raise TypeError(msg)

        if not isinstance_guard(version_dict, VersionDict):
            msg = f"Invalid argument {args=} and {kwargs=}. (invalid argument types, expected (major=int, minor=int, patch=int, prerelease={PreRelease}, buildmetadata={BuildMetadata}))"
            raise ValueError(msg)

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
        """Create an instance of Version with Python version."""
        return Version(
            major=sys.version_info.major,
            minor=sys.version_info.minor,
            patch=sys.version_info.micro,
            buildmetadata=sys.version_info.releaselevel,
        )

    @classmethod
    def from_dict(cls, version_dict: VersionDictLike) -> "Version":
        return Version(version_dict)

    @classmethod
    def from_str(cls, version_str: str) -> "Version":
        version_dict = _parse_version_str(version_str)
        return Version(**version_dict)

    @classmethod
    def from_tuple(cls, version_tuple: VersionTupleLike) -> "Version":
        return Version(version_tuple)

    def without_prerelease(self) -> "Version":
        return Version(self.major, self.minor, self.patch, None, self.buildmetadata)

    def without_buildmetadata(self) -> "Version":
        return Version(self.major, self.minor, self.patch, self.prerelease, None)

    def next_major(
        self,
        keep_prerelease: bool = False,
        keep_buildmetadata: bool = False,
    ) -> "Version":
        prerelease = self.prerelease if keep_prerelease else None
        buildmetadata = self.buildmetadata if keep_buildmetadata else None
        return Version(
            major=self.major + 1,
            minor=0,
            patch=0,
            prerelease=prerelease,
            buildmetadata=buildmetadata,
        )

    def next_minor(
        self,
        keep_prerelease: bool = False,
        keep_buildmetadata: bool = False,
    ) -> "Version":
        prerelease = self.prerelease if keep_prerelease else None
        buildmetadata = self.buildmetadata if keep_buildmetadata else None
        return Version(
            major=self.major,
            minor=self.minor + 1,
            patch=0,
            prerelease=prerelease,
            buildmetadata=buildmetadata,
        )

    def next_patch(
        self,
        keep_prerelease: bool = False,
        keep_buildmetadata: bool = False,
    ) -> "Version":
        prerelease = self.prerelease if keep_prerelease else None
        buildmetadata = self.buildmetadata if keep_buildmetadata else None
        return Version(
            major=self.major,
            minor=self.minor,
            patch=self.patch + 1,
            prerelease=prerelease,
            buildmetadata=buildmetadata,
        )

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
        version_tuple = tuple(self.to_dict(exclude_none).values())
        return version_tuple  # type: ignore

    def __str__(self) -> str:
        return self.to_str()

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

    def __lt__(self, other: VersionLike) -> bool:
        if isinstance(other, (dict, tuple, str)):
            other = Version(other)
        elif not isinstance(other, Version):
            raise TypeError(f"Invalid argument type {type(other)}.")

        self_tuple = self.to_tuple(exclude_none=False)
        other_tuple = other.to_tuple(exclude_none=False)

        for self_v, other_v in zip(self_tuple, other_tuple):
            if self_v == other_v:
                continue
            if self_v is None and other_v is not None:
                return False
            if self_v is not None and other_v is None:
                return True

            if isinstance(self_v, (int, str, NoneType)):
                self_v = [self_v]
            elif not isinstance(self_v, list):
                raise TypeError(f"Invalid argument type {type(self_v)}.")

            if isinstance(other_v, (int, str, NoneType)):
                other_v = [other_v]
            elif not isinstance(other_v, list):
                raise TypeError(f"Invalid argument type {type(other_v)}.")

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

        return False

    def __le__(self, other: VersionLike) -> bool:
        return (self == other) or (self < other)

    def __gt__(self, other: VersionLike) -> bool:
        return (self != other) and not (self < other)

    def __ge__(self, other: VersionLike) -> bool:
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
