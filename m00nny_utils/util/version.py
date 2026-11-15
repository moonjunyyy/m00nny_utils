from typing import overload


class Version:
    @overload
    def __init__(
        self, version: str) -> None: ...

    @overload
    def __init__(
        self, major: int, minor: int = None, patch: int = None
    ) -> None: ...

    def __init__(self, *args) -> None:
        self.major = 0
        self.minor = None
        self.patch = None
        if len(args) == 1:
            version = args[0]
            if version.startswith("v"):
                version = version[1:]
            version = version.split(".")
            if len(version) == 0:
                raise ValueError("Major version is required")
            if len(version) > 0:
                self.major = int(version[0].split()[0])
            if len(version) > 1:
                self.minor = int(version[1].split()[0])
            if len(version) > 2:
                self.patch = int(version[2].split()[0])
        elif len(args) == 3:
            self.major, self.minor, self.patch = args
        else:
            raise ValueError("Invalid arguments")

    def __str__(self) -> str:
        if self.minor is None:
            return f"{self.major}"
        if self.patch is None:
            return f"{self.major}.{self.minor}"
        return f"{self.major}.{self.minor}.{self.patch}"

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            _version_of_other = Version(other)
        elif isinstance(other, Version):
            _version_of_other = other
        else:
            raise ValueError(
                f"Invalid comparison type {type(other)} for Version")
        if _version_of_other.major != self.major:
            return False
        if (
            _version_of_other.minor
            and self.minor
            and _version_of_other.minor != self.minor
        ):
            return False
        if (
            _version_of_other.patch
            and self.patch
            and _version_of_other.patch != self.patch
        ):
            return False
        return True

    def __gt__(self, other) -> bool:
        if isinstance(other, str):
            _version_of_other = Version(other)
        elif isinstance(other, Version):
            _version_of_other = other
        else:
            raise ValueError(
                f"Invalid comparison type {type(other)} for Version")
        if self.major > _version_of_other.major:
            return True
        if self.major < _version_of_other.major:
            return False
        if self.minor and _version_of_other.minor:
            if self.minor > _version_of_other.minor:
                return True
            if self.minor < _version_of_other.minor:
                return False
        if self.patch and _version_of_other.patch:
            if self.patch > _version_of_other.patch:
                return True
            if self.patch < _version_of_other.patch:
                return False
        return False

    def __lt__(self, other) -> bool:
        if isinstance(other, str):
            _version_of_other = Version(other)
        elif isinstance(other, Version):
            _version_of_other = other
        else:
            raise ValueError(
                f"Invalid comparison type {type(other)} for Version")
        if self.major < _version_of_other.major:
            return True
        if self.major > _version_of_other.major:
            return False
        if self.minor and _version_of_other.minor:
            if self.minor < _version_of_other.minor:
                return True
            if self.minor > _version_of_other.minor:
                return False
        if self.patch and _version_of_other.patch:
            if self.patch < _version_of_other.patch:
                return True
            if self.patch > _version_of_other.patch:
                return False
        return False

    def __ge__(self, other) -> bool:
        return self.__gt__(other) or self.__eq__(other)

    def __le__(self, other) -> bool:
        return self.__lt__(other) or self.__eq__(other)

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)
