import os as _os


SAFE_OS_PATH_FUNCTIONS = frozenset(
    {
        "abspath",
        "basename",
        "commonpath",
        "commonprefix",
        "dirname",
        "isabs",
        "join",
        "normcase",
        "normpath",
        "relpath",
        "split",
        "splitdrive",
        "splitext",
    }
)

SAFE_OS_PATH_CONSTANTS = frozenset(
    {
        "altsep",
        "curdir",
        "extsep",
        "pardir",
        "pathsep",
        "sep",
    }
)

SAFE_OS_PATH_ATTRIBUTES = SAFE_OS_PATH_FUNCTIONS | SAFE_OS_PATH_CONSTANTS


def is_safe_os_path_attribute(name: str) -> bool:
    return str(name or "").strip() in SAFE_OS_PATH_ATTRIBUTES


class SafeOSPath:
    """
    Read-only os.path-compatible proxy for sandboxed path composition.

    Only pure path string manipulation helpers are exposed. Functions that
    inspect the filesystem or environment, such as exists/isfile/getmtime and
    expanduser/expandvars, are intentionally omitted.
    """

    __slots__ = ()

    @property
    def altsep(self):
        return _os.altsep

    @property
    def curdir(self):
        return _os.curdir

    @property
    def extsep(self):
        return _os.extsep

    @property
    def pardir(self):
        return _os.pardir

    @property
    def pathsep(self):
        return _os.pathsep

    @property
    def sep(self):
        return _os.sep

    def abspath(self, path):
        return _os.path.abspath(path)

    def basename(self, path):
        return _os.path.basename(path)

    def commonpath(self, paths):
        return _os.path.commonpath(paths)

    def commonprefix(self, paths):
        return _os.path.commonprefix(paths)

    def dirname(self, path):
        return _os.path.dirname(path)

    def isabs(self, path):
        return _os.path.isabs(path)

    def join(self, path, *paths):
        return _os.path.join(path, *paths)

    def normcase(self, path):
        return _os.path.normcase(path)

    def normpath(self, path):
        return _os.path.normpath(path)

    def relpath(self, path, start=None):
        if start is None:
            return _os.path.relpath(path)
        return _os.path.relpath(path, start)

    def split(self, path):
        return _os.path.split(path)

    def splitdrive(self, path):
        return _os.path.splitdrive(path)

    def splitext(self, path):
        return _os.path.splitext(path)

    def __setattr__(self, name, value):
        raise TypeError("os.path proxy is read-only")

    def __delattr__(self, name):
        raise TypeError("os.path proxy is read-only")

    def __repr__(self):
        return "<SafeOSPath>"


class SafeOS:
    """Read-only os proxy that exposes only the safe path wrapper."""

    __slots__ = ("path",)

    def __init__(self, path_proxy: SafeOSPath):
        object.__setattr__(self, "path", path_proxy)

    def __setattr__(self, name, value):
        raise TypeError("os proxy is read-only")

    def __delattr__(self, name):
        raise TypeError("os proxy is read-only")

    def __getattr__(self, name):
        raise AttributeError(f"sandbox os proxy only exposes os.path, blocked: {name}")

    def __repr__(self):
        return "<SafeOS path=<SafeOSPath>>"


safe_os_path = SafeOSPath()
safe_os = SafeOS(safe_os_path)
