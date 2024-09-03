from importlib.metadata import PackageNotFoundError, version


try:
    __version__: str | None = version(__package__)
except PackageNotFoundError:
    __version__ = None
