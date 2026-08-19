from __future__ import annotations


class UnsupportedFileTypeError(ValueError):
    """Raised when the router rejects a file type (e.g. json/yaml/xml)."""


class LoaderError(RuntimeError):
    """Raised when a loader fails in a non-recoverable way."""
