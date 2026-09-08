"""Package-specific exceptions with actionable messages."""

from __future__ import annotations


class PixcLabError(Exception):
    """Base exception for SWOT PIXC Lab errors."""


class AoiError(PixcLabError, ValueError):
    """Raised when an area of interest cannot be searched safely."""


class DateRangeError(PixcLabError, ValueError):
    """Raised when temporal search bounds are invalid."""


class DiscoveryError(PixcLabError):
    """Raised when NASA CMR discovery fails or returns malformed data."""


class DownloadError(PixcLabError):
    """Raised when one or more granules cannot be downloaded safely."""


class CacheIntegrityError(DownloadError):
    """Raised when a cached file does not match its CMR metadata."""


class LocalFilesUnavailableError(PixcLabError):
    """Raised when Phase-1 ``open`` cannot find all local granules."""
