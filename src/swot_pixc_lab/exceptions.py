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
    """Raised when ``resolve_local`` cannot find all local granules."""


class PixcOpenError(PixcLabError):
    """Raised when a local PIXC NetCDF file cannot be opened safely."""


class PixcSchemaError(PixcOpenError):
    """Raised when a PIXC NetCDF file has an incompatible or invalid schema."""


class ObservationMismatchError(PixcOpenError):
    """Raised when files from different SWOT observations are combined."""


class QualityMetadataError(PixcLabError, ValueError):
    """Raised when quality metadata cannot be decoded without guessing."""


class TransectError(PixcLabError, ValueError):
    """Raised when a manual transect or corridor cannot be sampled safely."""


class WidthError(PixcLabError, ValueError):
    """Raised when explicit wet-interval measurements are invalid."""


class BankInferenceError(PixcLabError, ValueError):
    """Raised when experimental candidate-interval inference is invalid."""
