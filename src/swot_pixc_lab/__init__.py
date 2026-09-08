"""Research-grade tools for SWOT L2 HR PIXC workflows."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .constants import (
    DEFAULT_PIXC_COLLECTION_CONCEPT_ID,
    DEFAULT_PIXC_SHORT_NAME,
    DEFAULT_PIXC_VERSION,
)
from .discovery import GranuleRecord
from .exceptions import (
    AoiError,
    CacheIntegrityError,
    DateRangeError,
    DiscoveryError,
    DownloadError,
    LocalFilesUnavailableError,
    ObservationMismatchError,
    PixcLabError,
    PixcOpenError,
    PixcSchemaError,
)
from .mosaic import PixcObservation, open_pixc
from .pixc import LocalPixcCollection, PixcCollection, SearchProvenance
from .reader import PixcSourceMetadata, PixcVariableMetadata

try:
    __version__ = version("swot-pixc-lab")
except PackageNotFoundError:  # source checkout without an editable install
    __version__ = "0+unknown"

__all__ = [
    "AoiError",
    "CacheIntegrityError",
    "DEFAULT_PIXC_COLLECTION_CONCEPT_ID",
    "DEFAULT_PIXC_SHORT_NAME",
    "DEFAULT_PIXC_VERSION",
    "DateRangeError",
    "DiscoveryError",
    "DownloadError",
    "GranuleRecord",
    "LocalFilesUnavailableError",
    "LocalPixcCollection",
    "ObservationMismatchError",
    "PixcCollection",
    "PixcLabError",
    "PixcObservation",
    "PixcOpenError",
    "PixcSchemaError",
    "PixcSourceMetadata",
    "PixcVariableMetadata",
    "SearchProvenance",
    "__version__",
    "open_pixc",
]
