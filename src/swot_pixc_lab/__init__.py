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
    QualityMetadataError,
    TransectError,
)
from .mosaic import PixcObservation, open_pixc
from .pixc import LocalPixcCollection, PixcCollection, SearchProvenance
from .qc import (
    QC_PROFILES,
    DecodedQualityFlags,
    FlagDefinition,
    HeightReferenceResult,
    QCProfile,
    QCResult,
    QCRule,
    QCRuleOutcome,
    apply_qc,
    decode_quality_flags,
)
from .reader import (
    PHASE3_POINT_VARIABLES,
    PixcSourceMetadata,
    PixcVariableMetadata,
)
from .transect import TransectSample, sample_transect
from .visualization import (
    CLASSIFICATION_COLORS,
    CLASSIFICATION_LABELS,
    SUPPORTED_COLOR_VARIABLES,
    plot_classification_comparison,
    plot_pixc_map,
    plot_transect_classification,
    plot_transect_corridor,
    plot_transect_height,
)

try:
    __version__ = version("swot-pixc-lab")
except PackageNotFoundError:  # source checkout without an editable install
    __version__ = "0+unknown"

__all__ = [
    "AoiError",
    "CacheIntegrityError",
    "CLASSIFICATION_COLORS",
    "CLASSIFICATION_LABELS",
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
    "PHASE3_POINT_VARIABLES",
    "PixcCollection",
    "PixcLabError",
    "PixcObservation",
    "PixcOpenError",
    "PixcSchemaError",
    "PixcSourceMetadata",
    "PixcVariableMetadata",
    "QC_PROFILES",
    "DecodedQualityFlags",
    "FlagDefinition",
    "HeightReferenceResult",
    "QCProfile",
    "QCResult",
    "QCRule",
    "QCRuleOutcome",
    "QualityMetadataError",
    "SearchProvenance",
    "SUPPORTED_COLOR_VARIABLES",
    "TransectError",
    "TransectSample",
    "__version__",
    "apply_qc",
    "decode_quality_flags",
    "open_pixc",
    "plot_classification_comparison",
    "plot_pixc_map",
    "plot_transect_classification",
    "plot_transect_corridor",
    "plot_transect_height",
    "sample_transect",
]
