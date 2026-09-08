"""Stable identifiers and table schema for the Phase-1 PIXC workflow."""

from __future__ import annotations

DEFAULT_PIXC_COLLECTION_CONCEPT_ID = "C3233944986-POCLOUD"
DEFAULT_PIXC_SHORT_NAME = "SWOT_L2_HR_PIXC_D"
DEFAULT_PIXC_VERSION = "D"

METADATA_COLUMNS = (
    "observation_datetime",
    "observation_start",
    "observation_end",
    "cycle",
    "pass",
    "tile",
    "granule_id",
    "native_id",
    "filename",
    "product_short_name",
    "product_version",
    "concept_id",
    "collection_concept_id",
    "provider",
    "revision_id",
    "revision_date",
    "metadata_specification_version",
    "source_url",
    "s3_url",
    "size_bytes",
    "checksum",
    "checksum_algorithm",
    "pge_version",
    "production_datetime",
    "granule_bbox",
    "local_path",
    "metadata_warnings",
)
