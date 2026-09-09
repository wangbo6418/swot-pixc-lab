"""Auditable Phase 5A.3b benchmark preparation and validation automation.

This module coordinates existing Phase 4 and Phase 5A.1--3a contracts. It
does not infer reference boundaries, select a preferred configuration, or
implement a new bank/width method.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from itertools import product
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pyproj import Geod
from shapely.geometry import LineString, mapping, shape

from ._pixel_input import PixelData, resolve_pixel_input
from .bank_inference import (
    CandidateIntervalConfiguration,
    CandidateIntervalInferenceResult,
    CandidateIntervalSensitivityResult,
    infer_candidate_wet_intervals,
    run_candidate_interval_sensitivity,
)
from .exceptions import AnnotationRequiredError, BenchmarkError
from .transect import TransectSample, sample_transect
from .validation import (
    ManualBenchmarkMetadata,
    ManualBenchmarkRecord,
    create_manual_benchmark_record,
    evaluate_sensitivity_against_explicit,
)
from .visualization import (
    plot_candidate_wet_interval_inference,
    plot_pixc_map,
    plot_transect_classification,
)
from .width import measure_explicit_wet_intervals

BENCHMARK_CONFIG_SCHEMA_VERSION = "1.0"
TRANSECT_MANIFEST_SCHEMA_VERSION = "1.0"
ANNOTATION_INPUT_SCHEMA_VERSION = "1.0"
RUN_MANIFEST_SCHEMA_VERSION = "1.0"
PROPOSED_TRANSECT_STATUS = "PROPOSED / REQUIRES SCIENTIST REVIEW"
APPROVED_TRANSECT_STATUS = "APPROVED"
PILOT_GRID_NOTICE = "EXPLORATORY PILOT GRID / NOT A RECOMMENDED FINAL CONFIGURATION"
ANNOTATION_REQUIRED_MESSAGE = (
    "Annotation packets created. Manual reference annotations are required "
    "before quantitative validation can run."
)
_BENCHMARK_SUBSETS = frozenset({"pilot", "development", "holdout", "unassigned"})
_TRANSECT_STATUSES = frozenset({PROPOSED_TRANSECT_STATUS, APPROVED_TRANSECT_STATUS})
_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True, slots=True)
class BenchmarkObservation:
    """Configured identity and local files for one benchmark observation."""

    identifier: str
    date: str
    cycle: int
    pass_number: int
    tiles: tuple[str, ...]
    aoi: tuple[float, float, float, float]
    local_files: tuple[Path, ...]

    def __post_init__(self) -> None:
        _nonempty_text(self.identifier, "observation.identifier")
        _nonempty_text(self.date, "observation.date")
        _positive_integer(self.cycle, "observation.cycle")
        _positive_integer(self.pass_number, "observation.pass")
        if not self.tiles or any(
            not isinstance(item, str) or not item.strip() for item in self.tiles
        ):
            raise BenchmarkError(
                "observation.tiles must contain non-empty tile strings."
            )
        if len(set(self.tiles)) != len(self.tiles):
            raise BenchmarkError("observation.tiles must not contain duplicates.")
        if len(self.aoi) != 4 or not all(math.isfinite(value) for value in self.aoi):
            raise BenchmarkError("observation.aoi must be four finite bbox values.")
        west, south, east, north = self.aoi
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise BenchmarkError(
                "observation.aoi must be a valid non-crossing EPSG:4326 bbox."
            )
        if not self.local_files:
            raise BenchmarkError(
                "observation.local_files must contain at least one file."
            )

    def as_dict(self) -> dict[str, object]:
        """Return JSON-friendly configured observation metadata."""

        return {
            "identifier": self.identifier,
            "date": self.date,
            "cycle": self.cycle,
            "pass": self.pass_number,
            "tiles": list(self.tiles),
            "aoi": list(self.aoi),
            "local_files": [str(path) for path in self.local_files],
        }


@dataclass(frozen=True, slots=True)
class ReferenceImageryMetadata:
    """Independent imagery metadata used only as a human annotation aid."""

    status: str
    source_provider: str | None = None
    acquisition_date: str | None = None
    temporal_offset_hours: float | None = None
    cloud_quality_note: str | None = None
    local_image_path: Path | None = None
    source_identifier: str | None = None
    analyst_notes: str | None = None

    def __post_init__(self) -> None:
        _nonempty_text(self.status, "reference_imagery.status")
        for name in (
            "source_provider",
            "acquisition_date",
            "cloud_quality_note",
            "source_identifier",
            "analyst_notes",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise BenchmarkError(f"reference_imagery.{name} must be text or null.")
        if self.temporal_offset_hours is not None:
            _finite_number(
                self.temporal_offset_hours,
                "reference_imagery.temporal_offset_hours",
            )

    @property
    def pending(self) -> bool:
        """Return whether independent imagery remains unavailable."""

        return "pending" in self.status.lower()

    def as_dict(self) -> dict[str, object]:
        """Return JSON-friendly imagery metadata."""

        result = asdict(self)
        result["local_image_path"] = (
            str(self.local_image_path) if self.local_image_path is not None else None
        )
        return result


@dataclass(frozen=True, slots=True)
class SensitivityGrid:
    """Explicit ordered axes for an unranked Cartesian sensitivity grid."""

    extent_class_sets: tuple[tuple[int, ...], ...]
    station_bin_widths_m: tuple[float, ...]
    min_extent_pixels_per_bin: tuple[int, ...]
    max_bridge_gaps_m: tuple[float, ...]
    notice: str

    def __post_init__(self) -> None:
        if self.notice != PILOT_GRID_NOTICE:
            raise BenchmarkError(
                "sensitivity_grid.notice must explicitly say that the pilot grid "
                "is exploratory and not recommended."
            )
        axes: tuple[tuple[object, ...], ...] = (
            self.extent_class_sets,
            self.station_bin_widths_m,
            self.min_extent_pixels_per_bin,
            self.max_bridge_gaps_m,
        )
        if any(not axis for axis in axes):
            raise BenchmarkError("Every sensitivity-grid axis must be non-empty.")
        for values in product(*axes):
            CandidateIntervalConfiguration(
                extent_classes=values[0],  # type: ignore[arg-type]
                station_bin_width_m=values[1],  # type: ignore[arg-type]
                min_extent_pixels_per_bin=values[2],  # type: ignore[arg-type]
                max_bridge_gap_m=values[3],  # type: ignore[arg-type]
            )

    def as_dict(self) -> dict[str, object]:
        """Return the four ordered grid axes and required notice."""

        return {
            "notice": self.notice,
            "extent_class_sets": [list(values) for values in self.extent_class_sets],
            "station_bin_widths_m": list(self.station_bin_widths_m),
            "min_extent_pixels_per_bin": list(self.min_extent_pixels_per_bin),
            "max_bridge_gaps_m": list(self.max_bridge_gaps_m),
        }


@dataclass(frozen=True, slots=True)
class TransectProposalSettings:
    """Explicit placement parameters for optional proposal-only transects."""

    enabled: bool
    count: int
    stratification_axis: Literal["latitude", "longitude"]
    transect_azimuth_degrees: float
    line_length_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise BenchmarkError("transect_proposals.enabled must be boolean.")
        _positive_integer(self.count, "transect_proposals.count")
        if self.stratification_axis not in {"latitude", "longitude"}:
            raise BenchmarkError(
                "transect_proposals.stratification_axis must be latitude or longitude."
            )
        azimuth = _finite_number(
            self.transect_azimuth_degrees,
            "transect_proposals.transect_azimuth_degrees",
        )
        if not (0.0 <= azimuth < 360.0):
            raise BenchmarkError(
                "transect_proposals.transect_azimuth_degrees must be in [0, 360)."
            )
        _positive_number(self.line_length_m, "transect_proposals.line_length_m")

    def as_dict(self) -> dict[str, object]:
        """Return all proposal parameters without hidden values."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Validated, path-resolved Phase 5A.3b benchmark configuration."""

    benchmark_id: str
    benchmark_subset: str
    status: str
    observation: BenchmarkObservation
    qc_profile: str
    corridor_half_width_m: float
    transect_manifest_path: Path
    annotation_directory: Path
    output_directory: Path
    reference_imagery: ReferenceImageryMetadata
    packet_configuration: CandidateIntervalConfiguration
    sensitivity_grid: SensitivityGrid
    transect_proposals: TransectProposalSettings
    source_path: Path | None = None
    schema_version: str = BENCHMARK_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BENCHMARK_CONFIG_SCHEMA_VERSION:
            raise BenchmarkError(
                f"Unsupported benchmark config schema {self.schema_version!r}."
            )
        _nonempty_text(self.benchmark_id, "benchmark_id")
        if self.benchmark_subset not in _BENCHMARK_SUBSETS:
            raise BenchmarkError(
                "benchmark_subset must be pilot, development, holdout, or unassigned."
            )
        _nonempty_text(self.status, "status")
        _nonempty_text(self.qc_profile, "qc_profile")
        _positive_number(self.corridor_half_width_m, "corridor_half_width_m")

    @property
    def sensitivity_configurations(
        self,
    ) -> tuple[CandidateIntervalConfiguration, ...]:
        """Expand the caller-ordered grid without ranking or selection."""

        return expand_sensitivity_grid(self.sensitivity_grid)

    def as_dict(self) -> dict[str, object]:
        """Return a complete JSON-friendly configuration snapshot."""

        return {
            "schema_version": self.schema_version,
            "benchmark_id": self.benchmark_id,
            "benchmark_subset": self.benchmark_subset,
            "status": self.status,
            "observation": self.observation.as_dict(),
            "qc_profile": self.qc_profile,
            "corridor_half_width_m": self.corridor_half_width_m,
            "transect_manifest_path": str(self.transect_manifest_path),
            "annotation_directory": str(self.annotation_directory),
            "output_directory": str(self.output_directory),
            "reference_imagery": self.reference_imagery.as_dict(),
            "packet_configuration": self.packet_configuration.as_dict(),
            "sensitivity_grid": self.sensitivity_grid.as_dict(),
            "transect_proposals": self.transect_proposals.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkTransect:
    """One explicit EPSG:4326 benchmark or review-proposal transect."""

    transect_id: str
    geometry: LineString
    corridor_half_width_m: float
    review_status: str
    site_label: str | None = None
    morphology_label: str | None = None
    selection_notes: str | None = None
    proposal_diagnostics: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        _nonempty_text(self.transect_id, "transect_id")
        if not isinstance(self.geometry, LineString):
            raise BenchmarkError("Transect geometry must be a LineString.")
        coordinates = np.asarray(self.geometry.coords, dtype=np.float64)
        if (
            self.geometry.is_empty
            or not self.geometry.is_valid
            or not self.geometry.is_simple
            or self.geometry.is_closed
            or coordinates.shape[0] < 2
            or coordinates.shape[1] != 2
            or not np.all(np.isfinite(coordinates))
        ):
            raise BenchmarkError("Transect must be a valid open 2-D LineString.")
        if np.any((coordinates[:, 0] < -180) | (coordinates[:, 0] > 180)) or np.any(
            (coordinates[:, 1] < -90) | (coordinates[:, 1] > 90)
        ):
            raise BenchmarkError("Transect coordinates must be valid EPSG:4326 values.")
        _positive_number(self.corridor_half_width_m, "corridor_half_width_m")
        if self.review_status not in _TRANSECT_STATUSES:
            raise BenchmarkError(
                f"review_status must be one of {sorted(_TRANSECT_STATUSES)!r}."
            )
        for name in ("site_label", "morphology_label", "selection_notes"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise BenchmarkError(f"{name} must be text or null.")

    @property
    def approved(self) -> bool:
        """Return whether the owner has explicitly approved this transect."""

        return self.review_status == APPROVED_TRANSECT_STATUS

    def as_feature(self) -> dict[str, object]:
        """Serialize the transect as one GeoJSON feature."""

        properties: dict[str, object] = {
            "transect_id": self.transect_id,
            "corridor_half_width_m": self.corridor_half_width_m,
            "review_status": self.review_status,
            "site_label": self.site_label,
            "morphology_label": self.morphology_label,
            "selection_notes": self.selection_notes,
        }
        if self.proposal_diagnostics is not None:
            properties["proposal_diagnostics"] = dict(self.proposal_diagnostics)
        return {
            "type": "Feature",
            "id": self.transect_id,
            "geometry": mapping(self.geometry),
            "properties": properties,
        }


@dataclass(frozen=True, slots=True)
class TransectManifest:
    """Ordered collection of explicit transects in EPSG:4326."""

    benchmark_id: str
    transects: tuple[BenchmarkTransect, ...]
    schema_version: str = TRANSECT_MANIFEST_SCHEMA_VERSION
    crs: str = "EPSG:4326"

    def __post_init__(self) -> None:
        if self.schema_version != TRANSECT_MANIFEST_SCHEMA_VERSION:
            raise BenchmarkError("Unsupported transect manifest schema version.")
        _nonempty_text(self.benchmark_id, "benchmark_id")
        if self.crs != "EPSG:4326":
            raise BenchmarkError("Transect manifest CRS must be EPSG:4326.")
        if not self.transects:
            raise BenchmarkError(
                "Transect manifest must contain at least one transect."
            )
        identifiers = [item.transect_id for item in self.transects]
        if len(set(identifiers)) != len(identifiers):
            raise BenchmarkError("Transect identifiers must be unique.")

    def as_geojson(self) -> dict[str, object]:
        """Return an ordered GeoJSON FeatureCollection."""

        return {
            "type": "FeatureCollection",
            "schema_version": self.schema_version,
            "benchmark_id": self.benchmark_id,
            "crs": {"type": "name", "properties": {"name": self.crs}},
            "features": [item.as_feature() for item in self.transects],
        }


@dataclass(frozen=True, slots=True)
class ManualAnnotation:
    """Validated analyst boundary input, excluding all derived widths."""

    benchmark_id: str
    transect_id: str
    annotation_id: str
    analyst_id: str
    benchmark_subset: str
    manual_intervals: tuple[tuple[float, float], ...]
    confidence: str | None
    notes: str | None
    reference_imagery: ReferenceImageryMetadata
    source_path: Path
    station_frame_hash: str

    def __post_init__(self) -> None:
        for name in ("benchmark_id", "transect_id", "annotation_id", "analyst_id"):
            _nonempty_text(getattr(self, name), name)
        if self.benchmark_subset not in _BENCHMARK_SUBSETS:
            raise BenchmarkError(
                "benchmark_subset must be pilot, development, holdout, or unassigned."
            )
        _nonempty_text(self.station_frame_hash, "station_frame_hash")


@dataclass(frozen=True, slots=True)
class AnnotationPacket:
    """Paths and cached sample for one generated annotation packet."""

    transect: BenchmarkTransect
    sample: TransectSample
    directory: Path
    files: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkPreparationResult:
    """Packet-generation result retaining reusable transect samples."""

    packets: tuple[AnnotationPacket, ...]
    transect_manifest_path: Path
    preparation_manifest_path: Path

    @property
    def samples(self) -> Mapping[str, TransectSample]:
        """Return samples keyed in manifest order."""

        return {packet.transect.transect_id: packet.sample for packet in self.packets}


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    """Paths and in-memory tables from one quantitative benchmark run."""

    individual_results: pd.DataFrame
    aggregate_results: pd.DataFrame
    benchmark_records: tuple[ManualBenchmarkRecord, ...]
    output_directory: Path
    results_csv: Path
    aggregate_csv: Path
    report_path: Path
    manifest_path: Path
    figure_paths: tuple[Path, ...]


def load_benchmark_config(path: str | os.PathLike[str]) -> BenchmarkConfig:
    """Load and strictly validate a versioned JSON benchmark configuration."""

    source = Path(path).expanduser().resolve()
    payload = _read_json_object(source, "benchmark configuration")
    _require_keys(
        payload,
        {
            "schema_version",
            "benchmark_id",
            "benchmark_subset",
            "status",
            "observation",
            "local_data_directory",
            "qc_profile",
            "corridor_half_width_m",
            "transect_manifest_path",
            "annotation_directory",
            "output_directory",
            "reference_imagery",
            "packet_configuration",
            "sensitivity_grid",
            "transect_proposals",
        },
        "benchmark configuration",
    )
    base = source.parent
    observation_raw = _mapping(payload["observation"], "observation")
    _require_keys(
        observation_raw,
        {"identifier", "date", "cycle", "pass", "tiles", "aoi", "filenames"},
        "observation",
    )
    local_directory = _resolve_path(base, payload["local_data_directory"])
    filenames = _text_sequence(observation_raw["filenames"], "observation.filenames")
    observation = BenchmarkObservation(
        identifier=_required_text(
            observation_raw["identifier"], "observation.identifier"
        ),
        date=_required_text(observation_raw["date"], "observation.date"),
        cycle=observation_raw["cycle"],  # type: ignore[arg-type]
        pass_number=observation_raw["pass"],  # type: ignore[arg-type]
        tiles=_text_sequence(observation_raw["tiles"], "observation.tiles"),
        aoi=_bbox(observation_raw["aoi"]),
        local_files=tuple((local_directory / name).resolve() for name in filenames),
    )
    imagery = _parse_imagery(payload["reference_imagery"], base=base)
    grid_raw = _mapping(payload["sensitivity_grid"], "sensitivity_grid")
    _require_keys(
        grid_raw,
        {
            "notice",
            "extent_class_sets",
            "station_bin_widths_m",
            "min_extent_pixels_per_bin",
            "max_bridge_gaps_m",
        },
        "sensitivity_grid",
    )
    proposal_raw = _mapping(payload["transect_proposals"], "transect_proposals")
    _require_keys(
        proposal_raw,
        {
            "enabled",
            "count",
            "stratification_axis",
            "transect_azimuth_degrees",
            "line_length_m",
        },
        "transect_proposals",
    )
    return BenchmarkConfig(
        schema_version=payload["schema_version"],  # type: ignore[arg-type]
        benchmark_id=_required_text(payload["benchmark_id"], "benchmark_id"),
        benchmark_subset=_required_text(
            payload["benchmark_subset"], "benchmark_subset"
        ),
        status=_required_text(payload["status"], "status"),
        observation=observation,
        qc_profile=_required_text(payload["qc_profile"], "qc_profile"),
        corridor_half_width_m=payload["corridor_half_width_m"],  # type: ignore[arg-type]
        transect_manifest_path=_resolve_path(base, payload["transect_manifest_path"]),
        annotation_directory=_resolve_path(base, payload["annotation_directory"]),
        output_directory=_resolve_path(base, payload["output_directory"]),
        reference_imagery=imagery,
        packet_configuration=_configuration(
            payload["packet_configuration"], "packet_configuration"
        ),
        sensitivity_grid=SensitivityGrid(
            notice=_required_text(grid_raw["notice"], "sensitivity_grid.notice"),
            extent_class_sets=tuple(
                tuple(values)
                for values in _sequence(
                    grid_raw["extent_class_sets"],
                    "sensitivity_grid.extent_class_sets",
                )
            ),
            station_bin_widths_m=tuple(
                _sequence(
                    grid_raw["station_bin_widths_m"],
                    "sensitivity_grid.station_bin_widths_m",
                )
            ),  # type: ignore[arg-type]
            min_extent_pixels_per_bin=tuple(
                _sequence(
                    grid_raw["min_extent_pixels_per_bin"],
                    "sensitivity_grid.min_extent_pixels_per_bin",
                )
            ),  # type: ignore[arg-type]
            max_bridge_gaps_m=tuple(
                _sequence(
                    grid_raw["max_bridge_gaps_m"],
                    "sensitivity_grid.max_bridge_gaps_m",
                )
            ),  # type: ignore[arg-type]
        ),
        transect_proposals=TransectProposalSettings(
            enabled=proposal_raw["enabled"],  # type: ignore[arg-type]
            count=proposal_raw["count"],  # type: ignore[arg-type]
            stratification_axis=proposal_raw["stratification_axis"],  # type: ignore[arg-type]
            transect_azimuth_degrees=proposal_raw["transect_azimuth_degrees"],  # type: ignore[arg-type]
            line_length_m=proposal_raw["line_length_m"],  # type: ignore[arg-type]
        ),
        source_path=source,
    )


def expand_sensitivity_grid(
    grid: SensitivityGrid,
) -> tuple[CandidateIntervalConfiguration, ...]:
    """Expand four axes in declared order without sorting or ranking."""

    if not isinstance(grid, SensitivityGrid):
        raise TypeError("grid must be a SensitivityGrid.")
    return tuple(
        CandidateIntervalConfiguration(
            extent_classes=extent_classes,
            station_bin_width_m=bin_width,
            min_extent_pixels_per_bin=minimum,
            max_bridge_gap_m=bridge_gap,
        )
        for extent_classes, bin_width, minimum, bridge_gap in product(
            grid.extent_class_sets,
            grid.station_bin_widths_m,
            grid.min_extent_pixels_per_bin,
            grid.max_bridge_gaps_m,
        )
    )


def load_transect_manifest(path: str | os.PathLike[str]) -> TransectManifest:
    """Load an ordered EPSG:4326 GeoJSON transect manifest."""

    source = Path(path).expanduser().resolve()
    payload = _read_json_object(source, "transect manifest")
    if payload.get("type") != "FeatureCollection":
        raise BenchmarkError("Transect manifest type must be FeatureCollection.")
    crs_raw = _mapping(payload.get("crs"), "transect manifest crs")
    crs_properties = _mapping(
        crs_raw.get("properties"), "transect manifest crs properties"
    )
    features = _sequence(payload.get("features"), "transect manifest features")
    transects: list[BenchmarkTransect] = []
    for index, raw in enumerate(features, start=1):
        feature = _mapping(raw, f"transect feature {index}")
        if feature.get("type") != "Feature":
            raise BenchmarkError(f"Transect feature {index} must have type Feature.")
        properties = _mapping(
            feature.get("properties"), f"transect feature {index} properties"
        )
        try:
            geometry = shape(feature.get("geometry"))
        except (TypeError, ValueError) as exc:
            raise BenchmarkError(
                f"Transect feature {index} has invalid geometry."
            ) from exc
        transects.append(
            BenchmarkTransect(
                transect_id=_required_text(
                    properties.get("transect_id"),
                    f"transect feature {index} transect_id",
                ),
                geometry=geometry,  # type: ignore[arg-type]
                corridor_half_width_m=properties.get("corridor_half_width_m"),  # type: ignore[arg-type]
                review_status=_required_text(
                    properties.get("review_status"),
                    f"transect feature {index} review_status",
                ),
                site_label=_optional_text(properties.get("site_label")),
                morphology_label=_optional_text(properties.get("morphology_label")),
                selection_notes=_optional_text(properties.get("selection_notes")),
                proposal_diagnostics=(
                    dict(
                        _mapping(
                            properties["proposal_diagnostics"],
                            "proposal diagnostics",
                        )
                    )
                    if properties.get("proposal_diagnostics") is not None
                    else None
                ),
            )
        )
    return TransectManifest(
        benchmark_id=_required_text(payload.get("benchmark_id"), "benchmark_id"),
        transects=tuple(transects),
        schema_version=_required_text(
            payload.get("schema_version"), "transect manifest schema_version"
        ),
        crs=_required_text(crs_properties.get("name"), "transect manifest crs name"),
    )


def write_transect_manifest(
    manifest: TransectManifest,
    path: str | os.PathLike[str],
) -> Path:
    """Write a small, deterministic GeoJSON transect manifest."""

    if not isinstance(manifest, TransectManifest):
        raise TypeError("manifest must be a TransectManifest.")
    destination = Path(path).expanduser().resolve()
    _write_json(destination, manifest.as_geojson())
    return destination


def propose_candidate_transects(
    data: PixelData,
    *,
    benchmark_id: str,
    settings: TransectProposalSettings,
    corridor_half_width_m: float,
    diagnostic_configuration: CandidateIntervalConfiguration,
) -> TransectManifest:
    """Propose systematic review locations from coarse PIXC spatial evidence.

    Proposal centers are selected from equal spatial strata along one explicit
    longitude/latitude axis. The transect orientation and length are supplied
    by the caller. Diagnostics describe PIXC support only; they are not branch,
    bank, width, centerline, or morphology interpretations. Every output remains
    unapproved until a scientist edits its review status in the manifest.
    """

    _nonempty_text(benchmark_id, "benchmark_id")
    if not isinstance(settings, TransectProposalSettings):
        raise TypeError("settings must be a TransectProposalSettings instance.")
    if not settings.enabled:
        raise BenchmarkError(
            "Candidate transect proposal is disabled by configuration."
        )
    _positive_number(corridor_half_width_m, "corridor_half_width_m")
    if not isinstance(diagnostic_configuration, CandidateIntervalConfiguration):
        raise TypeError(
            "diagnostic_configuration must be a CandidateIntervalConfiguration."
        )

    resolved = resolve_pixel_input(data)
    dataset = resolved.dataset
    if "classification" not in dataset:
        raise BenchmarkError(
            "Candidate transect proposals require PIXC classification evidence."
        )
    longitude = _valid_coordinate_values(dataset, "longitude")
    latitude = _valid_coordinate_values(dataset, "latitude")
    valid = longitude[1] & latitude[1]
    if np.count_nonzero(valid) < settings.count:
        raise BenchmarkError(
            "Not enough valid PIXC coordinates to create the requested number "
            "of distinct proposal centers."
        )
    longitudes = longitude[0][valid]
    latitudes = latitude[0][valid]
    axis_values = (
        latitudes if settings.stratification_axis == "latitude" else longitudes
    )
    lower = float(np.min(axis_values))
    upper = float(np.max(axis_values))
    if lower == upper:
        raise BenchmarkError(
            f"PIXC {settings.stratification_axis} has no range for spatial strata."
        )
    edges = np.linspace(lower, upper, settings.count + 1)
    used: set[int] = set()
    transects: list[BenchmarkTransect] = []
    half_length = settings.line_length_m / 2.0
    for index in range(settings.count):
        in_stratum = (axis_values >= edges[index]) & (
            axis_values <= edges[index + 1]
            if index == settings.count - 1
            else axis_values < edges[index + 1]
        )
        offsets = np.flatnonzero(in_stratum)
        target = float((edges[index] + edges[index + 1]) / 2.0)
        if offsets.size:
            center_lon = float(np.median(longitudes[offsets]))
            center_lat = float(np.median(latitudes[offsets]))
            center_source = "median PIXC coordinate in equal spatial stratum"
        else:
            order = np.argsort(np.abs(axis_values - target), kind="stable")
            selected = next(
                (int(item) for item in order if int(item) not in used), None
            )
            if selected is None:  # pragma: no cover - count guard makes this defensive
                raise BenchmarkError("Could not select a distinct proposal center.")
            used.add(selected)
            center_lon = float(longitudes[selected])
            center_lat = float(latitudes[selected])
            center_source = "nearest PIXC coordinate to empty spatial stratum"

        start_lon, start_lat, _ = _GEOD.fwd(
            center_lon,
            center_lat,
            settings.transect_azimuth_degrees + 180.0,
            half_length,
        )
        end_lon, end_lat, _ = _GEOD.fwd(
            center_lon,
            center_lat,
            settings.transect_azimuth_degrees,
            half_length,
        )
        geometry = LineString(((start_lon, start_lat), (end_lon, end_lat)))
        sample = sample_transect(
            data,
            geometry,
            corridor_half_width_m=corridor_half_width_m,
        )
        candidate = infer_candidate_wet_intervals(
            sample,
            **diagnostic_configuration.as_dict(),
        )
        unsampled_fraction = (
            candidate.unsampled_bin_count / candidate.bin_count
            if candidate.bin_count
            else None
        )
        diagnostics: dict[str, object] = {
            "proposal_method": "equal_spatial_strata_fixed_azimuth_v1",
            "stratification_axis": settings.stratification_axis,
            "stratum_index": index + 1,
            "stratum_bounds": [float(edges[index]), float(edges[index + 1])],
            "center_source": center_source,
            "center_longitude": center_lon,
            "center_latitude": center_lat,
            "selected_pixc_count": sample.selected_pixel_count,
            "classification_diversity": len(sample.classification_counts),
            "preliminary_wet_support_component_count": _observed_run_count(candidate),
            "unsampled_bin_fraction": unsampled_fraction,
            "candidate_diagnostic_configuration": (diagnostic_configuration.as_dict()),
            "interpretation_warning": (
                "PIXC evidence diversity diagnostic only; no physical branch, "
                "bank, width, centerline, or optimal orientation is inferred."
            ),
        }
        reason = (
            f"Systematic {settings.stratification_axis} stratum {index + 1}/"
            f"{settings.count}; {sample.selected_pixel_count} corridor pixels, "
            f"{len(sample.classification_counts)} observed classes, "
            f"{_observed_run_count(candidate)} preliminary wet-support "
            f"component(s), unsampled fraction "
            f"{_format_optional(unsampled_fraction)}. Requires scientist review."
        )
        transects.append(
            BenchmarkTransect(
                transect_id=f"T{index + 1:03d}",
                geometry=geometry,
                corridor_half_width_m=float(corridor_half_width_m),
                review_status=PROPOSED_TRANSECT_STATUS,
                selection_notes=reason,
                proposal_diagnostics=diagnostics,
            )
        )
    return TransectManifest(benchmark_id=benchmark_id, transects=tuple(transects))


def station_frame_hash(sample: TransectSample) -> str:
    """Hash the immutable station-frame fields used by manual intervals."""

    if not isinstance(sample, TransectSample):
        raise TypeError("sample must be a TransectSample.")
    payload = {
        "transect_wkt": sample.transect.wkt,
        "transect_length_m": sample.transect_length_m,
        "metric_crs_wkt": sample.local_crs.to_wkt(),
        "projection_center": list(sample.projection_center),
        "corridor_half_width_m": sample.corridor_half_width_m,
        "profile_name": sample.profile_name,
        "profile_status": sample.profile_status,
        "source_identity": [
            {
                "source_index": source.source_index,
                "granule_id": source.granule_id,
                "tile": source.tile,
                "cycle": source.cycle,
                "pass": source.pass_number,
            }
            for source in sample.sources
        ],
    }
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def generate_annotation_packets(
    data: PixelData,
    config: BenchmarkConfig,
    manifest: TransectManifest,
) -> BenchmarkPreparationResult:
    """Create independent-review packets and cache one sample per transect."""

    if not isinstance(config, BenchmarkConfig):
        raise TypeError("config must be a BenchmarkConfig.")
    if not isinstance(manifest, TransectManifest):
        raise TypeError("manifest must be a TransectManifest.")
    if manifest.benchmark_id != config.benchmark_id:
        raise BenchmarkError(
            "Transect manifest benchmark_id does not match the configuration."
        )
    packet_root = config.output_directory / "annotation_packets"
    packet_root.mkdir(parents=True, exist_ok=True)
    packets: list[AnnotationPacket] = []
    for transect in manifest.transects:
        sample = sample_transect(
            data,
            transect.geometry,
            corridor_half_width_m=transect.corridor_half_width_m,
        )
        candidate = infer_candidate_wet_intervals(
            sample,
            **config.packet_configuration.as_dict(),
        )
        directory = packet_root / transect.transect_id
        directory.mkdir(parents=True, exist_ok=True)
        planview = directory / "planview_pixc.png"
        classification = directory / "transect_classification.png"
        candidate_figure = directory / "candidate_diagnostic.png"
        evidence = directory / "station_evidence.csv"
        template = directory / "annotation_template.json"
        metadata = directory / "metadata.json"

        plan_axes = _packet_planview(data, sample, transect)
        _save_figure(plan_axes.figure, planview)
        station_axes = plot_transect_classification(
            sample,
            title=(
                f"{transect.transect_id}: station classification evidence\n"
                "Analyst boundaries are not inferred"
            ),
        )
        station_axes.set_xlim(0.0, sample.transect_length_m)
        _save_figure(station_axes.figure, classification)
        candidate_axes = plot_candidate_wet_interval_inference(
            candidate,
            title=(
                f"{transect.transect_id}: separate Phase 5A.2 candidate diagnostic\n"
                "NOT MANUAL TRUTH"
            ),
        )
        _save_figure(candidate_axes.figure, candidate_figure)
        candidate.bins_dataframe().to_csv(evidence, index=False)
        frame_hash = station_frame_hash(sample)
        _write_json(
            template,
            _blank_annotation_payload(
                config=config,
                transect=transect,
                station_frame_hash_value=frame_hash,
            ),
        )
        _write_json(
            metadata,
            _packet_metadata(
                config=config,
                transect=transect,
                sample=sample,
                candidate=candidate,
                station_frame_hash_value=frame_hash,
            ),
        )
        files = (
            planview,
            classification,
            candidate_figure,
            evidence,
            template,
            metadata,
        )
        packets.append(
            AnnotationPacket(
                transect=transect,
                sample=sample,
                directory=directory,
                files=files,
            )
        )
    packet_index = packet_root / "packet_index.json"
    _write_json(
        packet_index,
        {
            "benchmark_id": config.benchmark_id,
            "benchmark_subset": config.benchmark_subset,
            "packet_count": len(packets),
            "candidate_layer_warning": (
                "Phase 5A.2 candidate support is a separate diagnostic and is "
                "not independent reference truth."
            ),
            "independent_imagery": config.reference_imagery.as_dict(),
            "packets": [
                {
                    "transect_id": packet.transect.transect_id,
                    "review_status": packet.transect.review_status,
                    "directory": str(packet.directory),
                }
                for packet in packets
            ],
        },
    )
    preparation_manifest = config.output_directory / "preparation_manifest.json"
    packet_outputs = tuple(path for packet in packets for path in packet.files)
    manifest_payload = _run_manifest(
        config=config,
        manifest=manifest,
        samples={packet.transect.transect_id: packet.sample for packet in packets},
        annotations=(),
        configurations=config.sensitivity_configurations,
        evaluation_count=0,
        outputs=(*packet_outputs, packet_index, preparation_manifest),
    )
    manifest_payload.update(
        {
            "stage": "annotation_packet_preparation",
            "packet_count": len(packets),
            "quantitative_validation_ran": False,
            "ground_truth_fabricated": False,
            "status_message": ANNOTATION_REQUIRED_MESSAGE,
        }
    )
    _write_json(preparation_manifest, manifest_payload)
    return BenchmarkPreparationResult(
        packets=tuple(packets),
        transect_manifest_path=config.transect_manifest_path,
        preparation_manifest_path=preparation_manifest,
    )


def save_manual_annotation(
    sample: TransectSample,
    template_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    annotation_id: str,
    analyst_id: str,
    manual_intervals: Iterable[Sequence[float]],
    confidence: str | None = None,
    notes: str | None = None,
    reference_imagery: ReferenceImageryMetadata | None = None,
    confirmed: bool,
) -> ManualBenchmarkRecord:
    """Validate explicit analyst intervals through Phase 5A.1 and save inputs.

    The saved annotation contains only analyst-entered interval boundaries and
    descriptive metadata. Derived widths, spans, and gaps are deliberately not
    serialized as editable input fields.
    """

    if not confirmed:
        raise BenchmarkError("Manual annotation must be visibly confirmed before save.")
    template = _read_json_object(
        Path(template_path).expanduser().resolve(), "annotation template"
    )
    _validate_annotation_input_keys(template, pending=True)
    if template.get("review_status") != APPROVED_TRANSECT_STATUS:
        raise BenchmarkError(
            "Manual annotation requires a packet regenerated from an explicitly "
            "approved transect manifest."
        )
    expected_hash = station_frame_hash(sample)
    if template.get("station_frame_hash") != expected_hash:
        raise BenchmarkError(
            "Annotation template station frame does not match this TransectSample."
        )
    intervals = tuple(tuple(item) for item in manual_intervals)
    reference = measure_explicit_wet_intervals(sample, intervals)
    imagery = reference_imagery or _parse_imagery(
        template.get("reference_imagery"), base=Path(template_path).resolve().parent
    )
    metadata = ManualBenchmarkMetadata(
        benchmark_id=_required_text(template.get("benchmark_id"), "benchmark_id"),
        transect_id=_required_text(template.get("transect_id"), "transect_id"),
        annotation_id=_required_text(annotation_id, "annotation_id"),
        observation_identifier=_optional_text(template.get("observation_identifier")),
        observation_date=_optional_text(template.get("observation_date")),
        site_label=_optional_text(template.get("site_label")),
        analyst_id=_required_text(analyst_id, "analyst_id"),
        reference_source_description=imagery.source_provider,
        reference_acquisition_date=imagery.acquisition_date,
        temporal_offset_hours=imagery.temporal_offset_hours,
        annotation_confidence=confidence,
        notes=notes,
    )
    record = create_manual_benchmark_record(metadata, reference)
    payload = {
        **template,
        "status": "complete",
        "confirmed": True,
        "annotation_id": annotation_id,
        "analyst_id": analyst_id,
        "manual_intervals": [list(item) for item in intervals],
        "confidence": confidence,
        "notes": notes,
        "reference_imagery": imagery.as_dict(),
    }
    _write_json(Path(output_path).expanduser().resolve(), payload)
    return record


def load_manual_annotations(
    directory: str | os.PathLike[str],
    *,
    benchmark_id: str,
) -> tuple[ManualAnnotation, ...]:
    """Load completed analyst boundary files in stable filename order."""

    _nonempty_text(benchmark_id, "benchmark_id")
    source = Path(directory).expanduser().resolve()
    if not source.exists():
        return ()
    if not source.is_dir():
        raise BenchmarkError(f"Annotation path is not a directory: {source}")
    annotations: list[ManualAnnotation] = []
    for path in sorted(source.glob("*.json"), key=lambda item: item.name.casefold()):
        payload = _read_json_object(path, "manual annotation")
        status = payload.get("status")
        if status == "pending":
            _validate_annotation_input_keys(payload, pending=True)
            continue
        _validate_annotation_input_keys(payload, pending=False)
        if status != "complete" or payload.get("confirmed") is not True:
            raise BenchmarkError(
                f"Annotation {path.name!r} must be status='complete' and "
                "confirmed=true."
            )
        if payload.get("benchmark_id") != benchmark_id:
            raise BenchmarkError(
                f"Annotation {path.name!r} benchmark_id does not match "
                f"{benchmark_id!r}."
            )
        raw_intervals = payload.get("manual_intervals")
        if raw_intervals is None:
            raise BenchmarkError(
                f"Completed annotation {path.name!r} must provide manual_intervals; "
                "an explicitly confirmed empty list is valid."
            )
        intervals = tuple(
            tuple(item) for item in _sequence(raw_intervals, "manual_intervals")
        )
        annotations.append(
            ManualAnnotation(
                benchmark_id=benchmark_id,
                transect_id=_required_text(payload.get("transect_id"), "transect_id"),
                annotation_id=_required_text(
                    payload.get("annotation_id"), "annotation_id"
                ),
                analyst_id=_required_text(payload.get("analyst_id"), "analyst_id"),
                benchmark_subset=_benchmark_subset(payload.get("benchmark_subset")),
                manual_intervals=intervals,  # type: ignore[arg-type]
                confidence=_optional_text(payload.get("confidence")),
                notes=_optional_text(payload.get("notes")),
                reference_imagery=_parse_imagery(
                    payload.get("reference_imagery"), base=path.parent
                ),
                source_path=path,
                station_frame_hash=_required_text(
                    payload.get("station_frame_hash"), "station_frame_hash"
                ),
            )
        )
    identifiers = [item.annotation_id for item in annotations]
    if len(set(identifiers)) != len(identifiers):
        raise BenchmarkError("annotation_id values must be unique across files.")
    return tuple(annotations)


def run_benchmark(
    config: BenchmarkConfig,
    manifest: TransectManifest,
    samples: Mapping[str, TransectSample],
    annotations: Sequence[ManualAnnotation],
) -> BenchmarkRunResult:
    """Run the ordered Phase 5A.2/5A.3a experiment for manual references.

    The observation, QC result, and transect samples are supplied once. One
    Phase 5A.2 sweep is cached per transect and reused for multiple independent
    analysts. Proposed transects are rejected from quantitative validation.
    """

    if not isinstance(config, BenchmarkConfig):
        raise TypeError("config must be a BenchmarkConfig.")
    if not isinstance(manifest, TransectManifest):
        raise TypeError("manifest must be a TransectManifest.")
    if manifest.benchmark_id != config.benchmark_id:
        raise BenchmarkError(
            "Transect manifest benchmark_id does not match the configuration."
        )
    if isinstance(samples, (str, bytes, bytearray)) or not isinstance(samples, Mapping):
        raise TypeError("samples must map transect_id to TransectSample.")
    if isinstance(annotations, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError("annotations must be an ordered sequence.")
    annotations = tuple(annotations)
    if not annotations:
        raise AnnotationRequiredError(ANNOTATION_REQUIRED_MESSAGE)

    transect_lookup = {item.transect_id: item for item in manifest.transects}
    annotation_lookup: dict[str, list[ManualAnnotation]] = {}
    annotation_ids: set[str] = set()
    for annotation in annotations:
        if not isinstance(annotation, ManualAnnotation):
            raise TypeError("annotations must contain ManualAnnotation instances.")
        if annotation.annotation_id in annotation_ids:
            raise BenchmarkError(
                f"Duplicate annotation_id {annotation.annotation_id!r}."
            )
        annotation_ids.add(annotation.annotation_id)
        if annotation.benchmark_id != config.benchmark_id:
            raise BenchmarkError(
                f"Annotation {annotation.annotation_id!r} has the wrong benchmark_id."
            )
        transect = transect_lookup.get(annotation.transect_id)
        if transect is None:
            raise BenchmarkError(
                f"Annotation {annotation.annotation_id!r} refers to unknown "
                f"transect {annotation.transect_id!r}."
            )
        if not transect.approved:
            raise BenchmarkError(
                f"Transect {transect.transect_id!r} is {transect.review_status!r}; "
                "quantitative validation requires explicit owner approval."
            )
        annotation_lookup.setdefault(annotation.transect_id, []).append(annotation)

    configurations = config.sensitivity_configurations
    sensitivity_by_transect: dict[str, CandidateIntervalSensitivityResult] = {}
    result_frames: list[pd.DataFrame] = []
    benchmark_records: list[ManualBenchmarkRecord] = []
    for transect in manifest.transects:
        transect_annotations = annotation_lookup.get(transect.transect_id, [])
        if not transect_annotations:
            continue
        sample = samples.get(transect.transect_id)
        if not isinstance(sample, TransectSample):
            raise BenchmarkError(
                f"A TransectSample is required for annotated transect "
                f"{transect.transect_id!r}."
            )
        sensitivity = run_candidate_interval_sensitivity(
            sample,
            configurations=configurations,
        )
        sensitivity_by_transect[transect.transect_id] = sensitivity
        candidate_summary = sensitivity.dataframe()
        for annotation in transect_annotations:
            current_hash = station_frame_hash(sample)
            if annotation.station_frame_hash != current_hash:
                raise BenchmarkError(
                    f"Annotation {annotation.annotation_id!r} was made in a "
                    "different station frame. Regenerate its packet and annotate "
                    "again; boundaries are never transformed automatically."
                )
            reference = measure_explicit_wet_intervals(
                sample, annotation.manual_intervals
            )
            metadata = ManualBenchmarkMetadata(
                benchmark_id=annotation.benchmark_id,
                transect_id=annotation.transect_id,
                annotation_id=annotation.annotation_id,
                observation_identifier=config.observation.identifier,
                observation_date=config.observation.date,
                site_label=transect.site_label,
                analyst_id=annotation.analyst_id,
                reference_source_description=(
                    annotation.reference_imagery.source_provider
                ),
                reference_acquisition_date=(
                    annotation.reference_imagery.acquisition_date
                ),
                temporal_offset_hours=(
                    annotation.reference_imagery.temporal_offset_hours
                ),
                annotation_confidence=annotation.confidence,
                notes=annotation.notes,
            )
            benchmark_records.append(
                create_manual_benchmark_record(metadata, reference)
            )
            validation = evaluate_sensitivity_against_explicit(reference, sensitivity)
            frame = validation.dataframe()
            for column in (
                "bin_count",
                "candidate_wet_bin_count",
                "sampled_noneligible_bin_count",
                "unsampled_bin_count",
                "candidate_interval_count",
                "bridge_count",
            ):
                frame[column] = candidate_summary[column].to_numpy(copy=True)
            identifiers: tuple[tuple[str, object], ...] = (
                ("benchmark_id", config.benchmark_id),
                ("benchmark_subset", annotation.benchmark_subset),
                ("transect_id", transect.transect_id),
                ("annotation_id", annotation.annotation_id),
                ("analyst_id", annotation.analyst_id),
                ("observation_identifier", config.observation.identifier),
                ("observation_date", config.observation.date),
                ("cycle", config.observation.cycle),
                ("pass", config.observation.pass_number),
                ("tiles", tuple(config.observation.tiles)),
                ("qc_profile", config.qc_profile),
                ("site_label", transect.site_label),
                ("morphology_label", transect.morphology_label),
                ("selection_notes", transect.selection_notes),
                ("transect_review_status", transect.review_status),
                ("annotation_file", str(annotation.source_path)),
            )
            for position, (name, value) in enumerate(identifiers):
                frame.insert(position, name, [value for _ in range(len(frame))])
            result_frames.append(frame)

    if not result_frames:
        raise AnnotationRequiredError(ANNOTATION_REQUIRED_MESSAGE)
    individual = pd.concat(result_frames, ignore_index=True)
    aggregate = aggregate_benchmark_results(individual)
    output = config.output_directory
    output.mkdir(parents=True, exist_ok=True)
    results_csv = output / "individual_validation_results.csv"
    results_json = output / "individual_validation_results.json"
    aggregate_csv = output / "aggregate_configuration_summary.csv"
    aggregate_json = output / "aggregate_configuration_summary.json"
    benchmark_records_json = output / "manual_benchmark_records.json"
    individual.to_csv(results_csv, index=False)
    aggregate.to_csv(aggregate_csv, index=False)
    _write_json(results_json, _dataframe_records(individual))
    _write_json(aggregate_json, _dataframe_records(aggregate))
    _write_json(
        benchmark_records_json,
        [record.as_dict() for record in benchmark_records],
    )
    figure_paths = _generate_benchmark_figures(individual, output / "figures")
    report = output / "phase5a3b_pilot_report.md"
    _write_report(
        report,
        config=config,
        manifest=manifest,
        annotations=annotations,
        individual=individual,
        aggregate=aggregate,
        figure_paths=figure_paths,
    )
    run_manifest = output / "run_manifest.json"
    outputs = (
        results_csv,
        results_json,
        aggregate_csv,
        aggregate_json,
        benchmark_records_json,
        report,
        *figure_paths,
        run_manifest,
    )
    _write_json(
        run_manifest,
        _run_manifest(
            config=config,
            manifest=manifest,
            samples=samples,
            annotations=annotations,
            configurations=configurations,
            evaluation_count=len(individual),
            outputs=outputs,
        ),
    )
    return BenchmarkRunResult(
        individual_results=individual.copy(deep=True),
        aggregate_results=aggregate.copy(deep=True),
        benchmark_records=tuple(benchmark_records),
        output_directory=output,
        results_csv=results_csv,
        aggregate_csv=aggregate_csv,
        report_path=report,
        manifest_path=run_manifest,
        figure_paths=figure_paths,
    )


def aggregate_benchmark_results(results: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive, unranked statistics in configuration order."""

    if not isinstance(results, pd.DataFrame):
        raise TypeError("results must be a pandas DataFrame.")
    required = {
        "configuration_id",
        "annotation_id",
        "transect_id",
        "extent_classes",
        "station_bin_width_m",
        "min_extent_pixels_per_bin",
        "max_bridge_gap_m",
        "observed_support_iou",
        "bridge_inclusive_iou",
        "observed_support_f1",
        "bridge_inclusive_f1",
        "observed_support_signed_total_width_error_m",
        "bridge_inclusive_signed_total_width_error_m",
        "observed_support_absolute_total_width_error_m",
        "bridge_inclusive_absolute_total_width_error_m",
        "observed_support_false_positive_length_m",
        "bridge_inclusive_false_positive_length_m",
        "observed_support_false_negative_length_m",
        "bridge_inclusive_false_negative_length_m",
        "observed_support_boundary_symmetric_boundary_mean_m",
        "bridge_inclusive_boundary_symmetric_boundary_mean_m",
        "observed_component_count_difference",
        "inferred_component_count_difference",
        "total_bridged_gap_m",
        "bridged_length_over_manual_wet_m",
        "bridged_length_over_manual_nonwet_m",
        "delta_iou_due_to_bridging",
        "delta_f1_due_to_bridging",
    }
    missing = required - set(results.columns)
    if missing:
        raise BenchmarkError(
            "Benchmark result table is missing columns: " + ", ".join(sorted(missing))
        )
    if results.empty:
        raise BenchmarkError("Cannot aggregate an empty benchmark result table.")

    scalar_metrics = (
        "observed_support_iou",
        "bridge_inclusive_iou",
        "observed_support_f1",
        "bridge_inclusive_f1",
        "observed_support_signed_total_width_error_m",
        "bridge_inclusive_signed_total_width_error_m",
        "observed_support_absolute_total_width_error_m",
        "bridge_inclusive_absolute_total_width_error_m",
        "observed_support_false_positive_length_m",
        "bridge_inclusive_false_positive_length_m",
        "observed_support_false_negative_length_m",
        "bridge_inclusive_false_negative_length_m",
        "observed_support_boundary_symmetric_boundary_mean_m",
        "bridge_inclusive_boundary_symmetric_boundary_mean_m",
        "observed_component_count_difference",
        "inferred_component_count_difference",
        "total_bridged_gap_m",
        "bridged_length_over_manual_wet_m",
        "bridged_length_over_manual_nonwet_m",
        "delta_iou_due_to_bridging",
        "delta_f1_due_to_bridging",
    )
    records: list[dict[str, object]] = []
    for configuration_id in results["configuration_id"].drop_duplicates().tolist():
        group = results.loc[results["configuration_id"] == configuration_id]
        first = group.iloc[0]
        record: dict[str, object] = {
            "configuration_id": configuration_id,
            "extent_classes": first["extent_classes"],
            "station_bin_width_m": first["station_bin_width_m"],
            "min_extent_pixels_per_bin": first["min_extent_pixels_per_bin"],
            "max_bridge_gap_m": first["max_bridge_gap_m"],
            "n_annotations": int(group["annotation_id"].nunique()),
            "n_transects": int(group["transect_id"].nunique()),
        }
        for metric in scalar_metrics:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(
                dtype=np.float64
            )
            finite = values[np.isfinite(values)]
            record[f"{metric}_valid_n"] = int(finite.size)
            record[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else None
            record[f"{metric}_median"] = (
                float(np.median(finite)) if finite.size else None
            )
        for source, label in (
            ("observed_component_count_difference", "observed_component_count_error"),
            ("inferred_component_count_difference", "inferred_component_count_error"),
        ):
            values = pd.to_numeric(group[source], errors="coerce").to_numpy(
                dtype=np.float64
            )
            finite = np.abs(values[np.isfinite(values)])
            record[f"{label}_mean_absolute"] = (
                float(np.mean(finite)) if finite.size else None
            )
            record[f"{label}_median_absolute"] = (
                float(np.median(finite)) if finite.size else None
            )
        records.append(record)
    return pd.DataFrame.from_records(records, columns=tuple(records[0]))


def _generate_benchmark_figures(
    results: pd.DataFrame,
    directory: Path,
) -> tuple[Path, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []
    specifications = (
        (
            "iou_across_configurations.png",
            ("observed_support_iou", "bridge_inclusive_iou"),
            "IoU",
            "IoU across caller-ordered configurations",
        ),
        (
            "f1_across_configurations.png",
            ("observed_support_f1", "bridge_inclusive_f1"),
            "F1",
            "F1 across caller-ordered configurations",
        ),
        (
            "signed_total_width_error.png",
            (
                "observed_support_signed_total_width_error_m",
                "bridge_inclusive_signed_total_width_error_m",
            ),
            "Signed total-width error (m)",
            "Signed total-width error (candidate minus manual)",
        ),
        (
            "absolute_total_width_error.png",
            (
                "observed_support_absolute_total_width_error_m",
                "bridge_inclusive_absolute_total_width_error_m",
            ),
            "Absolute total-width error (m)",
            "Absolute total-width error",
        ),
        (
            "boundary_distance_error.png",
            (
                "observed_support_boundary_symmetric_boundary_mean_m",
                "bridge_inclusive_boundary_symmetric_boundary_mean_m",
            ),
            "Symmetric mean boundary distance (m)",
            "Boundary-distance diagnostics",
        ),
        (
            "component_count_difference.png",
            (
                "observed_component_count_difference",
                "inferred_component_count_difference",
            ),
            "Candidate components minus manual intervals",
            "Component-count difference",
        ),
    )
    for filename, columns, ylabel, title in specifications:
        path = directory / filename
        _metric_figure(
            results,
            columns=columns,
            ylabel=ylabel,
            title=title,
            path=path,
        )
        figures.append(path)

    bridge_path = directory / "bridge_effect.png"
    _bridge_figure(results, bridge_path)
    figures.append(bridge_path)
    per_transect = directory / "per_transect"
    per_transect.mkdir(parents=True, exist_ok=True)
    for transect_id in results["transect_id"].drop_duplicates().tolist():
        path = per_transect / f"{_safe_filename(str(transect_id))}_comparison.png"
        _per_transect_figure(
            results.loc[results["transect_id"] == transect_id],
            transect_id=str(transect_id),
            path=path,
        )
        figures.append(path)
    return tuple(figures)


def _metric_figure(
    results: pd.DataFrame,
    *,
    columns: tuple[str, str],
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11.5, 5.5), constrained_layout=True)
    x = pd.to_numeric(results["configuration_id"], errors="coerce").to_numpy()
    labels = {
        columns[0]: "observed candidate support",
        columns[1]: "bridge-inclusive result",
    }
    colors = ("#0072b2", "#d55e00")
    for column, color in zip(columns, colors, strict=True):
        y = pd.to_numeric(results[column], errors="coerce").to_numpy()
        axis.scatter(
            x,
            y,
            s=14,
            alpha=0.55,
            linewidths=0,
            color=color,
            label=labels[column],
            rasterized=True,
        )
    axis.axhline(0.0, color="#777777", linewidth=0.7, zorder=0)
    axis.set_xlabel("Configuration ID (caller order; not rank)")
    axis.set_ylabel(ylabel)
    axis.set_title(f"{title}\nPilot descriptive distributions; no winner selected")
    axis.grid(True, color="#e0e0e0", linewidth=0.6)
    axis.legend(loc="best")
    _save_figure(figure, path)


def _bridge_figure(results: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1, 2, figsize=(13.0, 5.5), constrained_layout=True, sharex=True
    )
    x = pd.to_numeric(results["configuration_id"], errors="coerce").to_numpy()
    delta = pd.to_numeric(
        results["delta_iou_due_to_bridging"], errors="coerce"
    ).to_numpy()
    axes[0].scatter(x, delta, s=14, alpha=0.55, linewidths=0, color="#6a3d9a")
    axes[0].axhline(0.0, color="#777777", linewidth=0.8)
    axes[0].set_ylabel("Delta IoU due to bridging")
    axes[0].set_title("Bridge change in IoU")
    for column, label, color in (
        ("bridged_length_over_manual_wet_m", "bridge over manual wet", "#009e73"),
        (
            "bridged_length_over_manual_nonwet_m",
            "bridge over manual non-wet",
            "#cc79a7",
        ),
    ):
        axes[1].scatter(
            x,
            pd.to_numeric(results[column], errors="coerce").to_numpy(),
            s=14,
            alpha=0.55,
            linewidths=0,
            label=label,
            color=color,
        )
    axes[1].set_ylabel("Bridged length (m)")
    axes[1].set_title("Where accepted bridges fall")
    axes[1].legend(loc="best")
    for axis in axes:
        axis.set_xlabel("Configuration ID (caller order; not rank)")
        axis.grid(True, color="#e0e0e0", linewidth=0.6)
    figure.suptitle("Bridge effects; observed support remains separate")
    _save_figure(figure, path)


def _per_transect_figure(
    results: pd.DataFrame,
    *,
    transect_id: str,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), constrained_layout=True)
    metrics = (
        ("observed_support_iou", "Observed-support IoU"),
        ("bridge_inclusive_iou", "Bridge-inclusive IoU"),
        (
            "bridge_inclusive_signed_total_width_error_m",
            "Bridge-inclusive signed width error (m)",
        ),
        ("delta_iou_due_to_bridging", "Delta IoU due to bridging"),
    )
    annotations = results["annotation_id"].drop_duplicates().tolist()
    colors = plt.get_cmap("tab10")
    for axis, (column, label) in zip(axes.ravel(), metrics, strict=True):
        for index, annotation_id in enumerate(annotations):
            selected = results.loc[results["annotation_id"] == annotation_id]
            axis.scatter(
                selected["configuration_id"],
                pd.to_numeric(selected[column], errors="coerce"),
                s=14,
                alpha=0.6,
                linewidths=0,
                label=str(annotation_id),
                color=colors(index % 10),
                rasterized=True,
            )
        axis.set_xlabel("Configuration ID (caller order; not rank)")
        axis.set_ylabel(label)
        axis.grid(True, color="#e0e0e0", linewidth=0.6)
    axes[0, 0].legend(title="Independent annotation", loc="best")
    figure.suptitle(
        f"{transect_id}: all configured sensitivity results\n"
        "Descriptive comparison only; no preferred setting selected"
    )
    _save_figure(figure, path)


def _write_report(
    path: Path,
    *,
    config: BenchmarkConfig,
    manifest: TransectManifest,
    annotations: Sequence[ManualAnnotation],
    individual: pd.DataFrame,
    aggregate: pd.DataFrame,
    figure_paths: Sequence[Path],
) -> None:
    summary_columns = (
        "configuration_id",
        "extent_classes",
        "station_bin_width_m",
        "min_extent_pixels_per_bin",
        "max_bridge_gap_m",
        "n_annotations",
        "n_transects",
        "observed_support_iou_median",
        "bridge_inclusive_iou_median",
        "bridge_inclusive_absolute_total_width_error_m_mean",
    )
    annotation_lines = [
        (
            f"- `{item.annotation_id}` — transect `{item.transect_id}`, analyst "
            f"`{item.analyst_id}`, subset `{item.benchmark_subset}`"
        )
        for item in annotations
    ]
    transect_lookup = {item.transect_id: item for item in manifest.transects}
    morphology_lines = []
    for identifier in individual["transect_id"].drop_duplicates().tolist():
        item = transect_lookup[str(identifier)]
        morphology_lines.append(
            f"- `{item.transect_id}` — site: {item.site_label or 'unlabeled'}; "
            f"morphology: {item.morphology_label or 'unlabeled'}; selection: "
            f"{item.selection_notes or 'not supplied'}"
        )
    undefined_iou = int(individual["observed_support_iou"].isna().sum())
    unsampled_rows = int((individual["unsampled_bin_count"] > 0).sum())
    component_mismatch_rows = int(
        (individual["inferred_component_count_difference"] != 0).sum()
    )
    false_positive_rows = int(
        (individual["bridge_inclusive_false_positive_length_m"] > 0).sum()
    )
    false_negative_rows = int(
        (individual["bridge_inclusive_false_negative_length_m"] > 0).sum()
    )
    positive_bridge = int((individual["delta_iou_due_to_bridging"] > 0).sum())
    negative_bridge = int((individual["delta_iou_due_to_bridging"] < 0).sum())
    analysts_per_transect = individual.groupby("transect_id", sort=False)[
        "analyst_id"
    ].nunique()
    multi_analyst_transects = int((analysts_per_transect > 1).sum())
    relative_figures = [
        path_item.relative_to(config.output_directory).as_posix()
        for path_item in figure_paths
    ]
    lines = [
        "# Phase 5A.3b pilot benchmark report",
        "",
        "> Pilot sensitivity description only. The configuration order is the "
        "caller order, not a ranking, and no recommended configuration is selected.",
        "",
        "## Inputs",
        "",
        f"- Benchmark: `{config.benchmark_id}` (`{config.benchmark_subset}`)",
        f"- Observation: `{config.observation.identifier}` on "
        f"`{config.observation.date}`, cycle {config.observation.cycle}, pass "
        f"{config.observation.pass_number}, tiles "
        f"{', '.join(config.observation.tiles)}",
        f"- AOI: `{config.observation.aoi}` (EPSG:4326)",
        f"- QC profile: `{config.qc_profile}`",
        f"- Sensitivity configurations: {len(config.sensitivity_configurations)}",
        f"- Independent annotations: {len(annotations)}",
        f"- Transects with multiple independent analysts: {multi_analyst_transects}",
        "",
        "## Explicit sensitivity grid",
        "",
        "```json",
        json.dumps(config.sensitivity_grid.as_dict(), indent=2),
        "```",
        "",
        "## Independent annotations",
        "",
        *annotation_lines,
        "",
        "## Transect selection context",
        "",
        *morphology_lines,
        "",
        "## Metric definitions",
        "",
        "IoU, precision, recall, F1, false-positive length, and false-negative "
        "length compare continuous station-interval sets. Signed total-width "
        "error is candidate minus manual width. Boundary diagnostics use nearest "
        "boundary distances. Component-count difference is candidate components "
        "minus manual intervals. Observed candidate support and bridge-inclusive "
        "intervals remain separate in every table.",
        "",
        "## Aggregate configuration table",
        "",
        _markdown_table(aggregate.loc[:, summary_columns]),
        "",
        "## Diagnostic figures",
        "",
        *[f"- `{item}`" for item in relative_figures],
        "",
        "## Data-gap and bridge behavior",
        "",
        f"- {unsampled_rows} annotation/configuration rows contained one or more "
        "unsampled station bins; unsampled bins remain unknown, not dry.",
        f"- Observed-support IoU was undefined in {undefined_iou} rows under the "
        "Phase 5A.3a empty-set semantics; undefined values were not changed to zero.",
        f"- Bridging increased IoU in {positive_bridge} rows and decreased it in "
        f"{negative_bridge} rows. Bridge overlap with manual wet and non-wet "
        "regions remains separately reported.",
        "",
        "## Observed failure-mode diagnostics",
        "",
        f"- {component_mismatch_rows} rows had a nonzero bridge-inclusive "
        "component-count difference.",
        f"- {false_positive_rows} rows had positive bridge-inclusive "
        "false-positive extent.",
        f"- {false_negative_rows} rows had positive bridge-inclusive "
        "false-negative extent.",
        "- These counts flag cases for scientific review; they do not rank or "
        "select configurations.",
        "",
        "## Limitations",
        "",
        "This pilot is not a final independent validation, parameter recommendation, "
        "or physical-bank accuracy claim. Transect placement, corridor width, manual "
        "interpretation, imagery timing/quality, PIXC sampling gaps, classifications, "
        "and the limited observation all affect these descriptive results. Multiple "
        "analysts remain separate; no consensus reference is synthesized. Final "
        "evaluation should use owner-designated holdout annotations that were not "
        "used during parameter selection.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_manifest(
    *,
    config: BenchmarkConfig,
    manifest: TransectManifest,
    samples: Mapping[str, TransectSample],
    annotations: Sequence[ManualAnnotation],
    configurations: Sequence[CandidateIntervalConfiguration],
    evaluation_count: int,
    outputs: Sequence[Path],
) -> dict[str, object]:
    commit, dirty = _git_state(config)
    config_hash = (
        _sha256_file(config.source_path)
        if config.source_path is not None and config.source_path.is_file()
        else _sha256_bytes(_canonical_json(config.as_dict()).encode("utf-8"))
    )
    manifest_hash = (
        _sha256_file(config.transect_manifest_path)
        if config.transect_manifest_path.is_file()
        else _sha256_bytes(_canonical_json(manifest.as_geojson()).encode("utf-8"))
    )
    annotation_hashes = [
        {
            "annotation_id": item.annotation_id,
            "path": str(item.source_path),
            "sha256": _sha256_file(item.source_path),
        }
        for item in annotations
    ]
    sources: dict[tuple[object, ...], dict[str, object]] = {}
    for sample in samples.values():
        for source in sample.sources:
            key = (source.granule_id, source.tile, source.source_index)
            sources[key] = {
                "source_index": source.source_index,
                "granule_id": source.granule_id,
                "filename": source.filename,
                "tile": source.tile,
                "cycle": source.cycle,
                "pass": source.pass_number,
                "crid": source.crid,
                "netcdf_product_version": source.netcdf_product_version,
            }
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "package_version": _package_version(),
        "git_commit": commit,
        "git_working_tree_dirty": dirty,
        "benchmark_id": config.benchmark_id,
        "benchmark_subset": config.benchmark_subset,
        "observation": config.observation.as_dict(),
        "source_provenance": list(sources.values()),
        "aoi": list(config.observation.aoi),
        "config_sha256": config_hash,
        "transect_manifest_sha256": manifest_hash,
        "annotation_file_hashes": annotation_hashes,
        "sensitivity_configurations": [
            {"configuration_id": index, **item.as_dict()}
            for index, item in enumerate(configurations, start=1)
        ],
        "configuration_count": len(configurations),
        "annotation_count": len(annotations),
        "evaluation_count": evaluation_count,
        "qc_profile": config.qc_profile,
        "output_filenames": [
            path.relative_to(config.output_directory).as_posix() for path in outputs
        ],
        "ranking_performed": False,
        "recommended_configuration_selected": False,
    }


def _blank_annotation_payload(
    *,
    config: BenchmarkConfig,
    transect: BenchmarkTransect,
    station_frame_hash_value: str,
) -> dict[str, object]:
    return {
        "schema_version": ANNOTATION_INPUT_SCHEMA_VERSION,
        "status": "pending",
        "confirmed": False,
        "benchmark_id": config.benchmark_id,
        "transect_id": transect.transect_id,
        "annotation_id": None,
        "analyst_id": None,
        "benchmark_subset": config.benchmark_subset,
        "observation_identifier": config.observation.identifier,
        "observation_date": config.observation.date,
        "site_label": transect.site_label,
        "morphology_label": transect.morphology_label,
        "review_status": transect.review_status,
        "station_frame_hash": station_frame_hash_value,
        "manual_intervals": None,
        "confidence": None,
        "notes": None,
        "reference_imagery": config.reference_imagery.as_dict(),
    }


def _packet_planview(
    data: PixelData,
    sample: TransectSample,
    transect: BenchmarkTransect,
) -> Any:
    """Plot local PIXC context without collapsing a long, narrow corridor."""

    import matplotlib.pyplot as plt

    coordinates = np.asarray(sample.transect.coords, dtype=np.float64)
    center_latitude = float(np.mean(coordinates[:, 1]))
    context_padding_m = max(
        1_000.0,
        5.0 * sample.corridor_half_width_m,
        0.15 * sample.transect_length_m,
    )
    latitude_padding = context_padding_m / 110_574.0
    longitude_scale = max(111_320.0 * math.cos(math.radians(center_latitude)), 1.0)
    longitude_padding = context_padding_m / longitude_scale
    extent = (
        max(-180.0, float(np.min(coordinates[:, 0])) - longitude_padding),
        max(-90.0, float(np.min(coordinates[:, 1])) - latitude_padding),
        min(180.0, float(np.max(coordinates[:, 0])) + longitude_padding),
        min(90.0, float(np.max(coordinates[:, 1])) + latitude_padding),
    )
    figure, axes = plt.subplots(figsize=(12.0, 6.5), constrained_layout=True)
    plot_pixc_map(
        data,
        color_by="classification",
        ax=axes,
        title=(
            f"{transect.transect_id}: local PIXC context and sampling corridor\n"
            f"{transect.review_status}; no width inference"
        ),
        point_size=0.45,
        alpha=0.8,
        extent=extent,
        show_colorbar=False,
    )
    prior_legend = axes.get_legend()
    corridor_handles = []
    boundary = sample.corridor.boundary
    boundary_parts = (
        tuple(boundary.geoms) if hasattr(boundary, "geoms") else (boundary,)
    )
    for index, part in enumerate(boundary_parts):
        x, y = part.xy
        (line,) = axes.plot(
            x,
            y,
            color="#d95f02",
            linestyle="--",
            linewidth=1.2,
            label=(
                f"±{sample.corridor_half_width_m:g} m corridor" if index == 0 else None
            ),
            zorder=4,
        )
        if index == 0:
            corridor_handles.append(line)
    x, y = sample.transect.xy
    (line_handle,) = axes.plot(
        x,
        y,
        color="#111111",
        linewidth=1.6,
        label="explicit transect",
        zorder=5,
    )
    (start_handle,) = axes.plot(
        x[0],
        y[0],
        marker="o",
        color="#009e73",
        markersize=5,
        linestyle="none",
        label="station start",
        zorder=6,
    )
    (end_handle,) = axes.plot(
        x[-1],
        y[-1],
        marker="s",
        color="#d55e00",
        markersize=5,
        linestyle="none",
        label="station end",
        zorder=6,
    )
    if prior_legend is not None:
        axes.add_artist(prior_legend)
    axes.legend(
        handles=[*corridor_handles, line_handle, start_handle, end_handle],
        title="Sampling geometry",
        loc="lower left",
    )
    return axes


def _packet_metadata(
    *,
    config: BenchmarkConfig,
    transect: BenchmarkTransect,
    sample: TransectSample,
    candidate: CandidateIntervalInferenceResult,
    station_frame_hash_value: str,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "benchmark_id": config.benchmark_id,
        "benchmark_subset": config.benchmark_subset,
        "observation": config.observation.as_dict(),
        "qc_profile": config.qc_profile,
        "transect": transect.as_feature(),
        "station_frame_hash": station_frame_hash_value,
        "sample_summary": {
            "input_pixel_count": sample.input_pixel_count,
            "selected_pixel_count": sample.selected_pixel_count,
            "invalid_coordinate_count": sample.invalid_coordinate_count,
            "outside_corridor_count": sample.outside_corridor_count,
            "transect_length_m": sample.transect_length_m,
            "corridor_half_width_m": sample.corridor_half_width_m,
            "classification_counts": sample.classification_counts,
            "source_counts": sample.source_counts,
            "source_tile_counts": sample.source_tile_counts,
        },
        "candidate_diagnostic": {
            "status": candidate.method_status,
            "warning": (
                "Separate Phase 5A.2 diagnostic only; it is not manual reference "
                "truth and must not be copied into manual_intervals."
            ),
            "configuration": config.packet_configuration.as_dict(),
            "bin_count": candidate.bin_count,
            "candidate_wet_bin_count": candidate.candidate_wet_bin_count,
            "sampled_noneligible_bin_count": (candidate.sampled_noneligible_bin_count),
            "unsampled_bin_count": candidate.unsampled_bin_count,
            "candidate_interval_count": candidate.candidate_interval_count,
            "bridge_count": candidate.bridge_count,
        },
        "reference_imagery": config.reference_imagery.as_dict(),
        "annotation_instruction": (
            "Interpret independent imagery first, then enter every manual station "
            "boundary without automatic snapping. Candidate support is diagnostic."
        ),
    }


def _validate_annotation_input_keys(
    payload: Mapping[str, object],
    *,
    pending: bool,
) -> None:
    allowed = {
        "schema_version",
        "status",
        "confirmed",
        "benchmark_id",
        "transect_id",
        "annotation_id",
        "analyst_id",
        "benchmark_subset",
        "observation_identifier",
        "observation_date",
        "site_label",
        "morphology_label",
        "review_status",
        "station_frame_hash",
        "manual_intervals",
        "confidence",
        "notes",
        "reference_imagery",
    }
    _require_keys(payload, allowed, "annotation input")
    if payload.get("schema_version") != ANNOTATION_INPUT_SCHEMA_VERSION:
        raise BenchmarkError("Unsupported annotation input schema version.")
    if pending and (
        payload.get("status") != "pending"
        or payload.get("confirmed") is not False
        or payload.get("manual_intervals") is not None
        or payload.get("annotation_id") is not None
        or payload.get("analyst_id") is not None
    ):
        raise BenchmarkError(
            "A pending annotation template must be unconfirmed with null analyst, "
            "annotation ID, and manual_intervals."
        )


def _parse_imagery(
    value: object,
    *,
    base: Path,
) -> ReferenceImageryMetadata:
    raw = _mapping(value, "reference_imagery")
    keys = {
        "status",
        "source_provider",
        "acquisition_date",
        "temporal_offset_hours",
        "cloud_quality_note",
        "local_image_path",
        "source_identifier",
        "analyst_notes",
    }
    _require_keys(raw, keys, "reference_imagery")
    local = raw.get("local_image_path")
    return ReferenceImageryMetadata(
        status=_required_text(raw.get("status"), "reference_imagery.status"),
        source_provider=_optional_text(raw.get("source_provider")),
        acquisition_date=_optional_text(raw.get("acquisition_date")),
        temporal_offset_hours=raw.get("temporal_offset_hours"),  # type: ignore[arg-type]
        cloud_quality_note=_optional_text(raw.get("cloud_quality_note")),
        local_image_path=(
            _resolve_path(base, local) if _optional_text(local) is not None else None
        ),
        source_identifier=_optional_text(raw.get("source_identifier")),
        analyst_notes=_optional_text(raw.get("analyst_notes")),
    )


def _configuration(value: object, label: str) -> CandidateIntervalConfiguration:
    raw = _mapping(value, label)
    required = {
        "extent_classes",
        "station_bin_width_m",
        "min_extent_pixels_per_bin",
        "max_bridge_gap_m",
    }
    _require_keys(raw, required, label)
    return CandidateIntervalConfiguration(
        extent_classes=tuple(
            _sequence(raw["extent_classes"], f"{label}.extent_classes")
        ),  # type: ignore[arg-type]
        station_bin_width_m=raw["station_bin_width_m"],  # type: ignore[arg-type]
        min_extent_pixels_per_bin=raw["min_extent_pixels_per_bin"],  # type: ignore[arg-type]
        max_bridge_gap_m=raw["max_bridge_gap_m"],  # type: ignore[arg-type]
    )


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BenchmarkError(f"{label.capitalize()} does not exist: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"Could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkError(f"{label.capitalize()} root must be a JSON object.")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(value), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def _require_keys(
    value: Mapping[str, object],
    required: set[str],
    label: str,
) -> None:
    actual = set(value)
    missing = required - actual
    extra = actual - required
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("unexpected " + ", ".join(sorted(extra)))
        raise BenchmarkError(
            f"{label.capitalize()} has invalid keys: " + "; ".join(details)
        )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BenchmarkError(f"{label.capitalize()} must be a JSON object.")
    if any(not isinstance(key, str) for key in value):
        raise BenchmarkError(f"{label.capitalize()} keys must be strings.")
    return value  # type: ignore[return-value]


def _sequence(value: object, label: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping, Set)):
        raise BenchmarkError(f"{label} must be an ordered JSON array.")
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise BenchmarkError(f"{label} must be an ordered JSON array.") from exc


def _text_sequence(value: object, label: str) -> tuple[str, ...]:
    items = _sequence(value, label)
    if not items:
        raise BenchmarkError(f"{label} must not be empty.")
    return tuple(_required_text(item, f"{label} item") for item in items)


def _bbox(value: object) -> tuple[float, float, float, float]:
    items = _sequence(value, "observation.aoi")
    if len(items) != 4:
        raise BenchmarkError("observation.aoi must contain exactly four values.")
    try:
        return tuple(float(item) for item in items)  # type: ignore[return-value]
    except (TypeError, ValueError) as exc:
        raise BenchmarkError("observation.aoi values must be numeric.") from exc


def _resolve_path(base: Path, value: object) -> Path:
    text = _required_text(value, "path")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{label} must be a non-empty string.")
    return value.strip()


def _nonempty_text(value: object, label: str) -> None:
    _required_text(value, label)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise BenchmarkError("Optional text fields must be strings or null.")
    text = value.strip()
    return text or None


def _benchmark_subset(value: object) -> str:
    text = _required_text(value, "benchmark_subset")
    if text not in _BENCHMARK_SUBSETS:
        raise BenchmarkError(
            "benchmark_subset must be pilot, development, holdout, or unassigned."
        )
    return text


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise BenchmarkError(f"{label} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(f"{label} must be a finite number.") from exc
    if not math.isfinite(result):
        raise BenchmarkError(f"{label} must be a finite number.")
    return result


def _positive_number(value: object, label: str) -> float:
    result = _finite_number(value, label)
    if result <= 0:
        raise BenchmarkError(f"{label} must be positive.")
    return result


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int) or value <= 0:
        raise BenchmarkError(f"{label} must be a positive integer.")
    return value


def _valid_coordinate_values(
    dataset: Any,
    name: str,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    try:
        values = np.asarray(dataset[name].values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(f"PIXC {name} must be numeric.") from exc
    valid = np.isfinite(values)
    fill_value = dataset[name].attrs.get("_FillValue")
    if fill_value is not None:
        valid &= values != fill_value
    if name == "latitude":
        valid &= (values >= -90.0) & (values <= 90.0)
    return values, valid


def _format_optional(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.3f}"


def _observed_run_count(candidate: CandidateIntervalInferenceResult) -> int:
    return sum(
        item.state == "candidate_wet"
        and (index == 0 or candidate.bins[index - 1].state != "candidate_wet")
        for index, item in enumerate(candidate.bins)
    )


def _save_figure(figure: Any, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise BenchmarkError(
            f"Could not hash required input file {path}: {exc}"
        ) from exc
    return digest.hexdigest()


def _dataframe_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {str(key): _json_value(value) for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def _safe_filename(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )
    return safe or "transect"


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(item) for item in frame.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        rows.append(
            "| "
            + " | ".join(
                str(_json_value(value)).replace("|", "\\|") for value in values
            )
            + " |"
        )
    return "\n".join(rows)


def _package_version() -> str:
    try:
        return version("swot-pixc-lab")
    except PackageNotFoundError:
        return "0+unknown"


def _git_state(config: BenchmarkConfig) -> tuple[str | None, bool | None]:
    start = config.source_path.parent if config.source_path is not None else Path.cwd()
    try:
        commit = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(start), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit or None, bool(status.strip())
