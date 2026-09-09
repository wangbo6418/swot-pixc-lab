"""Experimental fixed-bin inference of candidate wet-support intervals.

This module turns the discrete evidence in a Phase 4 ``TransectSample`` into
candidate station-bin intervals. Candidate interval edges are not validated
physical river banks. No WSE, hull, raster, morphology, centerline, or branch
identity is calculated here.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .exceptions import BankInferenceError
from .transect import TransectSample

type StationBinState = Literal["candidate_wet", "sampled_noneligible", "unsampled"]
type _CandidateIntervalConfigurationInput = (
    CandidateIntervalConfiguration | Mapping[str, object]
)

CANDIDATE_INTERVAL_METHOD_ID = "fixed_station_bin_candidate_wet_intervals"
CANDIDATE_INTERVAL_METHOD_VERSION = "1.0"
CANDIDATE_INTERVAL_METHOD_STATUS = "EXPERIMENTAL / CANDIDATE"
CANDIDATE_INTERVAL_METHOD_WARNING = (
    "Candidate interval edges are not validated physical river banks. "
    "An unsampled bin is not treated as confirmed dry land."
)
GAP_BRIDGE_REASON = "caller-specified gap bridging"
_MAX_STATION_BIN_COUNT = 1_000_000
_FLOAT_COMPARISON_ULPS = 8
BIN_EDGE_SEMANTICS = (
    "Bins are left-closed and right-open; the final bin includes the transect "
    "endpoint. A point exactly on an internal edge belongs to the bin on its right. "
    "Every positive final remainder is retained as a partial bin; adjacent bins "
    "share the same stored edge value."
)
BRIDGE_COMPARISON_SEMANTICS = (
    "An internal gap is accepted when its bin-edge span is less than or equal to "
    "max_bridge_gap_m. A positive difference no larger than "
    f"{_FLOAT_COMPARISON_ULPS} binary floating-point ULPs at the compared length "
    "scale is treated only as numerical equality, not as physical tolerance."
)


@dataclass(frozen=True, slots=True)
class CandidateIntervalConfiguration:
    """One explicit, unranked candidate-inference configuration.

    All four fields are required. The class set is preserved in caller order;
    only membership tests ignore order.
    """

    extent_classes: tuple[int, ...]
    station_bin_width_m: float
    min_extent_pixels_per_bin: int
    max_bridge_gap_m: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "extent_classes", _validate_extent_classes(self.extent_classes)
        )
        object.__setattr__(
            self,
            "station_bin_width_m",
            _positive_finite_number(
                self.station_bin_width_m, parameter="station_bin_width_m"
            ),
        )
        object.__setattr__(
            self,
            "min_extent_pixels_per_bin",
            _positive_integer(
                self.min_extent_pixels_per_bin,
                parameter="min_extent_pixels_per_bin",
            ),
        )
        object.__setattr__(
            self,
            "max_bridge_gap_m",
            _nonnegative_finite_number(
                self.max_bridge_gap_m, parameter="max_bridge_gap_m"
            ),
        )

    def as_dict(self) -> dict[str, object]:
        """Return the four caller-controlled parameters."""

        return {
            "extent_classes": self.extent_classes,
            "station_bin_width_m": self.station_bin_width_m,
            "min_extent_pixels_per_bin": self.min_extent_pixels_per_bin,
            "max_bridge_gap_m": self.max_bridge_gap_m,
        }


@dataclass(frozen=True, slots=True)
class StationBinEvidence:
    """Evidence and state for one fixed bin on the complete transect domain."""

    bin_id: int
    start_station_m: float
    end_station_m: float
    center_station_m: float
    total_sampled_pixel_count: int
    extent_eligible_pixel_count: int
    extent_eligible_fraction: float | None
    classification_counts: Mapping[int, int]
    source_counts: Mapping[int, int]
    extent_eligible_source_counts: Mapping[int, int]
    state: StationBinState
    candidate_wet: bool
    bridge_id: int | None = None

    @property
    def bin_span_m(self) -> float:
        """Return this bin's exact span on the finite station domain."""

        return self.end_station_m - self.start_station_m

    @property
    def bridged(self) -> bool:
        """Return whether an accepted bridge crosses this non-wet bin."""

        return self.bridge_id is not None

    def as_dict(self) -> dict[str, object]:
        """Return a detached tabular/JSON-friendly bin record."""

        return {
            "bin_id": self.bin_id,
            "start_station_m": self.start_station_m,
            "end_station_m": self.end_station_m,
            "center_station_m": self.center_station_m,
            "bin_span_m": self.bin_span_m,
            "total_sampled_pixel_count": self.total_sampled_pixel_count,
            "extent_eligible_pixel_count": self.extent_eligible_pixel_count,
            "extent_eligible_fraction": self.extent_eligible_fraction,
            "classification_counts": dict(self.classification_counts),
            "source_counts": dict(self.source_counts),
            "extent_eligible_source_counts": dict(self.extent_eligible_source_counts),
            "state": self.state,
            "candidate_wet": self.candidate_wet,
            "bridge_id": self.bridge_id,
            "bridged": self.bridged,
        }


@dataclass(frozen=True, slots=True)
class GapBridgeRecord:
    """One explicit connection across an internal non-wet station-bin gap."""

    bridge_id: int
    left_wet_run_id: int
    right_wet_run_id: int
    left_wet_run_first_bin_id: int
    left_wet_run_last_bin_id: int
    right_wet_run_first_bin_id: int
    right_wet_run_last_bin_id: int
    first_gap_bin_id: int
    last_gap_bin_id: int
    gap_start_station_m: float
    gap_end_station_m: float
    gap_width_m: float
    gap_bin_count: int
    unsampled_bin_count: int
    sampled_noneligible_bin_count: int
    total_sampled_pixel_count: int
    extent_eligible_pixel_count: int
    classification_counts: Mapping[int, int]
    source_counts: Mapping[int, int]
    extent_eligible_source_counts: Mapping[int, int]
    reason: str

    def as_dict(self) -> dict[str, object]:
        """Return a detached tabular/JSON-friendly bridge record."""

        return {
            "bridge_id": self.bridge_id,
            "left_wet_run_id": self.left_wet_run_id,
            "right_wet_run_id": self.right_wet_run_id,
            "left_wet_run_first_bin_id": self.left_wet_run_first_bin_id,
            "left_wet_run_last_bin_id": self.left_wet_run_last_bin_id,
            "right_wet_run_first_bin_id": self.right_wet_run_first_bin_id,
            "right_wet_run_last_bin_id": self.right_wet_run_last_bin_id,
            "first_gap_bin_id": self.first_gap_bin_id,
            "last_gap_bin_id": self.last_gap_bin_id,
            "gap_start_station_m": self.gap_start_station_m,
            "gap_end_station_m": self.gap_end_station_m,
            "gap_width_m": self.gap_width_m,
            "gap_bin_count": self.gap_bin_count,
            "unsampled_bin_count": self.unsampled_bin_count,
            "sampled_noneligible_bin_count": self.sampled_noneligible_bin_count,
            "total_sampled_pixel_count": self.total_sampled_pixel_count,
            "extent_eligible_pixel_count": self.extent_eligible_pixel_count,
            "classification_counts": dict(self.classification_counts),
            "source_counts": dict(self.source_counts),
            "extent_eligible_source_counts": dict(self.extent_eligible_source_counts),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CandidateWetInterval:
    """One inferred candidate interval bounded only by station-bin edges.

    ``observed_candidate_wet_support_m`` counts candidate-wet bin spans only.
    ``inferred_candidate_interval_span_m`` additionally includes every recorded
    bridge inside the interval.
    """

    candidate_interval_id: int
    start_station_m: float
    end_station_m: float
    first_bin_id: int
    last_bin_id: int
    candidate_wet_bin_count: int
    bridged_bin_count: int
    extent_eligible_pixel_count: int
    total_sampled_pixel_count: int
    classification_counts: Mapping[int, int]
    source_counts: Mapping[int, int]
    extent_eligible_source_counts: Mapping[int, int]
    bridge_ids: tuple[int, ...]
    observed_candidate_wet_support_m: float
    bridged_gap_m: float

    @property
    def inferred_candidate_interval_span_m(self) -> float:
        """Return the bin-edge span, including any explicitly bridged gap."""

        return self.end_station_m - self.start_station_m

    def as_dict(self) -> dict[str, object]:
        """Return a detached tabular/JSON-friendly candidate record."""

        return {
            "candidate_interval_id": self.candidate_interval_id,
            "start_station_m": self.start_station_m,
            "end_station_m": self.end_station_m,
            "inferred_candidate_interval_span_m": (
                self.inferred_candidate_interval_span_m
            ),
            "first_bin_id": self.first_bin_id,
            "last_bin_id": self.last_bin_id,
            "candidate_wet_bin_count": self.candidate_wet_bin_count,
            "bridged_bin_count": self.bridged_bin_count,
            "extent_eligible_pixel_count": self.extent_eligible_pixel_count,
            "total_sampled_pixel_count": self.total_sampled_pixel_count,
            "classification_counts": dict(self.classification_counts),
            "source_counts": dict(self.source_counts),
            "extent_eligible_source_counts": dict(self.extent_eligible_source_counts),
            "bridge_ids": self.bridge_ids,
            "observed_candidate_wet_support_m": (self.observed_candidate_wet_support_m),
            "bridged_gap_m": self.bridged_gap_m,
        }


@dataclass(frozen=True, slots=True)
class CandidateSourceSummary:
    """Compact source-granule context for candidate inference."""

    source_index: int
    selected_pixel_count: int
    extent_eligible_pixel_count: int
    granule_id: str | None
    filename: str | None
    tile: str | None
    cycle: int | None
    pass_number: int | None
    crid: str | None
    netcdf_product_version: str | None
    collection_version: str | None
    cmr_concept_id: str | None
    cmr_revision_id: int | None


@dataclass(frozen=True, slots=True)
class CandidateIntervalInferenceResult:
    """Immutable, reference-free Phase 5A.2 candidate-inference audit."""

    bins: tuple[StationBinEvidence, ...]
    candidate_intervals: tuple[CandidateWetInterval, ...]
    bridge_records: tuple[GapBridgeRecord, ...]
    method_id: str
    method_version: str
    method_status: str
    method_warning: str
    extent_classes: tuple[int, ...]
    station_bin_width_m: float
    min_extent_pixels_per_bin: int
    max_bridge_gap_m: float
    boundary_resolution_m: float
    bin_edge_semantics: str
    bridge_comparison_semantics: str
    transect_wkt: str
    transect_crs: str
    transect_length_m: float
    metric_crs_wkt: str
    projection_center: tuple[float, float]
    corridor_half_width_m: float
    profile_name: str | None
    profile_label: str | None
    profile_status: str | None
    selected_pixel_count: int
    classification_counts: Mapping[int, int]
    extent_eligible_classification_counts: Mapping[int, int]
    source_counts: Mapping[int, int]
    extent_eligible_source_counts: Mapping[int, int]
    source_tile_counts: Mapping[str, int]
    source_summaries: tuple[CandidateSourceSummary, ...]

    @property
    def bin_count(self) -> int:
        """Return the number of bins covering the complete transect."""

        return len(self.bins)

    @property
    def candidate_interval_count(self) -> int:
        """Return the number of final candidate intervals."""

        return len(self.candidate_intervals)

    @property
    def candidate_wet_bin_count(self) -> int:
        """Count bins meeting the explicit eligible-pixel threshold."""

        return sum(item.state == "candidate_wet" for item in self.bins)

    @property
    def sampled_noneligible_bin_count(self) -> int:
        """Count sampled bins that do not meet the candidate threshold."""

        return sum(item.state == "sampled_noneligible" for item in self.bins)

    @property
    def unsampled_bin_count(self) -> int:
        """Count bins with no sampled PIXC point evidence."""

        return sum(item.state == "unsampled" for item in self.bins)

    @property
    def bridge_count(self) -> int:
        """Return the number of explicitly recorded internal bridges."""

        return len(self.bridge_records)

    @property
    def individual_candidate_widths_m(self) -> tuple[float, ...]:
        """Return observed candidate-wet support per final interval.

        Bridged gaps are excluded; use
        ``individual_inferred_candidate_interval_spans_m`` for bin-edge spans.
        """

        return tuple(
            item.observed_candidate_wet_support_m for item in self.candidate_intervals
        )

    @property
    def individual_observed_candidate_wet_supports_m(self) -> tuple[float, ...]:
        """Return observed wet-bin support for each final candidate interval."""

        return self.individual_candidate_widths_m

    @property
    def individual_inferred_candidate_interval_spans_m(self) -> tuple[float, ...]:
        """Return final bin-edge spans, including recorded bridges."""

        return tuple(
            item.inferred_candidate_interval_span_m for item in self.candidate_intervals
        )

    @property
    def total_candidate_wetted_width_m(self) -> float:
        """Return candidate-wet bin support only, excluding bridged gaps."""

        return math.fsum(self.individual_candidate_widths_m)

    @property
    def total_observed_candidate_wet_support_m(self) -> float:
        """Return observed candidate-wet bin support across all intervals."""

        return self.total_candidate_wetted_width_m

    @property
    def observed_candidate_wet_support_m(self) -> float:
        """Return observed candidate-wet bin support, excluding bridges."""

        return self.total_candidate_wetted_width_m

    @property
    def total_bridged_gap_m(self) -> float:
        """Return the sum of all explicitly inferred bridge spans."""

        return math.fsum(item.gap_width_m for item in self.bridge_records)

    @property
    def inferred_candidate_interval_span_m(self) -> float:
        """Return summed final interval spans, including accepted bridges."""

        return math.fsum(self.individual_inferred_candidate_interval_spans_m)

    @property
    def total_inferred_candidate_interval_span_m(self) -> float:
        """Return summed bridge-inclusive candidate interval spans."""

        return self.inferred_candidate_interval_span_m

    @property
    def outer_candidate_wetted_span_m(self) -> float | None:
        """Return first candidate edge to last candidate edge, including gaps."""

        if not self.candidate_intervals:
            return None
        return (
            self.candidate_intervals[-1].end_station_m
            - self.candidate_intervals[0].start_station_m
        )

    @property
    def outer_candidate_interval_span_m(self) -> float | None:
        """Return the outer span alias without implying observed wet terrain."""

        return self.outer_candidate_wetted_span_m

    def bins_dataframe(self) -> pd.DataFrame:
        """Return fixed-bin evidence as a fresh pandas table."""

        columns = (
            "bin_id",
            "start_station_m",
            "end_station_m",
            "center_station_m",
            "bin_span_m",
            "total_sampled_pixel_count",
            "extent_eligible_pixel_count",
            "extent_eligible_fraction",
            "classification_counts",
            "source_counts",
            "extent_eligible_source_counts",
            "state",
            "candidate_wet",
            "bridge_id",
            "bridged",
        )
        return pd.DataFrame.from_records(
            (item.as_dict() for item in self.bins), columns=columns
        )

    def intervals_dataframe(self) -> pd.DataFrame:
        """Return candidate interval records as a fresh pandas table."""

        columns = (
            "candidate_interval_id",
            "start_station_m",
            "end_station_m",
            "inferred_candidate_interval_span_m",
            "first_bin_id",
            "last_bin_id",
            "candidate_wet_bin_count",
            "bridged_bin_count",
            "extent_eligible_pixel_count",
            "total_sampled_pixel_count",
            "classification_counts",
            "source_counts",
            "extent_eligible_source_counts",
            "bridge_ids",
            "observed_candidate_wet_support_m",
            "bridged_gap_m",
        )
        return pd.DataFrame.from_records(
            (item.as_dict() for item in self.candidate_intervals), columns=columns
        )

    def bridges_dataframe(self) -> pd.DataFrame:
        """Return accepted bridge provenance as a fresh pandas table."""

        columns = (
            "bridge_id",
            "left_wet_run_id",
            "right_wet_run_id",
            "left_wet_run_first_bin_id",
            "left_wet_run_last_bin_id",
            "right_wet_run_first_bin_id",
            "right_wet_run_last_bin_id",
            "first_gap_bin_id",
            "last_gap_bin_id",
            "gap_start_station_m",
            "gap_end_station_m",
            "gap_width_m",
            "gap_bin_count",
            "unsampled_bin_count",
            "sampled_noneligible_bin_count",
            "total_sampled_pixel_count",
            "extent_eligible_pixel_count",
            "classification_counts",
            "source_counts",
            "extent_eligible_source_counts",
            "reason",
        )
        return pd.DataFrame.from_records(
            (item.as_dict() for item in self.bridge_records), columns=columns
        )

    def audit_summary(self) -> dict[str, Any]:
        """Return all parameters, derived scalars, and compact provenance."""

        return {
            "method_id": self.method_id,
            "method_version": self.method_version,
            "method_status": self.method_status,
            "method_warning": self.method_warning,
            "extent_classes": self.extent_classes,
            "station_bin_width_m": self.station_bin_width_m,
            "min_extent_pixels_per_bin": self.min_extent_pixels_per_bin,
            "max_bridge_gap_m": self.max_bridge_gap_m,
            "boundary_resolution_m": self.boundary_resolution_m,
            "bin_edge_semantics": self.bin_edge_semantics,
            "bridge_comparison_semantics": self.bridge_comparison_semantics,
            "bins": [item.as_dict() for item in self.bins],
            "candidate_intervals": [
                item.as_dict() for item in self.candidate_intervals
            ],
            "bridge_records": [item.as_dict() for item in self.bridge_records],
            "bin_count": self.bin_count,
            "candidate_interval_count": self.candidate_interval_count,
            "candidate_wet_bin_count": self.candidate_wet_bin_count,
            "sampled_noneligible_bin_count": self.sampled_noneligible_bin_count,
            "unsampled_bin_count": self.unsampled_bin_count,
            "bridge_count": self.bridge_count,
            "individual_candidate_widths_m": self.individual_candidate_widths_m,
            "individual_observed_candidate_wet_supports_m": (
                self.individual_observed_candidate_wet_supports_m
            ),
            "individual_inferred_candidate_interval_spans_m": (
                self.individual_inferred_candidate_interval_spans_m
            ),
            "total_candidate_wetted_width_m": (self.total_candidate_wetted_width_m),
            "observed_candidate_wet_support_m": (self.observed_candidate_wet_support_m),
            "total_observed_candidate_wet_support_m": (
                self.total_observed_candidate_wet_support_m
            ),
            "total_bridged_gap_m": self.total_bridged_gap_m,
            "inferred_candidate_interval_span_m": (
                self.inferred_candidate_interval_span_m
            ),
            "total_inferred_candidate_interval_span_m": (
                self.total_inferred_candidate_interval_span_m
            ),
            "outer_candidate_wetted_span_m": self.outer_candidate_wetted_span_m,
            "outer_candidate_interval_span_m": self.outer_candidate_interval_span_m,
            "transect_wkt": self.transect_wkt,
            "transect_crs": self.transect_crs,
            "transect_length_m": self.transect_length_m,
            "metric_crs_wkt": self.metric_crs_wkt,
            "projection_center": self.projection_center,
            "corridor_half_width_m": self.corridor_half_width_m,
            "profile_name": self.profile_name,
            "profile_label": self.profile_label,
            "profile_status": self.profile_status,
            "selected_pixel_count": self.selected_pixel_count,
            "classification_counts": dict(self.classification_counts),
            "extent_eligible_classification_counts": dict(
                self.extent_eligible_classification_counts
            ),
            "source_counts": dict(self.source_counts),
            "extent_eligible_source_counts": dict(self.extent_eligible_source_counts),
            "source_tile_counts": dict(self.source_tile_counts),
            "source_summaries": [asdict(item) for item in self.source_summaries],
        }


@dataclass(frozen=True, slots=True)
class CandidateIntervalSensitivityResult:
    """Ordered, unranked results from explicit sensitivity configurations."""

    results: tuple[CandidateIntervalInferenceResult, ...]

    @property
    def configuration_count(self) -> int:
        """Return the number of configurations evaluated."""

        return len(self.results)

    def dataframe(self) -> pd.DataFrame:
        """Return one unranked scalar summary per configuration."""

        columns = (
            "configuration_id",
            "extent_classes",
            "station_bin_width_m",
            "min_extent_pixels_per_bin",
            "max_bridge_gap_m",
            "bin_count",
            "candidate_wet_bin_count",
            "sampled_noneligible_bin_count",
            "unsampled_bin_count",
            "candidate_interval_count",
            "bridge_count",
            "observed_candidate_wet_support_m",
            "total_bridged_gap_m",
            "total_inferred_candidate_interval_span_m",
            "outer_candidate_wetted_span_m",
        )
        records = (
            {
                "configuration_id": index,
                "extent_classes": result.extent_classes,
                "station_bin_width_m": result.station_bin_width_m,
                "min_extent_pixels_per_bin": result.min_extent_pixels_per_bin,
                "max_bridge_gap_m": result.max_bridge_gap_m,
                "bin_count": result.bin_count,
                "candidate_wet_bin_count": result.candidate_wet_bin_count,
                "sampled_noneligible_bin_count": (result.sampled_noneligible_bin_count),
                "unsampled_bin_count": result.unsampled_bin_count,
                "candidate_interval_count": result.candidate_interval_count,
                "bridge_count": result.bridge_count,
                "observed_candidate_wet_support_m": (
                    result.observed_candidate_wet_support_m
                ),
                "total_bridged_gap_m": result.total_bridged_gap_m,
                "total_inferred_candidate_interval_span_m": (
                    result.total_inferred_candidate_interval_span_m
                ),
                "outer_candidate_wetted_span_m": (result.outer_candidate_wetted_span_m),
            }
            for index, result in enumerate(self.results, start=1)
        )
        return pd.DataFrame.from_records(records, columns=columns)


@dataclass(frozen=True, slots=True)
class _WetRun:
    run_id: int
    first_bin_index: int
    last_bin_index: int


def infer_candidate_wet_intervals(
    sample: TransectSample,
    *,
    extent_classes: Iterable[int],
    station_bin_width_m: float,
    min_extent_pixels_per_bin: int,
    max_bridge_gap_m: float,
) -> CandidateIntervalInferenceResult:
    """Infer experimental candidate intervals from fixed station-bin support.

    All scientific parameters are mandatory. A bin is candidate wet if and
    only if its valid ``classification`` values contain at least
    ``min_extent_pixels_per_bin`` members of ``extent_classes``. The method
    does not apply a second QC profile and does not use any other PIXC variable.

    Bins are left-closed/right-open, except that the final bin includes the
    transect endpoint. Candidate boundaries are bin edges, not PIXC extrema.
    """

    _validate_sample_type(sample)
    configuration = CandidateIntervalConfiguration(
        extent_classes=extent_classes,  # type: ignore[arg-type]
        station_bin_width_m=station_bin_width_m,
        min_extent_pixels_per_bin=min_extent_pixels_per_bin,
        max_bridge_gap_m=max_bridge_gap_m,
    )
    transect_length = _transect_length(sample)
    stations, classifications, source_indices = _sample_arrays(sample, transect_length)
    classification_valid = _classification_valid_mask(
        sample, classifications, source_indices
    )
    extent_eligible = classification_valid & np.isin(
        classifications, configuration.extent_classes
    )
    starts, ends = _station_bin_edges(
        transect_length, configuration.station_bin_width_m
    )
    bin_indices = _assign_bin_indices(stations, starts)
    bins = _station_bin_records(
        starts=starts,
        ends=ends,
        bin_indices=bin_indices,
        classifications=classifications,
        classification_valid=classification_valid,
        extent_eligible=extent_eligible,
        source_indices=source_indices,
        minimum_count=configuration.min_extent_pixels_per_bin,
    )
    wet_runs = _candidate_wet_runs(bins)
    bridges = _bridge_records(
        bins,
        wet_runs,
        max_bridge_gap_m=configuration.max_bridge_gap_m,
    )
    if bridges:
        bridge_by_bin = {
            bin_id: bridge.bridge_id
            for bridge in bridges
            for bin_id in range(bridge.first_gap_bin_id, bridge.last_gap_bin_id + 1)
        }
        bins = tuple(
            replace(item, bridge_id=bridge_by_bin.get(item.bin_id)) for item in bins
        )
    intervals = _candidate_interval_records(bins, wet_runs, bridges)

    source_counts = sample.source_counts
    eligible_source_counts = {source_index: 0 for source_index in source_counts}
    eligible_source_counts.update(_counts(source_indices[extent_eligible]))
    for source in sample.sources:
        source_counts.setdefault(source.source_index, 0)
        eligible_source_counts.setdefault(source.source_index, 0)

    valid_classes = classifications[classification_valid]
    eligible_classes = classifications[extent_eligible]
    return CandidateIntervalInferenceResult(
        bins=bins,
        candidate_intervals=intervals,
        bridge_records=bridges,
        method_id=CANDIDATE_INTERVAL_METHOD_ID,
        method_version=CANDIDATE_INTERVAL_METHOD_VERSION,
        method_status=CANDIDATE_INTERVAL_METHOD_STATUS,
        method_warning=CANDIDATE_INTERVAL_METHOD_WARNING,
        extent_classes=configuration.extent_classes,
        station_bin_width_m=configuration.station_bin_width_m,
        min_extent_pixels_per_bin=configuration.min_extent_pixels_per_bin,
        max_bridge_gap_m=configuration.max_bridge_gap_m,
        boundary_resolution_m=configuration.station_bin_width_m,
        bin_edge_semantics=BIN_EDGE_SEMANTICS,
        bridge_comparison_semantics=BRIDGE_COMPARISON_SEMANTICS,
        transect_wkt=sample.transect.wkt,
        transect_crs="EPSG:4326",
        transect_length_m=transect_length,
        metric_crs_wkt=sample.local_crs.to_wkt(),
        projection_center=tuple(float(value) for value in sample.projection_center),
        corridor_half_width_m=float(sample.corridor_half_width_m),
        profile_name=sample.profile_name,
        profile_label=sample.profile_label,
        profile_status=sample.profile_status,
        selected_pixel_count=sample.selected_pixel_count,
        classification_counts=_frozen_counts(_counts(valid_classes)),
        extent_eligible_classification_counts=_frozen_counts(_counts(eligible_classes)),
        source_counts=_frozen_counts(source_counts),
        extent_eligible_source_counts=_frozen_counts(eligible_source_counts),
        source_tile_counts=_frozen_counts(sample.source_tile_counts),
        source_summaries=_source_summaries(
            sample, source_counts, eligible_source_counts
        ),
    )


def run_candidate_interval_sensitivity(
    sample: TransectSample,
    *,
    configurations: Iterable[CandidateIntervalConfiguration | Mapping[str, object]],
) -> CandidateIntervalSensitivityResult:
    """Evaluate ordered explicit configurations without ranking or tuning them."""

    _validate_sample_type(sample)
    if isinstance(configurations, (str, bytes, bytearray, Mapping, Set)):
        raise BankInferenceError(
            "configurations must be an ordered iterable of explicit configuration "
            "records or mappings."
        )
    try:
        supplied = tuple(configurations)
    except TypeError as exc:
        raise BankInferenceError(
            "configurations must be an ordered iterable of explicit configuration "
            "records or mappings."
        ) from exc
    if not supplied:
        raise BankInferenceError(
            "configurations must contain at least one explicit configuration."
        )

    results: list[CandidateIntervalInferenceResult] = []
    for index, value in enumerate(supplied, start=1):
        configuration = _normalize_configuration(value, index=index)
        results.append(
            infer_candidate_wet_intervals(
                sample,
                extent_classes=configuration.extent_classes,
                station_bin_width_m=configuration.station_bin_width_m,
                min_extent_pixels_per_bin=(configuration.min_extent_pixels_per_bin),
                max_bridge_gap_m=configuration.max_bridge_gap_m,
            )
        )
    return CandidateIntervalSensitivityResult(tuple(results))


def _normalize_configuration(
    value: _CandidateIntervalConfigurationInput,
    *,
    index: int,
) -> CandidateIntervalConfiguration:
    if isinstance(value, CandidateIntervalConfiguration):
        return value
    if not isinstance(value, Mapping):
        raise BankInferenceError(
            f"Sensitivity configuration {index} must be a "
            "CandidateIntervalConfiguration or mapping."
        )
    required = {
        "extent_classes",
        "station_bin_width_m",
        "min_extent_pixels_per_bin",
        "max_bridge_gap_m",
    }
    missing = required - set(value)
    extra = set(value) - required
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append(
                "unexpected " + ", ".join(sorted(repr(item) for item in extra))
            )
        raise BankInferenceError(
            f"Sensitivity configuration {index} has invalid keys: "
            + "; ".join(details)
            + "."
        )
    return CandidateIntervalConfiguration(
        extent_classes=value["extent_classes"],  # type: ignore[arg-type]
        station_bin_width_m=value["station_bin_width_m"],  # type: ignore[arg-type]
        min_extent_pixels_per_bin=value[  # type: ignore[arg-type]
            "min_extent_pixels_per_bin"
        ],
        max_bridge_gap_m=value["max_bridge_gap_m"],  # type: ignore[arg-type]
    )


def _sample_arrays(
    sample: TransectSample,
    transect_length_m: float,
) -> tuple[NDArray[np.float64], NDArray[Any], NDArray[np.integer[Any]]]:
    dataset = sample.pixels
    point_count = int(dataset.sizes.get("points", -1))
    if point_count < 0:
        raise BankInferenceError("TransectSample.pixels has no 'points' dimension.")
    for name in ("station_m", "classification", "source_index", "source_point_index"):
        if name not in dataset:
            raise BankInferenceError(
                f"Candidate inference requires TransectSample.pixels[{name!r}]."
            )
        if dataset[name].dims != ("points",):
            raise BankInferenceError(
                f"TransectSample.pixels[{name!r}] must use exactly ('points',)."
            )

    station_values = np.asarray(dataset["station_m"].values)
    if not np.issubdtype(station_values.dtype, np.number) or np.issubdtype(
        station_values.dtype, np.complexfloating
    ):
        raise BankInferenceError("station_m must be a real numeric point variable.")
    stations = np.asarray(station_values, dtype=np.float64)
    if not np.all(np.isfinite(stations)):
        raise BankInferenceError("station_m contains a non-finite value.")
    outside = (stations < 0.0) | (stations > transect_length_m)
    if np.any(outside):
        first = float(stations[np.flatnonzero(outside)[0]])
        raise BankInferenceError(
            f"station_m value {first!r} lies outside the finite transect domain "
            f"[0, {transect_length_m!r}]."
        )

    classifications = np.asarray(dataset["classification"].values)
    if np.issubdtype(classifications.dtype, np.bool_) or not np.issubdtype(
        classifications.dtype, np.integer
    ):
        raise BankInferenceError(
            "classification must retain a one-dimensional integer dtype."
        )
    for name in ("source_index", "source_point_index"):
        if np.issubdtype(dataset[name].dtype, np.bool_) or not np.issubdtype(
            dataset[name].dtype, np.integer
        ):
            raise BankInferenceError(f"{name} must retain an integer dtype.")
    source_indices = np.asarray(dataset["source_index"].values)
    return stations, classifications, source_indices


def _classification_valid_mask(
    sample: TransectSample,
    classifications: NDArray[Any],
    source_indices: NDArray[np.integer[Any]],
) -> NDArray[np.bool_]:
    valid = np.ones(classifications.shape, dtype=np.bool_)
    covered = np.zeros(classifications.shape, dtype=np.bool_)
    for source in sample.sources:
        selected = source_indices == source.source_index
        covered |= selected
        metadata = source.variables.get("classification")
        if metadata is None and np.any(selected):
            raise BankInferenceError(
                f"Classification metadata is missing for source "
                f"{source.source_index}:{source.filename}."
            )
        if metadata is not None and metadata.fill_value is not None:
            valid[selected] &= classifications[selected] != metadata.fill_value
    if sample.sources and not np.all(covered):
        unknown = np.unique(source_indices[~covered]).tolist()
        raise BankInferenceError(
            f"TransectSample pixels reference unknown source_index values: {unknown}."
        )
    if not sample.sources:
        for attribute in ("_FillValue", "missing_value"):
            fill = sample.pixels["classification"].attrs.get(attribute)
            if fill is not None:
                try:
                    valid &= classifications != np.asarray(fill).reshape(-1)[0]
                except (TypeError, ValueError, IndexError):
                    raise BankInferenceError(
                        f"classification {attribute} metadata is not scalar."
                    ) from None
    return valid


def _station_bin_edges(
    transect_length_m: float,
    station_bin_width_m: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    ratio = transect_length_m / station_bin_width_m
    if not math.isfinite(ratio):
        raise BankInferenceError(
            "station_bin_width_m is too small to construct a finite bin domain."
        )
    bin_count = max(1, math.ceil(ratio))
    if bin_count > _MAX_STATION_BIN_COUNT + 1:
        raise BankInferenceError(
            f"station_bin_width_m would create more than "
            f"{_MAX_STATION_BIN_COUNT:,} bins. Choose a wider bin."
        )
    # Correct either direction of quotient rounding against the actual stored
    # edge values. The result is the smallest positive n for which n * width is
    # at or beyond the finite float domain endpoint.
    while bin_count > 1 and (bin_count - 1) * station_bin_width_m >= transect_length_m:
        bin_count -= 1
    while bin_count * station_bin_width_m < transect_length_m:
        bin_count += 1
        if bin_count > _MAX_STATION_BIN_COUNT:
            raise BankInferenceError(
                f"station_bin_width_m would create more than "
                f"{_MAX_STATION_BIN_COUNT:,} bins. Choose a wider bin."
            )
    if bin_count > _MAX_STATION_BIN_COUNT:
        raise BankInferenceError(
            f"station_bin_width_m would create {bin_count:,} bins; the safety "
            f"limit is {_MAX_STATION_BIN_COUNT:,}. Choose a wider bin."
        )
    starts = np.arange(bin_count, dtype=np.float64) * station_bin_width_m
    edges = np.empty(bin_count + 1, dtype=np.float64)
    edges[:-1] = starts
    edges[-1] = transect_length_m
    starts = edges[:-1]
    ends = edges[1:]
    if (
        starts[0] != 0.0
        or starts[-1] >= transect_length_m
        or np.any(np.diff(starts) <= 0.0)
        or np.any(ends <= starts)
    ):
        raise BankInferenceError(
            "station_bin_width_m cannot produce distinct monotonic bin edges at "
            "this transect length. Choose a wider bin."
        )
    return starts, ends


def _validate_sample_type(sample: object) -> None:
    if not isinstance(sample, TransectSample):
        raise TypeError(
            f"sample must be a Phase 4 TransectSample; got {type(sample).__name__}."
        )


def _assign_bin_indices(
    stations: NDArray[np.float64],
    starts: NDArray[np.float64],
) -> NDArray[np.int64]:
    # Searching the left edges with side='right' implements [left, right),
    # while there is no edge after the final bin, so the endpoint stays final.
    indices = np.searchsorted(starts, stations, side="right") - 1
    return np.asarray(indices, dtype=np.int64)


def _station_bin_records(
    *,
    starts: NDArray[np.float64],
    ends: NDArray[np.float64],
    bin_indices: NDArray[np.int64],
    classifications: NDArray[Any],
    classification_valid: NDArray[np.bool_],
    extent_eligible: NDArray[np.bool_],
    source_indices: NDArray[np.integer[Any]],
    minimum_count: int,
) -> tuple[StationBinEvidence, ...]:
    order = np.argsort(bin_indices, kind="stable")
    ordered_bins = bin_indices[order]
    offsets = np.searchsorted(
        ordered_bins, np.arange(starts.size + 1, dtype=np.int64), side="left"
    )
    records: list[StationBinEvidence] = []
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        positions = order[offsets[index] : offsets[index + 1]]
        eligible_positions = positions[extent_eligible[positions]]
        valid_class_positions = positions[classification_valid[positions]]
        total_count = int(positions.size)
        eligible_count = int(eligible_positions.size)
        candidate_wet = eligible_count >= minimum_count
        if candidate_wet:
            state: StationBinState = "candidate_wet"
        elif total_count:
            state = "sampled_noneligible"
        else:
            state = "unsampled"
        records.append(
            StationBinEvidence(
                bin_id=index + 1,
                start_station_m=float(start),
                end_station_m=float(end),
                center_station_m=0.5 * (float(start) + float(end)),
                total_sampled_pixel_count=total_count,
                extent_eligible_pixel_count=eligible_count,
                extent_eligible_fraction=(
                    eligible_count / total_count if total_count else None
                ),
                classification_counts=_frozen_counts(
                    _counts(classifications[valid_class_positions])
                ),
                source_counts=_frozen_counts(_counts(source_indices[positions])),
                extent_eligible_source_counts=_frozen_counts(
                    _counts(source_indices[eligible_positions])
                ),
                state=state,
                candidate_wet=candidate_wet,
            )
        )
    return tuple(records)


def _candidate_wet_runs(
    bins: Sequence[StationBinEvidence],
) -> tuple[_WetRun, ...]:
    runs: list[_WetRun] = []
    start: int | None = None
    for index, item in enumerate(bins):
        if item.candidate_wet and start is None:
            start = index
        if start is not None and (not item.candidate_wet or index == len(bins) - 1):
            end = index if item.candidate_wet else index - 1
            runs.append(_WetRun(len(runs) + 1, start, end))
            start = None
    return tuple(runs)


def _bridge_records(
    bins: Sequence[StationBinEvidence],
    runs: Sequence[_WetRun],
    *,
    max_bridge_gap_m: float,
) -> tuple[GapBridgeRecord, ...]:
    if max_bridge_gap_m == 0.0 or len(runs) < 2:
        return ()
    bridges: list[GapBridgeRecord] = []
    for left, right in zip(runs, runs[1:], strict=False):
        gap = bins[left.last_bin_index + 1 : right.first_bin_index]
        if not gap:  # pragma: no cover - consecutive wet bins form one run
            continue
        gap_width = gap[-1].end_station_m - gap[0].start_station_m
        if not _less_than_or_numerically_equal(gap_width, max_bridge_gap_m):
            continue
        bridges.append(
            GapBridgeRecord(
                bridge_id=len(bridges) + 1,
                left_wet_run_id=left.run_id,
                right_wet_run_id=right.run_id,
                left_wet_run_first_bin_id=bins[left.first_bin_index].bin_id,
                left_wet_run_last_bin_id=bins[left.last_bin_index].bin_id,
                right_wet_run_first_bin_id=bins[right.first_bin_index].bin_id,
                right_wet_run_last_bin_id=bins[right.last_bin_index].bin_id,
                first_gap_bin_id=gap[0].bin_id,
                last_gap_bin_id=gap[-1].bin_id,
                gap_start_station_m=gap[0].start_station_m,
                gap_end_station_m=gap[-1].end_station_m,
                gap_width_m=gap_width,
                gap_bin_count=len(gap),
                unsampled_bin_count=sum(item.state == "unsampled" for item in gap),
                sampled_noneligible_bin_count=sum(
                    item.state == "sampled_noneligible" for item in gap
                ),
                total_sampled_pixel_count=sum(
                    item.total_sampled_pixel_count for item in gap
                ),
                extent_eligible_pixel_count=sum(
                    item.extent_eligible_pixel_count for item in gap
                ),
                classification_counts=_sum_record_counts(gap, "classification_counts"),
                source_counts=_sum_record_counts(gap, "source_counts"),
                extent_eligible_source_counts=_sum_record_counts(
                    gap, "extent_eligible_source_counts"
                ),
                reason=GAP_BRIDGE_REASON,
            )
        )
    return tuple(bridges)


def _candidate_interval_records(
    bins: Sequence[StationBinEvidence],
    runs: Sequence[_WetRun],
    bridges: Sequence[GapBridgeRecord],
) -> tuple[CandidateWetInterval, ...]:
    if not runs:
        return ()
    bridge_lookup = {
        (item.left_wet_run_id, item.right_wet_run_id): item for item in bridges
    }
    groups: list[tuple[list[_WetRun], list[GapBridgeRecord]]] = []
    current_runs = [runs[0]]
    current_bridges: list[GapBridgeRecord] = []
    for left, right in zip(runs, runs[1:], strict=False):
        bridge = bridge_lookup.get((left.run_id, right.run_id))
        if bridge is None:
            groups.append((current_runs, current_bridges))
            current_runs = [right]
            current_bridges = []
        else:
            current_runs.append(right)
            current_bridges.append(bridge)
    groups.append((current_runs, current_bridges))

    intervals: list[CandidateWetInterval] = []
    for interval_id, (group_runs, group_bridges) in enumerate(groups, start=1):
        first_index = group_runs[0].first_bin_index
        last_index = group_runs[-1].last_bin_index
        span_bins = bins[first_index : last_index + 1]
        wet_bins = [item for item in span_bins if item.candidate_wet]
        observed_support = math.fsum(item.bin_span_m for item in wet_bins)
        bridged_gap = math.fsum(item.gap_width_m for item in group_bridges)
        intervals.append(
            CandidateWetInterval(
                candidate_interval_id=interval_id,
                start_station_m=span_bins[0].start_station_m,
                end_station_m=span_bins[-1].end_station_m,
                first_bin_id=span_bins[0].bin_id,
                last_bin_id=span_bins[-1].bin_id,
                candidate_wet_bin_count=len(wet_bins),
                bridged_bin_count=sum(item.gap_bin_count for item in group_bridges),
                extent_eligible_pixel_count=sum(
                    item.extent_eligible_pixel_count for item in span_bins
                ),
                total_sampled_pixel_count=sum(
                    item.total_sampled_pixel_count for item in span_bins
                ),
                classification_counts=_sum_record_counts(
                    span_bins, "classification_counts"
                ),
                source_counts=_sum_record_counts(span_bins, "source_counts"),
                extent_eligible_source_counts=_sum_record_counts(
                    span_bins, "extent_eligible_source_counts"
                ),
                bridge_ids=tuple(item.bridge_id for item in group_bridges),
                observed_candidate_wet_support_m=observed_support,
                bridged_gap_m=bridged_gap,
            )
        )
    return tuple(intervals)


def _sum_record_counts(
    records: Sequence[object],
    attribute: str,
) -> Mapping[Any, int]:
    result: dict[Any, int] = {}
    for record in records:
        values = getattr(record, attribute)
        for key, value in values.items():
            result[key] = result.get(key, 0) + int(value)
    return _frozen_counts(result)


def _transect_length(sample: TransectSample) -> float:
    try:
        value = sample.transect_length_m
    except Exception as exc:
        if isinstance(exc, BankInferenceError):  # pragma: no cover - defensive
            raise
        raise BankInferenceError(
            "TransectSample.transect_length_m must be finite and positive."
        ) from exc
    if isinstance(value, (bool, np.bool_)):
        raise BankInferenceError(
            "TransectSample.transect_length_m must be finite and positive."
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise BankInferenceError(
            "TransectSample.transect_length_m must be finite and positive."
        )
    return result


def _validate_extent_classes(values: Iterable[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray, Mapping, Set)):
        raise BankInferenceError(
            "extent_classes must be a non-empty ordered iterable of unique integers."
        )
    try:
        supplied = tuple(values)
    except TypeError as exc:
        raise BankInferenceError(
            "extent_classes must be a non-empty ordered iterable of unique integers."
        ) from exc
    if not supplied:
        raise BankInferenceError("extent_classes must not be empty.")
    result: list[int] = []
    for value in supplied:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise BankInferenceError(
                "extent_classes must contain integers only; booleans are invalid."
            )
        resolved = int(value)
        if resolved in result:
            raise BankInferenceError(
                f"extent_classes contains duplicate value {resolved}."
            )
        result.append(resolved)
    return tuple(result)


def _positive_finite_number(value: object, *, parameter: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise BankInferenceError(f"{parameter} must be a positive finite number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise BankInferenceError(f"{parameter} must be a positive finite number.")
    return result


def _nonnegative_finite_number(value: object, *, parameter: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise BankInferenceError(f"{parameter} must be a non-negative finite number.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise BankInferenceError(f"{parameter} must be a non-negative finite number.")
    return result


def _positive_integer(value: object, *, parameter: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise BankInferenceError(f"{parameter} must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise BankInferenceError(f"{parameter} must be a positive integer.")
    return result


def _less_than_or_numerically_equal(value: float, limit: float) -> bool:
    """Compare physical lengths while absorbing only binary-float roundoff."""

    if value <= limit:
        return True
    scale = max(abs(value), abs(limit))
    roundoff = _FLOAT_COMPARISON_ULPS * math.ulp(scale)
    return value - limit <= roundoff


def _counts(values: NDArray[Any]) -> dict[Any, int]:
    labels, frequencies = np.unique(values, return_counts=True)
    return {
        int(label): int(count) for label, count in zip(labels, frequencies, strict=True)
    }


def _frozen_counts(values: Mapping[Any, int]) -> Mapping[Any, int]:
    return MappingProxyType({key: int(value) for key, value in values.items()})


def _source_summaries(
    sample: TransectSample,
    counts: Mapping[int, int],
    eligible_counts: Mapping[int, int],
) -> tuple[CandidateSourceSummary, ...]:
    summaries: list[CandidateSourceSummary] = []
    known_indices: set[int] = set()
    for source in sample.sources:
        known_indices.add(source.source_index)
        summaries.append(
            CandidateSourceSummary(
                source_index=source.source_index,
                selected_pixel_count=counts.get(source.source_index, 0),
                extent_eligible_pixel_count=eligible_counts.get(source.source_index, 0),
                granule_id=source.granule_id,
                filename=source.filename,
                tile=source.tile,
                cycle=source.cycle,
                pass_number=source.pass_number,
                crid=source.crid,
                netcdf_product_version=source.netcdf_product_version,
                collection_version=(
                    source.cmr_record.product_version
                    if source.cmr_record is not None
                    else None
                ),
                cmr_concept_id=(
                    source.cmr_record.concept_id
                    if source.cmr_record is not None
                    else None
                ),
                cmr_revision_id=(
                    source.cmr_record.revision_id
                    if source.cmr_record is not None
                    else None
                ),
            )
        )
    summaries.extend(
        CandidateSourceSummary(
            source_index=source_index,
            selected_pixel_count=count,
            extent_eligible_pixel_count=eligible_counts.get(source_index, 0),
            granule_id=None,
            filename=None,
            tile=None,
            cycle=None,
            pass_number=None,
            crid=None,
            netcdf_product_version=None,
            collection_version=None,
            cmr_concept_id=None,
            cmr_revision_id=None,
        )
        for source_index, count in counts.items()
        if source_index not in known_indices
    )
    return tuple(summaries)


__all__ = [
    "BIN_EDGE_SEMANTICS",
    "BRIDGE_COMPARISON_SEMANTICS",
    "CANDIDATE_INTERVAL_METHOD_ID",
    "CANDIDATE_INTERVAL_METHOD_STATUS",
    "CANDIDATE_INTERVAL_METHOD_VERSION",
    "CANDIDATE_INTERVAL_METHOD_WARNING",
    "CandidateIntervalConfiguration",
    "CandidateIntervalInferenceResult",
    "CandidateIntervalSensitivityResult",
    "CandidateSourceSummary",
    "CandidateWetInterval",
    "GapBridgeRecord",
    "StationBinEvidence",
    "StationBinState",
    "infer_candidate_wet_intervals",
    "run_candidate_interval_sensitivity",
]
