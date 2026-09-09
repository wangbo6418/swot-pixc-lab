"""Quantitative validation of candidate wet intervals against manual reference.

Phase 5A.3a compares the continuous one-dimensional wet set recorded by the
Phase 5A.1 analyst contract with both Phase 5A.2 candidate representations.
It does not infer banks, match branches, tune candidate parameters, or inspect
PIXC pixel arrays.
"""

from __future__ import annotations

import math
import statistics
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from .bank_inference import (
    CandidateIntervalInferenceResult,
    CandidateIntervalSensitivityResult,
    CandidateSourceSummary,
)
from .exceptions import ValidationError
from .width import ExplicitIntervalWidthResult, WidthSourceSummary

type _Interval = tuple[float, float]

VALIDATION_METHOD_ID = "explicit_candidate_interval_set_validation"
VALIDATION_METHOD_VERSION = "1.0"
VALIDATION_METHOD_STATUS = "EXPERIMENTAL / QUANTITATIVE AGREEMENT FRAMEWORK"
VALIDATION_METHOD_WARNING = (
    "Agreement is measured against an analyst-supplied reference. These metrics "
    "do not prove that the manual reference is error-free or that candidate "
    "station-bin edges are physical river banks."
)
BOUNDARY_DIAGNOSTIC_WARNING = (
    "Nearest-boundary distances are diagnostics, not one-to-one bank "
    "correspondence or branch matching."
)
_FRAME_COMPARISON_ULPS = 8
FRAME_COMPARISON_SEMANTICS = (
    "Textual CRS and WKT records must match exactly. Recorded metric scalars may "
    f"differ by at most {_FRAME_COMPARISON_ULPS} binary floating-point ULPs at "
    "their compared magnitude; this absorbs representation roundoff only and is "
    "not a physical distance tolerance."
)
MANUAL_BENCHMARK_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class IntervalSetMetrics:
    """Continuous-length agreement metrics for one predicted wet set."""

    reference_wet_length_m: float
    predicted_wet_length_m: float
    true_positive_length_m: float
    false_positive_length_m: float
    false_negative_length_m: float
    union_length_m: float
    precision: float | None
    recall: float | None
    f1: float | None
    iou: float | None
    reference_empty: bool
    prediction_empty: bool
    exact_empty_match: bool
    signed_total_width_error_m: float
    absolute_total_width_error_m: float
    relative_total_width_error: float | None
    reference_outer_span_m: float | None
    predicted_outer_span_m: float | None
    signed_outer_span_error_m: float | None
    absolute_outer_span_error_m: float | None
    reference_interval_count: int
    predicted_component_count: int
    component_count_difference: int

    def as_dict(self) -> dict[str, float | int | bool | None]:
        """Return a detached tabular/JSON-friendly metric record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BoundaryDistanceMetrics:
    """Symmetric nearest-boundary diagnostics without boundary matching.

    Distances retain the order of the source boundary tuples. An empty tuple of
    nearest distances means that at least one boundary set was empty, so the
    corresponding summaries are ``None``.
    """

    manual_boundaries_m: tuple[float, ...]
    candidate_boundaries_m: tuple[float, ...]
    manual_to_candidate_boundary_distances_m: tuple[float, ...]
    candidate_to_manual_boundary_distances_m: tuple[float, ...]
    diagnostic_warning: str = BOUNDARY_DIAGNOSTIC_WARNING

    @property
    def manual_to_candidate_boundary_mean_m(self) -> float | None:
        """Return the mean nearest distance from manual to candidate edges."""

        return _mean_or_none(self.manual_to_candidate_boundary_distances_m)

    @property
    def manual_to_candidate_boundary_median_m(self) -> float | None:
        """Return the median nearest distance from manual to candidate edges."""

        return _median_or_none(self.manual_to_candidate_boundary_distances_m)

    @property
    def manual_to_candidate_boundary_max_m(self) -> float | None:
        """Return the maximum nearest distance from manual to candidate edges."""

        return _max_or_none(self.manual_to_candidate_boundary_distances_m)

    @property
    def candidate_to_manual_boundary_mean_m(self) -> float | None:
        """Return the mean nearest distance from candidate to manual edges."""

        return _mean_or_none(self.candidate_to_manual_boundary_distances_m)

    @property
    def candidate_to_manual_boundary_median_m(self) -> float | None:
        """Return the median nearest distance from candidate to manual edges."""

        return _median_or_none(self.candidate_to_manual_boundary_distances_m)

    @property
    def candidate_to_manual_boundary_max_m(self) -> float | None:
        """Return the maximum nearest distance from candidate to manual edges."""

        return _max_or_none(self.candidate_to_manual_boundary_distances_m)

    @property
    def symmetric_boundary_mean_m(self) -> float | None:
        """Return the mean over both directed nearest-boundary tuples."""

        if not (
            self.manual_to_candidate_boundary_distances_m
            and self.candidate_to_manual_boundary_distances_m
        ):
            return None
        return float(
            statistics.fmean(
                (
                    *self.manual_to_candidate_boundary_distances_m,
                    *self.candidate_to_manual_boundary_distances_m,
                )
            )
        )

    @property
    def symmetric_boundary_max_m(self) -> float | None:
        """Return the maximum over both directed nearest-boundary tuples."""

        if not (
            self.manual_to_candidate_boundary_distances_m
            and self.candidate_to_manual_boundary_distances_m
        ):
            return None
        return max(
            *self.manual_to_candidate_boundary_distances_m,
            *self.candidate_to_manual_boundary_distances_m,
        )

    def as_dict(self) -> dict[str, object]:
        """Return full ordered distances and their detached summaries."""

        return {
            "manual_boundaries_m": self.manual_boundaries_m,
            "candidate_boundaries_m": self.candidate_boundaries_m,
            "manual_to_candidate_boundary_distances_m": (
                self.manual_to_candidate_boundary_distances_m
            ),
            "candidate_to_manual_boundary_distances_m": (
                self.candidate_to_manual_boundary_distances_m
            ),
            "manual_to_candidate_boundary_mean_m": (
                self.manual_to_candidate_boundary_mean_m
            ),
            "manual_to_candidate_boundary_median_m": (
                self.manual_to_candidate_boundary_median_m
            ),
            "manual_to_candidate_boundary_max_m": (
                self.manual_to_candidate_boundary_max_m
            ),
            "candidate_to_manual_boundary_mean_m": (
                self.candidate_to_manual_boundary_mean_m
            ),
            "candidate_to_manual_boundary_median_m": (
                self.candidate_to_manual_boundary_median_m
            ),
            "candidate_to_manual_boundary_max_m": (
                self.candidate_to_manual_boundary_max_m
            ),
            "symmetric_boundary_mean_m": self.symmetric_boundary_mean_m,
            "symmetric_boundary_max_m": self.symmetric_boundary_max_m,
            "diagnostic_warning": self.diagnostic_warning,
        }


@dataclass(frozen=True, slots=True)
class IntervalValidationResult:
    """Immutable Phase 5A.3a comparison and compact provenance snapshot."""

    observed_support_metrics: IntervalSetMetrics
    bridge_inclusive_metrics: IntervalSetMetrics
    observed_support_boundary_metrics: BoundaryDistanceMetrics
    bridge_inclusive_boundary_metrics: BoundaryDistanceMetrics
    total_bridged_gap_m: float
    bridged_length_over_manual_wet_m: float
    bridged_length_over_manual_nonwet_m: float
    delta_iou_due_to_bridging: float | None
    delta_f1_due_to_bridging: float | None
    method_id: str
    method_version: str
    method_status: str
    method_warning: str
    frame_comparison_semantics: str
    reference_method_id: str
    reference_method_status: str
    reference_method_warning: str
    candidate_method_id: str
    candidate_method_version: str
    candidate_method_status: str
    candidate_method_warning: str
    extent_classes: tuple[int, ...]
    station_bin_width_m: float
    min_extent_pixels_per_bin: int
    max_bridge_gap_m: float
    transect_wkt: str
    transect_crs: str
    transect_length_m: float
    metric_crs_wkt: str
    projection_center: tuple[float, float]
    corridor_half_width_m: float
    reference_profile_name: str | None
    reference_profile_label: str | None
    reference_profile_status: str | None
    candidate_profile_name: str | None
    candidate_profile_label: str | None
    candidate_profile_status: str | None
    reference_source_summaries: tuple[WidthSourceSummary, ...]
    candidate_source_summaries: tuple[CandidateSourceSummary, ...]
    reference_source_counts: Mapping[int, int]
    candidate_source_counts: Mapping[int, int]

    @property
    def reference_interval_count(self) -> int:
        """Return the manual wet-interval count."""

        return self.observed_support_metrics.reference_interval_count

    @property
    def observed_candidate_run_count(self) -> int:
        """Return the count of contiguous candidate-wet support runs."""

        return self.observed_support_metrics.predicted_component_count

    @property
    def inferred_candidate_interval_count(self) -> int:
        """Return the bridge-inclusive inferred interval count."""

        return self.bridge_inclusive_metrics.predicted_component_count

    @property
    def observed_component_count_difference(self) -> int:
        """Return observed-run count minus manual interval count."""

        return self.observed_support_metrics.component_count_difference

    @property
    def inferred_component_count_difference(self) -> int:
        """Return inferred-interval count minus manual interval count."""

        return self.bridge_inclusive_metrics.component_count_difference

    @property
    def observed_support_iou(self) -> float | None:
        """Return IoU for observed candidate-wet support only."""

        return self.observed_support_metrics.iou

    @property
    def bridge_inclusive_iou(self) -> float | None:
        """Return IoU after including caller-authorized bridges."""

        return self.bridge_inclusive_metrics.iou

    @property
    def observed_support_f1(self) -> float | None:
        """Return F1 for observed candidate-wet support only."""

        return self.observed_support_metrics.f1

    @property
    def bridge_inclusive_f1(self) -> float | None:
        """Return F1 after including caller-authorized bridges."""

        return self.bridge_inclusive_metrics.f1

    def summary_dataframe(self) -> pd.DataFrame:
        """Return one row with separately prefixed candidate representations."""

        record = self._summary_record()
        return pd.DataFrame.from_records([record], columns=tuple(record))

    def as_dataframe(self) -> pd.DataFrame:
        """Return the one-row summary table alias."""

        return self.summary_dataframe()

    def audit_summary(self) -> dict[str, Any]:
        """Return nested metrics, frame details, and compact provenance."""

        return {
            "method_id": self.method_id,
            "method_version": self.method_version,
            "method_status": self.method_status,
            "method_warning": self.method_warning,
            "frame_comparison_semantics": self.frame_comparison_semantics,
            "reference_method_id": self.reference_method_id,
            "reference_method_status": self.reference_method_status,
            "reference_method_warning": self.reference_method_warning,
            "candidate_method_id": self.candidate_method_id,
            "candidate_method_version": self.candidate_method_version,
            "candidate_method_status": self.candidate_method_status,
            "candidate_method_warning": self.candidate_method_warning,
            "candidate_configuration": {
                "extent_classes": self.extent_classes,
                "station_bin_width_m": self.station_bin_width_m,
                "min_extent_pixels_per_bin": self.min_extent_pixels_per_bin,
                "max_bridge_gap_m": self.max_bridge_gap_m,
            },
            "observed_support_metrics": self.observed_support_metrics.as_dict(),
            "bridge_inclusive_metrics": self.bridge_inclusive_metrics.as_dict(),
            "observed_support_boundary_metrics": (
                self.observed_support_boundary_metrics.as_dict()
            ),
            "bridge_inclusive_boundary_metrics": (
                self.bridge_inclusive_boundary_metrics.as_dict()
            ),
            "total_bridged_gap_m": self.total_bridged_gap_m,
            "bridged_length_over_manual_wet_m": (self.bridged_length_over_manual_wet_m),
            "bridged_length_over_manual_nonwet_m": (
                self.bridged_length_over_manual_nonwet_m
            ),
            "delta_iou_due_to_bridging": self.delta_iou_due_to_bridging,
            "delta_f1_due_to_bridging": self.delta_f1_due_to_bridging,
            "transect_wkt": self.transect_wkt,
            "transect_crs": self.transect_crs,
            "transect_length_m": self.transect_length_m,
            "metric_crs_wkt": self.metric_crs_wkt,
            "projection_center": self.projection_center,
            "corridor_half_width_m": self.corridor_half_width_m,
            "reference_profile_name": self.reference_profile_name,
            "reference_profile_label": self.reference_profile_label,
            "reference_profile_status": self.reference_profile_status,
            "candidate_profile_name": self.candidate_profile_name,
            "candidate_profile_label": self.candidate_profile_label,
            "candidate_profile_status": self.candidate_profile_status,
            "reference_source_counts": dict(self.reference_source_counts),
            "candidate_source_counts": dict(self.candidate_source_counts),
            "reference_source_summaries": [
                asdict(item) for item in self.reference_source_summaries
            ],
            "candidate_source_summaries": [
                asdict(item) for item in self.candidate_source_summaries
            ],
        }

    def _summary_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "method_id": self.method_id,
            "method_version": self.method_version,
            "reference_method_id": self.reference_method_id,
            "candidate_method_id": self.candidate_method_id,
            "candidate_method_version": self.candidate_method_version,
            "extent_classes": self.extent_classes,
            "station_bin_width_m": self.station_bin_width_m,
            "min_extent_pixels_per_bin": self.min_extent_pixels_per_bin,
            "max_bridge_gap_m": self.max_bridge_gap_m,
            "reference_interval_count": self.reference_interval_count,
            "observed_candidate_run_count": self.observed_candidate_run_count,
            "inferred_candidate_interval_count": (
                self.inferred_candidate_interval_count
            ),
            "observed_component_count_difference": (
                self.observed_component_count_difference
            ),
            "inferred_component_count_difference": (
                self.inferred_component_count_difference
            ),
        }
        record.update(
            _prefixed(self.observed_support_metrics.as_dict(), "observed_support_")
        )
        record.update(
            _prefixed(self.bridge_inclusive_metrics.as_dict(), "bridge_inclusive_")
        )
        record.update(
            _prefixed(
                self.observed_support_boundary_metrics.as_dict(),
                "observed_support_boundary_",
            )
        )
        record.update(
            _prefixed(
                self.bridge_inclusive_boundary_metrics.as_dict(),
                "bridge_inclusive_boundary_",
            )
        )
        record.update(
            {
                "total_bridged_gap_m": self.total_bridged_gap_m,
                "bridged_length_over_manual_wet_m": (
                    self.bridged_length_over_manual_wet_m
                ),
                "bridged_length_over_manual_nonwet_m": (
                    self.bridged_length_over_manual_nonwet_m
                ),
                "delta_iou_due_to_bridging": self.delta_iou_due_to_bridging,
                "delta_f1_due_to_bridging": self.delta_f1_due_to_bridging,
                "transect_wkt": self.transect_wkt,
                "transect_crs": self.transect_crs,
                "transect_length_m": self.transect_length_m,
                "metric_crs_wkt": self.metric_crs_wkt,
                "projection_center": self.projection_center,
                "corridor_half_width_m": self.corridor_half_width_m,
                "reference_profile_name": self.reference_profile_name,
                "candidate_profile_name": self.candidate_profile_name,
                "method_status": self.method_status,
                "method_warning": self.method_warning,
            }
        )
        return record


@dataclass(frozen=True, slots=True)
class SensitivityValidationResult:
    """Ordered, unranked validations for Phase 5A.2 sensitivity results."""

    results: tuple[IntervalValidationResult, ...]

    @property
    def configuration_count(self) -> int:
        """Return the number of candidate configurations evaluated."""

        return len(self.results)

    def dataframe(self) -> pd.DataFrame:
        """Return caller-ordered validation rows without ranking or selection."""

        records: list[dict[str, object]] = []
        for index, result in enumerate(self.results, start=1):
            records.append(
                {
                    "configuration_id": index,
                    **result._summary_record(),
                }
            )
        if not records:
            return pd.DataFrame(columns=("configuration_id",))
        return pd.DataFrame.from_records(records, columns=tuple(records[0]))

    def summary_dataframe(self) -> pd.DataFrame:
        """Return the ordered sensitivity table alias."""

        return self.dataframe()


@dataclass(frozen=True, slots=True)
class ManualBenchmarkMetadata:
    """Descriptive metadata for one independent manual annotation record."""

    benchmark_id: str
    transect_id: str
    annotation_id: str
    observation_identifier: str | None = None
    observation_date: str | None = None
    site_label: str | None = None
    analyst_id: str | None = None
    analyst_label: str | None = None
    reference_source_description: str | None = None
    reference_acquisition_date: str | None = None
    temporal_offset_hours: float | None = None
    annotation_confidence: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        for name in ("benchmark_id", "transect_id", "annotation_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"{name} must be a non-empty string.")
        optional_text = (
            "observation_identifier",
            "observation_date",
            "site_label",
            "analyst_id",
            "analyst_label",
            "reference_source_description",
            "reference_acquisition_date",
            "annotation_confidence",
            "notes",
        )
        for name in optional_text:
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValidationError(f"{name} must be a string or None.")
        if self.temporal_offset_hours is not None:
            value = self.temporal_offset_hours
            if isinstance(value, (bool, np.bool_)):
                raise ValidationError(
                    "temporal_offset_hours must be a finite number or None."
                )
            try:
                resolved = float(value)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    "temporal_offset_hours must be a finite number or None."
                ) from exc
            if not math.isfinite(resolved):
                raise ValidationError(
                    "temporal_offset_hours must be a finite number or None."
                )
            object.__setattr__(self, "temporal_offset_hours", resolved)

    def as_dict(self) -> dict[str, object]:
        """Return descriptive metadata without width measurements."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ManualBenchmarkRecord:
    """Associate metadata with one Phase 5A.1 explicit reference result.

    Multiple analysts remain separate records sharing a ``benchmark_id`` and
    ``transect_id``. This object performs no consensus or averaging.
    """

    metadata: ManualBenchmarkMetadata
    reference: ExplicitIntervalWidthResult
    schema_version: str = MANUAL_BENCHMARK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, ManualBenchmarkMetadata):
            raise TypeError("metadata must be a ManualBenchmarkMetadata instance.")
        if not isinstance(self.reference, ExplicitIntervalWidthResult):
            raise TypeError("reference must be an ExplicitIntervalWidthResult.")
        if self.schema_version != MANUAL_BENCHMARK_SCHEMA_VERSION:
            raise ValidationError(
                "schema_version must match the supported manual benchmark schema "
                f"{MANUAL_BENCHMARK_SCHEMA_VERSION!r}."
            )

    def as_dict(self) -> dict[str, object]:
        """Return a human-readable, JSON-serializable benchmark record."""

        reference = self.reference
        return {
            "schema_version": self.schema_version,
            "metadata": self.metadata.as_dict(),
            "explicit_interval_reference": {
                "method_id": reference.method_id,
                "method_status": reference.method_status,
                "method_warning": reference.method_warning,
                "manual_intervals": [item.as_dict() for item in reference.intervals],
                "measurement_summary": {
                    "interval_count": reference.interval_count,
                    "total_wetted_width_m": reference.total_wetted_width_m,
                    "outer_wetted_span_m": reference.outer_wetted_span_m,
                    "total_internal_dry_gap_m": (reference.total_internal_dry_gap_m),
                },
                "station_frame": {
                    "transect_wkt": reference.transect_wkt,
                    "transect_crs": reference.transect_crs,
                    "transect_length_m": reference.transect_length_m,
                    "metric_crs_wkt": reference.metric_crs_wkt,
                    "projection_center": reference.projection_center,
                    "corridor_half_width_m": reference.corridor_half_width_m,
                },
                "upstream_context": {
                    "profile_name": reference.profile_name,
                    "profile_label": reference.profile_label,
                    "profile_status": reference.profile_status,
                    "selected_pixel_count": reference.selected_pixel_count,
                    "classification_counts": dict(reference.classification_counts),
                    "source_counts": dict(reference.source_counts),
                    "source_tile_counts": dict(reference.source_tile_counts),
                    "source_summaries": [
                        asdict(item) for item in reference.source_summaries
                    ],
                },
            },
        }


def evaluate_candidate_against_explicit(
    reference: ExplicitIntervalWidthResult,
    candidate: CandidateIntervalInferenceResult,
) -> IntervalValidationResult:
    """Compare both Phase 5A.2 wet sets with a Phase 5A.1 manual reference.

    Manual intervals remain continuous and are never discretized to candidate
    bins. Candidate-wet bins form the observed-support set, while final
    candidate intervals form the separate bridge-inclusive set.
    """

    if not isinstance(reference, ExplicitIntervalWidthResult):
        raise TypeError("reference must be an ExplicitIntervalWidthResult.")
    if not isinstance(candidate, CandidateIntervalInferenceResult):
        raise TypeError("candidate must be a CandidateIntervalInferenceResult.")
    _validate_shared_frame(reference, candidate)

    manual = _ordered_interval_set(
        ((item.start_station_m, item.end_station_m) for item in reference.intervals),
        label="manual reference intervals",
        domain_length=reference.transect_length_m,
        merge_touching=False,
    )
    observed = _ordered_interval_set(
        (
            (item.start_station_m, item.end_station_m)
            for item in candidate.bins
            if item.state == "candidate_wet"
        ),
        label="observed candidate-wet bins",
        domain_length=candidate.transect_length_m,
        merge_touching=True,
    )
    inferred = _ordered_interval_set(
        (
            (item.start_station_m, item.end_station_m)
            for item in candidate.candidate_intervals
        ),
        label="bridge-inclusive candidate intervals",
        domain_length=candidate.transect_length_m,
        merge_touching=False,
    )
    bridges = _ordered_interval_set(
        (
            (item.gap_start_station_m, item.gap_end_station_m)
            for item in candidate.bridge_records
        ),
        label="accepted bridge regions",
        domain_length=candidate.transect_length_m,
        merge_touching=False,
    )
    for item, (bridge_start, bridge_end) in zip(
        candidate.bridge_records, bridges, strict=True
    ):
        endpoint_width = bridge_end - bridge_start
        if not _roundoff_equal(endpoint_width, item.gap_width_m):
            raise ValidationError(
                f"Accepted bridge record {item.bridge_id!r} has "
                f"gap_width_m={item.gap_width_m!r}, but its endpoints reconstruct "
                f"{endpoint_width!r} m. Recreate the Phase 5A.2 result before "
                "validation."
            )

    observed_bridge_overlap = _intersection_length(observed, bridges)
    if _interval_sets_overlap_beyond_roundoff(observed, bridges):
        raise ValidationError(
            "Accepted bridge regions overlap observed candidate-wet support by "
            f"{observed_bridge_overlap!r} m; a bridge must represent only the "
            "non-observed gap between wet runs. Recreate the Phase 5A.2 result "
            "before validation."
        )
    reconstructed_inferred = _continuous_interval_union(observed, bridges)
    if not _interval_sets_roundoff_equal(inferred, reconstructed_inferred):
        raise ValidationError(
            "Bridge-inclusive candidate intervals do not equal the continuous "
            "union of observed candidate-wet support and accepted bridge regions; "
            "the result contains an unrecorded addition or omission. Recreate the "
            "Phase 5A.2 result before validation."
        )

    observed_metrics = _interval_metrics(manual, observed)
    inferred_metrics = _interval_metrics(manual, inferred)
    observed_boundaries = _boundary_metrics(manual, observed)
    inferred_boundaries = _boundary_metrics(manual, inferred)
    bridged_length = _interval_length(bridges)
    if not _roundoff_equal(bridged_length, candidate.total_bridged_gap_m):
        raise ValidationError(
            "Candidate bridge records do not reconstruct total_bridged_gap_m; "
            "recreate the Phase 5A.2 result before validation."
        )
    bridge_over_wet = _intersection_length(bridges, manual)
    bridge_over_nonwet = _nonnegative_difference(bridged_length, bridge_over_wet)

    return IntervalValidationResult(
        observed_support_metrics=observed_metrics,
        bridge_inclusive_metrics=inferred_metrics,
        observed_support_boundary_metrics=observed_boundaries,
        bridge_inclusive_boundary_metrics=inferred_boundaries,
        total_bridged_gap_m=bridged_length,
        bridged_length_over_manual_wet_m=bridge_over_wet,
        bridged_length_over_manual_nonwet_m=bridge_over_nonwet,
        delta_iou_due_to_bridging=_difference_when_defined(
            inferred_metrics.iou, observed_metrics.iou
        ),
        delta_f1_due_to_bridging=_difference_when_defined(
            inferred_metrics.f1, observed_metrics.f1
        ),
        method_id=VALIDATION_METHOD_ID,
        method_version=VALIDATION_METHOD_VERSION,
        method_status=VALIDATION_METHOD_STATUS,
        method_warning=VALIDATION_METHOD_WARNING,
        frame_comparison_semantics=FRAME_COMPARISON_SEMANTICS,
        reference_method_id=reference.method_id,
        reference_method_status=reference.method_status,
        reference_method_warning=reference.method_warning,
        candidate_method_id=candidate.method_id,
        candidate_method_version=candidate.method_version,
        candidate_method_status=candidate.method_status,
        candidate_method_warning=candidate.method_warning,
        extent_classes=tuple(candidate.extent_classes),
        station_bin_width_m=candidate.station_bin_width_m,
        min_extent_pixels_per_bin=candidate.min_extent_pixels_per_bin,
        max_bridge_gap_m=candidate.max_bridge_gap_m,
        transect_wkt=reference.transect_wkt,
        transect_crs=reference.transect_crs,
        transect_length_m=reference.transect_length_m,
        metric_crs_wkt=reference.metric_crs_wkt,
        projection_center=tuple(reference.projection_center),
        corridor_half_width_m=reference.corridor_half_width_m,
        reference_profile_name=reference.profile_name,
        reference_profile_label=reference.profile_label,
        reference_profile_status=reference.profile_status,
        candidate_profile_name=candidate.profile_name,
        candidate_profile_label=candidate.profile_label,
        candidate_profile_status=candidate.profile_status,
        reference_source_summaries=tuple(reference.source_summaries),
        candidate_source_summaries=tuple(candidate.source_summaries),
        reference_source_counts=_frozen_counts(reference.source_counts),
        candidate_source_counts=_frozen_counts(candidate.source_counts),
    )


def evaluate_sensitivity_against_explicit(
    reference: ExplicitIntervalWidthResult,
    sensitivity_results: CandidateIntervalSensitivityResult,
) -> SensitivityValidationResult:
    """Validate explicit candidate configurations in caller order, unranked."""

    if not isinstance(reference, ExplicitIntervalWidthResult):
        raise TypeError("reference must be an ExplicitIntervalWidthResult.")
    if not isinstance(sensitivity_results, CandidateIntervalSensitivityResult):
        raise TypeError(
            "sensitivity_results must be a CandidateIntervalSensitivityResult."
        )
    return SensitivityValidationResult(
        tuple(
            evaluate_candidate_against_explicit(reference, candidate)
            for candidate in sensitivity_results.results
        )
    )


def create_manual_benchmark_record(
    metadata: ManualBenchmarkMetadata,
    reference: ExplicitIntervalWidthResult,
) -> ManualBenchmarkRecord:
    """Associate descriptive metadata with an existing Phase 5A.1 reference."""

    if not isinstance(metadata, ManualBenchmarkMetadata):
        raise TypeError("metadata must be a ManualBenchmarkMetadata instance.")
    if not isinstance(reference, ExplicitIntervalWidthResult):
        raise TypeError("reference must be an ExplicitIntervalWidthResult.")
    return ManualBenchmarkRecord(metadata=metadata, reference=reference)


def _validate_shared_frame(
    reference: ExplicitIntervalWidthResult,
    candidate: CandidateIntervalInferenceResult,
) -> None:
    for name in ("transect_wkt", "transect_crs", "metric_crs_wkt"):
        left = getattr(reference, name)
        right = getattr(candidate, name)
        if left != right:
            _raise_frame_mismatch(name, left, right)
    for name in ("transect_length_m", "corridor_half_width_m"):
        left = getattr(reference, name)
        right = getattr(candidate, name)
        if not _roundoff_equal(left, right):
            _raise_frame_mismatch(name, left, right)
    if len(reference.projection_center) != 2 or len(candidate.projection_center) != 2:
        raise ValidationError(
            "Cannot validate an invalid projection_center; both results must "
            "record a two-value (longitude, latitude) center."
        )
    for index, label in enumerate(("longitude", "latitude")):
        left = reference.projection_center[index]
        right = candidate.projection_center[index]
        if not _roundoff_equal(left, right):
            _raise_frame_mismatch(f"projection_center_{label}", left, right)


def _raise_frame_mismatch(name: str, reference: object, candidate: object) -> None:
    raise ValidationError(
        "Cannot compare results from incompatible transect/station frames: "
        f"{name} differs (reference={reference!r}, candidate={candidate!r}). "
        "Recreate both records from the same Phase 4 TransectSample; validation "
        "does not silently rescale or reproject station coordinates."
    )


def _roundoff_equal(left: object, right: object) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        return False
    if left_value == right_value:
        return True
    scale = max(abs(left_value), abs(right_value))
    return abs(left_value - right_value) <= _FRAME_COMPARISON_ULPS * math.ulp(scale)


def _ordered_interval_set(
    values: object,
    *,
    label: str,
    domain_length: float,
    merge_touching: bool,
) -> tuple[_Interval, ...]:
    result: list[_Interval] = []
    try:
        supplied = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:  # pragma: no cover - internal callers are iterable
        raise ValidationError(f"{label} are not iterable.") from exc
    for index, item in enumerate(supplied, start=1):
        try:
            start, end = item
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"{label} record {index} must have exactly two boundaries."
            ) from exc
        start = float(start)
        end = float(end)
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValidationError(f"{label} record {index} has a non-finite edge.")
        if start < 0.0 or end > domain_length or end <= start:
            raise ValidationError(
                f"{label} record {index} must satisfy 0 <= start < end <= "
                f"{domain_length!r}."
            )
        if result:
            previous_start, previous_end = result[-1]
            if start < previous_end:
                raise ValidationError(
                    f"{label} overlap or are not in increasing station order."
                )
            if start == previous_end:
                if merge_touching:
                    result[-1] = (previous_start, end)
                    continue
                raise ValidationError(
                    f"{label} contain touching components that should already be "
                    "represented as one interval."
                )
        result.append((start, end))
    return tuple(result)


def _interval_metrics(
    reference: tuple[_Interval, ...],
    predicted: tuple[_Interval, ...],
) -> IntervalSetMetrics:
    reference_length = _interval_length(reference)
    predicted_length = _interval_length(predicted)
    true_positive = _intersection_length(reference, predicted)
    false_positive = _nonnegative_difference(predicted_length, true_positive)
    false_negative = _nonnegative_difference(reference_length, true_positive)
    union_length = math.fsum((true_positive, false_positive, false_negative))
    reference_empty = not reference
    prediction_empty = not predicted
    exact_empty_match = reference_empty and prediction_empty

    precision = None if prediction_empty else true_positive / predicted_length
    recall = None if reference_empty else true_positive / reference_length
    if exact_empty_match:
        f1 = None
    elif reference_empty or prediction_empty:
        f1 = 0.0
    else:
        f1 = (2.0 * true_positive) / (
            2.0 * true_positive + false_positive + false_negative
        )
    iou = None if union_length == 0.0 else true_positive / union_length

    signed_width_error = predicted_length - reference_length
    reference_outer = _outer_span(reference)
    predicted_outer = _outer_span(predicted)
    if reference_outer is None or predicted_outer is None:
        signed_outer_error = None
        absolute_outer_error = None
    else:
        signed_outer_error = predicted_outer - reference_outer
        absolute_outer_error = abs(signed_outer_error)

    return IntervalSetMetrics(
        reference_wet_length_m=reference_length,
        predicted_wet_length_m=predicted_length,
        true_positive_length_m=true_positive,
        false_positive_length_m=false_positive,
        false_negative_length_m=false_negative,
        union_length_m=union_length,
        precision=precision,
        recall=recall,
        f1=f1,
        iou=iou,
        reference_empty=reference_empty,
        prediction_empty=prediction_empty,
        exact_empty_match=exact_empty_match,
        signed_total_width_error_m=signed_width_error,
        absolute_total_width_error_m=abs(signed_width_error),
        relative_total_width_error=(
            signed_width_error / reference_length if reference_length > 0.0 else None
        ),
        reference_outer_span_m=reference_outer,
        predicted_outer_span_m=predicted_outer,
        signed_outer_span_error_m=signed_outer_error,
        absolute_outer_span_error_m=absolute_outer_error,
        reference_interval_count=len(reference),
        predicted_component_count=len(predicted),
        component_count_difference=len(predicted) - len(reference),
    )


def _continuous_interval_union(
    *interval_sets: tuple[_Interval, ...],
) -> tuple[_Interval, ...]:
    """Return a continuous union, merging roundoff-equivalent shared edges."""

    ordered = sorted(
        (interval for intervals in interval_sets for interval in intervals),
        key=lambda interval: (interval[0], interval[1]),
    )
    merged: list[_Interval] = []
    for start, end in ordered:
        if not merged:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        if start < previous_end or _roundoff_equal(start, previous_end):
            if end > previous_end:
                merged[-1] = (previous_start, end)
            continue
        merged.append((start, end))
    return tuple(merged)


def _interval_sets_overlap_beyond_roundoff(
    left: tuple[_Interval, ...],
    right: tuple[_Interval, ...],
) -> bool:
    """Return whether two sets overlap by more than shared-edge roundoff."""

    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        overlap_start = max(left_start, right_start)
        overlap_end = min(left_end, right_end)
        if overlap_end > overlap_start and not _roundoff_equal(
            overlap_start, overlap_end
        ):
            return True
        if left_end < right_end:
            left_index += 1
        elif right_end < left_end:
            right_index += 1
        else:
            left_index += 1
            right_index += 1
    return False


def _interval_sets_roundoff_equal(
    left: tuple[_Interval, ...],
    right: tuple[_Interval, ...],
) -> bool:
    """Return whether interval topology and edges agree within roundoff only."""

    return len(left) == len(right) and all(
        _roundoff_equal(left_start, right_start)
        and _roundoff_equal(left_end, right_end)
        for (left_start, left_end), (right_start, right_end) in zip(
            left, right, strict=True
        )
    )


def _interval_length(intervals: tuple[_Interval, ...]) -> float:
    return math.fsum(end - start for start, end in intervals)


def _intersection_length(
    left: tuple[_Interval, ...],
    right: tuple[_Interval, ...],
) -> float:
    pieces: list[float] = []
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        start = max(left_start, right_start)
        end = min(left_end, right_end)
        if end > start:
            pieces.append(end - start)
        if left_end < right_end:
            left_index += 1
        elif right_end < left_end:
            right_index += 1
        else:
            left_index += 1
            right_index += 1
    return math.fsum(pieces)


def _nonnegative_difference(total: float, part: float) -> float:
    result = total - part
    if result >= 0.0:
        return result
    scale = max(abs(total), abs(part))
    if abs(result) <= _FRAME_COMPARISON_ULPS * math.ulp(scale):
        return 0.0
    raise ValidationError(
        "Interval arithmetic produced an invalid negative set length; the input "
        "records may be inconsistent."
    )


def _outer_span(intervals: tuple[_Interval, ...]) -> float | None:
    if not intervals:
        return None
    return intervals[-1][1] - intervals[0][0]


def _boundary_metrics(
    manual: tuple[_Interval, ...],
    candidate: tuple[_Interval, ...],
) -> BoundaryDistanceMetrics:
    manual_boundaries = tuple(value for interval in manual for value in interval)
    candidate_boundaries = tuple(value for interval in candidate for value in interval)
    if manual_boundaries and candidate_boundaries:
        manual_to_candidate = tuple(
            _nearest_distance(value, candidate_boundaries)
            for value in manual_boundaries
        )
        candidate_to_manual = tuple(
            _nearest_distance(value, manual_boundaries)
            for value in candidate_boundaries
        )
    else:
        manual_to_candidate = ()
        candidate_to_manual = ()
    return BoundaryDistanceMetrics(
        manual_boundaries_m=manual_boundaries,
        candidate_boundaries_m=candidate_boundaries,
        manual_to_candidate_boundary_distances_m=manual_to_candidate,
        candidate_to_manual_boundary_distances_m=candidate_to_manual,
    )


def _nearest_distance(value: float, sorted_boundaries: tuple[float, ...]) -> float:
    index = bisect_left(sorted_boundaries, value)
    candidates: list[float] = []
    if index:
        candidates.append(abs(value - sorted_boundaries[index - 1]))
    if index < len(sorted_boundaries):
        candidates.append(abs(sorted_boundaries[index] - value))
    return min(candidates)


def _difference_when_defined(
    after: float | None,
    before: float | None,
) -> float | None:
    if after is None or before is None:
        return None
    return after - before


def _mean_or_none(values: tuple[float, ...]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _median_or_none(values: tuple[float, ...]) -> float | None:
    return float(statistics.median(values)) if values else None


def _max_or_none(values: tuple[float, ...]) -> float | None:
    return max(values) if values else None


def _prefixed(values: Mapping[str, object], prefix: str) -> dict[str, object]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def _frozen_counts(values: Mapping[int, int]) -> Mapping[int, int]:
    return MappingProxyType({int(key): int(value) for key, value in values.items()})


__all__ = [
    "BOUNDARY_DIAGNOSTIC_WARNING",
    "BoundaryDistanceMetrics",
    "FRAME_COMPARISON_SEMANTICS",
    "IntervalSetMetrics",
    "IntervalValidationResult",
    "MANUAL_BENCHMARK_SCHEMA_VERSION",
    "ManualBenchmarkMetadata",
    "ManualBenchmarkRecord",
    "SensitivityValidationResult",
    "VALIDATION_METHOD_ID",
    "VALIDATION_METHOD_STATUS",
    "VALIDATION_METHOD_VERSION",
    "VALIDATION_METHOD_WARNING",
    "create_manual_benchmark_record",
    "evaluate_candidate_against_explicit",
    "evaluate_sensitivity_against_explicit",
]
