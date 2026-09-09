"""Explicit, analyst-supplied wet-interval width measurements.

Phase 5A.1 records measurements that a scientist supplies along an existing
manual transect.  It does not inspect PIXC station values to infer banks,
components, branches, or widths.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import WidthError
from .transect import TransectSample

EXPLICIT_INTERVAL_METHOD_ID = "explicit_analyst_wet_intervals_v1"
EXPLICIT_INTERVAL_METHOD_STATUS = "EXPERIMENTAL / MANUAL BENCHMARK CONTRACT"
EXPLICIT_INTERVAL_METHOD_WARNING = (
    "Wet interval boundaries are supplied by the analyst in Phase 5A.1. "
    "They are not automatically inferred from SWOT PIXC."
)


@dataclass(frozen=True, slots=True)
class WetInterval:
    """One ordered analyst-supplied wet interval on the transect."""

    interval_id: int
    start_station_m: float
    end_station_m: float

    @property
    def width_m(self) -> float:
        """Return the explicit interval length in metres."""

        return self.end_station_m - self.start_station_m

    def as_dict(self) -> dict[str, int | float]:
        """Return a tabular/JSON-friendly interval record."""

        return {
            "interval_id": self.interval_id,
            "start_station_m": self.start_station_m,
            "end_station_m": self.end_station_m,
            "width_m": self.width_m,
        }


@dataclass(frozen=True, slots=True)
class DryGap:
    """One explicit internal dry gap between consecutive wet intervals."""

    gap_id: int
    left_interval_id: int
    right_interval_id: int
    start_station_m: float
    end_station_m: float

    @property
    def width_m(self) -> float:
        """Return the explicit dry-gap length in metres."""

        return self.end_station_m - self.start_station_m

    def as_dict(self) -> dict[str, int | float]:
        """Return a tabular/JSON-friendly gap record."""

        return {
            "gap_id": self.gap_id,
            "left_interval_id": self.left_interval_id,
            "right_interval_id": self.right_interval_id,
            "start_station_m": self.start_station_m,
            "end_station_m": self.end_station_m,
            "width_m": self.width_m,
        }


@dataclass(frozen=True, slots=True)
class WidthSourceSummary:
    """Compact source-granule context without retaining pixel arrays or paths."""

    source_index: int
    selected_pixel_count: int
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
class ExplicitIntervalWidthResult:
    """Immutable, reference-free audit record for explicit wet intervals.

    The original :class:`~swot_pixc_lab.TransectSample` remains the pixel-level
    source.  This object stores only interval/gap measurements and compact
    summaries copied from that sample.
    """

    intervals: tuple[WetInterval, ...]
    dry_gaps: tuple[DryGap, ...]
    method_id: str
    method_status: str
    method_warning: str
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
    source_counts: Mapping[int, int]
    source_tile_counts: Mapping[str, int]
    source_summaries: tuple[WidthSourceSummary, ...]

    @property
    def interval_count(self) -> int:
        """Return the number of supplied wet intervals."""

        return len(self.intervals)

    @property
    def individual_interval_widths_m(self) -> tuple[float, ...]:
        """Return ordered wet-interval widths without physical branch claims."""

        return tuple(interval.width_m for interval in self.intervals)

    @property
    def total_wetted_width_m(self) -> float:
        """Return the sum of all explicit wet-interval widths."""

        return math.fsum(self.individual_interval_widths_m)

    @property
    def outer_wetted_span_m(self) -> float | None:
        """Return the first-start to last-end span, including internal gaps."""

        if not self.intervals:
            return None
        return self.intervals[-1].end_station_m - self.intervals[0].start_station_m

    @property
    def total_internal_dry_gap_m(self) -> float:
        """Return the sum of explicit positive gaps between wet intervals."""

        return math.fsum(gap.width_m for gap in self.dry_gaps)

    def intervals_dataframe(self) -> pd.DataFrame:
        """Return the ordered wet-interval records as a pandas table."""

        columns = ("interval_id", "start_station_m", "end_station_m", "width_m")
        return pd.DataFrame.from_records(
            (interval.as_dict() for interval in self.intervals), columns=columns
        )

    def gaps_dataframe(self) -> pd.DataFrame:
        """Return the ordered internal dry-gap records as a pandas table."""

        columns = (
            "gap_id",
            "left_interval_id",
            "right_interval_id",
            "start_station_m",
            "end_station_m",
            "width_m",
        )
        return pd.DataFrame.from_records(
            (gap.as_dict() for gap in self.dry_gaps), columns=columns
        )

    def audit_summary(self) -> dict[str, Any]:
        """Return a compact reconstruction and provenance summary."""

        return {
            "method_id": self.method_id,
            "method_status": self.method_status,
            "method_warning": self.method_warning,
            "intervals": [interval.as_dict() for interval in self.intervals],
            "dry_gaps": [gap.as_dict() for gap in self.dry_gaps],
            "interval_count": self.interval_count,
            "individual_interval_widths_m": self.individual_interval_widths_m,
            "total_wetted_width_m": self.total_wetted_width_m,
            "outer_wetted_span_m": self.outer_wetted_span_m,
            "total_internal_dry_gap_m": self.total_internal_dry_gap_m,
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
            "source_counts": dict(self.source_counts),
            "source_tile_counts": dict(self.source_tile_counts),
            "source_summaries": [asdict(source) for source in self.source_summaries],
        }


def measure_explicit_wet_intervals(
    sample: TransectSample,
    intervals: Iterable[Sequence[float]],
) -> ExplicitIntervalWidthResult:
    """Record ordered wet intervals supplied by an analyst.

    Boundaries use the ``station_m`` coordinate of ``sample`` and must be
    finite, non-negative, strictly increasing, disjoint, separated by a
    positive dry gap, and contained in the finite projected transect.  Input
    order is authoritative: intervals are never sorted, merged, deleted, or
    derived from PIXC samples.
    """

    if not isinstance(sample, TransectSample):
        raise TypeError(
            f"sample must be a Phase 4 TransectSample; got {type(sample).__name__}."
        )
    transect_length = _transect_length(sample)
    boundaries = _validate_intervals(intervals, transect_length)
    wet_intervals = tuple(
        WetInterval(
            interval_id=index,
            start_station_m=start,
            end_station_m=end,
        )
        for index, (start, end) in enumerate(boundaries, start=1)
    )
    dry_gaps = tuple(
        DryGap(
            gap_id=index,
            left_interval_id=left.interval_id,
            right_interval_id=right.interval_id,
            start_station_m=left.end_station_m,
            end_station_m=right.start_station_m,
        )
        for index, (left, right) in enumerate(
            zip(wet_intervals, wet_intervals[1:], strict=False), start=1
        )
    )

    source_counts = _frozen_counts(sample.source_counts)
    return ExplicitIntervalWidthResult(
        intervals=wet_intervals,
        dry_gaps=dry_gaps,
        method_id=EXPLICIT_INTERVAL_METHOD_ID,
        method_status=EXPLICIT_INTERVAL_METHOD_STATUS,
        method_warning=EXPLICIT_INTERVAL_METHOD_WARNING,
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
        classification_counts=_frozen_counts(sample.classification_counts),
        source_counts=source_counts,
        source_tile_counts=_frozen_counts(sample.source_tile_counts),
        source_summaries=_source_summaries(sample, source_counts),
    )


def _transect_length(sample: TransectSample) -> float:
    value = sample.transect_length_m
    if isinstance(value, (bool, np.bool_)):
        raise WidthError(
            "TransectSample.transect_length_m must be finite and positive."
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise WidthError(
            "TransectSample.transect_length_m must be finite and positive."
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise WidthError(
            "TransectSample.transect_length_m must be finite and positive."
        )
    return result


def _validate_intervals(
    intervals: Iterable[Sequence[float]],
    transect_length_m: float,
) -> tuple[tuple[float, float], ...]:
    if isinstance(intervals, (str, bytes, bytearray, Mapping, Set)):
        raise WidthError(
            "intervals must be an ordered iterable of (start_station_m, "
            "end_station_m) pairs."
        )
    try:
        supplied = tuple(intervals)
    except TypeError as exc:
        raise WidthError(
            "intervals must be an ordered iterable of (start_station_m, "
            "end_station_m) pairs."
        ) from exc

    validated: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for index, raw_interval in enumerate(supplied, start=1):
        if isinstance(raw_interval, (str, bytes, bytearray, Mapping, Set)):
            raise WidthError(
                f"Wet interval {index} must contain exactly two numeric stations."
            )
        try:
            pair = tuple(raw_interval)
        except TypeError as exc:
            raise WidthError(
                f"Wet interval {index} must contain exactly two numeric stations."
            ) from exc
        if len(pair) != 2:
            raise WidthError(
                f"Wet interval {index} must contain exactly two numeric stations."
            )
        start = _station_value(pair[0], interval_id=index, boundary="start")
        end = _station_value(pair[1], interval_id=index, boundary="end")
        if start < 0.0 or end < 0.0:
            raise WidthError(
                f"Wet interval {index} stations must be non-negative; got "
                f"({start!r}, {end!r})."
            )
        if end == start:
            raise WidthError(
                f"Wet interval {index} has zero width; end_station_m must be "
                "greater than start_station_m."
            )
        if end < start:
            raise WidthError(
                f"Wet interval {index} is reversed; end_station_m must be greater "
                "than start_station_m."
            )
        if end > transect_length_m:
            raise WidthError(
                f"Wet interval {index} ends at {end!r} m, beyond the projected "
                f"transect length {transect_length_m!r} m."
            )

        current = (start, end)
        if current in seen:
            raise WidthError(
                f"Wet interval {index} duplicates an earlier interval; duplicate "
                "intervals are not allowed."
            )
        if validated:
            previous_start, previous_end = validated[-1]
            if start < previous_start:
                raise WidthError(
                    f"Wet interval {index} is out of station order. Supply "
                    "intervals in increasing order; they are not sorted automatically."
                )
            if start < previous_end:
                raise WidthError(
                    f"Wet interval {index} overlaps interval {index - 1}. Explicit "
                    "wet intervals must be disjoint and are not merged automatically."
                )
            if start == previous_end:
                raise WidthError(
                    f"Wet interval {index} touches interval {index - 1}. Separate "
                    "wet components require an explicit positive dry gap; otherwise "
                    "supply one combined interval."
                )
        validated.append(current)
        seen.add(current)
    return tuple(validated)


def _station_value(value: object, *, interval_id: int, boundary: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise WidthError(
            f"Wet interval {interval_id} {boundary}_station_m must be a finite number."
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise WidthError(
            f"Wet interval {interval_id} {boundary}_station_m must be a finite number."
        ) from exc
    if not math.isfinite(result):
        raise WidthError(
            f"Wet interval {interval_id} {boundary}_station_m must be finite; "
            f"got {result!r}."
        )
    return result


def _frozen_counts(values: Mapping[Any, int]) -> Mapping[Any, int]:
    return MappingProxyType({key: int(value) for key, value in values.items()})


def _source_summaries(
    sample: TransectSample,
    counts: Mapping[int, int],
) -> tuple[WidthSourceSummary, ...]:
    summaries: list[WidthSourceSummary] = []
    known_indices: set[int] = set()
    for source in sample.sources:
        known_indices.add(source.source_index)
        summaries.append(
            WidthSourceSummary(
                source_index=source.source_index,
                selected_pixel_count=counts.get(source.source_index, 0),
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
        WidthSourceSummary(
            source_index=source_index,
            selected_pixel_count=count,
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
    "DryGap",
    "EXPLICIT_INTERVAL_METHOD_ID",
    "EXPLICIT_INTERVAL_METHOD_STATUS",
    "EXPLICIT_INTERVAL_METHOD_WARNING",
    "ExplicitIntervalWidthResult",
    "WetInterval",
    "WidthSourceSummary",
    "measure_explicit_wet_intervals",
]
