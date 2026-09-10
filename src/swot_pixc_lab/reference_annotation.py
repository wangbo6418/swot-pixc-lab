"""Blind, image-first collection of analyst-supplied wet intervals.

This module contains only geometry and annotation state.  It does not inspect
image pixels, infer optical edges, import Phase 5A.2 candidate inference, or
turn PIXC classifications into reference boundaries.  A click is snapped only
to the nearest point on the finite, approved transect and is measured in the
existing Phase 4 station frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path
from typing import Literal

from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError
from shapely import line_locate_point
from shapely.geometry import Point
from shapely.ops import transform

from .benchmark import (
    BenchmarkTransect,
    ReferenceImageryMetadata,
    save_manual_annotation,
)
from .exceptions import BenchmarkError
from .transect import TransectSample
from .validation import ManualBenchmarkRecord
from .width import ExplicitIntervalWidthResult, measure_explicit_wet_intervals

BoundaryRole = Literal["wet_start", "wet_end"]

_WGS84 = CRS.from_epsg(4326)


class ReferenceAnnotationError(BenchmarkError):
    """Report invalid or incomplete blind-reference annotation state."""


@dataclass(frozen=True, slots=True)
class ProjectedBoundaryPick:
    """One analyst click projected geometrically onto the fixed transect.

    ``input_x`` and ``input_y`` retain the clicked map coordinate.  The
    ``nearest_*`` fields identify the nearest position on the finite transect;
    no image values, PIXC values, or inferred edges participate in that
    projection.
    """

    input_x: float
    input_y: float
    input_crs: str
    nearest_longitude: float
    nearest_latitude: float
    station_m: float
    distance_to_transect_m: float
    role: BoundaryRole | None = None


@dataclass(frozen=True, slots=True)
class ReferenceAnnotationPreview:
    """Immutable preview derived by the authoritative Phase 5A.1 method."""

    measurement: ExplicitIntervalWidthResult
    reference_imagery: ReferenceImageryMetadata
    confidence: str | None
    notes: str | None
    pixc_context_viewed: bool

    @property
    def manual_intervals(self) -> tuple[tuple[float, float], ...]:
        """Return the exact analyst interval endpoints used in the preview."""

        return tuple(
            (interval.start_station_m, interval.end_station_m)
            for interval in self.measurement.intervals
        )


def project_click_to_station(
    sample: TransectSample,
    *,
    x: float,
    y: float,
    click_crs: str | CRS,
) -> ProjectedBoundaryPick:
    """Project one georeferenced map click onto ``sample``'s finite transect.

    The click and the ordered EPSG:4326 transect are transformed into the
    exact local metric CRS recorded by Phase 4.  Shapely then locates the
    nearest point on that finite line.  Consequently, station zero is always
    the first manifest endpoint and clicks beyond an endpoint project to that
    endpoint.  The operation never examines raster values or PIXC evidence.
    """

    if not isinstance(sample, TransectSample):
        raise TypeError("sample must be a TransectSample.")
    click_x = _finite_coordinate(x, "x")
    click_y = _finite_coordinate(y, "y")
    try:
        source_crs = CRS.from_user_input(click_crs)
    except (CRSError, TypeError, ValueError) as exc:
        raise ReferenceAnnotationError(
            f"click_crs is not a valid coordinate reference system: {click_crs!r}."
        ) from exc

    line_transformer = Transformer.from_crs(_WGS84, sample.local_crs, always_xy=True)
    click_transformer = Transformer.from_crs(
        source_crs, sample.local_crs, always_xy=True
    )
    reverse_transformer = Transformer.from_crs(sample.local_crs, _WGS84, always_xy=True)
    metric_line = transform(line_transformer.transform, sample.transect)
    metric_x, metric_y = click_transformer.transform(click_x, click_y)
    if not math.isfinite(metric_x) or not math.isfinite(metric_y):
        raise ReferenceAnnotationError(
            "The clicked coordinate cannot be transformed into the transect's "
            "recorded metric CRS."
        )
    metric_click = Point(metric_x, metric_y)
    station = float(line_locate_point(metric_line, metric_click))
    nearest = metric_line.interpolate(station)
    offset = float(metric_click.distance(nearest))
    longitude, latitude = reverse_transformer.transform(nearest.x, nearest.y)
    values = (station, offset, longitude, latitude)
    if not all(math.isfinite(value) for value in values):
        raise ReferenceAnnotationError(
            "The clicked coordinate did not produce a finite transect station."
        )
    return ProjectedBoundaryPick(
        input_x=click_x,
        input_y=click_y,
        input_crs=source_crs.to_string(),
        nearest_longitude=float(longitude),
        nearest_latitude=float(latitude),
        station_m=station,
        distance_to_transect_m=offset,
    )


class BlindReferenceAnnotationSession:
    """Mutable UI-neutral state for one approved, independent annotation.

    The selected image is fixed for the life of a session.  PIXC context is
    hidden by default and can only be marked visible by an explicit method
    call.  No candidate-inference object is accepted by this API.
    """

    def __init__(
        self,
        sample: TransectSample,
        transect: BenchmarkTransect,
        reference_imagery: ReferenceImageryMetadata,
    ) -> None:
        if not isinstance(sample, TransectSample):
            raise TypeError("sample must be a TransectSample.")
        if not isinstance(transect, BenchmarkTransect):
            raise TypeError("transect must be a BenchmarkTransect.")
        if not transect.approved:
            raise ReferenceAnnotationError(
                f"Transect {transect.transect_id!r} is "
                f"{transect.review_status!r}; blind annotation requires an "
                "explicitly approved geometry."
            )
        if sample.transect.wkt != transect.geometry.wkt:
            raise ReferenceAnnotationError(
                "The TransectSample geometry does not exactly match the approved "
                "manifest geometry. Regenerate the sample and annotation packet."
            )
        if not math.isclose(
            sample.corridor_half_width_m,
            transect.corridor_half_width_m,
            rel_tol=1.0e-12,
            abs_tol=1.0e-9,
        ):
            raise ReferenceAnnotationError(
                "The TransectSample corridor does not match the approved manifest."
            )
        _validate_reference_imagery(reference_imagery)

        self.sample = sample
        self.transect = transect
        self.reference_imagery = reference_imagery
        self._picks: list[ProjectedBoundaryPick] = []
        self._empty_declared = False
        self._pixc_context_visible = False
        self._pixc_context_ever_viewed = False
        self._preview: ReferenceAnnotationPreview | None = None

    @property
    def picks(self) -> tuple[ProjectedBoundaryPick, ...]:
        """Return analyst clicks in authoritative input order."""

        return tuple(self._picks)

    @property
    def next_role(self) -> BoundaryRole:
        """Return whether the next click is a wet start or wet end."""

        return "wet_start" if len(self._picks) % 2 == 0 else "wet_end"

    @property
    def empty_declared(self) -> bool:
        """Return whether the analyst deliberately declared no wet interval."""

        return self._empty_declared

    @property
    def pixc_context_visible(self) -> bool:
        """Return whether the analyst explicitly revealed PIXC context."""

        return self._pixc_context_visible

    @property
    def pixc_context_viewed(self) -> bool:
        """Return whether PIXC context has been revealed at any time."""

        return self._pixc_context_ever_viewed

    @property
    def current_preview(self) -> ReferenceAnnotationPreview | None:
        """Return the current preview, invalidated whenever review state changes."""

        return self._preview

    @property
    def completed_intervals(self) -> tuple[tuple[float, float], ...]:
        """Return complete start/end pairs, excluding an unmatched final start."""

        stations = tuple(pick.station_m for pick in self._picks)
        complete_count = len(stations) - (len(stations) % 2)
        return tuple(
            (stations[index], stations[index + 1])
            for index in range(0, complete_count, 2)
        )

    def add_click(
        self,
        *,
        x: float,
        y: float,
        click_crs: str | CRS,
    ) -> ProjectedBoundaryPick:
        """Project and append one analyst-selected boundary click."""

        pick = project_click_to_station(
            self.sample,
            x=x,
            y=y,
            click_crs=click_crs,
        )
        return self.add_projected_pick(pick)

    def add_projected_pick(self, pick: ProjectedBoundaryPick) -> ProjectedBoundaryPick:
        """Append a pre-projected click after ordered-boundary validation."""

        if not isinstance(pick, ProjectedBoundaryPick):
            raise TypeError("pick must be a ProjectedBoundaryPick.")
        station = _station_in_domain(pick.station_m, self.sample.transect_length_m)
        if self._picks and station <= self._picks[-1].station_m:
            role = self.next_role.replace("_", " ")
            raise ReferenceAnnotationError(
                f"The next {role} must be at a station greater than the previous "
                "boundary. Click order is never sorted automatically."
            )
        resolved = replace(pick, station_m=station, role=self.next_role)
        self._picks.append(resolved)
        self._empty_declared = False
        self._invalidate_preview()
        return resolved

    def undo(self) -> ProjectedBoundaryPick | None:
        """Remove and return the final click, or clear an empty declaration."""

        if self._picks:
            result = self._picks.pop()
        else:
            result = None
        self._empty_declared = False
        self._invalidate_preview()
        return result

    def reset(self) -> None:
        """Clear all draft boundaries and any explicit empty declaration."""

        self._picks.clear()
        self._empty_declared = False
        self._invalidate_preview()

    def declare_empty(self) -> None:
        """Create a deliberate empty draft without inferring absence of water."""

        self._picks.clear()
        self._empty_declared = True
        self._invalidate_preview()

    def set_pixc_context_visible(self, visible: bool) -> None:
        """Record an explicit PIXC-context toggle; default is always hidden."""

        if not isinstance(visible, bool):
            raise TypeError("visible must be boolean.")
        if visible != self._pixc_context_visible:
            self._pixc_context_visible = visible
            self._pixc_context_ever_viewed |= visible
            self._invalidate_preview()

    def preview(
        self,
        *,
        confidence: str | None = None,
        notes: str | None = None,
    ) -> ReferenceAnnotationPreview:
        """Measure the draft through Phase 5A.1 and freeze a review preview."""

        intervals = self._intervals_for_preview()
        measurement = measure_explicit_wet_intervals(self.sample, intervals)
        preview = ReferenceAnnotationPreview(
            measurement=measurement,
            reference_imagery=self.reference_imagery,
            confidence=confidence,
            notes=notes,
            pixc_context_viewed=self._pixc_context_ever_viewed,
        )
        self._preview = preview
        return preview

    def confirm_and_save(
        self,
        template_path: str | PathLike[str],
        output_path: str | PathLike[str],
        *,
        annotation_id: str,
        analyst_id: str,
        confidence: str | None = None,
        notes: str | None = None,
        confirmed: bool,
    ) -> ManualBenchmarkRecord:
        """Save only when the unchanged preview is explicitly confirmed."""

        if not confirmed:
            raise ReferenceAnnotationError(
                "Blind-reference annotation must be explicitly confirmed before save."
            )
        intervals = self._intervals_for_preview()
        preview = self._preview
        if preview is None:
            raise ReferenceAnnotationError(
                "Preview the selected intervals before confirming and saving."
            )
        if (
            preview.manual_intervals != intervals
            or preview.reference_imagery != self.reference_imagery
            or preview.confidence != confidence
            or preview.notes != notes
            or preview.pixc_context_viewed != self._pixc_context_ever_viewed
        ):
            self._invalidate_preview()
            raise ReferenceAnnotationError(
                "Annotation details changed after preview; preview again before save."
            )
        return save_manual_annotation(
            self.sample,
            template_path,
            output_path,
            annotation_id=annotation_id,
            analyst_id=analyst_id,
            manual_intervals=intervals,
            confidence=confidence,
            notes=notes,
            reference_imagery=self._imagery_with_context_audit(),
            confirmed=True,
        )

    def _intervals_for_preview(self) -> tuple[tuple[float, float], ...]:
        if self._empty_declared:
            return ()
        if not self._picks:
            raise ReferenceAnnotationError(
                "No boundary is selected. Add ordered wet intervals or explicitly "
                "declare no visible wet interval."
            )
        if len(self._picks) % 2:
            raise ReferenceAnnotationError(
                "Add an end boundary for the final wet interval before preview."
            )
        return self.completed_intervals

    def _invalidate_preview(self) -> None:
        self._preview = None

    def _imagery_with_context_audit(self) -> ReferenceImageryMetadata:
        audit = (
            "Blind-reference annotation UI: optional PIXC context viewed="
            f"{str(self._pixc_context_ever_viewed).lower()}."
        )
        existing = self.reference_imagery.analyst_notes
        notes = f"{existing} {audit}" if existing else audit
        return replace(self.reference_imagery, analyst_notes=notes)


def _validate_reference_imagery(value: ReferenceImageryMetadata) -> None:
    if not isinstance(value, ReferenceImageryMetadata):
        raise TypeError("reference_imagery must be ReferenceImageryMetadata.")
    if value.pending:
        raise ReferenceAnnotationError(
            "A selected independent reference image is required before annotation."
        )
    missing_metadata = [
        name
        for name in ("source_provider", "acquisition_date", "source_identifier")
        if not getattr(value, name)
    ]
    if missing_metadata:
        raise ReferenceAnnotationError(
            "Selected reference imagery is missing required provenance: "
            + ", ".join(missing_metadata)
            + "."
        )
    if value.local_image_path is None:
        raise ReferenceAnnotationError(
            "Selected reference imagery has no local georeferenced image path."
        )
    path = Path(value.local_image_path).expanduser()
    if not path.is_file():
        raise ReferenceAnnotationError(
            f"Selected local reference image is missing: {path}."
        )
    if value.temporal_offset_hours is None:
        raise ReferenceAnnotationError(
            "Selected reference imagery has no finite Sentinel-minus-SWOT "
            "temporal offset."
        )


def _finite_coordinate(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ReferenceAnnotationError(f"Click {label} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReferenceAnnotationError(
            f"Click {label} must be a finite number."
        ) from exc
    if not math.isfinite(result):
        raise ReferenceAnnotationError(f"Click {label} must be a finite number.")
    return result


def _station_in_domain(value: object, transect_length_m: float) -> float:
    station = _finite_coordinate(value, "station")
    if not 0.0 <= station <= transect_length_m:
        raise ReferenceAnnotationError(
            "Projected station is outside the finite transect domain "
            f"[0, {transect_length_m!r}]."
        )
    return station


__all__ = [
    "BlindReferenceAnnotationSession",
    "BoundaryRole",
    "ProjectedBoundaryPick",
    "ReferenceAnnotationError",
    "ReferenceAnnotationPreview",
    "project_click_to_station",
]
