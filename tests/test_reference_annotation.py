from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS, Transformer
from shapely.geometry import LineString

import swot_pixc_lab.reference_annotation as annotation_module
from swot_pixc_lab.benchmark import (
    ANNOTATION_INPUT_SCHEMA_VERSION,
    APPROVED_TRANSECT_STATUS,
    PROPOSED_TRANSECT_STATUS,
    BenchmarkTransect,
    ReferenceImageryMetadata,
    station_frame_hash,
)
from swot_pixc_lab.reference_annotation import (
    BlindReferenceAnnotationSession,
    ProjectedBoundaryPick,
    ReferenceAnnotationError,
    project_click_to_station,
)
from swot_pixc_lab.transect import TransectSample


def _sample(*, reverse: bool = False) -> TransectSample:
    center = (87.0, 26.6)
    local_crs = CRS.from_proj4(
        "+proj=aeqd +lat_0=26.6 +lon_0=87 +datum=WGS84 +units=m +no_defs"
    )
    reverse_transformer = Transformer.from_crs(
        local_crs, CRS.from_epsg(4326), always_xy=True
    )
    endpoints = [
        reverse_transformer.transform(-500.0, 0.0),
        reverse_transformer.transform(500.0, 0.0),
    ]
    if reverse:
        endpoints.reverse()
    line = LineString(endpoints)
    pixels = xr.Dataset(
        {"source_index": xr.DataArray(np.array([], dtype=np.int32), dims=("points",))}
    )
    return TransectSample(
        pixels=pixels,
        transect=line,
        corridor=line.buffer(0.001),
        corridor_half_width_m=50.0,
        local_crs=local_crs,
        projection_center=center,
        profile_name="synthetic_reference_test",
        profile_label="Synthetic reference test",
        profile_status="TEST ONLY",
        sources=(),
        input_source_indices=(0,),
        input_pixel_count=0,
        invalid_coordinate_count=0,
    )


def _transect(sample: TransectSample, *, approved: bool = True) -> BenchmarkTransect:
    return BenchmarkTransect(
        transect_id="T002F",
        geometry=sample.transect,
        corridor_half_width_m=sample.corridor_half_width_m,
        review_status=(
            APPROVED_TRANSECT_STATUS if approved else PROPOSED_TRANSECT_STATUS
        ),
    )


def _imagery(tmp_path: Path, *, pending: bool = False) -> ReferenceImageryMetadata:
    image = tmp_path / "independent_rgb.tif"
    image.write_bytes(b"synthetic georeferenced raster placeholder")
    return ReferenceImageryMetadata(
        status=(
            "independent imagery pending"
            if pending
            else "selected independent Sentinel-2 L2A reference imagery"
        ),
        source_provider="synthetic STAC provider",
        acquisition_date="2024-01-15T05:00:00Z",
        temporal_offset_hours=20.0,
        local_image_path=image,
        source_identifier="sentinel-2-l2a:SYNTHETIC_ITEM",
    )


def _session(
    tmp_path: Path, *, reverse: bool = False
) -> BlindReferenceAnnotationSession:
    sample = _sample(reverse=reverse)
    return BlindReferenceAnnotationSession(
        sample,
        _transect(sample),
        _imagery(tmp_path),
    )


def _pick(station: float) -> ProjectedBoundaryPick:
    return ProjectedBoundaryPick(
        input_x=station,
        input_y=0.0,
        input_crs="synthetic metric",
        nearest_longitude=87.0,
        nearest_latitude=26.6,
        station_m=station,
        distance_to_transect_m=0.0,
    )


def _write_template(
    path: Path,
    sample: TransectSample,
    imagery: ReferenceImageryMetadata,
) -> None:
    payload = {
        "schema_version": ANNOTATION_INPUT_SCHEMA_VERSION,
        "status": "pending",
        "confirmed": False,
        "benchmark_id": "synthetic-reference-benchmark",
        "transect_id": "T002F",
        "annotation_id": None,
        "analyst_id": None,
        "benchmark_subset": "pilot",
        "observation_identifier": "synthetic-observation",
        "observation_date": "2024-01-14",
        "site_label": None,
        "morphology_label": None,
        "review_status": APPROVED_TRANSECT_STATUS,
        "station_frame_hash": station_frame_hash(sample),
        "manual_intervals": None,
        "confidence": None,
        "notes": None,
        "reference_imagery": imagery.as_dict(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_click_projection_uses_metric_nearest_point_and_finite_endpoints() -> None:
    sample = _sample()

    middle = project_click_to_station(
        sample,
        x=0.0,
        y=100.0,
        click_crs=sample.local_crs,
    )
    before_start = project_click_to_station(
        sample,
        x=-750.0,
        y=0.0,
        click_crs=sample.local_crs,
    )
    after_end = project_click_to_station(
        sample,
        x=750.0,
        y=0.0,
        click_crs=sample.local_crs,
    )

    assert middle.station_m == pytest.approx(500.0, abs=0.02)
    assert middle.distance_to_transect_m == pytest.approx(100.0, abs=0.02)
    assert before_start.station_m == pytest.approx(0.0, abs=0.02)
    assert before_start.distance_to_transect_m == pytest.approx(250.0, abs=0.02)
    assert after_end.station_m == pytest.approx(1000.0, abs=0.02)
    assert after_end.distance_to_transect_m == pytest.approx(250.0, abs=0.02)


def test_click_projection_accepts_non_wgs84_image_crs_and_preserves_direction() -> None:
    sample = _sample()
    reverse_sample = _sample(reverse=True)
    local_point = (-250.0, 80.0)
    image_crs = CRS.from_epsg(32645)
    transformer = Transformer.from_crs(sample.local_crs, image_crs, always_xy=True)
    image_x, image_y = transformer.transform(*local_point)

    forward = project_click_to_station(
        sample, x=image_x, y=image_y, click_crs=image_crs
    )
    backward = project_click_to_station(
        reverse_sample, x=image_x, y=image_y, click_crs=image_crs
    )

    assert forward.station_m == pytest.approx(250.0, abs=0.03)
    assert backward.station_m == pytest.approx(750.0, abs=0.03)
    assert forward.distance_to_transect_m == pytest.approx(80.0, abs=0.03)
    assert backward.distance_to_transect_m == pytest.approx(80.0, abs=0.03)


@pytest.mark.parametrize(
    ("x", "y", "crs"),
    [
        (float("nan"), 0.0, "EPSG:4326"),
        (0.0, float("inf"), "EPSG:4326"),
        (0.0, 0.0, "not-a-crs"),
    ],
)
def test_click_projection_rejects_invalid_georeferencing(
    x: float, y: float, crs: str
) -> None:
    with pytest.raises(ReferenceAnnotationError, match="finite|coordinate reference"):
        project_click_to_station(_sample(), x=x, y=y, click_crs=crs)


def test_session_requires_approved_geometry_before_imagery(tmp_path: Path) -> None:
    sample = _sample()
    missing = ReferenceImageryMetadata(
        status="selected independent reference imagery",
        source_provider="provider",
        acquisition_date="2024-01-15",
        source_identifier="item",
        local_image_path=tmp_path / "missing.tif",
    )

    with pytest.raises(ReferenceAnnotationError, match="approved geometry"):
        BlindReferenceAnnotationSession(
            sample,
            _transect(sample, approved=False),
            missing,
        )


def test_session_rejects_pending_or_missing_reference_imagery(tmp_path: Path) -> None:
    sample = _sample()
    with pytest.raises(ReferenceAnnotationError, match="selected independent"):
        BlindReferenceAnnotationSession(
            sample,
            _transect(sample),
            _imagery(tmp_path, pending=True),
        )

    missing = ReferenceImageryMetadata(
        status="selected independent reference imagery",
        source_provider="provider",
        acquisition_date="2024-01-15",
        source_identifier="item",
        local_image_path=tmp_path / "missing.tif",
    )
    with pytest.raises(ReferenceAnnotationError, match="missing"):
        BlindReferenceAnnotationSession(sample, _transect(sample), missing)

    no_offset = replace(_imagery(tmp_path), temporal_offset_hours=None)
    with pytest.raises(ReferenceAnnotationError, match="temporal offset"):
        BlindReferenceAnnotationSession(sample, _transect(sample), no_offset)


def test_blind_session_starts_empty_and_pixc_requires_explicit_toggle(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)

    assert session.picks == ()
    assert session.completed_intervals == ()
    assert session.empty_declared is False
    assert session.pixc_context_visible is False
    assert session.pixc_context_viewed is False
    assert session.current_preview is None

    session.set_pixc_context_visible(True)
    session.set_pixc_context_visible(False)

    assert session.pixc_context_visible is False
    assert session.pixc_context_viewed is True
    assert session.picks == ()
    assert session.completed_intervals == ()
    source = Path(annotation_module.__file__).read_text(encoding="utf-8")
    assert "infer_candidate_wet_intervals" not in source
    assert "CandidateIntervalInferenceResult" not in source


def test_click_roles_are_alternating_ordered_and_never_sorted(tmp_path: Path) -> None:
    session = _session(tmp_path)

    first = session.add_projected_pick(_pick(100.0))
    second = session.add_projected_pick(_pick(200.0))

    assert first.role == "wet_start"
    assert second.role == "wet_end"
    assert session.next_role == "wet_start"
    with pytest.raises(ReferenceAnnotationError, match="never sorted"):
        session.add_projected_pick(_pick(150.0))
    assert tuple(pick.station_m for pick in session.picks) == (100.0, 200.0)


def test_preview_uses_phase5a1_and_edits_invalidate_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path)
    session.add_projected_pick(_pick(100.0))
    session.add_projected_pick(_pick(225.0))
    original = annotation_module.measure_explicit_wet_intervals
    calls: list[tuple[tuple[float, float], ...]] = []

    def counted_measure(sample, intervals):
        resolved = tuple(tuple(item) for item in intervals)
        calls.append(resolved)
        return original(sample, resolved)

    monkeypatch.setattr(
        annotation_module, "measure_explicit_wet_intervals", counted_measure
    )
    preview = session.preview(confidence="medium", notes="image interpretation")

    assert calls == [((100.0, 225.0),)]
    assert preview.measurement.total_wetted_width_m == 125.0
    assert preview.measurement.outer_wetted_span_m == 125.0
    assert session.current_preview is preview

    session.add_projected_pick(_pick(300.0))
    assert session.current_preview is None
    with pytest.raises(ReferenceAnnotationError, match="end boundary"):
        session.preview()
    removed = session.undo()
    assert removed is not None and removed.station_m == 300.0
    assert session.current_preview is None


def test_explicit_empty_is_distinct_from_unstarted_and_can_be_previewed(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    with pytest.raises(ReferenceAnnotationError, match="No boundary"):
        session.preview()

    session.declare_empty()
    preview = session.preview(notes="No visible wet interval in selected image.")

    assert preview.manual_intervals == ()
    assert preview.measurement.total_wetted_width_m == 0.0
    assert preview.measurement.outer_wetted_span_m is None
    session.reset()
    assert session.empty_declared is False
    assert session.current_preview is None


def test_confirm_requires_unchanged_preview_and_existing_save_machinery(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    session.add_projected_pick(_pick(100.0))
    session.add_projected_pick(_pick(200.0))
    template = tmp_path / "annotation_template.json"
    output = tmp_path / "annotation.json"
    _write_template(template, session.sample, session.reference_imagery)

    with pytest.raises(ReferenceAnnotationError, match="Preview"):
        session.confirm_and_save(
            template,
            output,
            annotation_id="T002F-analyst-a",
            analyst_id="analyst-a",
            confirmed=True,
        )
    with pytest.raises(ReferenceAnnotationError, match="explicitly confirmed"):
        session.confirm_and_save(
            template,
            output,
            annotation_id="T002F-analyst-a",
            analyst_id="analyst-a",
            confirmed=False,
        )

    session.preview(confidence="high", notes="selected independent RGB")
    with pytest.raises(ReferenceAnnotationError, match="changed after preview"):
        session.confirm_and_save(
            template,
            output,
            annotation_id="T002F-analyst-a",
            analyst_id="analyst-a",
            confidence="low",
            notes="selected independent RGB",
            confirmed=True,
        )
    assert not output.exists()

    session.preview(confidence="high", notes="selected independent RGB")
    record = session.confirm_and_save(
        template,
        output,
        annotation_id="T002F-analyst-a",
        analyst_id="analyst-a",
        confidence="high",
        notes="selected independent RGB",
        confirmed=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["manual_intervals"] == [[100.0, 200.0]]
    assert payload["reference_imagery"]["source_identifier"].endswith("SYNTHETIC_ITEM")
    assert "PIXC context viewed=false" in payload["reference_imagery"]["analyst_notes"]
    assert record.reference.total_wetted_width_m == 100.0
    assert record.metadata.annotation_id == "T002F-analyst-a"
    assert not {
        "total_wetted_width_m",
        "outer_wetted_span_m",
        "total_internal_dry_gap_m",
    }.intersection(payload)


def test_session_does_not_mutate_approved_geometry_or_selected_image(
    tmp_path: Path,
) -> None:
    sample = _sample()
    transect = _transect(sample)
    imagery = _imagery(tmp_path)
    geometry_before = tuple(transect.geometry.coords)
    imagery_before = imagery.as_dict()
    session = BlindReferenceAnnotationSession(sample, transect, imagery)

    session.add_click(x=0.0, y=25.0, click_crs=sample.local_crs)
    session.undo()
    session.declare_empty()
    session.preview()

    assert tuple(transect.geometry.coords) == geometry_before
    assert tuple(sample.transect.coords) == geometry_before
    assert imagery.as_dict() == imagery_before
