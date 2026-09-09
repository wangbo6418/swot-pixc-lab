from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import CRS, Transformer
from shapely.geometry import LineString

from swot_pixc_lab.bank_inference import CandidateIntervalConfiguration
from swot_pixc_lab.benchmark import (
    ANNOTATION_REQUIRED_MESSAGE,
    APPROVED_TRANSECT_STATUS,
    PILOT_GRID_NOTICE,
    PROPOSED_TRANSECT_STATUS,
    BenchmarkConfig,
    BenchmarkError,
    BenchmarkTransect,
    ReferenceImageryMetadata,
    SensitivityGrid,
    TransectManifest,
    TransectProposalSettings,
    aggregate_benchmark_results,
    expand_sensitivity_grid,
    generate_annotation_packets,
    load_benchmark_config,
    load_manual_annotations,
    load_transect_manifest,
    propose_candidate_transects,
    run_benchmark,
    save_manual_annotation,
    write_transect_manifest,
)
from swot_pixc_lab.exceptions import AnnotationRequiredError
from swot_pixc_lab.transect import sample_transect

matplotlib.use("Agg", force=True)

_TEST_CRS = CRS.from_proj4(
    "+proj=aeqd +lat_0=26.7 +lon_0=87 +ellps=WGS84 +units=m +type=crs"
)
_TO_GEOGRAPHIC = Transformer.from_crs(_TEST_CRS, "EPSG:4326", always_xy=True)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _geographic(x: object, y: object) -> tuple[np.ndarray, np.ndarray]:
    longitude, latitude = _TO_GEOGRAPHIC.transform(x, y)
    return np.asarray(longitude), np.asarray(latitude)


def _pixel_dataset(
    *,
    stations: tuple[float, ...] = (25.0, 75.0, 225.0, 275.0),
    classifications: tuple[int, ...] = (4, 4, 4, 4),
    y_m: float = 0.0,
) -> xr.Dataset:
    assert len(stations) == len(classifications)
    longitude, latitude = _geographic(
        np.asarray(stations, dtype=np.float64) - 500.0,
        np.full(len(stations), y_m, dtype=np.float64),
    )
    return xr.Dataset(
        {
            "longitude": xr.DataArray(longitude, dims="points"),
            "latitude": xr.DataArray(latitude, dims="points"),
            "classification": xr.DataArray(
                np.asarray(classifications, dtype=np.uint8),
                dims="points",
                attrs={"_FillValue": np.uint8(255)},
            ),
            "height": xr.DataArray(
                np.linspace(100.0, 101.0, len(stations), dtype=np.float32),
                dims="points",
                attrs={"units": "m"},
            ),
            "source_index": xr.DataArray(
                np.zeros(len(stations), dtype=np.int32), dims="points"
            ),
            "source_point_index": xr.DataArray(
                np.arange(len(stations), dtype=np.int64), dims="points"
            ),
        },
        attrs={
            "swot_pixc_lab_qc_profile": "channel_extent_candidate",
            "swot_pixc_lab_qc_label": "Channel extent candidate",
            "swot_pixc_lab_qc_status": "experimental / unvalidated",
            "test_marker": "unchanged",
        },
    )


def _line(*, y_m: float = 0.0) -> LineString:
    longitude, latitude = _geographic([-500.0, 500.0], [y_m, y_m])
    return LineString(zip(longitude, latitude, strict=True))


def _sample(
    *,
    y_m: float = 0.0,
    stations: tuple[float, ...] = (25.0, 75.0, 225.0, 275.0),
    classifications: tuple[int, ...] = (4, 4, 4, 4),
):
    sample = sample_transect(
        _pixel_dataset(
            stations=stations,
            classifications=classifications,
            y_m=y_m,
        ),
        _line(y_m=y_m),
        corridor_half_width_m=50.0,
    )
    pixels = sample.pixels.copy(deep=True)
    pixels["station_m"] = xr.DataArray(
        np.asarray(stations, dtype=np.float64),
        dims="points",
        attrs=sample.station_m.attrs,
    )
    return replace(sample, pixels=pixels)


def _config_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "benchmark_id": "synthetic-pilot",
        "benchmark_subset": "pilot",
        "status": "EXPERIMENTAL / PILOT",
        "observation": {
            "identifier": "SYNTHETIC_OBSERVATION",
            "date": "2024-01-14",
            "cycle": 9,
            "pass": 286,
            "tiles": ["107L", "108L"],
            "aoi": [86.87, 26.49, 87.20, 26.90],
            "filenames": ["synthetic-107L.nc", "synthetic-108L.nc"],
        },
        "local_data_directory": "local-data",
        "qc_profile": "channel_extent_candidate",
        "corridor_half_width_m": 50.0,
        "transect_manifest_path": "inputs/transects.geojson",
        "annotation_directory": "annotations",
        "output_directory": "outputs",
        "reference_imagery": {
            "status": "independent imagery pending",
            "source_provider": None,
            "acquisition_date": None,
            "temporal_offset_hours": None,
            "cloud_quality_note": None,
            "local_image_path": None,
            "source_identifier": None,
            "analyst_notes": None,
        },
        "packet_configuration": {
            "extent_classes": [4],
            "station_bin_width_m": 50.0,
            "min_extent_pixels_per_bin": 1,
            "max_bridge_gap_m": 50.0,
        },
        "sensitivity_grid": {
            "notice": PILOT_GRID_NOTICE,
            "extent_class_sets": [[4], [3, 4]],
            "station_bin_widths_m": [50.0, 100.0],
            "min_extent_pixels_per_bin": [1],
            "max_bridge_gaps_m": [0.0, 50.0],
        },
        "transect_proposals": {
            "enabled": True,
            "count": 2,
            "stratification_axis": "latitude",
            "transect_azimuth_degrees": 90.0,
            "line_length_m": 1000.0,
        },
    }


def _write_config(tmp_path: Path, payload: dict[str, object] | None = None) -> Path:
    path = tmp_path / "config" / "pilot.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload or _config_payload()), encoding="utf-8")
    return path


def _manifest(*, proposed: bool = False) -> TransectManifest:
    status = PROPOSED_TRANSECT_STATUS if proposed else APPROVED_TRANSECT_STATUS
    return TransectManifest(
        benchmark_id="synthetic-pilot",
        transects=(
            BenchmarkTransect(
                transect_id="T001",
                geometry=_line(),
                corridor_half_width_m=50.0,
                review_status=status,
                site_label="Synthetic reach A",
                morphology_label="separated support",
                selection_notes="Synthetic test geometry only.",
            ),
            BenchmarkTransect(
                transect_id="T002",
                geometry=_line(y_m=100.0),
                corridor_half_width_m=50.0,
                review_status=status,
                site_label="Synthetic reach B",
                morphology_label="sparse evidence",
                selection_notes="Synthetic test geometry only.",
            ),
        ),
    )


def _spatial_dataset() -> xr.Dataset:
    x = np.asarray([-400, -200, 0, 200, 400] * 3, dtype=np.float64)
    y = np.repeat(np.asarray([-300, 0, 300], dtype=np.float64), 5)
    longitude, latitude = _geographic(x, y)
    count = len(x)
    return xr.Dataset(
        {
            "longitude": xr.DataArray(longitude, dims="points"),
            "latitude": xr.DataArray(latitude, dims="points"),
            "classification": xr.DataArray(
                np.resize(np.asarray([1, 3, 4, 5, 7], dtype=np.uint8), count),
                dims="points",
                attrs={"_FillValue": np.uint8(255)},
            ),
            "source_index": xr.DataArray(
                np.zeros(count, dtype=np.int32), dims="points"
            ),
            "source_point_index": xr.DataArray(
                np.arange(count, dtype=np.int64), dims="points"
            ),
        },
        attrs={"test_marker": "proposal-input-unchanged"},
    )


def _prepared_run(tmp_path: Path):
    payload = _config_payload()
    grid = payload["sensitivity_grid"]
    assert isinstance(grid, dict)
    grid["extent_class_sets"] = [[4], [3, 4]]
    grid["station_bin_widths_m"] = [100.0]
    grid["min_extent_pixels_per_bin"] = [1]
    grid["max_bridge_gaps_m"] = [0.0]
    config = load_benchmark_config(_write_config(tmp_path, payload))
    manifest = _manifest()
    write_transect_manifest(manifest, config.transect_manifest_path)
    preparation = generate_annotation_packets(_pixel_dataset(), config, manifest)
    config.annotation_directory.mkdir(parents=True, exist_ok=True)

    definitions = (
        (
            "T001",
            "01-T001-a.json",
            "annotation-a",
            "analyst-a",
            "pilot",
            ((0.0, 100.0), (200.0, 300.0)),
        ),
        (
            "T001",
            "02-T001-b.json",
            "annotation-b",
            "analyst-b",
            "development",
            ((25.0, 125.0), (200.0, 300.0)),
        ),
        ("T002", "03-T002-a.json", "annotation-c", "analyst-a", "holdout", ()),
    )
    packet_by_id = {
        packet.transect.transect_id: packet for packet in preparation.packets
    }
    for (
        transect_id,
        filename,
        annotation_id,
        analyst_id,
        subset,
        intervals,
    ) in definitions:
        packet = packet_by_id[transect_id]
        output = config.annotation_directory / filename
        save_manual_annotation(
            packet.sample,
            packet.directory / "annotation_template.json",
            output,
            annotation_id=annotation_id,
            analyst_id=analyst_id,
            manual_intervals=intervals,
            confidence="synthetic-test-only",
            notes="Synthetic annotation; not real reference truth.",
            confirmed=True,
        )
        annotation_payload = json.loads(output.read_text(encoding="utf-8"))
        annotation_payload["benchmark_subset"] = subset
        output.write_text(json.dumps(annotation_payload, indent=2), encoding="utf-8")
    annotations = load_manual_annotations(
        config.annotation_directory,
        benchmark_id=config.benchmark_id,
    )
    return config, manifest, preparation, annotations


def test_config_paths_are_resolved_from_file_and_grid_is_explicit(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)

    config = load_benchmark_config(config_path)

    assert isinstance(config, BenchmarkConfig)
    assert config.source_path == config_path.resolve()
    assert config.observation.local_files == (
        (config_path.parent / "local-data" / "synthetic-107L.nc").resolve(),
        (config_path.parent / "local-data" / "synthetic-108L.nc").resolve(),
    )
    assert (
        config.transect_manifest_path
        == (config_path.parent / "inputs" / "transects.geojson").resolve()
    )
    assert config.annotation_directory == (config_path.parent / "annotations").resolve()
    assert config.output_directory == (config_path.parent / "outputs").resolve()
    assert config.reference_imagery.pending is True
    assert config.sensitivity_grid.notice == PILOT_GRID_NOTICE


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"unexpected": True}), "unexpected"),
        (lambda value: value.update({"benchmark_subset": "random"}), "subset"),
        (
            lambda value: value["sensitivity_grid"].update(  # type: ignore[union-attr]
                {"notice": "recommended defaults"}
            ),
            "exploratory|recommended",
        ),
        (
            lambda value: value["sensitivity_grid"].update(  # type: ignore[union-attr]
                {"station_bin_widths_m": []}
            ),
            "non-empty",
        ),
        (
            lambda value: value["observation"].update(  # type: ignore[union-attr]
                {"aoi": [87.2, 26.49, 86.87, 26.90]}
            ),
            "bbox",
        ),
    ],
)
def test_config_validation_is_strict_and_actionable(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    payload = _config_payload()
    mutation(payload)

    with pytest.raises(BenchmarkError, match=message):
        load_benchmark_config(_write_config(tmp_path, payload))


def test_cartesian_grid_expansion_preserves_declared_order_without_ranking() -> None:
    grid = SensitivityGrid(
        extent_class_sets=((4,), (3, 4)),
        station_bin_widths_m=(10.0, 25.0),
        min_extent_pixels_per_bin=(1, 2),
        max_bridge_gaps_m=(0.0, 50.0),
        notice=PILOT_GRID_NOTICE,
    )

    first = expand_sensitivity_grid(grid)
    second = expand_sensitivity_grid(grid)

    assert first == second
    assert len(first) == 16
    assert all(isinstance(item, CandidateIntervalConfiguration) for item in first)
    assert [item.as_dict() for item in first[:5]] == [
        {
            "extent_classes": (4,),
            "station_bin_width_m": 10.0,
            "min_extent_pixels_per_bin": 1,
            "max_bridge_gap_m": 0.0,
        },
        {
            "extent_classes": (4,),
            "station_bin_width_m": 10.0,
            "min_extent_pixels_per_bin": 1,
            "max_bridge_gap_m": 50.0,
        },
        {
            "extent_classes": (4,),
            "station_bin_width_m": 10.0,
            "min_extent_pixels_per_bin": 2,
            "max_bridge_gap_m": 0.0,
        },
        {
            "extent_classes": (4,),
            "station_bin_width_m": 10.0,
            "min_extent_pixels_per_bin": 2,
            "max_bridge_gap_m": 50.0,
        },
        {
            "extent_classes": (4,),
            "station_bin_width_m": 25.0,
            "min_extent_pixels_per_bin": 1,
            "max_bridge_gap_m": 0.0,
        },
    ]
    assert not hasattr(first[0], "rank")
    assert not hasattr(first[0], "best")


def test_transect_manifest_round_trip_preserves_order_status_and_geometry(
    tmp_path: Path,
) -> None:
    manifest = _manifest(proposed=True)

    path = write_transect_manifest(manifest, tmp_path / "transects.geojson")
    loaded = load_transect_manifest(path)

    assert loaded.benchmark_id == manifest.benchmark_id
    assert [item.transect_id for item in loaded.transects] == ["T001", "T002"]
    assert all(
        item.review_status == PROPOSED_TRANSECT_STATUS for item in loaded.transects
    )
    assert all(not item.approved for item in loaded.transects)
    assert [item.geometry.wkt for item in loaded.transects] == [
        item.geometry.wkt for item in manifest.transects
    ]
    assert (
        json.loads(path.read_text(encoding="utf-8"))["crs"]["properties"]["name"]
        == "EPSG:4326"
    )


def test_manifest_rejects_duplicate_identifiers_and_unreviewed_status() -> None:
    first = _manifest().transects[0]

    with pytest.raises(BenchmarkError, match="unique"):
        TransectManifest(
            benchmark_id="synthetic-pilot",
            transects=(first, replace(first, geometry=_line(y_m=100.0))),
        )
    with pytest.raises(BenchmarkError, match="review_status"):
        replace(first, review_status="silently approved")


def test_candidate_proposals_are_deterministic_and_never_approved() -> None:
    data = _spatial_dataset()
    before = data.copy(deep=True)
    settings = TransectProposalSettings(
        enabled=True,
        count=3,
        stratification_axis="latitude",
        transect_azimuth_degrees=90.0,
        line_length_m=1000.0,
    )
    diagnostic = CandidateIntervalConfiguration(
        extent_classes=(4,),
        station_bin_width_m=100.0,
        min_extent_pixels_per_bin=1,
        max_bridge_gap_m=0.0,
    )

    first = propose_candidate_transects(
        data,
        benchmark_id="synthetic-pilot",
        settings=settings,
        corridor_half_width_m=50.0,
        diagnostic_configuration=diagnostic,
    )
    second = propose_candidate_transects(
        data,
        benchmark_id="synthetic-pilot",
        settings=settings,
        corridor_half_width_m=50.0,
        diagnostic_configuration=diagnostic,
    )

    assert first.as_geojson() == second.as_geojson()
    assert [item.transect_id for item in first.transects] == ["T001", "T002", "T003"]
    assert all(
        item.review_status == PROPOSED_TRANSECT_STATUS for item in first.transects
    )
    assert all(not item.approved for item in first.transects)
    assert all(
        "Requires scientist review" in (item.selection_notes or "")
        for item in first.transects
    )
    for item in first.transects:
        diagnostics = dict(item.proposal_diagnostics or {})
        assert diagnostics["proposal_method"] == "equal_spatial_strata_fixed_azimuth_v1"
        assert "no physical" in str(diagnostics["interpretation_warning"]).lower()
        assert not {"branch_id", "bank_position", "width_m"}.intersection(diagnostics)
    xr.testing.assert_identical(data, before)


def test_annotation_packets_are_pending_separate_candidate_diagnostics(
    tmp_path: Path,
) -> None:
    config = load_benchmark_config(_write_config(tmp_path))
    manifest = _manifest(proposed=True)
    data = _pixel_dataset()
    before = data.copy(deep=True)

    result = generate_annotation_packets(data, config, manifest)

    assert len(result.packets) == 2
    assert list(result.samples) == ["T001", "T002"]
    for packet in result.packets:
        assert packet.transect.review_status == PROPOSED_TRANSECT_STATUS
        assert {path.name for path in packet.files} == {
            "planview_pixc.png",
            "transect_classification.png",
            "candidate_diagnostic.png",
            "station_evidence.csv",
            "annotation_template.json",
            "metadata.json",
        }
        assert all(path.is_file() and path.stat().st_size > 0 for path in packet.files)
        template = json.loads(
            (packet.directory / "annotation_template.json").read_text(encoding="utf-8")
        )
        assert template["status"] == "pending"
        assert template["confirmed"] is False
        assert template["manual_intervals"] is None
        assert template["reference_imagery"]["status"] == "independent imagery pending"
        assert not {
            "width_m",
            "total_wetted_width_m",
            "outer_wetted_span_m",
            "total_internal_dry_gap_m",
        }.intersection(template)
        metadata = json.loads(
            (packet.directory / "metadata.json").read_text(encoding="utf-8")
        )
        assert (
            metadata["transect"]["properties"]["review_status"]
            == PROPOSED_TRANSECT_STATUS
        )
        assert (
            "not manual reference truth" in metadata["candidate_diagnostic"]["warning"]
        )
        evidence = pd.read_csv(packet.directory / "station_evidence.csv")
        assert {"state", "candidate_wet", "bridged"} <= set(evidence.columns)
    packet_index = json.loads(
        (
            config.output_directory / "annotation_packets" / "packet_index.json"
        ).read_text(encoding="utf-8")
    )
    assert packet_index["packet_count"] == 2
    assert "not independent reference truth" in packet_index["candidate_layer_warning"]
    preparation_manifest = json.loads(
        result.preparation_manifest_path.read_text(encoding="utf-8")
    )
    assert preparation_manifest["stage"] == "annotation_packet_preparation"
    assert preparation_manifest["packet_count"] == 2
    assert preparation_manifest["evaluation_count"] == 0
    assert preparation_manifest["quantitative_validation_ran"] is False
    assert preparation_manifest["ground_truth_fabricated"] is False
    assert preparation_manifest["annotation_file_hashes"] == []
    with pytest.raises(BenchmarkError, match="approved transect"):
        save_manual_annotation(
            result.packets[0].sample,
            result.packets[0].directory / "annotation_template.json",
            tmp_path / "must-not-save.json",
            annotation_id="unapproved",
            analyst_id="analyst-a",
            manual_intervals=((0.0, 100.0),),
            confirmed=True,
        )
    xr.testing.assert_identical(data, before)


def test_manual_annotation_save_uses_phase5a1_and_round_trips_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import swot_pixc_lab.benchmark as benchmark_module

    config = load_benchmark_config(_write_config(tmp_path))
    manifest = TransectManifest(
        benchmark_id=config.benchmark_id,
        transects=(_manifest().transects[0],),
    )
    packet = generate_annotation_packets(_pixel_dataset(), config, manifest).packets[0]
    output = config.annotation_directory / "T001-analyst-a.json"
    original = benchmark_module.measure_explicit_wet_intervals
    calls: list[tuple[tuple[float, float], ...]] = []

    def counted_measure(sample, intervals):
        supplied = tuple(tuple(item) for item in intervals)
        calls.append(supplied)
        return original(sample, supplied)

    monkeypatch.setattr(
        benchmark_module, "measure_explicit_wet_intervals", counted_measure
    )
    sample_before = packet.sample.pixels.copy(deep=True)

    record = save_manual_annotation(
        packet.sample,
        packet.directory / "annotation_template.json",
        output,
        annotation_id="T001-analyst-a",
        analyst_id="analyst-a",
        manual_intervals=((0.0, 100.0), (200.0, 300.0)),
        confidence="synthetic-test-only",
        notes="Independent synthetic interpretation.",
        confirmed=True,
    )
    annotations = load_manual_annotations(
        config.annotation_directory,
        benchmark_id=config.benchmark_id,
    )

    assert calls == [((0.0, 100.0), (200.0, 300.0))]
    assert record.reference.total_wetted_width_m == 200.0
    assert record.reference.total_internal_dry_gap_m == 100.0
    assert len(annotations) == 1
    assert annotations[0].manual_intervals == ((0.0, 100.0), (200.0, 300.0))
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["manual_intervals"] == [[0.0, 100.0], [200.0, 300.0]]
    assert not {
        "width_m",
        "total_wetted_width_m",
        "outer_wetted_span_m",
        "total_internal_dry_gap_m",
    }.intersection(payload)
    xr.testing.assert_identical(packet.sample.pixels, sample_before)


def test_manual_annotation_requires_confirmation_and_rejects_derived_fields(
    tmp_path: Path,
) -> None:
    config = load_benchmark_config(_write_config(tmp_path))
    manifest = TransectManifest(
        benchmark_id=config.benchmark_id,
        transects=(_manifest().transects[0],),
    )
    packet = generate_annotation_packets(_pixel_dataset(), config, manifest).packets[0]
    rejected = config.annotation_directory / "rejected.json"

    with pytest.raises(BenchmarkError, match="confirmed"):
        save_manual_annotation(
            packet.sample,
            packet.directory / "annotation_template.json",
            rejected,
            annotation_id="rejected",
            analyst_id="analyst-a",
            manual_intervals=((0.0, 100.0),),
            confirmed=False,
        )
    assert not rejected.exists()

    valid = config.annotation_directory / "derived.json"
    save_manual_annotation(
        packet.sample,
        packet.directory / "annotation_template.json",
        valid,
        annotation_id="derived",
        analyst_id="analyst-a",
        manual_intervals=((0.0, 100.0),),
        confirmed=True,
    )
    payload = json.loads(valid.read_text(encoding="utf-8"))
    payload["total_wetted_width_m"] = 999.0
    valid.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BenchmarkError, match="unexpected|derived"):
        load_manual_annotations(
            config.annotation_directory,
            benchmark_id=config.benchmark_id,
        )


def test_confirmed_empty_reference_is_distinct_from_pending_annotation(
    tmp_path: Path,
) -> None:
    config = load_benchmark_config(_write_config(tmp_path))
    manifest = TransectManifest(
        benchmark_id=config.benchmark_id,
        transects=(_manifest().transects[0],),
    )
    packet = generate_annotation_packets(_pixel_dataset(), config, manifest).packets[0]
    config.annotation_directory.mkdir(parents=True)
    pending_path = config.annotation_directory / "00-pending.json"
    pending_path.write_text(
        (packet.directory / "annotation_template.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    save_manual_annotation(
        packet.sample,
        packet.directory / "annotation_template.json",
        config.annotation_directory / "01-confirmed-empty.json",
        annotation_id="confirmed-empty",
        analyst_id="analyst-a",
        manual_intervals=(),
        notes="Analyst explicitly confirmed no wet interval.",
        confirmed=True,
    )

    annotations = load_manual_annotations(
        config.annotation_directory,
        benchmark_id=config.benchmark_id,
    )

    assert len(annotations) == 1
    assert annotations[0].annotation_id == "confirmed-empty"
    assert annotations[0].manual_intervals == ()


def test_missing_annotations_stop_before_quantitative_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import swot_pixc_lab.benchmark as benchmark_module

    config = load_benchmark_config(_write_config(tmp_path))
    manifest = _manifest()
    samples = {"T001": _sample(), "T002": _sample(y_m=100.0)}

    def forbidden_sensitivity(*args, **kwargs):
        raise AssertionError("candidate output must never become reference truth")

    monkeypatch.setattr(
        benchmark_module,
        "run_candidate_interval_sensitivity",
        forbidden_sensitivity,
    )

    with pytest.raises(AnnotationRequiredError, match=ANNOTATION_REQUIRED_MESSAGE):
        run_benchmark(config, manifest, samples, ())

    assert not config.output_directory.exists()


def test_proposed_transect_cannot_enter_quantitative_validation(tmp_path: Path) -> None:
    config, manifest, preparation, annotations = _prepared_run(tmp_path)
    proposed = TransectManifest(
        benchmark_id=manifest.benchmark_id,
        transects=(
            replace(
                manifest.transects[0],
                review_status=PROPOSED_TRANSECT_STATUS,
            ),
            manifest.transects[1],
        ),
    )

    with pytest.raises(BenchmarkError, match="explicit owner approval"):
        run_benchmark(config, proposed, preparation.samples, annotations)


def test_runner_preserves_multiple_transects_analysts_and_configuration_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import swot_pixc_lab.benchmark as benchmark_module

    config, manifest, preparation, annotations = _prepared_run(tmp_path)
    samples_before = {
        key: value.pixels.copy(deep=True) for key, value in preparation.samples.items()
    }
    annotation_before = copy.deepcopy(annotations)
    original = benchmark_module.run_candidate_interval_sensitivity
    sweep_calls: list[str] = []

    def counted_sweep(sample, *, configurations):
        sweep_calls.append(sample.transect.wkt)
        return original(sample, configurations=configurations)

    monkeypatch.setattr(
        benchmark_module,
        "run_candidate_interval_sensitivity",
        counted_sweep,
    )

    result = run_benchmark(config, manifest, preparation.samples, annotations)
    individual = result.individual_results
    aggregate = result.aggregate_results

    assert len(sweep_calls) == 2
    assert individual.shape[0] == len(annotations) * 2 == 6
    assert individual[["annotation_id", "configuration_id"]].to_records(
        index=False
    ).tolist() == [
        ("annotation-a", 1),
        ("annotation-a", 2),
        ("annotation-b", 1),
        ("annotation-b", 2),
        ("annotation-c", 1),
        ("annotation-c", 2),
    ]
    assert individual["benchmark_subset"].tolist() == [
        "pilot",
        "pilot",
        "development",
        "development",
        "holdout",
        "holdout",
    ]
    assert set(individual["transect_id"]) == {"T001", "T002"}
    assert set(individual["analyst_id"]) == {"analyst-a", "analyst-b"}
    assert individual["extent_classes"].tolist() == [
        (4,),
        (3, 4),
        (4,),
        (3, 4),
        (4,),
        (3, 4),
    ]
    required = {
        "observed_support_iou",
        "observed_support_f1",
        "observed_support_precision",
        "observed_support_recall",
        "observed_support_false_positive_length_m",
        "observed_support_false_negative_length_m",
        "observed_support_signed_total_width_error_m",
        "observed_support_absolute_total_width_error_m",
        "observed_support_boundary_symmetric_boundary_mean_m",
        "observed_component_count_difference",
        "bridge_inclusive_iou",
        "bridge_inclusive_f1",
        "bridge_inclusive_false_positive_length_m",
        "bridge_inclusive_false_negative_length_m",
        "bridge_inclusive_signed_total_width_error_m",
        "bridge_inclusive_absolute_total_width_error_m",
        "bridge_inclusive_boundary_symmetric_boundary_mean_m",
        "inferred_component_count_difference",
        "total_bridged_gap_m",
        "bridged_length_over_manual_wet_m",
        "bridged_length_over_manual_nonwet_m",
        "delta_iou_due_to_bridging",
        "delta_f1_due_to_bridging",
    }
    assert required <= set(individual.columns)
    assert not {"rank", "score", "best", "winner"}.intersection(individual.columns)
    assert aggregate["configuration_id"].tolist() == [1, 2]
    assert aggregate["n_annotations"].tolist() == [3, 3]
    assert aggregate["n_transects"].tolist() == [2, 2]
    assert not {"rank", "score", "best", "winner"}.intersection(aggregate.columns)
    assert len(result.benchmark_records) == len(annotations)
    assert [item.metadata.annotation_id for item in result.benchmark_records] == [
        "annotation-a",
        "annotation-b",
        "annotation-c",
    ]
    for key, before in samples_before.items():
        xr.testing.assert_identical(preparation.samples[key].pixels, before)
    assert annotations == annotation_before


def test_aggregate_is_unranked_ordered_and_handles_undefined_metrics(
    tmp_path: Path,
) -> None:
    config, manifest, preparation, annotations = _prepared_run(tmp_path)
    individual = run_benchmark(
        config, manifest, preparation.samples, annotations
    ).individual_results
    before = individual.copy(deep=True)

    reordered = pd.concat(
        [
            individual.loc[individual["configuration_id"] == 2],
            individual.loc[individual["configuration_id"] == 1],
        ],
        ignore_index=True,
    )
    undefined_column = "observed_support_boundary_symmetric_boundary_mean_m"
    reordered[undefined_column] = np.nan
    aggregate = aggregate_benchmark_results(reordered)

    assert aggregate["configuration_id"].tolist() == [2, 1]
    assert aggregate[f"{undefined_column}_valid_n"].tolist() == [0, 0]
    assert aggregate[f"{undefined_column}_mean"].isna().all()
    assert aggregate[f"{undefined_column}_median"].isna().all()
    assert not {"rank", "score", "best", "winner"}.intersection(aggregate.columns)
    pd.testing.assert_frame_equal(individual, before)

    with pytest.raises(BenchmarkError, match="empty"):
        aggregate_benchmark_results(individual.iloc[:0])
    with pytest.raises(BenchmarkError, match="missing columns"):
        aggregate_benchmark_results(individual.drop(columns="observed_support_iou"))


def test_runner_outputs_have_hashes_figures_report_and_no_winner_claim(
    tmp_path: Path,
) -> None:
    config, manifest, preparation, annotations = _prepared_run(tmp_path)

    result = run_benchmark(config, manifest, preparation.samples, annotations)

    expected_tables = {
        "individual_validation_results.csv",
        "individual_validation_results.json",
        "aggregate_configuration_summary.csv",
        "aggregate_configuration_summary.json",
        "manual_benchmark_records.json",
    }
    assert expected_tables <= {
        item.name for item in config.output_directory.iterdir() if item.is_file()
    }
    assert result.results_csv.is_file()
    assert result.aggregate_csv.is_file()
    assert pd.read_csv(result.results_csv).shape[0] == 6
    assert pd.read_csv(result.aggregate_csv).shape[0] == 2

    expected_aggregate_figures = {
        "iou_across_configurations.png",
        "f1_across_configurations.png",
        "signed_total_width_error.png",
        "absolute_total_width_error.png",
        "boundary_distance_error.png",
        "component_count_difference.png",
        "bridge_effect.png",
    }
    assert expected_aggregate_figures <= {path.name for path in result.figure_paths}
    assert {
        path.name for path in result.figure_paths if "per_transect" in path.parts
    } == {
        "T001_comparison.png",
        "T002_comparison.png",
    }
    assert all(
        path.is_file() and path.stat().st_size > 0 for path in result.figure_paths
    )

    manifest_payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert (
        datetime.fromisoformat(manifest_payload["created_utc"]).utcoffset() is not None
    )
    assert manifest_payload["benchmark_id"] == config.benchmark_id
    assert manifest_payload["aoi"] == list(config.observation.aoi)
    assert manifest_payload["qc_profile"] == config.qc_profile
    assert manifest_payload["configuration_count"] == 2
    assert manifest_payload["annotation_count"] == 3
    assert manifest_payload["evaluation_count"] == 6
    assert manifest_payload["ranking_performed"] is False
    assert manifest_payload["recommended_configuration_selected"] is False
    assert (
        manifest_payload["config_sha256"]
        == hashlib.sha256(config.source_path.read_bytes()).hexdigest()
    )
    assert (
        manifest_payload["transect_manifest_sha256"]
        == hashlib.sha256(config.transect_manifest_path.read_bytes()).hexdigest()
    )
    assert [
        item["annotation_id"] for item in manifest_payload["annotation_file_hashes"]
    ] == [
        "annotation-a",
        "annotation-b",
        "annotation-c",
    ]
    for item, annotation in zip(
        manifest_payload["annotation_file_hashes"], annotations, strict=True
    ):
        assert (
            item["sha256"]
            == hashlib.sha256(annotation.source_path.read_bytes()).hexdigest()
        )
    assert manifest_payload["sensitivity_configurations"] == [
        {
            "configuration_id": 1,
            "extent_classes": [4],
            "station_bin_width_m": 100.0,
            "min_extent_pixels_per_bin": 1,
            "max_bridge_gap_m": 0.0,
        },
        {
            "configuration_id": 2,
            "extent_classes": [3, 4],
            "station_bin_width_m": 100.0,
            "min_extent_pixels_per_bin": 1,
            "max_bridge_gap_m": 0.0,
        },
    ]
    assert "run_manifest.json" in manifest_payload["output_filenames"]

    report = result.report_path.read_text(encoding="utf-8")
    assert "Pilot sensitivity description only" in report
    assert "Observed candidate support and bridge-inclusive" in report
    assert "unsampled bins remain unknown, not dry" in report
    assert "no recommended configuration is selected" in report
    assert "best configuration is" not in report.lower()
    assert "winner is" not in report.lower()

    records = json.loads(
        (config.output_directory / "manual_benchmark_records.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(records) == 3
    assert [item["metadata"]["annotation_id"] for item in records] == [
        "annotation-a",
        "annotation-b",
        "annotation-c",
    ]
    assert (
        records[0]["explicit_interval_reference"]["measurement_summary"][
            "total_wetted_width_m"
        ]
        == 200.0
    )


def test_station_frame_mismatch_is_rejected_without_transforming_boundaries(
    tmp_path: Path,
) -> None:
    config, manifest, preparation, annotations = _prepared_run(tmp_path)
    mismatched = replace(annotations[0], station_frame_hash="different-frame")

    with pytest.raises(BenchmarkError, match="different station frame"):
        run_benchmark(
            config,
            manifest,
            preparation.samples,
            (mismatched,),
        )


def test_reference_imagery_metadata_tolerates_pending_but_rejects_nonfinite() -> None:
    pending = ReferenceImageryMetadata(status="independent imagery pending")

    assert pending.pending is True
    assert pending.local_image_path is None
    assert pending.as_dict()["source_provider"] is None

    with pytest.raises(BenchmarkError, match="finite"):
        ReferenceImageryMetadata(
            status="available",
            source_provider="synthetic",
            temporal_offset_hours=np.nan,
        )
