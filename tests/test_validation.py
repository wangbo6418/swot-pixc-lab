from __future__ import annotations

import copy
import json
import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pyproj import CRS, Transformer
from shapely.geometry import LineString, box

from swot_pixc_lab import (
    BoundaryDistanceMetrics,
    IntervalSetMetrics,
    IntervalValidationResult,
    ManualBenchmarkMetadata,
    SensitivityValidationResult,
    ValidationError,
    create_manual_benchmark_record,
    evaluate_candidate_against_explicit,
    evaluate_sensitivity_against_explicit,
    infer_candidate_wet_intervals,
    measure_explicit_wet_intervals,
    plot_interval_validation,
    run_candidate_interval_sensitivity,
    sample_transect,
)
from swot_pixc_lab.mosaic import PixcObservation
from swot_pixc_lab.reader import PixcSourceMetadata, PixcVariableMetadata
from swot_pixc_lab.subset import ExactAoi

_TEST_CRS = CRS.from_proj4(
    "+proj=aeqd +lat_0=26.7 +lon_0=87 +ellps=WGS84 +units=m +type=crs"
)
_TO_GEOGRAPHIC = Transformer.from_crs(_TEST_CRS, "EPSG:4326", always_xy=True)

_METRIC_FIELDS = (
    "reference_wet_length_m",
    "predicted_wet_length_m",
    "true_positive_length_m",
    "false_positive_length_m",
    "false_negative_length_m",
    "union_length_m",
    "precision",
    "recall",
    "f1",
    "iou",
    "reference_empty",
    "prediction_empty",
    "exact_empty_match",
    "signed_total_width_error_m",
    "absolute_total_width_error_m",
    "relative_total_width_error",
    "reference_outer_span_m",
    "predicted_outer_span_m",
    "signed_outer_span_error_m",
    "absolute_outer_span_error_m",
    "reference_interval_count",
    "predicted_component_count",
    "component_count_difference",
)
_BOUNDARY_FIELDS = (
    "manual_boundaries_m",
    "candidate_boundaries_m",
    "manual_to_candidate_boundary_distances_m",
    "candidate_to_manual_boundary_distances_m",
    "manual_to_candidate_boundary_mean_m",
    "manual_to_candidate_boundary_median_m",
    "manual_to_candidate_boundary_max_m",
    "candidate_to_manual_boundary_mean_m",
    "candidate_to_manual_boundary_median_m",
    "candidate_to_manual_boundary_max_m",
    "symmetric_boundary_mean_m",
    "symmetric_boundary_max_m",
    "diagnostic_warning",
)


def _geographic(x: object, y: object) -> tuple[np.ndarray, np.ndarray]:
    longitude, latitude = _TO_GEOGRAPHIC.transform(x, y)
    return np.asarray(longitude), np.asarray(latitude)


def _transect() -> LineString:
    longitude, latitude = _geographic([-500.0, 500.0], [0.0, 0.0])
    return LineString(zip(longitude, latitude, strict=True))


def _source_metadata(
    source_index: int,
    tile: str,
    point_count: int,
) -> PixcSourceMetadata:
    classification = PixcVariableMetadata(
        name="classification",
        dimensions=("points",),
        shape=(point_count,),
        dtype="uint8",
        attributes=MappingProxyType({"_FillValue": np.uint8(255)}),
    )
    return PixcSourceMetadata(
        source_index=source_index,
        path=Path(f"synthetic-validation-{tile}.nc"),
        filename=f"synthetic-validation-{tile}.nc",
        granule_id=f"synthetic-validation-{tile}",
        cycle=9,
        pass_number=286,
        tile=tile,
        crid="PGD0",
        netcdf_product_version="D",
        netcdf_pge_version="3.1.0",
        cmr_record=None,
        data_model="NETCDF4",
        groups=("pixel_cloud",),
        dimensions=MappingProxyType({"points": point_count}),
        root_attributes=MappingProxyType({}),
        pixel_cloud_attributes=MappingProxyType({}),
        variables=MappingProxyType({"classification": classification}),
        loaded_variables=("longitude", "latitude", "classification"),
        points_before=point_count,
        points_after=point_count,
        invalid_coordinate_count=0,
        normalized_longitude_count=0,
    )


def _sample_for_candidate_intervals(
    intervals: list[tuple[float, float]],
    *,
    station_bin_width_m: float = 25.0,
    sampled_noneligible_stations: tuple[float, ...] = (),
    with_source_metadata: bool = False,
):
    stations: list[float] = []
    classifications: list[int] = []
    for start, end in intervals:
        span_in_bins = (end - start) / station_bin_width_m
        assert span_in_bins == pytest.approx(round(span_in_bins))
        for bin_offset in range(round(span_in_bins)):
            stations.append(start + (bin_offset + 0.5) * station_bin_width_m)
            classifications.append(4)
    stations.extend(sampled_noneligible_stations)
    classifications.extend([1] * len(sampled_noneligible_stations))

    longitude, latitude = _geographic(
        np.asarray(stations, dtype=np.float64) - 500.0,
        np.zeros(len(stations), dtype=np.float64),
    )
    source_indices = (
        np.arange(len(stations), dtype=np.int32) % 2
        if with_source_metadata
        else np.zeros(len(stations), dtype=np.int32)
    )
    source_point_indices = np.empty(len(stations), dtype=np.int64)
    for source_index in np.unique(source_indices):
        selected = source_indices == source_index
        source_point_indices[selected] = np.arange(np.count_nonzero(selected))
    pixels = xr.Dataset(
        {
            "longitude": xr.DataArray(longitude, dims="points"),
            "latitude": xr.DataArray(latitude, dims="points"),
            "classification": xr.DataArray(
                np.asarray(classifications, dtype=np.uint8),
                dims="points",
                attrs={"_FillValue": np.uint8(255)},
            ),
            "source_index": xr.DataArray(source_indices, dims="points"),
            "source_point_index": xr.DataArray(source_point_indices, dims="points"),
        },
        attrs={
            "test_marker": "validation-input-unchanged",
            "swot_pixc_lab_qc_profile": "synthetic_validation_profile",
            "swot_pixc_lab_qc_label": "Synthetic validation profile",
            "swot_pixc_lab_qc_status": "experimental / test only",
        },
    )
    data: xr.Dataset | PixcObservation = pixels
    if with_source_metadata:
        sources = tuple(
            _source_metadata(
                source_index,
                tile,
                int(np.count_nonzero(source_indices == source_index)),
            )
            for source_index, tile in ((0, "107L"), (1, "108L"))
        )
        data = PixcObservation(
            pixels=pixels,
            sources=sources,
            aoi=ExactAoi(
                geometry=box(86.0, 26.0, 88.0, 28.0),
                kind="bounding_box",
                bounding_box=(86.0, 26.0, 88.0, 28.0),
            ),
        )
    sample = sample_transect(
        data,
        _transect(),
        corridor_half_width_m=25.0,
    )
    selected = sample.pixels.copy(deep=True)
    selected["station_m"] = xr.DataArray(
        np.asarray(stations, dtype=np.float64),
        dims="points",
        attrs=sample.station_m.attrs,
    )
    return replace(sample, pixels=selected)


def _validation(
    manual_intervals: list[tuple[float, float]],
    observed_intervals: list[tuple[float, float]],
    *,
    station_bin_width_m: float = 25.0,
    max_bridge_gap_m: float = 0.0,
    sampled_noneligible_stations: tuple[float, ...] = (),
):
    sample = _sample_for_candidate_intervals(
        observed_intervals,
        station_bin_width_m=station_bin_width_m,
        sampled_noneligible_stations=sampled_noneligible_stations,
    )
    reference = measure_explicit_wet_intervals(sample, manual_intervals)
    candidate = infer_candidate_wet_intervals(
        sample,
        extent_classes=(4,),
        station_bin_width_m=station_bin_width_m,
        min_extent_pixels_per_bin=1,
        max_bridge_gap_m=max_bridge_gap_m,
    )
    result = evaluate_candidate_against_explicit(reference, candidate)
    return sample, reference, candidate, result


def _assert_perfect(
    metrics: IntervalSetMetrics,
    *,
    wet_length_m: float,
    component_count: int,
) -> None:
    assert metrics.reference_wet_length_m == pytest.approx(wet_length_m)
    assert metrics.predicted_wet_length_m == pytest.approx(wet_length_m)
    assert metrics.true_positive_length_m == pytest.approx(wet_length_m)
    assert metrics.false_positive_length_m == pytest.approx(0.0)
    assert metrics.false_negative_length_m == pytest.approx(0.0)
    assert metrics.union_length_m == pytest.approx(wet_length_m)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.iou == pytest.approx(1.0)
    assert metrics.signed_total_width_error_m == pytest.approx(0.0)
    assert metrics.absolute_total_width_error_m == pytest.approx(0.0)
    assert metrics.relative_total_width_error == pytest.approx(0.0)
    assert metrics.reference_interval_count == component_count
    assert metrics.predicted_component_count == component_count
    assert metrics.component_count_difference == 0


def test_a_perfect_one_interval_match() -> None:
    _, _, _, result = _validation([(100.0, 300.0)], [(100.0, 300.0)])

    _assert_perfect(
        result.observed_support_metrics,
        wet_length_m=200.0,
        component_count=1,
    )
    _assert_perfect(
        result.bridge_inclusive_metrics,
        wet_length_m=200.0,
        component_count=1,
    )
    assert result.observed_support_boundary_metrics.symmetric_boundary_max_m == 0.0
    assert result.bridge_inclusive_boundary_metrics.symmetric_boundary_max_m == 0.0
    assert result.delta_iou_due_to_bridging == 0.0
    assert result.delta_f1_due_to_bridging == 0.0


def test_b_perfect_three_interval_match() -> None:
    intervals = [(0.0, 100.0), (150.0, 250.0), (400.0, 500.0)]
    _, _, _, result = _validation(intervals, intervals)

    _assert_perfect(
        result.observed_support_metrics,
        wet_length_m=300.0,
        component_count=3,
    )
    assert result.observed_support_metrics.reference_outer_span_m == 500.0
    assert result.observed_support_metrics.predicted_outer_span_m == 500.0


def test_c_candidate_too_wide_records_false_positive_extent() -> None:
    _, _, _, result = _validation([(100.0, 200.0)], [(50.0, 250.0)])
    metrics = result.observed_support_metrics

    assert metrics.true_positive_length_m == 100.0
    assert metrics.false_positive_length_m == 100.0
    assert metrics.false_negative_length_m == 0.0
    assert metrics.precision == 0.5
    assert metrics.recall == 1.0
    assert metrics.iou == 0.5


def test_d_candidate_too_narrow_records_false_negative_extent() -> None:
    _, _, _, result = _validation([(100.0, 300.0)], [(150.0, 250.0)])
    metrics = result.observed_support_metrics

    assert metrics.true_positive_length_m == 100.0
    assert metrics.false_positive_length_m == 0.0
    assert metrics.false_negative_length_m == 100.0
    assert metrics.precision == 1.0
    assert metrics.recall == 0.5
    assert metrics.iou == 0.5


def test_e_lateral_shift_can_have_correct_width_but_wrong_location() -> None:
    _, _, _, result = _validation([(100.0, 200.0)], [(150.0, 250.0)])
    metrics = result.observed_support_metrics

    assert metrics.reference_wet_length_m == metrics.predicted_wet_length_m == 100.0
    assert metrics.signed_total_width_error_m == 0.0
    assert metrics.true_positive_length_m == 50.0
    assert metrics.false_positive_length_m == 50.0
    assert metrics.false_negative_length_m == 50.0
    assert metrics.iou == pytest.approx(1.0 / 3.0)


def test_f_one_manual_interval_split_into_two_observed_runs() -> None:
    _, _, _, result = _validation(
        [(100.0, 300.0)],
        [(100.0, 175.0), (225.0, 300.0)],
    )
    metrics = result.observed_support_metrics

    assert metrics.reference_interval_count == 1
    assert metrics.predicted_component_count == 2
    assert metrics.component_count_difference == 1
    assert metrics.true_positive_length_m == 150.0
    assert metrics.false_negative_length_m == 50.0


def test_g_two_manual_intervals_are_merged_only_in_bridge_inclusive_view() -> None:
    _, _, _, result = _validation(
        [(0.0, 100.0), (150.0, 250.0)],
        [(0.0, 100.0), (150.0, 250.0)],
        max_bridge_gap_m=50.0,
    )

    assert result.observed_support_metrics.predicted_component_count == 2
    assert result.observed_support_metrics.component_count_difference == 0
    assert result.bridge_inclusive_metrics.predicted_component_count == 1
    assert result.bridge_inclusive_metrics.component_count_difference == -1


def test_h_bridge_across_manual_gap_is_false_positive_bridge_length() -> None:
    _, _, _, result = _validation(
        [(0.0, 100.0), (150.0, 250.0)],
        [(0.0, 100.0), (150.0, 250.0)],
        max_bridge_gap_m=50.0,
    )

    assert result.total_bridged_gap_m == 50.0
    assert result.bridged_length_over_manual_wet_m == 0.0
    assert result.bridged_length_over_manual_nonwet_m == 50.0
    assert result.observed_support_metrics.iou == 1.0
    assert result.bridge_inclusive_metrics.iou == 0.8
    assert result.delta_iou_due_to_bridging == pytest.approx(-0.2)
    assert result.delta_f1_due_to_bridging == pytest.approx(400.0 / 450.0 - 1.0)


def test_i_bridge_restores_missing_support_inside_manual_wet_region() -> None:
    _, _, _, result = _validation(
        [(0.0, 250.0)],
        [(0.0, 100.0), (150.0, 250.0)],
        max_bridge_gap_m=50.0,
    )

    assert result.bridged_length_over_manual_wet_m == 50.0
    assert result.bridged_length_over_manual_nonwet_m == 0.0
    assert result.observed_support_metrics.iou == 0.8
    assert result.bridge_inclusive_metrics.iou == 1.0
    assert result.delta_iou_due_to_bridging == pytest.approx(0.2)
    assert result.delta_f1_due_to_bridging == pytest.approx(1.0 - 400.0 / 450.0)
    assert result.observed_support_boundary_metrics.candidate_boundaries_m == (
        0.0,
        100.0,
        150.0,
        250.0,
    )
    assert (
        result.observed_support_boundary_metrics.candidate_to_manual_boundary_mean_m
        == 50.0
    )
    assert result.observed_support_boundary_metrics.symmetric_boundary_mean_m == (
        pytest.approx(100.0 / 3.0)
    )
    assert result.bridge_inclusive_boundary_metrics.candidate_boundaries_m == (
        0.0,
        250.0,
    )
    assert result.bridge_inclusive_boundary_metrics.symmetric_boundary_mean_m == 0.0


def test_bridge_invariant_rejects_final_geometry_that_omits_recorded_bridge() -> None:
    _, reference, bridged_candidate, _ = _validation(
        [(0.0, 100.0), (150.0, 250.0)],
        [(0.0, 100.0), (150.0, 250.0)],
        max_bridge_gap_m=50.0,
    )
    _, _, unbridged_candidate, _ = _validation(
        [(0.0, 100.0), (150.0, 250.0)],
        [(0.0, 100.0), (150.0, 250.0)],
        max_bridge_gap_m=0.0,
    )
    inconsistent = replace(
        bridged_candidate,
        candidate_intervals=unbridged_candidate.candidate_intervals,
    )

    with pytest.raises(ValidationError, match="unrecorded addition or omission"):
        evaluate_candidate_against_explicit(reference, inconsistent)


def test_bridge_invariant_rejects_unrecorded_final_interval_addition() -> None:
    _, reference, unbridged_candidate, _ = _validation(
        [(0.0, 100.0), (150.0, 250.0)],
        [(0.0, 100.0), (150.0, 250.0)],
        max_bridge_gap_m=0.0,
    )
    _, _, bridged_candidate, _ = _validation(
        [(0.0, 100.0), (150.0, 250.0)],
        [(0.0, 100.0), (150.0, 250.0)],
        max_bridge_gap_m=50.0,
    )
    inconsistent = replace(
        unbridged_candidate,
        candidate_intervals=bridged_candidate.candidate_intervals,
    )

    with pytest.raises(ValidationError, match="unrecorded addition or omission"):
        evaluate_candidate_against_explicit(reference, inconsistent)


def test_bridge_invariant_rejects_gap_width_inconsistent_with_endpoints() -> None:
    _, reference, candidate, _ = _validation(
        [(0.0, 100.0), (150.0, 250.0)],
        [(0.0, 100.0), (150.0, 250.0)],
        max_bridge_gap_m=50.0,
    )
    bad_bridge = replace(
        candidate.bridge_records[0],
        gap_width_m=candidate.bridge_records[0].gap_width_m + 1.0,
    )
    inconsistent = replace(candidate, bridge_records=(bad_bridge,))

    with pytest.raises(ValidationError, match="endpoints reconstruct"):
        evaluate_candidate_against_explicit(reference, inconsistent)


def test_bridge_invariant_rejects_overlap_with_observed_wet_support() -> None:
    _, reference, candidate, _ = _validation(
        [(0.0, 100.0), (150.0, 250.0)],
        [(0.0, 100.0), (150.0, 250.0)],
        max_bridge_gap_m=50.0,
    )
    bridge = candidate.bridge_records[0]
    overlapping_bridge = replace(
        bridge,
        gap_start_station_m=75.0,
        gap_width_m=bridge.gap_end_station_m - 75.0,
    )
    inconsistent = replace(candidate, bridge_records=(overlapping_bridge,))

    with pytest.raises(ValidationError, match="overlap observed candidate-wet"):
        evaluate_candidate_against_explicit(reference, inconsistent)


def test_j_reference_empty_and_prediction_empty_have_explicit_none_scores() -> None:
    _, _, _, result = _validation([], [])

    for metrics in (
        result.observed_support_metrics,
        result.bridge_inclusive_metrics,
    ):
        assert metrics.reference_empty is True
        assert metrics.prediction_empty is True
        assert metrics.exact_empty_match is True
        assert metrics.precision is None
        assert metrics.recall is None
        assert metrics.f1 is None
        assert metrics.iou is None
        assert metrics.relative_total_width_error is None
        assert metrics.signed_outer_span_error_m is None
        assert metrics.absolute_outer_span_error_m is None
        assert metrics.true_positive_length_m == 0.0
        assert metrics.false_positive_length_m == 0.0
        assert metrics.false_negative_length_m == 0.0
        assert metrics.union_length_m == 0.0
    assert result.delta_iou_due_to_bridging is None
    assert result.delta_f1_due_to_bridging is None


def test_k_nonempty_reference_and_empty_prediction_semantics() -> None:
    _, _, _, result = _validation([(100.0, 200.0)], [])
    metrics = result.observed_support_metrics

    assert metrics.reference_empty is False
    assert metrics.prediction_empty is True
    assert metrics.exact_empty_match is False
    assert metrics.precision is None
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert metrics.iou == 0.0
    assert metrics.relative_total_width_error == -1.0
    assert metrics.signed_outer_span_error_m is None
    assert metrics.absolute_outer_span_error_m is None
    boundaries = result.observed_support_boundary_metrics
    assert boundaries.manual_boundaries_m == (100.0, 200.0)
    assert boundaries.candidate_boundaries_m == ()
    assert boundaries.manual_to_candidate_boundary_distances_m == ()
    assert boundaries.candidate_to_manual_boundary_distances_m == ()
    assert boundaries.symmetric_boundary_mean_m is None


def test_l_empty_reference_and_nonempty_prediction_semantics() -> None:
    _, _, _, result = _validation([], [(100.0, 200.0)])
    metrics = result.observed_support_metrics

    assert metrics.reference_empty is True
    assert metrics.prediction_empty is False
    assert metrics.exact_empty_match is False
    assert metrics.precision == 0.0
    assert metrics.recall is None
    assert metrics.f1 == 0.0
    assert metrics.iou == 0.0
    assert metrics.relative_total_width_error is None
    assert metrics.signed_outer_span_error_m is None
    assert metrics.absolute_outer_span_error_m is None
    boundaries = result.observed_support_boundary_metrics
    assert boundaries.manual_boundaries_m == ()
    assert boundaries.candidate_boundaries_m == (100.0, 200.0)
    assert boundaries.manual_to_candidate_boundary_distances_m == ()
    assert boundaries.candidate_to_manual_boundary_distances_m == ()
    assert boundaries.symmetric_boundary_mean_m is None


def test_m_observed_and_bridge_inclusive_predictions_remain_separate() -> None:
    _, _, _, result = _validation(
        [(0.0, 250.0)],
        [(0.0, 100.0), (150.0, 250.0)],
        max_bridge_gap_m=50.0,
    )

    assert result.observed_support_metrics.predicted_wet_length_m == 200.0
    assert result.bridge_inclusive_metrics.predicted_wet_length_m == 250.0
    assert result.observed_support_metrics.predicted_component_count == 2
    assert result.bridge_inclusive_metrics.predicted_component_count == 1


def test_n_required_numerical_regression_example() -> None:
    _, _, _, result = _validation(
        [(0.0, 100.0), (150.0, 250.0)],
        [(0.0, 100.0), (175.0, 250.0)],
    )
    metrics = result.observed_support_metrics

    assert metrics.reference_wet_length_m == 200.0
    assert metrics.predicted_wet_length_m == 175.0
    assert metrics.true_positive_length_m == 175.0
    assert metrics.false_positive_length_m == 0.0
    assert metrics.false_negative_length_m == 25.0
    assert metrics.union_length_m == 200.0
    assert metrics.precision == 1.0
    assert metrics.recall == 0.875
    assert metrics.f1 == pytest.approx(350.0 / 375.0)
    assert metrics.iou == 0.875
    assert metrics.signed_total_width_error_m == -25.0


def test_continuous_interval_arithmetic_does_not_rasterize_manual_edges() -> None:
    _, _, _, result = _validation([(12.5, 87.5)], [(0.0, 100.0)])
    metrics = result.observed_support_metrics

    assert metrics.reference_wet_length_m == 75.0
    assert metrics.predicted_wet_length_m == 100.0
    assert metrics.true_positive_length_m == 75.0
    assert metrics.false_positive_length_m == 25.0
    assert metrics.false_negative_length_m == 0.0
    assert metrics.union_length_m == 100.0


def test_o_signed_total_width_error_retains_direction() -> None:
    _, _, _, narrow = _validation([(100.0, 300.0)], [(100.0, 250.0)])
    _, _, _, wide = _validation([(100.0, 300.0)], [(50.0, 300.0)])

    assert narrow.observed_support_metrics.signed_total_width_error_m == -50.0
    assert narrow.observed_support_metrics.absolute_total_width_error_m == 50.0
    assert wide.observed_support_metrics.signed_total_width_error_m == 50.0
    assert wide.observed_support_metrics.absolute_total_width_error_m == 50.0


def test_p_relative_width_error_uses_reference_wet_length() -> None:
    _, _, _, result = _validation([(100.0, 300.0)], [(100.0, 250.0)])

    assert result.observed_support_metrics.relative_total_width_error == -0.25


def test_q_outer_span_error_is_distinct_from_total_width_error() -> None:
    _, _, _, result = _validation(
        [(100.0, 200.0), (300.0, 400.0)],
        [(50.0, 150.0), (300.0, 450.0)],
    )
    metrics = result.observed_support_metrics

    assert metrics.reference_outer_span_m == 300.0
    assert metrics.predicted_outer_span_m == 400.0
    assert metrics.signed_outer_span_error_m == 100.0
    assert metrics.absolute_outer_span_error_m == 100.0
    assert metrics.signed_total_width_error_m == 50.0


def test_r_component_count_differences_are_not_branch_errors() -> None:
    _, _, _, result = _validation(
        [(0.0, 50.0), (100.0, 250.0)],
        [(0.0, 50.0), (100.0, 150.0), (200.0, 250.0)],
        max_bridge_gap_m=50.0,
    )

    assert result.observed_support_metrics.reference_interval_count == 2
    assert result.observed_support_metrics.predicted_component_count == 3
    assert result.observed_support_metrics.component_count_difference == 1
    assert result.bridge_inclusive_metrics.predicted_component_count == 1
    assert result.bridge_inclusive_metrics.component_count_difference == -1


def test_s_symmetric_nearest_boundary_distances_have_exact_values() -> None:
    _, _, _, result = _validation(
        [(100.0, 300.0)],
        [(90.0, 150.0), (250.0, 330.0)],
        station_bin_width_m=10.0,
    )
    boundaries = result.observed_support_boundary_metrics

    assert isinstance(boundaries, BoundaryDistanceMetrics)
    assert boundaries.manual_to_candidate_boundary_distances_m == (10.0, 30.0)
    assert boundaries.candidate_to_manual_boundary_distances_m == (
        10.0,
        50.0,
        50.0,
        30.0,
    )
    assert boundaries.manual_to_candidate_boundary_mean_m == 20.0
    assert boundaries.manual_to_candidate_boundary_median_m == 20.0
    assert boundaries.manual_to_candidate_boundary_max_m == 30.0
    assert boundaries.candidate_to_manual_boundary_mean_m == 35.0
    assert boundaries.candidate_to_manual_boundary_median_m == 40.0
    assert boundaries.candidate_to_manual_boundary_max_m == 50.0
    assert boundaries.symmetric_boundary_mean_m == 30.0
    assert boundaries.symmetric_boundary_max_m == 50.0


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("transect_wkt", "LINESTRING (0 0, 1 1)", "transect_wkt"),
        ("transect_crs", "EPSG:3857", "transect_crs"),
        ("metric_crs_wkt", "different metric CRS", "metric_crs_wkt"),
        ("projection_center", (87.1, 26.7), "projection_center"),
        ("corridor_half_width_m", 30.0, "corridor_half_width_m"),
    ],
)
def test_t_incompatible_station_frames_are_rejected(
    field: str,
    replacement: object,
    message: str,
) -> None:
    _, reference, candidate, _ = _validation([(100.0, 200.0)], [(100.0, 200.0)])
    incompatible = replace(candidate, **{field: replacement})

    with pytest.raises(ValidationError, match=message):
        evaluate_candidate_against_explicit(reference, incompatible)


def test_u_different_station_length_is_rejected_without_rescaling() -> None:
    _, reference, candidate, _ = _validation([(100.0, 200.0)], [(100.0, 200.0)])
    incompatible = replace(
        candidate,
        transect_length_m=candidate.transect_length_m + 1.0,
    )

    with pytest.raises(ValidationError, match="transect_length_m"):
        evaluate_candidate_against_explicit(reference, incompatible)


def test_recorded_metric_roundoff_is_not_treated_as_physical_mismatch() -> None:
    _, reference, candidate, _ = _validation([(100.0, 200.0)], [(100.0, 200.0)])
    roundoff_only = replace(
        candidate,
        transect_length_m=math.nextafter(reference.transect_length_m, math.inf),
        corridor_half_width_m=math.nextafter(reference.corridor_half_width_m, math.inf),
        projection_center=(
            math.nextafter(reference.projection_center[0], math.inf),
            math.nextafter(reference.projection_center[1], math.inf),
        ),
    )

    result = evaluate_candidate_against_explicit(reference, roundoff_only)

    assert isinstance(result, IntervalValidationResult)


def test_recorded_metric_roundoff_allowance_is_bounded_to_eight_ulps() -> None:
    _, reference, candidate, _ = _validation(
        [(100.0, 200.0)],
        [(100.0, 200.0)],
    )
    within_bound = reference.transect_length_m
    for _ in range(8):
        within_bound = math.nextafter(within_bound, math.inf)
    outside_bound = math.nextafter(within_bound, math.inf)

    accepted = evaluate_candidate_against_explicit(
        reference,
        replace(candidate, transect_length_m=within_bound),
    )

    assert isinstance(accepted, IntervalValidationResult)
    with pytest.raises(ValidationError, match="transect_length_m"):
        evaluate_candidate_against_explicit(
            reference,
            replace(candidate, transect_length_m=outside_bound),
        )


def test_v_inputs_and_nested_records_remain_immutable_and_unchanged() -> None:
    _, reference, candidate, _ = _validation(
        [(0.0, 100.0), (150.0, 250.0)],
        [(0.0, 100.0), (150.0, 250.0)],
        max_bridge_gap_m=50.0,
    )
    reference_before = copy.deepcopy(reference.audit_summary())
    candidate_before = copy.deepcopy(candidate.audit_summary())

    result = evaluate_candidate_against_explicit(reference, candidate)

    assert reference.audit_summary() == reference_before
    assert candidate.audit_summary() == candidate_before
    with pytest.raises(FrozenInstanceError):
        result.total_bridged_gap_m = 999.0
    with pytest.raises(FrozenInstanceError):
        result.observed_support_metrics.iou = 0.0


def test_w_phase5a1_manual_benchmark_behavior_is_unchanged() -> None:
    _, reference, candidate, _ = _validation(
        [(0.0, 120.0), (180.0, 260.0), (310.0, 350.0)],
        [],
    )
    before = reference.audit_summary()

    evaluate_candidate_against_explicit(reference, candidate)

    assert reference.audit_summary() == before
    assert reference.individual_interval_widths_m == (120.0, 80.0, 40.0)
    assert reference.total_wetted_width_m == 240.0
    assert reference.total_internal_dry_gap_m == 110.0
    assert reference.outer_wetted_span_m == 350.0


def test_x_phase5a2_candidate_inference_behavior_is_unchanged() -> None:
    _, reference, candidate, _ = _validation(
        [(0.0, 250.0)],
        [(0.0, 100.0), (150.0, 250.0)],
        max_bridge_gap_m=50.0,
    )
    before = candidate.audit_summary()

    evaluate_candidate_against_explicit(reference, candidate)

    assert candidate.audit_summary() == before
    assert candidate.observed_candidate_wet_support_m == 200.0
    assert candidate.total_bridged_gap_m == 50.0
    assert candidate.total_inferred_candidate_interval_span_m == 250.0


def test_y_summary_dataframe_has_separate_stable_metric_namespaces() -> None:
    _, _, _, result = _validation(
        [(0.0, 250.0)],
        [(0.0, 100.0), (150.0, 250.0)],
        max_bridge_gap_m=50.0,
    )

    frame = result.summary_dataframe()

    assert frame.shape[0] == 1
    assert frame.columns.is_unique
    expected_columns = [
        "method_id",
        "method_version",
        "reference_method_id",
        "candidate_method_id",
        "candidate_method_version",
        "extent_classes",
        "station_bin_width_m",
        "min_extent_pixels_per_bin",
        "max_bridge_gap_m",
        "reference_interval_count",
        "observed_candidate_run_count",
        "inferred_candidate_interval_count",
        "observed_component_count_difference",
        "inferred_component_count_difference",
        *(f"observed_support_{name}" for name in _METRIC_FIELDS),
        *(f"bridge_inclusive_{name}" for name in _METRIC_FIELDS),
        *(f"observed_support_boundary_{name}" for name in _BOUNDARY_FIELDS),
        *(f"bridge_inclusive_boundary_{name}" for name in _BOUNDARY_FIELDS),
        "total_bridged_gap_m",
        "bridged_length_over_manual_wet_m",
        "bridged_length_over_manual_nonwet_m",
        "delta_iou_due_to_bridging",
        "delta_f1_due_to_bridging",
        "transect_wkt",
        "transect_crs",
        "transect_length_m",
        "metric_crs_wkt",
        "projection_center",
        "corridor_half_width_m",
        "reference_profile_name",
        "candidate_profile_name",
        "method_status",
        "method_warning",
    ]
    assert frame.columns.tolist() == expected_columns
    assert "iou" not in frame.columns
    assert frame.at[0, "observed_support_iou"] == 0.8
    assert frame.at[0, "bridge_inclusive_iou"] == 1.0
    assert result.as_dataframe().equals(frame)


def test_z_validation_plot_uses_separate_bands_without_mutating_results() -> None:
    _, reference, candidate, validation = _validation(
        [(0.0, 250.0)],
        [(0.0, 100.0), (150.0, 250.0)],
        max_bridge_gap_m=50.0,
        sampled_noneligible_stations=(112.5,),
    )
    reference_before = reference.audit_summary()
    candidate_before = candidate.audit_summary()

    figure, axes = plot_interval_validation(
        reference,
        candidate,
        validation=validation,
    )

    assert isinstance(figure, Figure)
    assert np.asarray(axes).shape == (4,)
    assert all(isinstance(axis, Axes) for axis in axes)
    titles = " ".join(axis.get_title().lower() for axis in axes)
    assert "manual" in titles
    assert "observed" in titles
    assert "bridge" in titles
    assert "iou=" in titles
    assert "f1=" in titles
    patch_ids = {
        patch.get_gid() for axis in axes for patch in axis.patches if patch.get_gid()
    }
    assert "swot-pixc-lab:validation-bridge-sampled_noneligible-bin-5" in patch_ids
    assert "swot-pixc-lab:validation-bridge-unsampled-bin-6" in patch_ids
    assert "swot-pixc-lab:validation-bridge-1" in patch_ids
    assert all(
        axis.get_xlim() == pytest.approx((0.0, reference.transect_length_m))
        for axis in axes
    )
    assert "not validated physical river banks" in figure._suptitle.get_text()
    assert "unsampled bins remain unknown" in figure._suptitle.get_text()
    assert reference.audit_summary() == reference_before
    assert candidate.audit_summary() == candidate_before


def test_empty_boundary_sets_return_none_summaries_not_nan() -> None:
    _, _, _, result = _validation([], [])

    for boundaries in (
        result.observed_support_boundary_metrics,
        result.bridge_inclusive_boundary_metrics,
    ):
        assert boundaries.manual_to_candidate_boundary_distances_m == ()
        assert boundaries.candidate_to_manual_boundary_distances_m == ()
        assert boundaries.manual_to_candidate_boundary_mean_m is None
        assert boundaries.manual_to_candidate_boundary_median_m is None
        assert boundaries.manual_to_candidate_boundary_max_m is None
        assert boundaries.candidate_to_manual_boundary_mean_m is None
        assert boundaries.candidate_to_manual_boundary_median_m is None
        assert boundaries.candidate_to_manual_boundary_max_m is None
        assert boundaries.symmetric_boundary_mean_m is None
        assert boundaries.symmetric_boundary_max_m is None


def test_validation_audit_retains_methods_configuration_frame_and_provenance() -> None:
    _, reference, candidate, result = _validation([(100.0, 200.0)], [(100.0, 200.0)])

    audit = result.audit_summary()

    assert result.reference_method_id == reference.method_id
    assert result.candidate_method_id == candidate.method_id
    assert result.candidate_method_version == candidate.method_version
    assert audit["candidate_configuration"] == {
        "extent_classes": (4,),
        "station_bin_width_m": 25.0,
        "min_extent_pixels_per_bin": 1,
        "max_bridge_gap_m": 0.0,
    }
    assert audit["transect_wkt"] == reference.transect_wkt
    assert audit["transect_crs"] == reference.transect_crs
    assert audit["metric_crs_wkt"] == reference.metric_crs_wkt
    assert audit["projection_center"] == reference.projection_center
    assert audit["corridor_half_width_m"] == 25.0
    assert audit["reference_profile_name"] == "synthetic_validation_profile"
    assert audit["candidate_profile_name"] == "synthetic_validation_profile"
    assert audit["reference_source_counts"] == {0: 4}
    assert audit["candidate_source_counts"] == {0: 4}
    assert len(audit["reference_source_summaries"]) == 1
    assert audit["reference_source_summaries"][0]["source_index"] == 0
    assert audit["reference_source_summaries"][0]["selected_pixel_count"] == 4
    assert len(audit["candidate_source_summaries"]) == 1
    assert audit["candidate_source_summaries"][0]["source_index"] == 0
    assert audit["candidate_source_summaries"][0]["selected_pixel_count"] == 4
    assert "analyst-supplied reference" in audit["method_warning"]
    assert (
        "not one-to-one"
        in audit["observed_support_boundary_metrics"]["diagnostic_warning"]
    )
    assert not hasattr(result, "pixels")
    assert not hasattr(result, "sample")
    with pytest.raises(TypeError):
        result.reference_source_counts[0] = 999


def test_validation_preserves_real_source_granule_and_tile_provenance() -> None:
    sample = _sample_for_candidate_intervals(
        [(100.0, 200.0)],
        with_source_metadata=True,
    )
    reference = measure_explicit_wet_intervals(sample, [(100.0, 200.0)])
    candidate = infer_candidate_wet_intervals(
        sample,
        extent_classes=(4,),
        station_bin_width_m=25.0,
        min_extent_pixels_per_bin=1,
        max_bridge_gap_m=0.0,
    )

    result = evaluate_candidate_against_explicit(reference, candidate)

    assert dict(result.reference_source_counts) == {0: 2, 1: 2}
    assert dict(result.candidate_source_counts) == {0: 2, 1: 2}
    assert [item.source_index for item in result.reference_source_summaries] == [0, 1]
    assert [item.tile for item in result.reference_source_summaries] == [
        "107L",
        "108L",
    ]
    assert [item.granule_id for item in result.reference_source_summaries] == [
        "synthetic-validation-107L",
        "synthetic-validation-108L",
    ]
    assert [item.filename for item in result.reference_source_summaries] == [
        "synthetic-validation-107L.nc",
        "synthetic-validation-108L.nc",
    ]
    assert [
        item.selected_pixel_count for item in result.reference_source_summaries
    ] == [
        2,
        2,
    ]
    assert [item.source_index for item in result.candidate_source_summaries] == [0, 1]
    assert [item.tile for item in result.candidate_source_summaries] == [
        "107L",
        "108L",
    ]
    assert [item.granule_id for item in result.candidate_source_summaries] == [
        "synthetic-validation-107L",
        "synthetic-validation-108L",
    ]
    assert [item.filename for item in result.candidate_source_summaries] == [
        "synthetic-validation-107L.nc",
        "synthetic-validation-108L.nc",
    ]
    assert [
        item.extent_eligible_pixel_count for item in result.candidate_source_summaries
    ] == [2, 2]

    audit = result.audit_summary()
    assert [item["tile"] for item in audit["reference_source_summaries"]] == [
        "107L",
        "108L",
    ]
    assert [item["filename"] for item in audit["candidate_source_summaries"]] == [
        "synthetic-validation-107L.nc",
        "synthetic-validation-108L.nc",
    ]
    assert all("path" not in item for item in audit["reference_source_summaries"])
    assert all("path" not in item for item in audit["candidate_source_summaries"])


def test_benchmark_record_serializes_synthetic_metadata_and_manual_contract() -> None:
    _, reference, _, _ = _validation(
        [(100.0, 200.0), (250.0, 300.0)],
        [],
    )
    metadata = ManualBenchmarkMetadata(
        benchmark_id="synthetic-benchmark",
        transect_id="synthetic-transect-001",
        annotation_id="synthetic-annotation-a",
        observation_identifier="SYNTHETIC_OBSERVATION",
        observation_date="2024-01-14",
        site_label="Synthetic multi-channel reach",
        analyst_id="synthetic-analyst-a",
        analyst_label="Synthetic Analyst A",
        reference_source_description="Synthetic diagram; not real imagery",
        reference_acquisition_date="2024-01-14",
        temporal_offset_hours=0.0,
        annotation_confidence="synthetic high",
        notes="SYNTHETIC EXAMPLE ONLY; no Koshi boundaries.",
    )

    record = create_manual_benchmark_record(metadata, reference)
    payload = record.as_dict()

    assert record.metadata == metadata
    assert record.reference == reference
    assert payload["metadata"]["benchmark_id"] == "synthetic-benchmark"
    assert payload["metadata"]["analyst_id"] == "synthetic-analyst-a"
    assert "manual_intervals" not in payload["metadata"]
    assert payload["schema_version"] == "1.0"
    explicit = payload["explicit_interval_reference"]
    assert explicit["method_id"] == reference.method_id
    assert "analyst_id" not in explicit
    assert explicit["manual_intervals"] == [
        {
            "interval_id": 1,
            "start_station_m": 100.0,
            "end_station_m": 200.0,
            "width_m": 100.0,
        },
        {
            "interval_id": 2,
            "start_station_m": 250.0,
            "end_station_m": 300.0,
            "width_m": 50.0,
        },
    ]
    assert "pixels" not in json.dumps(payload)
    json.dumps(payload)
    with pytest.raises(FrozenInstanceError):
        record.schema_version = "changed"
    with pytest.raises(FrozenInstanceError):
        metadata.notes = "changed"


def test_committed_manual_benchmark_json_loads_with_complete_schema() -> None:
    example_path = (
        Path(__file__).parents[1]
        / "examples"
        / "manual_interval_benchmark_example.json"
    )

    payload = json.loads(example_path.read_text(encoding="utf-8"))

    assert set(payload) == {
        "format_name",
        "format_version",
        "synthetic_example",
        "warning",
        "generation_instruction",
        "records",
        "multi_analyst_policy",
    }
    assert payload["format_name"] == (
        "swot_pixc_lab_manual_interval_benchmark_collection"
    )
    assert payload["format_version"] == "1.0"
    assert payload["synthetic_example"] is True
    assert "SYNTHETIC EXAMPLE ONLY" in payload["warning"]
    assert len(payload["records"]) == 2

    expected_metadata_fields = {
        "benchmark_id",
        "transect_id",
        "annotation_id",
        "observation_identifier",
        "observation_date",
        "site_label",
        "analyst_id",
        "analyst_label",
        "reference_source_description",
        "reference_acquisition_date",
        "temporal_offset_hours",
        "annotation_confidence",
        "notes",
    }
    expected_reference_fields = {
        "method_id",
        "method_status",
        "method_warning",
        "manual_intervals",
        "measurement_summary",
        "station_frame",
        "upstream_context",
    }
    expected_summary_fields = {
        "interval_count",
        "total_wetted_width_m",
        "outer_wetted_span_m",
        "total_internal_dry_gap_m",
    }
    expected_station_frame_fields = {
        "transect_wkt",
        "transect_crs",
        "transect_length_m",
        "metric_crs_wkt",
        "projection_center",
        "corridor_half_width_m",
    }
    expected_upstream_fields = {
        "profile_name",
        "profile_label",
        "profile_status",
        "selected_pixel_count",
        "classification_counts",
        "source_counts",
        "source_tile_counts",
        "source_summaries",
    }

    annotation_ids: list[str] = []
    analyst_ids: list[str] = []
    for record in payload["records"]:
        assert set(record) == {
            "schema_version",
            "metadata",
            "explicit_interval_reference",
        }
        assert record["schema_version"] == "1.0"
        metadata = record["metadata"]
        reference = record["explicit_interval_reference"]
        assert set(metadata) == expected_metadata_fields
        assert set(reference) == expected_reference_fields
        assert set(reference["measurement_summary"]) == expected_summary_fields
        assert set(reference["station_frame"]) == expected_station_frame_fields
        assert set(reference["upstream_context"]) == expected_upstream_fields
        assert metadata["benchmark_id"]
        assert metadata["transect_id"]
        assert metadata["annotation_id"]
        assert metadata["analyst_id"]
        assert metadata["observation_identifier"] == "SYNTHETIC_OBSERVATION_NOT_REAL"
        assert reference["method_id"] == "explicit_analyst_wet_intervals_v1"

        intervals = reference["manual_intervals"]
        assert all(
            set(interval)
            == {
                "interval_id",
                "start_station_m",
                "end_station_m",
                "width_m",
            }
            for interval in intervals
        )
        assert all(
            interval["width_m"]
            == pytest.approx(interval["end_station_m"] - interval["start_station_m"])
            for interval in intervals
        )
        summary = reference["measurement_summary"]
        assert summary["interval_count"] == len(intervals)
        assert summary["total_wetted_width_m"] == pytest.approx(
            sum(interval["width_m"] for interval in intervals)
        )
        assert summary["outer_wetted_span_m"] == pytest.approx(
            intervals[-1]["end_station_m"] - intervals[0]["start_station_m"]
        )
        assert summary["total_internal_dry_gap_m"] == pytest.approx(
            sum(
                right["start_station_m"] - left["end_station_m"]
                for left, right in zip(intervals, intervals[1:], strict=False)
            )
        )
        annotation_ids.append(metadata["annotation_id"])
        analyst_ids.append(metadata["analyst_id"])

    assert len(set(annotation_ids)) == 2
    assert len(set(analyst_ids)) == 2
    assert (
        len({record["metadata"]["benchmark_id"] for record in payload["records"]}) == 1
    )
    assert (
        len({record["metadata"]["transect_id"] for record in payload["records"]}) == 1
    )
    assert all(
        not {"rank", "score", "best", "winner", "consensus"}.intersection(record)
        for record in payload["records"]
    )


def test_benchmark_records_preserve_analysts_independently_without_consensus() -> None:
    _, reference_a, _, _ = _validation([(100.0, 200.0)], [])
    _, reference_b, _, _ = _validation([(110.0, 210.0)], [])
    common = {
        "benchmark_id": "synthetic-multi-analyst",
        "transect_id": "synthetic-transect-001",
        "observation_identifier": "SYNTHETIC_OBSERVATION",
    }
    record_a = create_manual_benchmark_record(
        ManualBenchmarkMetadata(
            annotation_id="annotation-a",
            analyst_id="analyst-a",
            **common,
        ),
        reference_a,
    )
    record_b = create_manual_benchmark_record(
        ManualBenchmarkMetadata(
            annotation_id="annotation-b",
            analyst_id="analyst-b",
            **common,
        ),
        reference_b,
    )

    assert record_a.metadata.analyst_id == "analyst-a"
    assert record_b.metadata.analyst_id == "analyst-b"
    assert record_a.reference.intervals != record_b.reference.intervals
    assert not hasattr(record_a, "consensus")
    assert not hasattr(record_a, "average")


def test_sensitivity_validation_preserves_configuration_order_without_ranking() -> None:
    sample = _sample_for_candidate_intervals([(0.0, 100.0), (150.0, 250.0)])
    reference = measure_explicit_wet_intervals(
        sample,
        [(0.0, 100.0), (150.0, 250.0)],
    )
    sensitivity = run_candidate_interval_sensitivity(
        sample,
        configurations=[
            {
                "extent_classes": (4,),
                "station_bin_width_m": 25.0,
                "min_extent_pixels_per_bin": 1,
                "max_bridge_gap_m": 0.0,
            },
            {
                "extent_classes": (3, 4),
                "station_bin_width_m": 50.0,
                "min_extent_pixels_per_bin": 1,
                "max_bridge_gap_m": 50.0,
            },
        ],
    )
    reference_before = reference.audit_summary()
    candidates_before = [item.audit_summary() for item in sensitivity.results]

    result = evaluate_sensitivity_against_explicit(reference, sensitivity)
    frame = result.dataframe()

    assert isinstance(result, SensitivityValidationResult)
    assert result.configuration_count == 2
    assert [item.extent_classes for item in result.results] == [
        (4,),
        (3, 4),
    ]
    assert frame["configuration_id"].tolist() == [1, 2]
    assert frame["extent_classes"].tolist() == [(4,), (3, 4)]
    assert frame["station_bin_width_m"].tolist() == [25.0, 50.0]
    assert frame["min_extent_pixels_per_bin"].tolist() == [1, 1]
    assert frame["max_bridge_gap_m"].tolist() == [0.0, 50.0]
    assert not {"rank", "score", "best", "winner"}.intersection(frame.columns)
    assert result.summary_dataframe().equals(frame)
    assert reference.audit_summary() == reference_before
    assert [item.audit_summary() for item in sensitivity.results] == candidates_before
    with pytest.raises(FrozenInstanceError):
        result.results = ()


def test_validation_rejects_reversed_input_types_actionably() -> None:
    _, reference, candidate, _ = _validation([(100.0, 200.0)], [(100.0, 200.0)])

    with pytest.raises(TypeError, match="ExplicitIntervalWidthResult"):
        evaluate_candidate_against_explicit(candidate, reference)
    with pytest.raises(TypeError, match="CandidateIntervalInferenceResult"):
        evaluate_candidate_against_explicit(reference, reference)
