from __future__ import annotations

import copy
import inspect
from collections import Counter
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS, Transformer
from shapely.geometry import LineString, box

import swot_pixc_lab.bank_inference as bank_inference
from swot_pixc_lab import (
    BankInferenceError,
    CandidateIntervalConfiguration,
    ExplicitIntervalWidthResult,
    infer_candidate_wet_intervals,
    measure_explicit_wet_intervals,
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
_FLOAT_FILL = np.float32(9.96921e36)


def _geographic(x: object, y: object) -> tuple[np.ndarray, np.ndarray]:
    longitude, latitude = _TO_GEOGRAPHIC.transform(x, y)
    return np.asarray(longitude), np.asarray(latitude)


def _transect() -> LineString:
    longitude, latitude = _geographic([-500.0, 500.0], [0.0, 0.0])
    return LineString(zip(longitude, latitude, strict=True))


def _source(source_index: int, tile: str, point_count: int) -> PixcSourceMetadata:
    classification = PixcVariableMetadata(
        name="classification",
        dimensions=("points",),
        shape=(point_count,),
        dtype="uint8",
        attributes=MappingProxyType({"_FillValue": np.uint8(255)}),
    )
    return PixcSourceMetadata(
        source_index=source_index,
        path=Path(f"synthetic-{tile}.nc"),
        filename=f"synthetic-{tile}.nc",
        granule_id=f"synthetic-{tile}",
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
        loaded_variables=(
            "longitude",
            "latitude",
            "classification",
            "height",
            "water_frac",
            "sig0",
        ),
        points_before=point_count,
        points_after=point_count,
        invalid_coordinate_count=0,
        normalized_longitude_count=0,
    )


def _sample(
    stations: list[float],
    classifications: list[int],
    *,
    source_indices: list[int] | None = None,
    with_source_metadata: bool = False,
    heights: list[float] | None = None,
    water_fractions: list[float] | None = None,
    backscatter: list[float] | None = None,
    pixel_areas: list[float] | None = None,
    classification_quality: list[int] | None = None,
):
    assert len(stations) == len(classifications)
    point_count = len(stations)
    sources = source_indices if source_indices is not None else [0] * point_count
    assert len(sources) == point_count
    longitude, latitude = _geographic(
        np.asarray(stations, dtype=np.float64) - 500.0,
        np.zeros(point_count, dtype=np.float64),
    )
    dataset = xr.Dataset(
        {
            "longitude": xr.DataArray(longitude, dims="points"),
            "latitude": xr.DataArray(latitude, dims="points"),
            "classification": xr.DataArray(
                np.asarray(classifications, dtype=np.uint8),
                dims="points",
                attrs={"_FillValue": np.uint8(255)},
            ),
            "height": xr.DataArray(
                np.asarray(
                    heights if heights is not None else [100.0] * point_count,
                    dtype=np.float32,
                ),
                dims="points",
                attrs={"_FillValue": _FLOAT_FILL, "units": "m"},
            ),
            "water_frac": xr.DataArray(
                np.asarray(
                    water_fractions
                    if water_fractions is not None
                    else [1.0] * point_count,
                    dtype=np.float32,
                ),
                dims="points",
                attrs={"_FillValue": _FLOAT_FILL, "units": "1"},
            ),
            "sig0": xr.DataArray(
                np.asarray(
                    backscatter if backscatter is not None else [-10.0] * point_count,
                    dtype=np.float32,
                ),
                dims="points",
                attrs={"_FillValue": _FLOAT_FILL, "units": "1"},
            ),
            "pixel_area": xr.DataArray(
                np.asarray(
                    pixel_areas if pixel_areas is not None else [10.0] * point_count,
                    dtype=np.float32,
                ),
                dims="points",
                attrs={"_FillValue": _FLOAT_FILL, "units": "m^2"},
            ),
            "classification_qual": xr.DataArray(
                np.asarray(
                    classification_quality
                    if classification_quality is not None
                    else [0] * point_count,
                    dtype=np.uint32,
                ),
                dims="points",
                attrs={"_FillValue": np.uint32(4294967295)},
            ),
            "source_index": xr.DataArray(
                np.asarray(sources, dtype=np.int32), dims="points"
            ),
            "source_point_index": xr.DataArray(
                np.arange(point_count, dtype=np.int64), dims="points"
            ),
        },
        attrs={
            "test_marker": "unchanged",
            "swot_pixc_lab_qc_profile": "synthetic_extent_profile",
            "swot_pixc_lab_qc_label": "Synthetic extent profile",
            "swot_pixc_lab_qc_status": "experimental / test only",
        },
    )

    data: xr.Dataset | PixcObservation = dataset
    if with_source_metadata:
        counts = Counter(sources)
        known_sources = tuple(
            _source(index, tile, counts.get(index, 0))
            for index, tile in ((0, "107L"), (1, "108L"))
        )
        data = PixcObservation(
            pixels=dataset,
            sources=known_sources,
            aoi=ExactAoi(
                geometry=box(86.0, 26.0, 88.0, 28.0),
                kind="bounding_box",
                bounding_box=(86.0, 26.0, 88.0, 28.0),
            ),
        )

    result = sample_transect(
        data,
        _transect(),
        corridor_half_width_m=25.0,
    )
    assert result.selected_pixel_count == point_count
    # Binning tests require exact station-edge values. The replacement remains
    # a valid Phase 4 sample and changes only its detached derived station array.
    pixels = result.pixels.copy(deep=True)
    pixels["station_m"] = xr.DataArray(
        np.asarray(stations, dtype=np.float64),
        dims="points",
        attrs=result.station_m.attrs,
    )
    return replace(result, pixels=pixels)


def _with_stations(sample, stations: list[float]):
    pixels = sample.pixels.copy(deep=True)
    pixels["station_m"] = xr.DataArray(
        np.asarray(stations, dtype=np.float64),
        dims="points",
        attrs=sample.station_m.attrs,
    )
    return replace(sample, pixels=pixels)


def _infer(sample, **overrides):
    parameters = {
        "extent_classes": (4,),
        "station_bin_width_m": 100.0,
        "min_extent_pixels_per_bin": 1,
        "max_bridge_gap_m": 0.0,
    }
    parameters.update(overrides)
    return infer_candidate_wet_intervals(sample, **parameters)


def test_a_simple_single_candidate_wet_run_uses_bin_edges() -> None:
    result = _infer(_sample([10.0, 110.0, 150.0], [4, 4, 4]))

    assert [item.state for item in result.bins[:3]] == [
        "candidate_wet",
        "candidate_wet",
        "unsampled",
    ]
    assert result.candidate_interval_count == 1
    interval = result.candidate_intervals[0]
    assert (interval.start_station_m, interval.end_station_m) == (0.0, 200.0)
    assert (interval.first_bin_id, interval.last_bin_id) == (1, 2)
    assert interval.candidate_wet_bin_count == 2
    assert interval.bridged_bin_count == 0
    assert interval.observed_candidate_wet_support_m == 200.0
    assert interval.inferred_candidate_interval_span_m == 200.0


def test_b_three_separate_wet_runs_remain_ordered_intervals() -> None:
    result = _infer(_sample([10.0, 210.0, 510.0], [4, 4, 4]))

    assert [item.candidate_interval_id for item in result.candidate_intervals] == [
        1,
        2,
        3,
    ]
    assert [
        (item.start_station_m, item.end_station_m)
        for item in result.candidate_intervals
    ] == [(0.0, 100.0), (200.0, 300.0), (500.0, 600.0)]
    assert result.individual_candidate_widths_m == (100.0, 100.0, 100.0)
    assert result.total_candidate_wetted_width_m == 300.0
    assert result.outer_candidate_wetted_span_m == 600.0


def test_c_sampled_noneligible_gap_is_not_unsampled() -> None:
    result = _infer(_sample([10.0, 110.0, 210.0], [4, 1, 4]))
    gap = result.bins[1]

    assert gap.state == "sampled_noneligible"
    assert gap.total_sampled_pixel_count == 1
    assert gap.extent_eligible_pixel_count == 0
    assert gap.extent_eligible_fraction == 0.0
    assert dict(gap.classification_counts) == {1: 1}


def test_d_completely_unsampled_gap_is_not_confirmed_dry() -> None:
    result = _infer(_sample([10.0, 210.0], [4, 4]))
    gap = result.bins[1]

    assert gap.state == "unsampled"
    assert gap.total_sampled_pixel_count == 0
    assert gap.extent_eligible_pixel_count == 0
    assert gap.extent_eligible_fraction is None
    assert dict(gap.classification_counts) == {}


def test_e_zero_bridge_tolerance_leaves_runs_separate() -> None:
    result = _infer(
        _sample([10.0, 210.0], [4, 4]),
        max_bridge_gap_m=0.0,
    )

    assert result.candidate_interval_count == 2
    assert result.bridge_count == 0
    assert result.bridge_records == ()
    assert not any(item.bridged for item in result.bins)


def test_f_positive_bridge_tolerance_applies_only_at_allowed_gap_width() -> None:
    sample = _sample([10.0, 210.0], [4, 4])

    below = _infer(sample, max_bridge_gap_m=99.999)
    at_limit = _infer(sample, max_bridge_gap_m=100.0)

    assert below.candidate_interval_count == 2
    assert below.bridge_count == 0
    assert at_limit.candidate_interval_count == 1
    assert at_limit.bridge_count == 1
    assert at_limit.total_bridged_gap_m == 100.0


def test_f_bridge_limit_equality_is_stable_for_decimal_bin_widths() -> None:
    sample = _sample([0.05, 0.45], [4, 4])

    result = _infer(
        sample,
        station_bin_width_m=0.1,
        max_bridge_gap_m=0.3,
    )

    assert result.bridge_count == 1
    assert result.bridge_records[0].gap_bin_count == 3
    assert result.bridge_records[0].gap_width_m == pytest.approx(0.3)
    assert result.candidate_interval_count == 1
    assert all(
        left.end_station_m == right.start_station_m
        for left, right in zip(result.bins, result.bins[1:], strict=False)
    )


def test_f_bridge_roundoff_allowance_is_bounded_to_eight_ulps() -> None:
    sample = _sample([0.05, 0.45], [4, 4])
    represented_gap = 0.4 - 0.1
    inside = represented_gap
    outside = represented_gap
    for _ in range(8):
        inside = float(np.nextafter(inside, -np.inf))
        outside = float(np.nextafter(outside, -np.inf))
    outside = float(np.nextafter(outside, -np.inf))

    assert (
        _infer(
            sample,
            station_bin_width_m=0.1,
            max_bridge_gap_m=inside,
        ).bridge_count
        == 1
    )
    assert (
        _infer(
            sample,
            station_bin_width_m=0.1,
            max_bridge_gap_m=outside,
        ).bridge_count
        == 0
    )


def test_g_bridged_gap_provenance_and_support_span_are_reconstructable() -> None:
    result = _infer(
        _sample([10.0, 110.0, 210.0], [4, 1, 4]),
        max_bridge_gap_m=100.0,
    )
    bridge = result.bridge_records[0]
    interval = result.candidate_intervals[0]

    assert bridge.bridge_id == 1
    assert (bridge.left_wet_run_id, bridge.right_wet_run_id) == (1, 2)
    assert (bridge.first_gap_bin_id, bridge.last_gap_bin_id) == (2, 2)
    assert (bridge.gap_start_station_m, bridge.gap_end_station_m) == (100.0, 200.0)
    assert bridge.gap_width_m == 100.0
    assert bridge.reason == "caller-specified gap bridging"
    assert interval.bridge_ids == (1,)
    assert interval.observed_candidate_wet_support_m == 200.0
    assert interval.bridged_gap_m == 100.0
    assert interval.inferred_candidate_interval_span_m == 300.0
    assert result.individual_candidate_widths_m == (200.0,)
    assert result.total_candidate_wetted_width_m == 200.0
    assert (
        result.observed_candidate_wet_support_m + result.total_bridged_gap_m
        == result.inferred_candidate_interval_span_m
    )
    assert result.bins[1].bridge_id == 1
    assert result.bins[1].bridged is True


def test_h_bridge_distinguishes_unsampled_and_sampled_noneligible_gaps() -> None:
    sampled = _infer(
        _sample([10.0, 110.0, 210.0], [4, 1, 4]),
        max_bridge_gap_m=100.0,
    ).bridge_records[0]
    unsampled = _infer(
        _sample([10.0, 210.0], [4, 4]),
        max_bridge_gap_m=100.0,
    ).bridge_records[0]

    assert (
        sampled.unsampled_bin_count,
        sampled.sampled_noneligible_bin_count,
    ) == (0, 1)
    assert (
        unsampled.unsampled_bin_count,
        unsampled.sampled_noneligible_bin_count,
    ) == (1, 0)
    assert sampled.total_sampled_pixel_count == 1
    assert unsampled.total_sampled_pixel_count == 0


def test_i_and_w_exact_and_adjacent_floating_bin_edges_are_deterministic() -> None:
    edge = 100.0
    stations = [
        0.0,
        np.nextafter(edge, -np.inf),
        edge,
        np.nextafter(edge, np.inf),
        999.0,
    ]
    sample = _sample(stations, [4] * len(stations))
    sample = _with_stations(
        sample,
        [*stations[:-1], sample.transect_length_m],
    )

    result = _infer(sample)

    assert result.bins[0].total_sampled_pixel_count == 2
    assert result.bins[1].total_sampled_pixel_count == 2
    assert result.bins[-1].total_sampled_pixel_count == 1
    assert sum(item.total_sampled_pixel_count for item in result.bins) == 5
    assert "final bin includes the transect endpoint" in result.bin_edge_semantics
    assert "belongs to the bin on its right" in result.bin_edge_semantics


def test_j_final_partial_bin_is_retained_with_actual_edges_and_center() -> None:
    sample = _sample([999.0], [4])
    sample = _with_stations(sample, [sample.transect_length_m])

    result = _infer(sample, station_bin_width_m=300.0)
    final_bin = result.bins[-1]

    assert result.bin_count == 4
    assert final_bin.start_station_m == 900.0
    assert final_bin.end_station_m == sample.transect_length_m
    assert final_bin.bin_span_m == pytest.approx(100.0, abs=1.0e-6)
    assert final_bin.center_station_m == pytest.approx(950.0, abs=1.0e-6)
    assert final_bin.state == "candidate_wet"
    assert result.candidate_intervals[0].end_station_m == sample.transect_length_m


def test_every_positive_roundtrip_remainder_is_retained_as_a_partial_bin() -> None:
    sample = _sample([], [])

    result = _infer(sample, station_bin_width_m=25.0)

    assert sample.transect_length_m > 1000.0
    assert result.bin_count == 41
    assert result.bins[-1].start_station_m == 1000.0
    assert 0.0 < result.bins[-1].bin_span_m < 1.0e-6
    assert result.bins[-1].end_station_m == sample.transect_length_m
    assert sum(item.bin_span_m for item in result.bins) == pytest.approx(
        sample.transect_length_m
    )


def test_quotient_rounding_down_cannot_hide_a_positive_final_remainder() -> None:
    station_bin_width_m = 0.6378553829473124
    transect_length_m = 3.189276914736562
    assert transect_length_m / station_bin_width_m == 5.0
    assert 5 * station_bin_width_m < transect_length_m

    starts, ends = bank_inference._station_bin_edges(
        transect_length_m, station_bin_width_m
    )

    assert len(starts) == len(ends) == 6
    assert starts[-1] == 5 * station_bin_width_m
    assert starts[-1] < ends[-1] == transect_length_m
    np.testing.assert_array_equal(ends[:-1], starts[1:])
    edge = starts[-1]
    assigned = bank_inference._assign_bin_indices(
        np.asarray([np.nextafter(edge, -np.inf), edge, np.nextafter(edge, np.inf)]),
        starts,
    )
    assert assigned.tolist() == [4, 5, 5]


def test_k_no_extent_eligible_pixels_produces_no_candidate_interval() -> None:
    result = _infer(_sample([10.0, 110.0], [1, 2]))

    assert [item.state for item in result.bins[:2]] == [
        "sampled_noneligible",
        "sampled_noneligible",
    ]
    assert result.candidate_intervals == ()
    assert result.candidate_interval_count == 0
    assert result.total_candidate_wetted_width_m == 0.0
    assert result.inferred_candidate_interval_span_m == 0.0
    assert result.outer_candidate_wetted_span_m is None


def test_l_empty_transect_sample_retains_the_geometry_defined_domain() -> None:
    sample = _sample([], [])

    result = _infer(sample)

    assert result.selected_pixel_count == 0
    assert result.bin_count == 11
    assert result.unsampled_bin_count == 11
    assert all(item.state == "unsampled" for item in result.bins)
    assert all(item.extent_eligible_fraction is None for item in result.bins)
    assert result.candidate_intervals == ()
    assert result.bridge_records == ()


def test_m_unsampled_domain_portions_are_preserved_and_not_externally_bridged() -> None:
    result = _infer(
        _sample([210.0, 610.0], [4, 4]),
        max_bridge_gap_m=300.0,
    )

    assert [item.state for item in result.bins[:7]] == [
        "unsampled",
        "unsampled",
        "candidate_wet",
        "unsampled",
        "unsampled",
        "unsampled",
        "candidate_wet",
    ]
    assert result.bridge_count == 1
    assert result.bridge_records[0].unsampled_bin_count == 3
    assert all(item.bridge_id is None for item in result.bins[:2])
    assert all(item.state == "unsampled" for item in result.bins[7:])
    assert all(item.bridge_id is None for item in result.bins[7:])
    assert result.candidate_intervals[0].start_station_m == 200.0
    assert result.candidate_intervals[0].end_station_m == 700.0


def test_n_extent_class_sensitivity_changes_only_explicit_membership() -> None:
    sample = _sample([10.0, 110.0], [3, 4])

    open_water = _infer(sample, extent_classes=(4,))
    near_land_and_open = _infer(sample, extent_classes=(3, 4))

    assert open_water.extent_classes == (4,)
    assert open_water.candidate_wet_bin_count == 1
    assert open_water.candidate_intervals[0].start_station_m == 100.0
    assert near_land_and_open.extent_classes == (3, 4)
    assert near_land_and_open.candidate_wet_bin_count == 2
    assert near_land_and_open.candidate_intervals[0].start_station_m == 0.0


def test_o_minimum_count_changes_candidate_support_deterministically() -> None:
    sample = _sample([10.0, 15.0, 110.0], [4, 4, 4])

    minimum_one = _infer(sample, min_extent_pixels_per_bin=1)
    minimum_two = _infer(sample, min_extent_pixels_per_bin=2)

    assert minimum_one.candidate_wet_bin_count == 2
    assert minimum_one.candidate_intervals[0].end_station_m == 200.0
    assert minimum_two.candidate_wet_bin_count == 1
    assert minimum_two.bins[1].state == "sampled_noneligible"
    assert minimum_two.bins[1].extent_eligible_pixel_count == 1
    assert minimum_two.candidate_intervals[0].end_station_m == 100.0


@pytest.mark.parametrize(
    "extent_classes",
    [[], [4, 4], [True], [np.bool_(False)], [4.0], ["4"], {3, 4}, {"a": 4}],
)
def test_p_invalid_extent_class_inputs_are_rejected(extent_classes) -> None:
    with pytest.raises(BankInferenceError, match="extent_classes"):
        _infer(_sample([], []), extent_classes=extent_classes)


@pytest.mark.parametrize(
    "station_bin_width_m",
    [0.0, -1.0, np.nan, np.inf, True, np.bool_(True), "100"],
)
def test_q_invalid_station_bin_width_is_rejected(station_bin_width_m) -> None:
    with pytest.raises(BankInferenceError, match="station_bin_width_m"):
        _infer(_sample([], []), station_bin_width_m=station_bin_width_m)


def test_positive_bin_width_that_would_create_too_many_bins_fails_promptly() -> None:
    with pytest.raises(BankInferenceError, match="Choose a wider bin"):
        _infer(_sample([], []), station_bin_width_m=1.0e-300)


@pytest.mark.parametrize(
    "minimum_count",
    [0, -1, 1.0, True, np.bool_(True), "1"],
)
def test_r_invalid_minimum_count_is_rejected(minimum_count) -> None:
    with pytest.raises(BankInferenceError, match="min_extent_pixels_per_bin"):
        _infer(_sample([], []), min_extent_pixels_per_bin=minimum_count)


@pytest.mark.parametrize(
    "max_bridge_gap_m",
    [-1.0, np.nan, np.inf, True, np.bool_(True), "0"],
)
def test_s_invalid_gap_tolerance_is_rejected(max_bridge_gap_m) -> None:
    with pytest.raises(BankInferenceError, match="max_bridge_gap_m"):
        _infer(_sample([], []), max_bridge_gap_m=max_bridge_gap_m)


def test_t_input_transect_sample_and_pixels_remain_unchanged() -> None:
    sample = _sample([210.0, 10.0, 110.0], [4, 3, 1])
    before = sample.pixels.copy(deep=True)
    before_attrs = copy.deepcopy(sample.pixels.attrs)
    before_station = sample.station_m.values.copy()

    _infer(
        sample,
        extent_classes=(3, 4),
        min_extent_pixels_per_bin=1,
        max_bridge_gap_m=100.0,
    )

    xr.testing.assert_identical(sample.pixels, before)
    assert sample.pixels.attrs == before_attrs
    np.testing.assert_array_equal(sample.station_m.values, before_station)


def test_u_phase5a1_manual_contract_remains_distinct_and_unchanged() -> None:
    sample = _sample([10.0, 210.0], [4, 4])

    candidate = _infer(sample, max_bridge_gap_m=100.0)
    manual = measure_explicit_wet_intervals(
        sample,
        [(0.0, 120.0), (180.0, 260.0), (310.0, 350.0)],
    )

    assert not isinstance(candidate, ExplicitIntervalWidthResult)
    assert candidate.method_id != manual.method_id
    assert "candidate" in candidate.method_status.lower()
    assert "analyst" in manual.method_warning.lower()
    assert manual.individual_interval_widths_m == (120.0, 80.0, 40.0)
    assert manual.total_wetted_width_m == 240.0
    assert manual.total_internal_dry_gap_m == 110.0
    assert manual.outer_wetted_span_m == 350.0
    assert manual.total_wetted_width_m + manual.total_internal_dry_gap_m == 350.0


def test_v_multi_source_provenance_is_traceable_at_every_level() -> None:
    sample = _sample(
        [10.0, 20.0, 110.0, 210.0],
        [4, 1, 4, 4],
        source_indices=[0, 0, 1, 1],
        with_source_metadata=True,
    )

    result = _infer(sample)

    assert dict(result.source_counts) == {0: 2, 1: 2}
    assert dict(result.extent_eligible_source_counts) == {0: 1, 1: 2}
    assert dict(result.bins[0].source_counts) == {0: 2}
    assert dict(result.bins[0].extent_eligible_source_counts) == {0: 1}
    assert [item.source_index for item in result.source_summaries] == [0, 1]
    assert [item.tile for item in result.source_summaries] == ["107L", "108L"]
    assert [item.granule_id for item in result.source_summaries] == [
        "synthetic-107L",
        "synthetic-108L",
    ]
    assert [item.filename for item in result.source_summaries] == [
        "synthetic-107L.nc",
        "synthetic-108L.nc",
    ]
    assert [item.selected_pixel_count for item in result.source_summaries] == [2, 2]
    assert [item.extent_eligible_pixel_count for item in result.source_summaries] == [
        1,
        2,
    ]
    interval = result.candidate_intervals[0]
    assert dict(interval.source_counts) == {0: 2, 1: 2}
    assert dict(interval.extent_eligible_source_counts) == {0: 1, 1: 2}
    audit_sources = result.audit_summary()["source_summaries"]
    assert [item["granule_id"] for item in audit_sources] == [
        "synthetic-107L",
        "synthetic-108L",
    ]
    assert [item["tile"] for item in audit_sources] == ["107L", "108L"]
    np.testing.assert_array_equal(sample.pixels["source_point_index"], [0, 1, 2, 3])


def test_zero_selected_input_source_remains_in_compact_provenance() -> None:
    sample = replace(_sample([], []), input_source_indices=(5,))

    result = _infer(sample)

    assert dict(result.source_counts) == {5: 0}
    assert dict(result.extent_eligible_source_counts) == {5: 0}
    assert dict(result.source_tile_counts) == {"source_5": 0}
    assert len(result.source_summaries) == 1
    assert result.source_summaries[0].source_index == 5
    assert result.source_summaries[0].selected_pixel_count == 0


def test_x_nonclassification_variables_do_not_affect_inference() -> None:
    baseline = _sample(
        [10.0, 110.0, 210.0],
        [4, 1, 4],
        heights=[100.0, 101.0, 102.0],
        water_fractions=[0.0, 0.5, 1.0],
        backscatter=[-20.0, -10.0, 0.0],
        pixel_areas=[1.0, 2.0, 3.0],
        classification_quality=[0, 1, 2],
    )
    perturbed = _sample(
        [10.0, 110.0, 210.0],
        [4, 1, 4],
        heights=[np.nan, _FLOAT_FILL, -9999.0],
        water_fractions=[_FLOAT_FILL, -100.0, 100.0],
        backscatter=[np.nan, _FLOAT_FILL, 9999.0],
        pixel_areas=[_FLOAT_FILL, -1.0, 1.0e20],
        classification_quality=[4294967295, 2147483648, 4294967294],
    )

    baseline_result = _infer(baseline, max_bridge_gap_m=100.0)
    perturbed_result = _infer(perturbed, max_bridge_gap_m=100.0)

    assert baseline_result.audit_summary() == perturbed_result.audit_summary()


def test_classification_fill_is_never_extent_evidence_even_if_requested() -> None:
    result = _infer(
        _sample([10.0, 110.0], [4, 255]),
        extent_classes=(4, 255),
    )

    assert dict(result.classification_counts) == {4: 1}
    assert dict(result.extent_eligible_classification_counts) == {4: 1}
    assert result.bins[1].state == "sampled_noneligible"
    assert result.bins[1].classification_counts == {}


def test_required_scientific_parameters_have_no_defaults() -> None:
    parameters = inspect.signature(infer_candidate_wet_intervals).parameters

    for name in (
        "extent_classes",
        "station_bin_width_m",
        "min_extent_pixels_per_bin",
        "max_bridge_gap_m",
    ):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is inspect.Parameter.empty


def test_sensitivity_preserves_configuration_order_without_ranking() -> None:
    sample = _sample([10.0, 110.0, 210.0], [3, 4, 4])
    first = CandidateIntervalConfiguration(
        extent_classes=(4,),
        station_bin_width_m=100.0,
        min_extent_pixels_per_bin=1,
        max_bridge_gap_m=0.0,
    )
    second = {
        "extent_classes": (3, 4),
        "station_bin_width_m": 200.0,
        "min_extent_pixels_per_bin": 2,
        "max_bridge_gap_m": 100.0,
    }

    sensitivity = run_candidate_interval_sensitivity(
        sample,
        configurations=[first, second],
    )
    table = sensitivity.dataframe()

    assert sensitivity.configuration_count == 2
    assert [item.extent_classes for item in sensitivity.results] == [
        (4,),
        (3, 4),
    ]
    assert table["configuration_id"].tolist() == [1, 2]
    assert table["station_bin_width_m"].tolist() == [100.0, 200.0]
    assert table["min_extent_pixels_per_bin"].tolist() == [1, 2]
    assert table["max_bridge_gap_m"].tolist() == [0.0, 100.0]
    assert not {"rank", "score", "best", "winner"}.intersection(table.columns)


@pytest.mark.parametrize(
    "configuration",
    [
        {"extent_classes": (4,)},
        {
            "extent_classes": (4,),
            "station_bin_width_m": 100.0,
            "min_extent_pixels_per_bin": 1,
            "max_bridge_gap_m": 0.0,
            "rank": 1,
        },
    ],
)
def test_sensitivity_never_fills_missing_or_accepts_ranking_keys(configuration) -> None:
    with pytest.raises(BankInferenceError, match="invalid keys"):
        run_candidate_interval_sensitivity(
            _sample([], []),
            configurations=[configuration],
        )


def test_sensitivity_rejects_empty_configuration_and_validates_sample_first() -> None:
    sample = _sample([], [])

    with pytest.raises(BankInferenceError, match="at least one"):
        run_candidate_interval_sensitivity(sample, configurations=[])
    with pytest.raises(TypeError, match="TransectSample"):
        run_candidate_interval_sensitivity(object(), configurations=[])


def test_sensitivity_reports_nontext_extra_mapping_keys_actionably() -> None:
    configuration = {
        "extent_classes": (4,),
        "station_bin_width_m": 100.0,
        "min_extent_pixels_per_bin": 1,
        "max_bridge_gap_m": 0.0,
        1: "unexpected",
    }

    with pytest.raises(BankInferenceError, match="unexpected 1"):
        run_candidate_interval_sensitivity(
            _sample([], []), configurations=[configuration]
        )


def test_result_is_immutable_reference_free_and_audit_exports_are_detached() -> None:
    result = _infer(
        _sample([10.0, 110.0, 210.0], [4, 1, 4]),
        max_bridge_gap_m=100.0,
    )

    with pytest.raises(FrozenInstanceError):
        result.method_id = "changed"
    with pytest.raises(TypeError):
        result.classification_counts[4] = 999
    with pytest.raises(TypeError):
        result.bins[0].source_counts[0] = 999
    assert not hasattr(result, "pixels")
    assert not hasattr(result, "sample")

    summary = result.audit_summary()
    summary["bins"][0]["classification_counts"][4] = 999
    assert dict(result.bins[0].classification_counts) == {4: 1}

    table = result.bins_dataframe()
    table.at[0, "classification_counts"][4] = 999
    assert dict(result.bins[0].classification_counts) == {4: 1}


def test_audit_summary_and_tables_have_reconstructable_stable_schemas() -> None:
    result = _infer(_sample([], []))
    summary = result.audit_summary()

    assert summary["extent_classes"] == (4,)
    assert summary["station_bin_width_m"] == 100.0
    assert summary["min_extent_pixels_per_bin"] == 1
    assert summary["max_bridge_gap_m"] == 0.0
    assert summary["boundary_resolution_m"] == 100.0
    assert "Every positive final remainder is retained" in summary["bin_edge_semantics"]
    assert "8 binary floating-point ULPs" in summary["bridge_comparison_semantics"]
    assert summary["method_id"] == "fixed_station_bin_candidate_wet_intervals"
    assert summary["method_version"] == "1.0"
    assert summary["method_status"] == "EXPERIMENTAL / CANDIDATE"
    assert summary["transect_wkt"] == result.transect_wkt
    assert summary["transect_length_m"] == result.transect_length_m
    assert summary["corridor_half_width_m"] == 25.0
    assert summary["profile_name"] == "synthetic_extent_profile"
    assert summary["profile_label"] == "Synthetic extent profile"
    assert summary["profile_status"] == "experimental / test only"
    assert summary["selected_pixel_count"] == 0
    assert summary["classification_counts"] == {}
    assert summary["source_counts"] == {}
    assert summary["source_tile_counts"] == {}
    assert summary["source_summaries"] == []
    assert summary["bin_count"] == 11
    assert summary["candidate_wet_bin_count"] == 0
    assert summary["sampled_noneligible_bin_count"] == 0
    assert summary["unsampled_bin_count"] == 11
    assert summary["candidate_interval_count"] == 0
    assert summary["bridge_count"] == 0
    assert "not validated physical river banks" in summary["method_warning"]
    assert "not treated as confirmed dry land" in summary["method_warning"]
    assert result.bins_dataframe().columns.tolist() == [
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
    ]
    assert result.intervals_dataframe().empty
    assert "candidate_interval_id" in result.intervals_dataframe().columns
    assert result.bridges_dataframe().empty
    assert "bridge_id" in result.bridges_dataframe().columns


def test_chained_bridges_remain_individual_audit_records() -> None:
    result = _infer(
        _sample([10.0, 210.0, 410.0], [4, 4, 4]),
        max_bridge_gap_m=100.0,
    )

    assert result.candidate_interval_count == 1
    assert result.bridge_count == 2
    assert [item.bridge_id for item in result.bridge_records] == [1, 2]
    assert [item.first_gap_bin_id for item in result.bridge_records] == [2, 4]
    assert result.candidate_intervals[0].bridge_ids == (1, 2)
    assert result.candidate_intervals[0].candidate_wet_bin_count == 3
    assert result.candidate_intervals[0].bridged_bin_count == 2
    assert result.observed_candidate_wet_support_m == 300.0
    assert result.total_bridged_gap_m == 200.0
    assert result.inferred_candidate_interval_span_m == 500.0


def test_invalid_sample_arrays_are_rejected_actionably() -> None:
    sample = _sample([10.0], [4])

    missing_classification = replace(
        sample, pixels=sample.pixels.drop_vars("classification")
    )
    with pytest.raises(BankInferenceError, match="classification"):
        _infer(missing_classification)

    nonfinite_station = _with_stations(sample, [np.nan])
    with pytest.raises(BankInferenceError, match="non-finite"):
        _infer(nonfinite_station)

    outside_station = _with_stations(sample, [sample.transect_length_m + 1.0])
    with pytest.raises(BankInferenceError, match="outside"):
        _infer(outside_station)

    floating_classification = sample.pixels.copy(deep=True)
    floating_classification["classification"] = xr.DataArray([4.0], dims="points")
    with pytest.raises(BankInferenceError, match="integer dtype"):
        _infer(replace(sample, pixels=floating_classification))
