from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS, Transformer
from shapely.geometry import LineString, box

from swot_pixc_lab import (
    DryGap,
    ExplicitIntervalWidthResult,
    WetInterval,
    WidthError,
    WidthSourceSummary,
    measure_explicit_wet_intervals,
    plot_wet_interval_summary,
    sample_transect,
)
from swot_pixc_lab.mosaic import PixcObservation
from swot_pixc_lab.qc import apply_qc
from swot_pixc_lab.reader import PixcSourceMetadata, PixcVariableMetadata
from swot_pixc_lab.subset import ExactAoi

_TEST_CRS = CRS.from_proj4(
    "+proj=aeqd +lat_0=26.7 +lon_0=87 +ellps=WGS84 +units=m +type=crs"
)
_TO_GEOGRAPHIC = Transformer.from_crs(_TEST_CRS, "EPSG:4326", always_xy=True)


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
        loaded_variables=("longitude", "latitude", "classification"),
        points_before=point_count,
        points_after=point_count,
        invalid_coordinate_count=0,
        normalized_longitude_count=0,
    )


def _observation(*, second_source_outside: bool = False) -> PixcObservation:
    x = [-400.0, -200.0, 0.0, 200.0, 400.0]
    y = [0.0, 5.0, 10.0, 15.0, 20.0]
    if second_source_outside:
        y[2:] = [200.0, 210.0, 220.0]
    longitude, latitude = _geographic(x, y)
    dataset = xr.Dataset(
        {
            "longitude": xr.DataArray(longitude, dims="points"),
            "latitude": xr.DataArray(latitude, dims="points"),
            "classification": xr.DataArray(
                np.asarray([3, 4, 5, 6, 7], dtype=np.uint8),
                dims="points",
                attrs={"_FillValue": np.uint8(255)},
            ),
            "source_index": xr.DataArray(
                np.asarray([0, 0, 1, 1, 1], dtype=np.int32), dims="points"
            ),
            "source_point_index": xr.DataArray(
                np.asarray([0, 1, 0, 1, 2], dtype=np.int64), dims="points"
            ),
        },
        attrs={"test_marker": "unchanged"},
    )
    return PixcObservation(
        pixels=dataset,
        sources=(_source(0, "107L", 2), _source(1, "108L", 3)),
        aoi=ExactAoi(
            geometry=box(86.0, 26.0, 88.0, 28.0),
            kind="bounding_box",
            bounding_box=(86.0, 26.0, 88.0, 28.0),
        ),
    )


def _sample(*, qc: bool = False, second_source_outside: bool = False):
    observation = _observation(second_source_outside=second_source_outside)
    data = (
        apply_qc(observation, profile="channel_extent_candidate") if qc else observation
    )
    return sample_transect(data, _transect(), corridor_half_width_m=50.0)


def test_canonical_three_interval_contract_and_exact_invariant() -> None:
    result = measure_explicit_wet_intervals(
        _sample(),
        intervals=[(0.0, 120.0), (180.0, 260.0), (310.0, 350.0)],
    )

    assert result.interval_count == 3
    assert result.individual_interval_widths_m == (120.0, 80.0, 40.0)
    assert [gap.width_m for gap in result.dry_gaps] == [60.0, 50.0]
    assert result.total_wetted_width_m == 240.0
    assert result.outer_wetted_span_m == 350.0
    assert result.total_internal_dry_gap_m == 110.0
    assert (
        result.total_wetted_width_m + result.total_internal_dry_gap_m
        == result.outer_wetted_span_m
    )
    assert [item.interval_id for item in result.intervals] == [1, 2, 3]
    assert [item.gap_id for item in result.dry_gaps] == [1, 2]
    assert [item.left_interval_id for item in result.dry_gaps] == [1, 2]
    assert [item.right_interval_id for item in result.dry_gaps] == [2, 3]


def test_one_interval_away_from_ends_has_no_internal_gap() -> None:
    result = measure_explicit_wet_intervals(_sample(), [(100.0, 220.0)])

    assert result.total_wetted_width_m == 120.0
    assert result.outer_wetted_span_m == 120.0
    assert result.dry_gaps == ()
    assert result.total_internal_dry_gap_m == 0.0


def test_intervals_away_from_ends_exclude_external_margins_from_dry_gaps() -> None:
    result = measure_explicit_wet_intervals(_sample(), [(100.0, 220.0), (280.0, 360.0)])

    assert result.individual_interval_widths_m == (120.0, 80.0)
    assert [gap.width_m for gap in result.dry_gaps] == [60.0]
    assert result.total_wetted_width_m == 200.0
    assert result.outer_wetted_span_m == 260.0
    assert result.total_internal_dry_gap_m == 60.0


def test_empty_interval_set_is_explicit_and_tables_are_well_formed() -> None:
    result = measure_explicit_wet_intervals(_sample(), [])

    assert result.interval_count == 0
    assert result.intervals == ()
    assert result.dry_gaps == ()
    assert result.total_wetted_width_m == 0.0
    assert result.total_internal_dry_gap_m == 0.0
    assert result.outer_wetted_span_m is None
    assert result.intervals_dataframe().columns.tolist() == [
        "interval_id",
        "start_station_m",
        "end_station_m",
        "width_m",
    ]
    assert result.intervals_dataframe().empty
    assert result.gaps_dataframe().columns.tolist() == [
        "gap_id",
        "left_interval_id",
        "right_interval_id",
        "start_station_m",
        "end_station_m",
        "width_m",
    ]
    assert result.gaps_dataframe().empty


@pytest.mark.parametrize(
    ("intervals", "message"),
    [
        ([(200.0, 250.0), (100.0, 150.0)], "out of station order"),
        ([(0.0, 100.0), (50.0, 120.0)], "overlaps"),
        ([(0.0, 100.0), (100.0, 200.0)], "touches"),
        ([(100.0, 50.0)], "reversed"),
        ([(100.0, 100.0)], "zero width"),
        ([(0.0, np.nan)], "must be finite"),
        ([(0.0, np.inf)], "must be finite"),
        ([(0.0, -np.inf)], "must be finite"),
        ([(np.nan, 100.0)], "must be finite"),
        ([(False, 100.0)], "finite number"),
        ([(np.bool_(True), 100.0)], "finite number"),
        ([(-1.0, 100.0)], "non-negative"),
        ([(0.0, 50.0), (100.0, 150.0), (0.0, 50.0)], "duplicates"),
    ],
)
def test_invalid_intervals_are_rejected_without_reordering(intervals, message) -> None:
    with pytest.raises(WidthError, match=message):
        measure_explicit_wet_intervals(_sample(), intervals)


def test_endpoint_beyond_actual_projected_transect_length_is_rejected() -> None:
    sample = _sample()

    with pytest.raises(WidthError, match="beyond the projected transect length"):
        measure_explicit_wet_intervals(sample, [(0.0, sample.transect_length_m + 1.0)])


def test_full_transect_interval_is_allowed_independently_of_pixel_spread() -> None:
    sample = _sample()

    result = measure_explicit_wet_intervals(sample, [(0.0, sample.transect_length_m)])

    assert float(sample.station_m.min()) > 0.0
    assert float(sample.station_m.max()) < sample.transect_length_m
    assert result.total_wetted_width_m == sample.transect_length_m


@pytest.mark.parametrize(
    "intervals",
    [
        None,
        "0, 100",
        {(0.0, 1.0), (2.0, 3.0)},
        [{0.0, 1.0}],
        [(0.0,)],
        [(0.0, 1.0, 2.0)],
        [3.0],
    ],
)
def test_malformed_interval_container_is_actionable(intervals) -> None:
    with pytest.raises(WidthError, match="intervals|exactly two"):
        measure_explicit_wet_intervals(_sample(), intervals)


def test_any_strictly_positive_gap_is_preserved_without_hidden_threshold() -> None:
    result = measure_explicit_wet_intervals(
        _sample(), [(0.0, 100.0), (100.0 + 1.0e-9, 200.0)]
    )

    assert result.dry_gaps[0].width_m > 0.0
    assert result.dry_gaps[0].width_m == pytest.approx(1.0e-9, rel=1.0e-5)


def test_floating_values_are_not_rounded_to_integer_metres() -> None:
    result = measure_explicit_wet_intervals(
        _sample(), [(0.125, 120.875), (180.25, 260.625)]
    )

    assert result.individual_interval_widths_m == (120.75, 80.375)
    assert result.total_wetted_width_m == pytest.approx(201.125)
    assert result.outer_wetted_span_m == pytest.approx(260.5)
    assert result.total_internal_dry_gap_m == pytest.approx(59.375)


def test_result_is_reference_free_immutable_and_does_not_mutate_sample() -> None:
    sample = _sample()
    before = sample.pixels.copy(deep=True)
    before_attributes = copy.deepcopy(sample.pixels.attrs)

    result = measure_explicit_wet_intervals(sample, [(10.0, 20.5)])

    xr.testing.assert_identical(sample.pixels, before)
    assert sample.pixels.attrs == before_attributes
    assert not hasattr(result, "pixels")
    assert not hasattr(result, "sample")
    with pytest.raises(FrozenInstanceError):
        result.method_id = "changed"
    with pytest.raises(TypeError):
        result.classification_counts[3] = 999


def test_qc_and_multisource_provenance_are_snapshotted() -> None:
    sample = _sample(qc=True)

    result = measure_explicit_wet_intervals(sample, [(0.0, 350.0)])

    assert result.profile_name == "channel_extent_candidate"
    assert result.profile_label == "Channel extent candidate"
    assert result.profile_status == "experimental / unvalidated"
    assert result.selected_pixel_count == 5
    assert dict(result.classification_counts) == {3: 1, 4: 1, 5: 1, 6: 1, 7: 1}
    assert dict(result.source_counts) == {0: 2, 1: 3}
    assert dict(result.source_tile_counts) == {"107L": 2, "108L": 3}
    assert [source.source_index for source in result.source_summaries] == [0, 1]
    assert [source.selected_pixel_count for source in result.source_summaries] == [
        2,
        3,
    ]
    assert [source.granule_id for source in result.source_summaries] == [
        "synthetic-107L",
        "synthetic-108L",
    ]
    assert [source.tile for source in result.source_summaries] == ["107L", "108L"]
    assert all(
        source.netcdf_product_version == "D" for source in result.source_summaries
    )


def test_zero_count_source_remains_in_provenance_summary() -> None:
    sample = _sample(second_source_outside=True)

    result = measure_explicit_wet_intervals(sample, [])

    assert dict(result.source_counts) == {0: 2, 1: 0}
    assert [source.selected_pixel_count for source in result.source_summaries] == [
        2,
        0,
    ]


def test_metric_geometry_and_full_audit_are_portable_and_reconstructable() -> None:
    sample = _sample()
    result = measure_explicit_wet_intervals(sample, [(0.0, 120.0), (180.0, 260.0)])

    summary = result.audit_summary()
    assert result.transect_crs == "EPSG:4326"
    assert result.transect_wkt == sample.transect.wkt
    assert result.metric_crs_wkt == sample.local_crs.to_wkt()
    assert result.projection_center == sample.projection_center
    assert result.transect_length_m == pytest.approx(1000.0, abs=0.02)
    assert summary["intervals"] == result.intervals_dataframe().to_dict("records")
    assert summary["dry_gaps"] == result.gaps_dataframe().to_dict("records")
    assert summary["method_status"] == "EXPERIMENTAL / MANUAL BENCHMARK CONTRACT"
    assert "supplied by the analyst" in summary["method_warning"]
    assert summary["total_wetted_width_m"] == 200.0
    assert summary["total_internal_dry_gap_m"] == 60.0
    assert summary["outer_wetted_span_m"] == 260.0
    assert "path" not in summary["source_summaries"][0]


def test_dataframe_accessors_return_detached_fresh_tables() -> None:
    result = measure_explicit_wet_intervals(_sample(), [(0.0, 10.25)])

    first = result.intervals_dataframe()
    first.loc[0, "width_m"] = -999.0
    second = result.intervals_dataframe()

    assert second.loc[0, "width_m"] == 10.25
    assert result.intervals[0].width_m == 10.25


def test_non_transect_sample_is_rejected() -> None:
    with pytest.raises(TypeError, match="Phase 4 TransectSample"):
        measure_explicit_wet_intervals(object(), [])


def test_phase5a_public_api_exports_stable_types_and_plot() -> None:
    sample = _sample()
    result = measure_explicit_wet_intervals(sample, [(0.0, 10.0)])

    assert isinstance(result, ExplicitIntervalWidthResult)
    assert isinstance(result.intervals[0], WetInterval)
    assert isinstance(result.source_summaries[0], WidthSourceSummary)
    assert DryGap.__name__ == "DryGap"
    assert callable(plot_wet_interval_summary)
