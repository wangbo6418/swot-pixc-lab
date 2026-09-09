from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import MappingProxyType, ModuleType

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS, Geod, Transformer
from shapely.geometry import LineString, box

from swot_pixc_lab.mosaic import PixcObservation
from swot_pixc_lab.qc import apply_qc
from swot_pixc_lab.reader import PixcSourceMetadata, PixcVariableMetadata
from swot_pixc_lab.subset import ExactAoi
from swot_pixc_lab.transect import TransectError, sample_transect

_GEOD = Geod(ellps="WGS84")
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


def _dataset(
    x: list[float],
    y: list[float],
    *,
    classification: list[int] | None = None,
    source_index: list[int] | None = None,
    source_point_index: list[int] | None = None,
) -> xr.Dataset:
    longitude, latitude = _geographic(x, y)
    count = len(x)
    classes = classification or [4] * count
    sources = source_index or [0] * count
    source_points = source_point_index or list(range(count))
    return xr.Dataset(
        data_vars={
            "longitude": xr.DataArray(
                longitude.astype(np.float64),
                dims=("points",),
                attrs={"units": "degrees_east"},
            ),
            "latitude": xr.DataArray(
                latitude.astype(np.float64),
                dims=("points",),
                attrs={"units": "degrees_north"},
            ),
            "classification": xr.DataArray(
                np.asarray(classes, dtype=np.uint8),
                dims=("points",),
                attrs={"_FillValue": np.uint8(255)},
            ),
            "height": xr.DataArray(
                np.linspace(90.0, 91.0, count, dtype=np.float32),
                dims=("points",),
                attrs={"units": "m", "reference": "ellipsoid"},
            ),
            "water_frac": xr.DataArray(
                np.linspace(0.2, 1.0, count, dtype=np.float32),
                dims=("points",),
                attrs={"units": "1"},
            ),
            "classification_qual": xr.DataArray(
                np.arange(count, dtype=np.uint32), dims=("points",)
            ),
            "source_index": xr.DataArray(
                np.asarray(sources, dtype=np.int32), dims=("points",)
            ),
            "source_point_index": xr.DataArray(
                np.asarray(source_points, dtype=np.int64), dims=("points",)
            ),
        },
        attrs={"test_marker": "unchanged"},
    )


def _source(source_index: int, tile: str) -> PixcSourceMetadata:
    classification = PixcVariableMetadata(
        name="classification",
        dimensions=("points",),
        shape=(2,),
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
        netcdf_pge_version=None,
        cmr_record=None,
        data_model="NETCDF4",
        groups=("pixel_cloud",),
        dimensions=MappingProxyType({"points": 2}),
        root_attributes=MappingProxyType({}),
        pixel_cloud_attributes=MappingProxyType({}),
        variables=MappingProxyType({"classification": classification}),
        loaded_variables=(
            "longitude",
            "latitude",
            "classification",
            "height",
            "water_frac",
            "classification_qual",
        ),
        points_before=2,
        points_after=2,
        invalid_coordinate_count=0,
        normalized_longitude_count=0,
    )


def _observation(dataset: xr.Dataset, *, two_sources: bool = False) -> PixcObservation:
    sources = (_source(0, "107L"), _source(1, "108L")) if two_sources else ()
    return PixcObservation(
        pixels=dataset,
        sources=sources,
        aoi=ExactAoi(
            geometry=box(86.0, 26.0, 88.0, 28.0),
            kind="bounding_box",
            bounding_box=(86.0, 26.0, 88.0, 28.0),
        ),
    )


@pytest.mark.parametrize(
    "line_input",
    [
        pytest.param(lambda: _transect(), id="shapely"),
        pytest.param(
            lambda: {
                "type": "LineString",
                "coordinates": [list(value) for value in _transect().coords],
            },
            id="geojson",
        ),
        pytest.param(lambda: tuple(_transect().coords), id="endpoints"),
    ],
)
def test_supported_user_supplied_transect_forms(line_input) -> None:
    sample = sample_transect(
        _dataset([0.0], [0.0]),
        line_input(),
        corridor_half_width_m=25.0,
    )

    assert sample.selected_pixel_count == 1
    assert isinstance(sample.transect, LineString)


def test_known_metric_station_distance_corridor_and_round_end_caps() -> None:
    data = _dataset(
        [-525.0, -500.0, -250.0, 0.0, 500.0, 525.0, 0.0],
        [0.0, 0.0, 30.0, 50.0, 0.0, 0.0, 50.1],
        source_point_index=[10, 11, 12, 13, 14, 15, 16],
    )

    sample = sample_transect(
        data,
        _transect(),
        corridor_half_width_m=50.0,
        chunk_size=2,
    )

    np.testing.assert_array_equal(
        sample.pixels["source_point_index"].values, [10, 11, 12, 13, 14, 15]
    )
    np.testing.assert_allclose(
        sample.station_m.values,
        [0.0, 0.0, 250.0, 500.0, 1000.0, 1000.0],
        atol=0.02,
    )
    np.testing.assert_allclose(
        sample.distance_to_transect_m.values,
        [25.0, 0.0, 30.0, 50.0, 0.0, 25.0],
        atol=0.02,
    )
    assert sample.station_m.dtype == np.dtype("float64")
    assert sample.distance_to_transect_m.dtype == np.dtype("float64")
    assert sample.corridor_half_width_m == 50.0
    assert sample.outside_corridor_count == 1
    assert not sample.corridor.is_empty


def test_input_order_and_provenance_are_preserved_not_station_sorted() -> None:
    data = _dataset(
        [400.0, -400.0, 0.0],
        [5.0, 10.0, 15.0],
        source_index=[1, 0, 1],
        source_point_index=[42, 7, 99],
    )

    sample = sample_transect(data, _transect(), corridor_half_width_m=25.0)

    np.testing.assert_allclose(
        sample.station_m.values, [900.0, 100.0, 500.0], atol=0.02
    )
    np.testing.assert_array_equal(sample.pixels["source_index"], [1, 0, 1])
    np.testing.assert_array_equal(sample.pixels["source_point_index"], [42, 7, 99])


def test_every_point_variable_is_detached_and_input_is_not_mutated() -> None:
    data = _dataset([-100.0, 0.0, 100.0], [0.0, 10.0, 75.0])
    original = data.copy(deep=True)
    original_attributes = copy.deepcopy(data.attrs)

    sample = sample_transect(data, _transect(), corridor_half_width_m=20.0)

    assert set(data.data_vars) <= set(sample.pixels.data_vars)
    assert sample.selected_pixel_count == 2
    np.testing.assert_array_equal(
        sample.pixels["classification_qual"].values,
        data["classification_qual"].values[:2],
    )
    sample.pixels["height"].values[0] = np.float32(-999.0)
    sample.pixels.attrs["test_marker"] = "changed-result-only"
    xr.testing.assert_identical(data, original)
    assert data.attrs == original_attributes


def test_pixc_observation_and_qc_result_inputs_retain_context() -> None:
    dataset = _dataset([-100.0, 100.0], [0.0, 0.0])
    observation = _observation(dataset)
    qc_result = apply_qc(observation, profile="raw")

    raw_sample = sample_transect(observation, _transect(), corridor_half_width_m=25.0)
    qc_sample = sample_transect(qc_result, _transect(), corridor_half_width_m=25.0)

    assert raw_sample.profile_name == "raw"
    assert raw_sample.profile_status == "baseline / no scientific filtering"
    assert qc_sample.profile_name == qc_result.profile.name
    assert qc_sample.profile_label == qc_result.profile.label
    assert qc_sample.profile_status == qc_result.profile.status
    np.testing.assert_array_equal(
        raw_sample.pixels["source_point_index"],
        qc_sample.pixels["source_point_index"],
    )
    qc_sample.pixels["classification"].values[0] = np.uint8(1)
    np.testing.assert_array_equal(qc_result.filtered["classification"], [4, 4])
    np.testing.assert_array_equal(observation.raw["classification"], [4, 4])


def test_classification_and_source_tile_counts_include_zero_sources() -> None:
    data = _dataset(
        [-100.0, 0.0, 100.0, 200.0],
        [0.0, 0.0, 0.0, 80.0],
        classification=[3, 255, 5, 7],
        source_index=[0, 0, 1, 1],
        source_point_index=[8, 9, 10, 11],
    )
    observation = _observation(data, two_sources=True)

    sample = sample_transect(observation, _transect(), corridor_half_width_m=25.0)

    assert sample.classification_counts == {3: 1, 5: 1}
    assert sample.source_counts == {0: 2, 1: 1}
    assert sample.source_tile_counts == {"107L": 2, "108L": 1}
    np.testing.assert_array_equal(sample.pixels["source_point_index"], [8, 9, 10])

    empty_for_second = sample_transect(
        _observation(
            _dataset(
                [-100.0, 100.0],
                [0.0, 100.0],
                source_index=[0, 1],
            ),
            two_sources=True,
        ),
        _transect(),
        corridor_half_width_m=25.0,
    )
    assert empty_for_second.source_counts == {0: 1, 1: 0}
    assert empty_for_second.source_tile_counts == {"107L": 1, "108L": 0}


def test_empty_result_is_well_formed() -> None:
    sample = sample_transect(
        _dataset([0.0, 100.0], [500.0, 600.0]),
        _transect(),
        corridor_half_width_m=25.0,
    )

    assert sample.selected_pixel_count == 0
    assert sample.classification_counts == {}
    assert sample.source_counts == {0: 0}
    assert sample.source_tile_counts == {"source_0": 0}
    assert sample.station_m.dtype == np.dtype("float64")
    assert sample.distance_to_transect_m.dtype == np.dtype("float64")
    assert sample.station_m.size == 0
    assert sample.outside_corridor_count == 2


def test_geodataframe_export_carries_corridor_qc_and_source_audit_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _dataset(
        [-100.0, 100.0],
        [0.0, 0.0],
        source_index=[0, 1],
        source_point_index=[8, 10],
    )
    sample = sample_transect(
        _observation(data, two_sources=True),
        _transect(),
        corridor_half_width_m=50.0,
    )
    fake_geopandas = ModuleType("geopandas")
    fake_geopandas.points_from_xy = lambda x, y, crs: list(zip(x, y, strict=True))
    fake_geopandas.GeoDataFrame = lambda frame, geometry, crs: frame.assign(
        geometry=geometry
    )
    monkeypatch.setitem(sys.modules, "geopandas", fake_geopandas)

    exported = sample.to_geodataframe()

    assert exported["corridor_half_width_m"].tolist() == [50.0, 50.0]
    assert exported["source_tile"].tolist() == ["107L", "108L"]
    assert exported["source_point_index"].tolist() == [8, 10]
    assert exported.attrs["swot_pixc_lab_corridor_half_width_m"] == 50.0
    assert exported.attrs["swot_pixc_lab_transect_wkt"] == sample.transect.wkt
    assert exported.attrs["swot_pixc_lab_qc_profile"] == "raw"
    assert [item["tile"] for item in exported.attrs["swot_pixc_lab_sources"]] == [
        "107L",
        "108L",
    ]


def test_geographic_to_local_aeqd_uses_geodesic_midpoint() -> None:
    line = _transect()
    sample = sample_transect(
        _dataset([0.0], [40.0]),
        line,
        corridor_half_width_m=50.0,
    )

    first, last = tuple(line.coords)
    azimuth, _, length = _GEOD.inv(*first, *last)
    midpoint = _GEOD.fwd(*first, azimuth, length / 2.0)
    assert sample.local_crs.coordinate_operation is not None
    assert sample.local_crs.coordinate_operation.method_name == "Azimuthal Equidistant"
    np.testing.assert_allclose(
        sample.projection_center,
        midpoint[:2],
        atol=1e-10,
    )
    np.testing.assert_allclose(sample.distance_to_transect_m, [40.0], atol=0.02)
    np.testing.assert_allclose(sample.station_m, [500.0], atol=0.02)


def test_invalid_pixel_coordinates_are_counted_not_selected() -> None:
    data = _dataset([0.0, 10.0, 20.0], [0.0, 0.0, 0.0])
    data["longitude"].values[1] = np.nan
    data["latitude"].values[2] = 91.0

    sample = sample_transect(data, _transect(), corridor_half_width_m=25.0)

    assert sample.input_pixel_count == 3
    assert sample.invalid_coordinate_count == 2
    assert sample.selected_pixel_count == 1
    assert sample.outside_corridor_count == 0


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ({"type": "Polygon", "coordinates": []}, "LineString"),
        (LineString([(87.0, 26.7), (87.0, 26.7)]), "non-repeated"),
        (LineString([(181.0, 26.7), (179.0, 26.7)]), "longitudes"),
        (LineString([(87.0, 91.0), (87.1, 89.0)]), "latitudes"),
        (LineString([(179.0, 26.7), (-179.0, 26.7)]), "antimeridian"),
        (
            LineString([(86.9, 26.6), (87.1, 26.8), (86.9, 26.8), (87.1, 26.6)]),
            "self-intersect",
        ),
        (LineString([(87.0, 26.7, 0.0), (87.1, 26.7, 0.0)]), "2-D"),
        (((87.0, 26.7),), "exactly two"),
    ],
)
def test_invalid_transect_is_rejected(line, message: str) -> None:
    with pytest.raises(TransectError, match=message):
        sample_transect(
            _dataset([0.0], [0.0]),
            line,
            corridor_half_width_m=25.0,
        )


@pytest.mark.parametrize("half_width", [-1.0, 0.0, np.inf, np.nan, True, "wide"])
def test_invalid_corridor_half_width_is_rejected(half_width) -> None:
    with pytest.raises(TransectError, match="positive finite"):
        sample_transect(
            _dataset([0.0], [0.0]),
            _transect(),
            corridor_half_width_m=half_width,
        )


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_invalid_chunk_size_is_rejected(chunk_size) -> None:
    with pytest.raises(TransectError, match="positive integer"):
        sample_transect(
            _dataset([0.0], [0.0]),
            _transect(),
            corridor_half_width_m=25.0,
            chunk_size=chunk_size,
        )


def test_missing_or_malformed_provenance_is_rejected() -> None:
    data = _dataset([0.0], [0.0]).drop_vars("source_point_index")
    with pytest.raises(TransectError, match="source_point_index"):
        sample_transect(data, _transect(), corridor_half_width_m=25.0)

    malformed = _dataset([0.0], [0.0])
    malformed["source_point_index"] = xr.DataArray(1)
    with pytest.raises(TransectError, match="points"):
        sample_transect(malformed, _transect(), corridor_half_width_m=25.0)


def test_result_exposes_no_branch_bank_or_width_calculation() -> None:
    sample = sample_transect(
        _dataset([0.0], [0.0]),
        _transect(),
        corridor_half_width_m=25.0,
    )

    prohibited = {"channel_width", "branch_width", "bank_position", "branch_id"}
    assert prohibited.isdisjoint(dir(sample))
    assert prohibited.isdisjoint(sample.pixels.variables)
    assert "width" not in sample.pixels
