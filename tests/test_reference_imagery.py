from __future__ import annotations

import importlib.util
import inspect
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, box, mapping, shape

from swot_pixc_lab.exceptions import ReferenceImageryError
from swot_pixc_lab.reference_imagery import (
    NDWI_INTERPRETATION,
    EarthSearchSentinelProvider,
    ReviewRegion,
    SentinelScene,
    StacAsset,
    apply_asset_scale,
    calculate_ndwi,
    calculate_scl_diagnostics,
    group_same_acquisition,
    make_review_region,
    mosaic_same_acquisition_chips,
    parse_stac_item,
    read_swot_temporal_context,
    read_windowed_cog_asset,
    review_region_pixel_mask,
    scale_rgb,
    scenes_intersecting_region,
    search_progressive_windows,
    shortlist_acquisitions,
    stable_source_url,
    temporal_provenance,
)

RASTERIO_AVAILABLE = importlib.util.find_spec("rasterio") is not None


def _polygon(
    west: float = 86.8,
    south: float = 26.4,
    east: float = 87.3,
    north: float = 27.0,
) -> dict[str, object]:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south],
            ]
        ],
    }


def _stac_item(
    *,
    item_id: str = "S2A_45RVK_20240112_0_L2A",
    acquired: str = "2024-01-12T05:01:13.589Z",
    datatake_id: str = "GS2A_20240112T045141_044689_N05.10",
) -> dict[str, object]:
    assets: dict[str, object] = {}
    for role, resolution in {
        "red": 10,
        "green": 10,
        "blue": 10,
        "nir": 10,
        "scl": 20,
    }.items():
        raster_band: dict[str, object] = {
            "nodata": 0,
            "data_type": "uint8" if role == "scl" else "uint16",
            "spatial_resolution": resolution,
        }
        if role != "scl":
            raster_band.update(scale=0.0001, offset=-0.1)
        assets[role] = {
            "href": f"https://example.test/{item_id}/{role}.tif?sig=secret#fragment",
            "type": "image/tiff; application=geotiff; profile=cloud-optimized",
            "raster:bands": [raster_band],
        }
    return {
        "type": "Feature",
        "id": item_id,
        "bbox": [86.8, 26.4, 87.3, 27.0],
        "geometry": _polygon(),
        "properties": {
            "datetime": acquired,
            "platform": "sentinel-2a",
            "proj:epsg": 32645,
            "grid:code": "MGRS-45RVK",
            "eo:cloud_cover": 0.078948,
            "s2:cloud_shadow_percentage": 0.000604,
            "s2:product_type": "S2MSI2A",
            "s2:processing_baseline": "05.10",
            "s2:datatake_id": datatake_id,
        },
        "assets": assets,
    }


def _scene(
    *,
    item_id: str = "scene-a",
    acquired: str = "2024-01-12T05:01:13Z",
    search_window_days: int = 7,
    geometry: dict[str, object] | None = None,
    datatake_id: str | None = "datatake-a",
) -> SentinelScene:
    parsed = parse_stac_item(
        _stac_item(item_id=item_id, acquired=acquired),
        provider="synthetic",
        stac_endpoint="https://example.test/stac",
        collection="sentinel-2-l2a",
        query_time="2024-01-20T00:00:00Z",
        search_window_days=search_window_days,
    )
    return replace(
        parsed,
        geometry=geometry or parsed.geometry,
        datatake_id=datatake_id,
    )


def test_parse_stac_item_preserves_metadata_and_redacts_access_urls() -> None:
    scene = parse_stac_item(
        _stac_item(),
        provider="Element84 Earth Search",
        stac_endpoint="https://earth-search.aws.element84.com/v1/",
        collection="sentinel-2-l2a",
        query_time="2024-02-01T12:00:00+00:00",
        search_window_days=7,
    )

    assert scene.item_id == "S2A_45RVK_20240112_0_L2A"
    assert scene.acquisition_datetime == datetime(
        2024, 1, 12, 5, 1, 13, 589000, tzinfo=UTC
    )
    assert scene.processing_level == "S2MSI2A"
    assert scene.processing_baseline == "05.10"
    assert scene.mgrs_tile == "45RVK"
    assert scene.epsg == 32645
    assert scene.scene_cloud_cover_percent == pytest.approx(0.078948)
    assert scene.assets["green"].band_name == "B03"
    assert scene.assets["green"].scale == pytest.approx(0.0001)
    assert scene.assets["green"].offset == pytest.approx(-0.1)
    assert scene.assets["scl"].spatial_resolution_m == 20

    serialized = scene.to_dict()
    encoded = json.dumps(serialized)
    assert "sig=secret" not in encoded
    assert "#fragment" not in encoded
    assert "href" not in serialized["assets"]["green"]
    assert serialized["assets"]["green"]["source_identifier"].endswith("/green.tif")


def test_earth_search_provider_parses_sorts_and_deduplicates_mock_response() -> None:
    requests: list[tuple[str, dict[str, object]]] = []
    later = _stac_item(item_id="later", acquired="2024-01-17T05:00:00Z")
    earlier = _stac_item(item_id="earlier", acquired="2024-01-07T05:00:00Z")

    def transport(url: str, payload: object) -> dict[str, object]:
        assert isinstance(payload, dict)
        requests.append((url, payload))
        return {"type": "FeatureCollection", "features": [later, earlier, later]}

    provider = EarthSearchSentinelProvider(transport=transport)
    scenes = provider.search(
        bbox_wgs84=(86.87, 26.49, 87.2, 26.9),
        start=datetime(2024, 1, 7, tzinfo=UTC),
        end=datetime(2024, 1, 21, 23, 59, 59, tzinfo=UTC),
        search_window_days=7,
    )

    assert [scene.item_id for scene in scenes] == ["earlier", "later"]
    assert requests[0][0].endswith("/v1/search")
    assert requests[0][1] == {
        "collections": ["sentinel-2-l2a"],
        "bbox": [86.87, 26.49, 87.2, 26.9],
        "datetime": "2024-01-07T00:00:00Z/2024-01-21T23:59:59Z",
        "limit": 100,
    }


def test_parse_stac_item_rejects_missing_required_asset() -> None:
    item = _stac_item()
    assert isinstance(item["assets"], dict)
    item["assets"].pop("scl")

    with pytest.raises(ReferenceImageryError, match="missing required asset 'scl'"):
        parse_stac_item(
            item,
            provider="synthetic",
            stac_endpoint="https://example.test/stac",
            collection="sentinel-2-l2a",
            query_time="2024-01-14T00:00:00Z",
            search_window_days=7,
        )


def test_stable_source_url_strips_all_query_and_fragment_information() -> None:
    signed = "https://blob.example.test/a/b.tif?se=soon&sig=secret#fragment"
    assert stable_source_url(signed) == "https://blob.example.test/a/b.tif"

    asset = StacAsset(
        key="green",
        band_name="B03",
        href=signed,
        source_identifier=stable_source_url(signed),
        media_type="image/tiff",
        scale=1.0,
        offset=0.0,
        nodata=0,
        spatial_resolution_m=10,
    )
    assert "sig=secret" not in repr(asset)
    assert "href" not in asset.to_dict()
    assert asset.to_dict(include_access_href=True)["href"] == signed

    with pytest.raises(ReferenceImageryError, match="embedded user information"):
        stable_source_url("https://user:secret@example.test/a.tif?sig=secret")


@pytest.mark.parametrize(
    ("sentinel", "expected_hours", "flag"),
    [
        ("2024-01-14T00:00:00Z", -12.0, "SAME_DAY"),
        ("2024-01-11T12:00:00Z", -72.0, "WITHIN_3_DAYS"),
        ("2024-01-07T12:00:00Z", -168.0, "WITHIN_7_DAYS"),
        ("2023-12-31T12:00:00Z", -336.0, "WITHIN_14_DAYS"),
        ("2023-12-15T12:00:00Z", -720.0, "WITHIN_30_DAYS"),
        ("2023-12-15T11:59:59Z", -720.0002777778, "BEYOND_30_DAYS"),
    ],
)
def test_temporal_provenance_boundaries(
    sentinel: str, expected_hours: float, flag: str
) -> None:
    result = temporal_provenance(sentinel, "2024-01-14T12:00:00Z")

    assert result.signed_offset_hours == pytest.approx(expected_hours)
    assert result.absolute_offset_hours == pytest.approx(abs(expected_hours))
    assert result.descriptive_flag == flag


def test_swot_temporal_context_reads_granule_coverage_and_records_midpoint(
    tmp_path: Path,
) -> None:
    from netCDF4 import Dataset

    paths = [tmp_path / "tile_a.nc", tmp_path / "tile_b.nc"]
    coverage = [
        ("2024-01-14T08:14:32Z", "2024-01-14T08:14:42Z"),
        ("2024-01-14T08:14:42Z", "2024-01-14T08:14:52Z"),
    ]
    for path, (start, end) in zip(paths, coverage, strict=True):
        with Dataset(path, "w") as dataset:
            dataset.setncattr("time_coverage_start", start)
            dataset.setncattr("time_coverage_end", end)

    result = read_swot_temporal_context(paths)

    assert result.coverage_start == "2024-01-14T08:14:32Z"
    assert result.coverage_end == "2024-01-14T08:14:52Z"
    assert result.reference_datetime == "2024-01-14T08:14:42Z"
    assert result.reference_basis == "midpoint_of_combined_netcdf_time_coverage"
    assert [row["filename"] for row in result.source_time_coverage] == [
        "tile_a.nc",
        "tile_b.nc",
    ]


def test_progressive_search_expands_only_to_approved_windows_and_deduplicates() -> None:
    first = _scene(
        item_id="first",
        acquired="2024-01-12T05:00:00Z",
        search_window_days=14,
    )
    second = _scene(
        item_id="second",
        acquired="2024-01-17T05:00:00Z",
        search_window_days=30,
    )

    class Provider:
        provider_name = "stub"
        endpoint = "https://example.test/stac"
        collection = "sentinel-2-l2a"

        def __init__(self) -> None:
            self.calls: list[int] = []

        def search(self, **kwargs: object) -> tuple[SentinelScene, ...]:
            days = int(kwargs["search_window_days"])
            self.calls.append(days)
            if days == 7:
                return ()
            if days == 14:
                return (first,)
            return (replace(first, search_window_days=30), second)

    provider = Provider()
    result = search_progressive_windows(
        provider,
        bbox_wgs84=(86.87, 26.49, 87.2, 26.9),
        swot_datetime="2024-01-14T08:14:32Z",
        is_sufficient=lambda scenes: len(scenes) >= 2,
    )

    assert provider.calls == [7, 14, 30]
    assert result.windows_queried_days == (7, 14, 30)
    assert [scene.item_id for scene in result.scenes] == ["first", "second"]
    assert result.scenes[0].search_window_days == 14


def test_review_region_is_metric_and_does_not_clip_t001_to_swot_aoi() -> None:
    # The downstream endpoint lies south of the exact SWOT AOI (latitude 26.49).
    t001_edge_case = LineString([(86.91, 26.53), (86.98, 26.47)])
    region = make_review_region(t001_edge_case, context_margin_m=1500)
    region_shape = shape(region.polygon_wgs84)

    assert region.context_margin_m == 1500
    assert "Azimuthal Equidistant" in region.construction_crs_wkt
    assert region.bounds_wgs84[1] < 26.47 < 26.49
    assert region.bounds_wgs84[0] < 86.91
    assert region.bounds_wgs84[2] > 86.98
    assert region_shape.covers(t001_edge_case)


def test_scene_intersection_and_same_acquisition_grouping_are_spatial() -> None:
    region = make_review_region(
        LineString([(86.94, 26.52), (86.97, 26.50)]),
        context_margin_m=1000,
    )
    first_tile = _scene(item_id="tile-vk", datatake_id="same-datatake")
    second_tile = _scene(
        item_id="tile-wk",
        acquired="2024-01-12T05:01:14Z",
        datatake_id="same-datatake",
    )
    distant = _scene(
        item_id="distant",
        geometry=_polygon(80, 20, 81, 21),
        datatake_id="other-datatake",
    )

    intersecting = scenes_intersecting_region(
        [distant, second_tile, first_tile], region
    )
    groups = group_same_acquisition(intersecting)

    assert [scene.item_id for scene in intersecting] == ["tile-vk", "tile-wk"]
    assert list(groups) == ["same-datatake"]
    assert [scene.item_id for scene in groups["same-datatake"]] == [
        "tile-vk",
        "tile-wk",
    ]


def test_scl_diagnostics_report_official_classes_without_creating_truth() -> None:
    scl = np.arange(13, dtype=np.uint8).reshape(1, 13)
    result = calculate_scl_diagnostics(scl)

    assert result.total_pixel_count == 13
    assert result.valid_pixel_count == 11
    assert result.unexpected_value_count == 1
    assert result.class_counts["water"] == 1
    assert result.valid_fraction == pytest.approx(11 / 13)
    assert result.cloud_fraction == pytest.approx(2 / 13)
    assert result.cloud_shadow_fraction == pytest.approx(1 / 13)
    assert result.cirrus_fraction == pytest.approx(1 / 13)
    assert result.snow_ice_fraction == pytest.approx(1 / 13)
    assert result.no_data_fraction == pytest.approx(1 / 13)
    assert result.saturated_defective_fraction == pytest.approx(1 / 13)
    assert result.non_obscured_diagnostic_fraction == pytest.approx(3 / 13)
    assert result.obscured_diagnostic_fraction == pytest.approx(8 / 13)
    assert "not reference truth" in result.interpretation
    assert not hasattr(result, "manual_intervals")


def test_asset_scaling_rgb_and_continuous_ndwi_have_no_threshold_output() -> None:
    raw = np.array([[0, 1000, 2000]], dtype=np.uint16)
    scaled = apply_asset_scale(raw, scale=0.0001, offset=-0.1, nodata=0)
    np.testing.assert_allclose(scaled[0, 1:], [0.0, 0.1], atol=1e-7)
    assert np.isnan(scaled[0, 0])

    red = np.array([[0.0, 1.0], [2.0, np.nan]], dtype=np.float32)
    green = np.array([[2.0, 1.0], [0.0, np.nan]], dtype=np.float32)
    blue = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    rgb = scale_rgb(
        red,
        green,
        blue,
        lower_percentile=0,
        upper_percentile=100,
    )
    assert rgb.dtype == np.uint8
    np.testing.assert_array_equal(
        rgb,
        np.array(
            [
                [[0, 255, 0], [128, 128, 85]],
                [[255, 0, 170], [0, 0, 255]],
            ],
            dtype=np.uint8,
        ),
    )

    ndwi = calculate_ndwi(
        np.array([3.0, 1.0, 0.0, np.nan]),
        np.array([1.0, 3.0, 0.0, 1.0]),
    )
    np.testing.assert_allclose(ndwi[:2], [0.5, -0.5])
    assert np.isnan(ndwi[2:]).all()
    assert ndwi.dtype == np.float32
    assert ndwi.dtype != np.bool_
    assert list(inspect.signature(calculate_ndwi).parameters) == [
        "green_reflectance",
        "nir_reflectance",
    ]
    assert "visualization aid only" in NDWI_INTERPRETATION


def test_shortlist_is_nearest_before_nearest_after_then_locally_clearest() -> None:
    candidates = [
        {
            "candidate_id": "before-far",
            "acquisition_datetime": "2024-01-10T12:00:00Z",
            "local_obscured_fraction": 0.10,
        },
        {
            "candidate_id": "before-near",
            "acquisition_datetime": "2024-01-13T12:00:00Z",
            "local_obscured_fraction": 0.90,
        },
        {
            "candidate_id": "after-near",
            "acquisition_datetime": "2024-01-15T12:00:00Z",
            "local_obscured_fraction": 0.80,
        },
        {
            "candidate_id": "after-clear",
            "acquisition_datetime": "2024-01-16T12:00:00Z",
            "local_obscured_fraction": 0.05,
        },
    ]

    selected = shortlist_acquisitions(
        candidates,
        swot_datetime="2024-01-14T12:00:00Z",
    )

    assert [candidate["candidate_id"] for candidate in selected] == [
        "before-near",
        "after-near",
        "after-clear",
    ]


@pytest.mark.skipif(not RASTERIO_AVAILABLE, reason="rasterio extra is not installed")
def test_local_raster_window_cache_and_no_clobber(tmp_path: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin

    source_path = tmp_path / "source.tif"
    values = np.arange(100, dtype=np.uint16).reshape(10, 10)
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="uint16",
        crs="EPSG:4326",
        transform=from_origin(0, 10, 1, 1),
        nodata=0,
    ) as destination:
        destination.write(values, 1)

    base = _scene()
    local_asset = replace(
        base.assets["green"],
        href=str(source_path),
        source_identifier="https://example.test/source/green.tif",
    )
    scene = replace(base, assets={**base.assets, "green": local_asset})
    target = tmp_path / "cache" / "green.tif"

    created = read_windowed_cog_asset(
        scene,
        "green",
        bounds_wgs84=(2.0, 4.0, 6.0, 8.0),
        output_path=target,
    )
    assert created.cache_status == "created"
    assert created.width == 4
    assert created.height == 4
    assert target.is_file()
    sidecar = target.with_suffix(".tif.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert "href" not in payload
    assert "sig=" not in json.dumps(payload)
    with rasterio.open(target) as cached:
        np.testing.assert_array_equal(cached.read(1), values[2:6, 2:6])

    hit = read_windowed_cog_asset(
        scene,
        "green",
        bounds_wgs84=(2.0, 4.0, 6.0, 8.0),
        output_path=target,
    )
    assert hit.cache_status == "hit"
    assert isinstance(hit.request_bounds_wgs84, tuple)
    assert isinstance(hit.raster_bounds, tuple)
    assert isinstance(hit.resolution, tuple)
    assert isinstance(hit.transform, tuple)
    original_bytes = target.read_bytes()

    single = mosaic_same_acquisition_chips(
        [hit], asset_role="green", output_path=tmp_path / "single.tif"
    )
    region = ReviewRegion(
        bounds_wgs84=(2.0, 4.0, 4.0, 8.0),
        polygon_wgs84=mapping(box(2.0, 4.0, 4.0, 8.0)),
        context_margin_m=1.0,
        construction_crs_wkt="synthetic test region",
    )
    included = review_region_pixel_mask(single, region)
    assert included.shape == (4, 4)
    assert int(np.count_nonzero(included)) == 8

    with pytest.raises(ReferenceImageryError, match="different request"):
        read_windowed_cog_asset(
            scene,
            "green",
            bounds_wgs84=(1.0, 4.0, 6.0, 8.0),
            output_path=target,
        )
    assert target.read_bytes() == original_bytes

    with pytest.raises(ReferenceImageryError, match="different Sentinel datatakes"):
        mosaic_same_acquisition_chips(
            [created, replace(created, acquisition_key="different-datatake")],
            asset_role="green",
            output_path=tmp_path / "invalid_mosaic.tif",
        )
    with pytest.raises(ReferenceImageryError, match="spectral asset roles"):
        mosaic_same_acquisition_chips(
            [created, replace(created, canonical_asset_role="red")],
            asset_role="green",
            output_path=tmp_path / "invalid_role_mosaic.tif",
        )
