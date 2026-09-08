from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from swot_pixc_lab import (
    ObservationMismatchError,
    PixcCollection,
    PixcOpenError,
    PixcSchemaError,
    open_pixc,
)
from swot_pixc_lab.discovery import parse_granule

FLOAT_FILL = np.float32(9.96921e36)
DOUBLE_FILL = np.float64(9.969209968386869e36)
UINT8_FILL = np.uint8(255)
UINT32_FILL = np.uint32(4294967295)
INT32_FILL = np.int32(2147483647)


def _write_pixc(
    path: Path,
    *,
    longitude: Sequence[float] = (1.0, 2.0, 3.0),
    latitude: Sequence[float] = (1.0, 2.0, 3.0),
    height: Sequence[float] | None = None,
    classification: Sequence[int] | None = None,
    cycle: int = 9,
    pass_number: int = 286,
    tile: str = "107L",
    include_pixel_cloud: bool = True,
    omitted_variables: Collection[str] = (),
    latitude_dimension: str = "points",
) -> Path:
    """Create a tiny NetCDF file with the observed Version-D PIXC layout."""

    longitude_values = np.asarray(longitude, dtype=np.float64)
    latitude_values = np.asarray(latitude, dtype=np.float64)
    if longitude_values.shape != latitude_values.shape:
        raise ValueError("Synthetic longitude and latitude must have equal lengths.")
    point_count = longitude_values.size
    height_values = (
        np.arange(point_count, dtype=np.float32) + np.float32(10.0)
        if height is None
        else np.asarray(height, dtype=np.float32)
    )
    classification_values = (
        (np.arange(point_count, dtype=np.uint8) % 7) + np.uint8(1)
        if classification is None
        else np.asarray(classification, dtype=np.uint8)
    )
    if height_values.size != point_count or classification_values.size != point_count:
        raise ValueError("Synthetic point variables must share the points length.")

    with netCDF4.Dataset(path, mode="w", format="NETCDF4") as root:
        root.setncattr("cycle_number", np.int16(cycle))
        root.setncattr("pass_number", np.int16(pass_number))
        root.setncattr("tile_number", np.int16(int(tile[:3])))
        root.setncattr("swath_side", tile[-1])
        root.setncattr("tile_name", f"{pass_number:03d}_{tile}")
        root.setncattr("crid", "PID0")
        root.setncattr("product_version", "01")
        root.setncattr("pge_version", "5.4.2")
        if not include_pixel_cloud:
            return path

        group = root.createGroup("pixel_cloud")
        group.setncattr("description", "synthetic pixel cloud")
        group.createDimension("points", point_count)
        group.createDimension("num_pixc_lines", 2)
        if latitude_dimension not in group.dimensions:
            group.createDimension(latitude_dimension, point_count)

        payloads: dict[str, tuple[str, tuple[str, ...], object, np.ndarray]] = {
            "azimuth_index": (
                "i4",
                ("points",),
                INT32_FILL,
                np.arange(point_count, dtype=np.int32),
            ),
            "range_index": (
                "i4",
                ("points",),
                INT32_FILL,
                np.arange(point_count, dtype=np.int32) + 100,
            ),
            "latitude": (
                "f8",
                (latitude_dimension,),
                DOUBLE_FILL,
                latitude_values,
            ),
            "longitude": (
                "f8",
                ("points",),
                DOUBLE_FILL,
                longitude_values,
            ),
            "height": ("f4", ("points",), FLOAT_FILL, height_values),
            "classification": (
                "u1",
                ("points",),
                UINT8_FILL,
                classification_values,
            ),
            "water_frac": (
                "f4",
                ("points",),
                FLOAT_FILL,
                np.linspace(0.1, 0.9, point_count, dtype=np.float32),
            ),
            "pixel_area": (
                "f4",
                ("points",),
                FLOAT_FILL,
                np.arange(point_count, dtype=np.float32) + np.float32(100.0),
            ),
            "sig0": (
                "f4",
                ("points",),
                FLOAT_FILL,
                np.arange(point_count, dtype=np.float32) - np.float32(2.0),
            ),
            "cross_track": (
                "f4",
                ("points",),
                FLOAT_FILL,
                np.arange(point_count, dtype=np.float32) * np.float32(10.0),
            ),
        }
        for name, (dtype, dimensions, fill_value, values) in payloads.items():
            if name in omitted_variables:
                continue
            variable = group.createVariable(
                name,
                dtype,
                dimensions,
                fill_value=fill_value,
            )
            variable[:] = values
            if name == "latitude":
                variable.setncatts(
                    {
                        "long_name": "latitude (positive N, negative S)",
                        "standard_name": "latitude",
                        "units": "degrees_north",
                    }
                )
            elif name == "longitude":
                variable.setncatts(
                    {
                        "long_name": "longitude (degrees East)",
                        "standard_name": "longitude",
                        "units": "degrees_east",
                    }
                )
            elif name == "height":
                variable.setncatts(
                    {
                        "long_name": "height above reference ellipsoid",
                        "units": "m",
                        "quality_flag": "geolocation_qual",
                    }
                )
            elif name == "classification":
                variable.setncatts(
                    {
                        "long_name": "classification",
                        "flag_values": np.arange(1, 8, dtype=np.uint8),
                        "flag_meanings": (
                            "land land_near_water water_near_land open_water "
                            "dark_water low_coh_water_near_land "
                            "open_low_coh_water"
                        ),
                    }
                )
            else:
                units = {
                    "azimuth_index": "1",
                    "range_index": "1",
                    "water_frac": "1",
                    "pixel_area": "m^2",
                    "sig0": "1",
                    "cross_track": "m",
                }
                variable.setncattr("units", units[name])

        for offset, name in enumerate(
            (
                "interferogram_qual",
                "classification_qual",
                "geolocation_qual",
                "sig0_qual",
            )
        ):
            if name in omitted_variables:
                continue
            variable = group.createVariable(
                name,
                "u4",
                ("points",),
                fill_value=UINT32_FILL,
            )
            variable[:] = np.arange(point_count, dtype=np.uint32) + offset
            variable.setncatts(
                {
                    "standard_name": "status_flag",
                    "flag_masks": np.asarray([1, 2], dtype=np.uint32),
                    "flag_meanings": "first second",
                }
            )

        line_quality = group.createVariable(
            "pixc_line_qual",
            "u4",
            ("num_pixc_lines",),
            fill_value=UINT32_FILL,
        )
        line_quality[:] = np.asarray([0, 1], dtype=np.uint32)
        line_quality.setncatts(
            {
                "standard_name": "status_flag",
                "flag_masks": np.asarray([1], dtype=np.uint32),
                "flag_meanings": "not_in_tile",
            }
        )
    return path


def test_bbox_clip_is_inclusive_and_accounts_for_invalid_coordinates(
    tmp_path: Path,
) -> None:
    path = _write_pixc(
        tmp_path / "bbox.nc",
        longitude=(0.0, 1.0, 2.0, 2.0, 3.0, DOUBLE_FILL, 1.5),
        latitude=(0.0, 1.0, 1.0, 2.0, 2.0, 1.5, np.nan),
    )

    observation = open_pixc([path], aoi=(1.0, 1.0, 2.0, 2.0))

    assert observation.pixel_count == 3
    np.testing.assert_array_equal(
        observation.pixels["source_point_index"].values,
        np.asarray([1, 2, 3], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        observation.pixels["longitude"].values,
        np.asarray([1.0, 2.0, 2.0]),
    )
    source = observation.sources[0]
    assert source.points_before == 7
    assert source.points_after == 3
    assert source.invalid_coordinate_count == 2


def test_bbox_clip_preserves_non_binary_decimal_boundaries(tmp_path: Path) -> None:
    path = _write_pixc(
        tmp_path / "decimal-boundaries.nc",
        longitude=(0.1, 0.2),
        latitude=(0.1, 0.2),
    )

    observation = open_pixc([path], aoi=(0.1, 0.1, 0.2, 0.2))

    np.testing.assert_array_equal(
        observation.pixels["source_point_index"].values,
        np.asarray([0, 1], dtype=np.int64),
    )
    assert observation.sources[0].normalized_longitude_count == 0


def test_polygon_hole_is_excluded_but_outer_and_hole_boundaries_are_retained(
    tmp_path: Path,
) -> None:
    path = _write_pixc(
        tmp_path / "polygon.nc",
        longitude=(0.0, 2.0, 1.0, 0.5, 5.0),
        latitude=(0.0, 2.0, 2.0, 0.5, 5.0),
    )
    polygon = {
        "type": "Polygon",
        "coordinates": [
            [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
            [[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]],
        ],
    }

    observation = open_pixc([path], aoi=polygon)

    np.testing.assert_array_equal(
        observation.pixels["source_point_index"].values,
        np.asarray([0, 2, 3], dtype=np.int64),
    )


def test_raw_dtypes_fill_values_attributes_and_quality_metadata_are_preserved(
    tmp_path: Path,
) -> None:
    path = _write_pixc(
        tmp_path / "metadata.nc",
        longitude=(1.0, 1.5, 2.0),
        latitude=(1.0, 1.5, 2.0),
        height=(10.25, FLOAT_FILL, 12.75),
        classification=(3, UINT8_FILL, 5),
    )

    observation = open_pixc([path], aoi=(0.0, 0.0, 3.0, 3.0))
    pixels = observation.pixels

    assert observation.raw is pixels
    assert pixels["latitude"].dtype == np.dtype("float64")
    assert pixels["height"].dtype == np.dtype("float32")
    assert pixels["classification"].dtype == np.dtype("uint8")
    assert pixels["geolocation_qual"].dtype == np.dtype("uint32")
    assert pixels["height"].attrs["_FillValue"] == FLOAT_FILL
    assert pixels["height"].attrs["long_name"] == "height above reference ellipsoid"
    assert pixels["height"].attrs["units"] == "m"
    assert pixels["height"].attrs["quality_flag"] == "geolocation_qual"
    assert pixels["classification"].attrs["_FillValue"] == UINT8_FILL
    np.testing.assert_array_equal(
        pixels["classification"].attrs["flag_values"],
        np.arange(1, 8, dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        pixels["height"].values,
        np.asarray([10.25, FLOAT_FILL, 12.75], dtype=np.float32),
    )
    assert {
        name: pixels[name].attrs["units"]
        for name in (
            "azimuth_index",
            "range_index",
            "water_frac",
            "pixel_area",
            "sig0",
            "cross_track",
        )
    } == {
        "azimuth_index": "1",
        "range_index": "1",
        "water_frac": "1",
        "pixel_area": "m^2",
        "sig0": "1",
        "cross_track": "m",
    }

    for name in (
        "interferogram_qual",
        "classification_qual",
        "geolocation_qual",
        "sig0_qual",
    ):
        assert name in pixels
        assert pixels[name].attrs["_FillValue"] == UINT32_FILL
        assert observation.sources[0].variables[name].dimensions == ("points",)

    assert "pixc_line_qual" not in pixels
    line_metadata = observation.sources[0].variables["pixc_line_qual"]
    assert line_metadata.dimensions == ("num_pixc_lines",)
    assert line_metadata.shape == (2,)
    assert line_metadata.fill_value == UINT32_FILL
    assert observation.sources[0].pixel_cloud_attributes["description"] == (
        "synthetic pixel cloud"
    )
    assert observation.sources[0].root_attributes["cycle_number"] == 9

    assert observation.classification_counts == {3: 1, 5: 1}
    assert observation.coordinate_ranges == {
        "longitude": (1.0, 2.0),
        "latitude": (1.0, 2.0),
    }
    assert observation.height_summary == {
        "count": 2,
        "min": 10.25,
        "max": 12.75,
        "mean": 11.5,
        "median": 11.5,
        "std": 1.25,
    }


def test_longitude_is_normalized_only_for_selection_not_in_raw_output(
    tmp_path: Path,
) -> None:
    path = _write_pixc(
        tmp_path / "longitude.nc",
        longitude=(360.0, 0.0, 1.0),
        latitude=(0.0, 0.0, 0.0),
    )

    observation = open_pixc([path], aoi=(-0.1, -0.1, 0.1, 0.1))

    np.testing.assert_array_equal(
        observation.pixels["longitude"].values,
        np.asarray([360.0, 0.0]),
    )
    assert observation.sources[0].normalized_longitude_count == 1


def test_antimeridian_bbox_clips_both_sides_without_changing_coordinates(
    tmp_path: Path,
) -> None:
    path = _write_pixc(
        tmp_path / "antimeridian.nc",
        longitude=(179.9, -179.9, 0.0),
        latitude=(0.0, 0.0, 0.0),
    )

    observation = open_pixc(path, aoi=(179.0, -1.0, -179.0, 1.0))

    np.testing.assert_array_equal(
        observation.pixels["longitude"].values,
        np.asarray([179.9, -179.9]),
    )
    assert observation.sources[0].normalized_longitude_count == 0


def test_tiles_are_concatenated_in_order_without_removing_duplicate_coordinates(
    tmp_path: Path,
) -> None:
    first = _write_pixc(
        tmp_path / "first.nc",
        longitude=(1.0, 2.0),
        latitude=(1.0, 2.0),
        height=(10.0, 20.0),
        classification=(3, 4),
        tile="107L",
    )
    second = _write_pixc(
        tmp_path / "second.nc",
        longitude=(2.0, 3.0),
        latitude=(2.0, 3.0),
        height=(200.0, 300.0),
        classification=(5, 6),
        tile="108L",
    )

    observation = open_pixc([first, second], aoi=(0.0, 0.0, 4.0, 4.0))

    assert observation.pixel_count == 4
    np.testing.assert_array_equal(
        observation.pixels["longitude"].values,
        np.asarray([1.0, 2.0, 2.0, 3.0]),
    )
    np.testing.assert_array_equal(
        observation.pixels["height"].values,
        np.asarray([10.0, 20.0, 200.0, 300.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        observation.pixels["source_index"].values,
        np.asarray([0, 0, 1, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        observation.pixels["source_point_index"].values,
        np.asarray([0, 1, 0, 1], dtype=np.int64),
    )
    assert [source.tile for source in observation.sources] == ["107L", "108L"]
    assert observation.source_table["filename"].tolist() == ["first.nc", "second.nc"]
    assert observation.source_table["granule_id"].tolist() == ["first", "second"]
    assert observation.source_tile_counts == {"107L": 2, "108L": 2}
    assert observation.summary()["pixels_before"] == 4
    assert observation.summary()["pixels_after"] == 4


def test_source_with_zero_clipped_pixels_remains_in_provenance(
    tmp_path: Path,
) -> None:
    empty = _write_pixc(
        tmp_path / "empty.nc",
        longitude=(10.0,),
        latitude=(10.0,),
        tile="107L",
    )
    retained = _write_pixc(
        tmp_path / "retained.nc",
        longitude=(1.0,),
        latitude=(1.0,),
        tile="108L",
    )

    observation = open_pixc([empty, retained], aoi=(0.0, 0.0, 2.0, 2.0))

    assert len(observation.sources) == 2
    assert observation.source_table["points_after"].tolist() == [0, 1]
    assert observation.source_tile_counts == {"107L": 0, "108L": 1}
    np.testing.assert_array_equal(
        observation.pixels["source_index"].values,
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        observation.pixels["source_point_index"].values,
        np.asarray([0], dtype=np.int64),
    )


def test_all_sources_can_clip_to_zero_without_losing_provenance(
    tmp_path: Path,
) -> None:
    first = _write_pixc(
        tmp_path / "empty-first.nc",
        longitude=(10.0,),
        latitude=(10.0,),
        tile="107L",
    )
    second = _write_pixc(
        tmp_path / "empty-second.nc",
        longitude=(20.0,),
        latitude=(20.0,),
        tile="108L",
    )

    observation = open_pixc([first, second], aoi=(0.0, 0.0, 2.0, 2.0))

    assert observation.pixel_count == 0
    assert observation.source_table["points_after"].tolist() == [0, 0]
    assert observation.source_tile_counts == {"107L": 0, "108L": 0}
    assert observation.coordinate_ranges == {"longitude": None, "latitude": None}
    assert observation.height_summary["count"] == 0


def test_collection_open_resolves_locally_and_passes_cmr_provenance(
    tmp_path: Path,
    granule_factory,
) -> None:
    raw_record = granule_factory(
        cycle=9,
        pass_number=286,
        tile="107L",
        start="2024-01-14T08:14:32Z",
    )
    filename = raw_record["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]
    path = _write_pixc(
        tmp_path / filename,
        longitude=(1.0,),
        latitude=(1.0,),
    )
    record = parse_granule(raw_record).with_local_path(path)
    collection = PixcCollection([record])

    observation = collection.open(
        aoi=(0.0, 0.0, 2.0, 2.0),
        verify="none",
    )

    assert observation.pixel_count == 1
    assert observation.sources[0].granule_id == record.granule_id
    assert observation.sources[0].tile == "107L"
    assert observation.sources[0].cmr_record == record
    assert observation.source_table.loc[0, "collection_version"] == "D"
    assert observation.source_table.loc[0, "netcdf_product_version"] == "01"


def test_collection_open_rejects_mismatched_cmr_filename(
    tmp_path: Path,
    granule_factory,
) -> None:
    path = _write_pixc(tmp_path / "wrong-local-file.nc")
    record = parse_granule(
        granule_factory(cycle=9, pass_number=286, tile="107L")
    ).with_local_path(path)

    with pytest.raises(PixcSchemaError, match="CMR filename .* conflicts"):
        PixcCollection([record]).open(
            aoi=(0.0, 0.0, 4.0, 4.0),
            verify="none",
        )


def test_collection_rejects_multi_observation_manifest_before_local_io(
    granule_factory,
) -> None:
    first = parse_granule(granule_factory(concept_id="G1", cycle=9))
    second = parse_granule(granule_factory(concept_id="G2", cycle=10))

    with pytest.raises(ObservationMismatchError, match="select one observation"):
        PixcCollection([first, second]).open(
            aoi=(0.0, 0.0, 4.0, 4.0),
            verify="none",
        )


def test_combined_attributes_do_not_alias_source_metadata(tmp_path: Path) -> None:
    first = _write_pixc(tmp_path / "first-attrs.nc", tile="107L")
    second = _write_pixc(tmp_path / "second-attrs.nc", tile="108L")

    observation = open_pixc([first, second], aoi=(0.0, 0.0, 4.0, 4.0))
    combined_masks = observation.pixels["geolocation_qual"].attrs["flag_masks"]
    source_masks = (
        observation.sources[0].variables["geolocation_qual"].attributes["flag_masks"]
    )

    assert not np.shares_memory(combined_masks, source_masks)
    assert not source_masks.flags.writeable
    np.testing.assert_array_equal(source_masks, np.asarray([1, 2], dtype=np.uint32))


def test_differing_attributes_remain_per_source_and_are_reported(
    tmp_path: Path,
) -> None:
    first = _write_pixc(tmp_path / "first-units.nc", tile="107L")
    second = _write_pixc(tmp_path / "second-units.nc", tile="108L")
    with netCDF4.Dataset(second, mode="a") as root:
        root["pixel_cloud"]["height"].setncattr("units", "synthetic_other_unit")
        root.setncattr("crid", "PGD0")

    observation = open_pixc([first, second], aoi=(0.0, 0.0, 4.0, 4.0))

    assert "units" not in observation.pixels["height"].attrs
    assert observation.sources[0].variables["height"].attributes["units"] == "m"
    assert (
        observation.sources[1].variables["height"].attributes["units"]
        == "synthetic_other_unit"
    )
    assert any("/pixel_cloud/height" in note for note in observation.notes)
    assert any("NetCDF CRID" in note for note in observation.notes)


def test_duplicate_source_path_is_rejected(tmp_path: Path) -> None:
    path = _write_pixc(tmp_path / "duplicate.nc")

    with pytest.raises(PixcOpenError, match="duplicate input"):
        open_pixc([path, path], aoi=(0.0, 0.0, 4.0, 4.0))


def test_duplicate_tile_requires_explicit_revision_selection(tmp_path: Path) -> None:
    first = _write_pixc(tmp_path / "tile-first.nc", tile="107L")
    second = _write_pixc(tmp_path / "tile-reprocessed.nc", tile="107L")

    with pytest.raises(ObservationMismatchError, match="same tile"):
        open_pixc([first, second], aoi=(0.0, 0.0, 4.0, 4.0))


@pytest.mark.parametrize(
    ("other_cycle", "other_pass"),
    [(10, 286), (9, 564)],
)
def test_different_cycle_or_pass_cannot_be_combined(
    tmp_path: Path,
    other_cycle: int,
    other_pass: int,
) -> None:
    first = _write_pixc(tmp_path / "cycle9.nc", cycle=9, pass_number=286)
    second = _write_pixc(
        tmp_path / "other-observation.nc",
        cycle=other_cycle,
        pass_number=other_pass,
    )

    with pytest.raises(
        ObservationMismatchError,
        match=r"different SWOT cycle/pass observations.*cycle9\.nc=cycle 9/pass 286",
    ):
        open_pixc([first, second], aoi=(0.0, 0.0, 4.0, 4.0))


def test_multiple_sources_require_resolved_cycle_and_pass(tmp_path: Path) -> None:
    first = _write_pixc(tmp_path / "known.nc", tile="107L")
    second = _write_pixc(tmp_path / "missing-pass.nc", tile="108L")
    with netCDF4.Dataset(second, mode="a") as root:
        root.delncattr("pass_number")
        root.delncattr("tile_name")

    with pytest.raises(
        ObservationMismatchError,
        match=r"resolved cycle and pass.*missing-pass\.nc",
    ):
        open_pixc([first, second], aoi=(0.0, 0.0, 4.0, 4.0))


def test_inconsistent_netcdf_tile_metadata_is_rejected(tmp_path: Path) -> None:
    path = _write_pixc(tmp_path / "inconsistent-tile.nc", tile="107L")
    with netCDF4.Dataset(path, mode="a") as root:
        root.setncattr("tile_number", np.int16(108))

    with pytest.raises(PixcSchemaError, match="tile_name .* conflicts"):
        open_pixc(path, aoi=(0.0, 0.0, 4.0, 4.0))


def test_missing_pixel_cloud_group_has_actionable_schema_error(
    tmp_path: Path,
) -> None:
    path = _write_pixc(tmp_path / "no-group.nc", include_pixel_cloud=False)

    with pytest.raises(PixcSchemaError, match=r"no /pixel_cloud group: no-group\.nc"):
        open_pixc([path], aoi=(0.0, 0.0, 4.0, 4.0))


def test_missing_required_coordinate_has_actionable_schema_error(
    tmp_path: Path,
) -> None:
    path = _write_pixc(
        tmp_path / "no-latitude.nc",
        omitted_variables={"latitude"},
    )

    with pytest.raises(
        PixcSchemaError,
        match=r"/pixel_cloud/latitude is required in no-latitude\.nc",
    ):
        open_pixc([path], aoi=(0.0, 0.0, 4.0, 4.0))


def test_coordinate_on_wrong_dimension_has_actionable_schema_error(
    tmp_path: Path,
) -> None:
    path = _write_pixc(
        tmp_path / "wrong-dimension.nc",
        longitude=(1.0, 2.0),
        latitude=(1.0, 2.0),
        latitude_dimension="wrong_points",
    )

    with pytest.raises(
        PixcSchemaError,
        match=r"/pixel_cloud/latitude must use only the points dimension",
    ):
        open_pixc([path], aoi=(0.0, 0.0, 4.0, 4.0))


def test_line_quality_cannot_be_requested_as_a_point_variable(
    tmp_path: Path,
) -> None:
    path = _write_pixc(tmp_path / "line-quality.nc")

    with pytest.raises(
        PixcSchemaError,
        match=r"pixc_line_qual.*only 1-D point variables",
    ):
        open_pixc(
            [path],
            aoi=(0.0, 0.0, 4.0, 4.0),
            variables=["pixc_line_qual"],
        )


def test_variables_must_be_a_sequence_of_complete_names(tmp_path: Path) -> None:
    path = _write_pixc(tmp_path / "variable-string.nc")

    with pytest.raises(TypeError, match="sequence of complete"):
        open_pixc(path, aoi=(0.0, 0.0, 4.0, 4.0), variables="height")
