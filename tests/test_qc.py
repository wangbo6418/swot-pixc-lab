from __future__ import annotations

import copy
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from swot_pixc_lab import (
    QualityMetadataError,
    apply_qc,
    decode_quality_flags,
    open_pixc,
)

FLOAT_FILL = np.float32(9.96921e36)
DOUBLE_FILL = np.float64(9.969209968386869e36)
UINT8_FILL = np.uint8(255)
UINT32_FILL = np.uint32(4294967295)

CLASSIFICATION_MEANINGS = (
    "land land_near_water water_near_land open_water dark_water "
    "low_coh_water_near_land open_low_coh_water"
)

QUALITY_DEFINITIONS: Mapping[str, tuple[tuple[str, int], ...]] = {
    "classification_qual": (
        ("water_false_detection_rate_suspect", 16),
        ("in_air_pixel_degraded", 262144),
        ("coherent_power_bad", 134217728),
        ("tvp_bad", 536870912),
        ("sc_event_bad", 1073741824),
        ("large_karin_gap", 2147483648),
    ),
    "geolocation_qual": (
        ("phase_noise_suspect", 2),
        ("specular_ringing_degraded", 524288),
        ("no_geolocation_bad", 134217728),
        ("medium_phase_bad", 268435456),
        ("tvp_bad", 536870912),
        ("sc_event_bad", 1073741824),
        ("large_karin_gap", 2147483648),
    ),
    "interferogram_qual": (
        ("rare_phase_suspect", 4096),
        ("in_air_pixel_degraded", 262144),
        ("specular_ringing_degraded", 524288),
        ("rare_power_bad", 134217728),
        ("rare_phase_bad", 268435456),
        ("tvp_bad", 536870912),
        ("sc_event_bad", 1073741824),
        ("large_karin_gap", 2147483648),
    ),
    "sig0_qual": (
        ("noise_power_suspect", 4),
        ("in_air_pixel_degraded", 262144),
        ("noise_power_bad", 33554432),
        ("tvp_bad", 536870912),
        ("sc_event_bad", 1073741824),
        ("large_karin_gap", 2147483648),
    ),
}

QC_POINT_VARIABLES = (
    "classification",
    "water_frac",
    "water_frac_uncert",
    "height",
    "geoid",
    "inc",
    "phase_noise_std",
    "bright_land_flag",
    "false_detection_rate",
    "missed_detection_rate",
    "prior_water_prob",
    "prior_water_change",
    "classification_qual",
    "geolocation_qual",
    "interferogram_qual",
    "sig0_qual",
)


def _values(
    count: int,
    value: float | int,
    dtype: np.dtype[np.generic] | type[np.generic],
) -> np.ndarray:
    return np.full(count, value, dtype=dtype)


def _write_qc_pixc(
    path: Path,
    *,
    count: int = 4,
    overrides: Mapping[str, Sequence[float | int]] | None = None,
    omitted: Collection[str] = (),
    tile: str = "107L",
    geoid_source: str | None = "EGM2008 (Pavlis et al., 2012)",
) -> Path:
    """Write a tiny file with the relevant Version-D /pixel_cloud layout."""

    payloads: dict[str, np.ndarray] = {
        "longitude": np.linspace(0.0, 0.3, count, dtype=np.float64),
        "latitude": np.linspace(0.0, 0.3, count, dtype=np.float64),
        "classification": _values(count, 4, np.uint8),
        "water_frac": _values(count, 0.95, np.float32),
        "water_frac_uncert": _values(count, 0.10, np.float32),
        "height": np.arange(count, dtype=np.float32) + np.float32(100.0),
        "geoid": _values(count, 50.0, np.float32),
        "inc": _values(count, 2.5, np.float32),
        "phase_noise_std": _values(count, 0.5, np.float32),
        "bright_land_flag": _values(count, 0, np.uint8),
        "false_detection_rate": _values(count, 0.05, np.float32),
        "missed_detection_rate": _values(count, 0.05, np.float32),
        "prior_water_prob": _values(count, 0.8, np.float32),
        "prior_water_change": _values(count, 0.0, np.float32),
        "classification_qual": _values(count, 0, np.uint32),
        "geolocation_qual": _values(count, 0, np.uint32),
        "interferogram_qual": _values(count, 0, np.uint32),
        "sig0_qual": _values(count, 0, np.uint32),
    }
    for name, values in (overrides or {}).items():
        if name not in payloads:
            raise ValueError(f"Unknown synthetic variable: {name}")
        payloads[name] = np.asarray(values, dtype=payloads[name].dtype)
        if payloads[name].shape != (count,):
            raise ValueError(f"Synthetic {name} must contain {count} values.")

    with netCDF4.Dataset(path, mode="w", format="NETCDF4") as root:
        root.setncattr("cycle_number", np.int16(9))
        root.setncattr("pass_number", np.int16(286))
        root.setncattr("tile_number", np.int16(int(tile[:3])))
        root.setncattr("swath_side", tile[-1])
        root.setncattr("tile_name", f"286_{tile}")
        root.setncattr("crid", "PGD0")
        root.setncattr("product_version", "01")

        group = root.createGroup("pixel_cloud")
        group.createDimension("points", count)
        group.createDimension("num_pixc_lines", 2)

        for name, values in payloads.items():
            if name in omitted:
                continue
            fill_value: np.generic
            if values.dtype == np.dtype("float64"):
                fill_value = DOUBLE_FILL
            elif values.dtype == np.dtype("float32"):
                fill_value = FLOAT_FILL
            elif values.dtype == np.dtype("uint8"):
                fill_value = UINT8_FILL
            elif values.dtype == np.dtype("uint32"):
                fill_value = UINT32_FILL
            else:  # pragma: no cover - fixture invariant
                raise AssertionError(values.dtype)
            variable = group.createVariable(
                name,
                values.dtype,
                ("points",),
                fill_value=fill_value,
            )
            variable[:] = values
            _set_variable_attributes(variable, name, geoid_source=geoid_source)

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
                "valid_min": np.uint32(0),
                "valid_max": np.uint32(1),
            }
        )
    return path


def _set_variable_attributes(
    variable: netCDF4.Variable,
    name: str,
    *,
    geoid_source: str | None,
) -> None:
    if name == "longitude":
        variable.setncatts({"standard_name": "longitude", "units": "degrees_east"})
        return
    if name == "latitude":
        variable.setncatts({"standard_name": "latitude", "units": "degrees_north"})
        return
    if name == "classification":
        variable.setncatts(
            {
                "flag_meanings": CLASSIFICATION_MEANINGS,
                "flag_values": np.arange(1, 8, dtype=np.uint8),
                "valid_min": np.uint8(1),
                "valid_max": np.uint8(7),
            }
        )
        return
    if name == "bright_land_flag":
        variable.setncatts(
            {
                "standard_name": "status_flag",
                "flag_meanings": "not_bright_land bright_land bright_land_or_water",
                "flag_values": np.arange(3, dtype=np.uint8),
                "valid_min": np.uint8(0),
                "valid_max": np.uint8(2),
            }
        )
        return
    if name in QUALITY_DEFINITIONS:
        definitions = QUALITY_DEFINITIONS[name]
        masks = np.asarray([mask for _, mask in definitions], dtype=np.uint32)
        variable.setncatts(
            {
                "standard_name": "status_flag",
                "flag_meanings": " ".join(meaning for meaning, _ in definitions),
                "flag_masks": masks,
                "valid_min": np.uint32(0),
                "valid_max": np.bitwise_or.reduce(masks),
            }
        )
        return

    units = {
        "water_frac": "1",
        "water_frac_uncert": "1",
        "height": "m",
        "geoid": "m",
        "inc": "degrees",
        "phase_noise_std": "radians",
        "false_detection_rate": "1",
        "missed_detection_rate": "1",
        "prior_water_prob": "1",
        "prior_water_change": "1",
    }
    attributes: dict[str, object] = {"units": units[name]}
    if name == "height":
        attributes.update(
            {
                "long_name": "height above reference ellipsoid",
                "valid_min": np.float32(-1500.0),
                "valid_max": np.float32(15000.0),
            }
        )
    elif name == "geoid":
        attributes.update(
            {
                "long_name": "geoid height",
                "standard_name": "geoid_height_above_reference_ellipsoid",
                "valid_min": np.float32(-150.0),
                "valid_max": np.float32(150.0),
                "comment": (
                    "Geoid height above the reference ellipsoid; reported for "
                    "reference and not applied to height."
                ),
            }
        )
        if geoid_source is not None:
            attributes["source"] = geoid_source
    elif name in {
        "water_frac_uncert",
        "phase_noise_std",
        "inc",
    }:
        attributes.update(
            {
                "valid_min": np.float32(0.0),
                "valid_max": np.float32(999999.0),
            }
        )
    elif name in {
        "false_detection_rate",
        "missed_detection_rate",
        "prior_water_prob",
    }:
        attributes.update({"valid_min": np.float32(0.0), "valid_max": np.float32(1.0)})
    variable.setncatts(attributes)


def _open_qc(path: Path, *, omitted: Collection[str] = ()):
    variables = tuple(name for name in QC_POINT_VARIABLES if name not in omitted)
    return open_pixc(path, aoi=(-1.0, -1.0, 1.0, 1.0), variables=variables)


def _bool_values(mask: object) -> np.ndarray:
    return np.asarray(mask, dtype=np.bool_)


def test_decode_mask_flags_and_fill_without_changing_unsigned_values() -> None:
    values = np.asarray([0, 1, 2, 3, UINT32_FILL], dtype=np.uint32)
    original = values.copy()
    decoded = decode_quality_flags(
        values,
        {
            "flag_masks": np.asarray([1, 2], dtype=np.uint32),
            "flag_meanings": "first second",
            "valid_min": np.uint32(0),
            "valid_max": np.uint32(3),
            "_FillValue": UINT32_FILL,
        },
        variable_name="synthetic_qual",
    )

    np.testing.assert_array_equal(
        _bool_values(decoded.flag("first")),
        [False, True, False, True, False],
    )
    np.testing.assert_array_equal(
        _bool_values(decoded.flag("second")),
        [False, False, True, True, False],
    )
    np.testing.assert_array_equal(
        _bool_values(decoded.fill_mask),
        [False, False, False, False, True],
    )
    assert not _bool_values(decoded.out_of_range_mask).any()
    assert decoded.frequencies() == {"first": 2, "second": 2}
    assert values.dtype == np.dtype("uint32")
    np.testing.assert_array_equal(values, original)


def test_decode_flag_values_and_combined_masks_values() -> None:
    enumerated = decode_quality_flags(
        np.asarray([0, 1, 2, 3, UINT8_FILL], dtype=np.uint8),
        {
            "flag_values": np.asarray([0, 1, 2], dtype=np.uint8),
            "flag_meanings": "clear one two",
            "valid_min": np.uint8(0),
            "valid_max": np.uint8(2),
            "_FillValue": UINT8_FILL,
        },
        variable_name="enumerated_flag",
    )
    np.testing.assert_array_equal(
        _bool_values(enumerated.flag("one")),
        [False, True, False, False, False],
    )
    np.testing.assert_array_equal(
        _bool_values(enumerated.out_of_range_mask),
        [False, False, False, True, False],
    )

    combined = decode_quality_flags(
        np.asarray([0, 1, 2, 3, 4, UINT8_FILL], dtype=np.uint8),
        {
            "flag_masks": np.asarray([3, 3], dtype=np.uint8),
            "flag_values": np.asarray([1, 2], dtype=np.uint8),
            "flag_meanings": "low_one low_two",
            "valid_min": np.uint8(0),
            "valid_max": np.uint8(4),
            "_FillValue": UINT8_FILL,
        },
        variable_name="combined_flag",
    )
    np.testing.assert_array_equal(
        _bool_values(combined.flag("low_one")),
        [False, True, False, False, False, False],
    )
    np.testing.assert_array_equal(
        _bool_values(combined.flag("low_two")),
        [False, False, True, False, False, False],
    )


def test_decoder_handles_signed_high_bit_and_shared_mask_unknown_state() -> None:
    signed = decode_quality_flags(
        np.asarray([0, -128], dtype=np.int8),
        {
            "flag_masks": np.asarray([128], dtype=np.int16),
            "flag_meanings": "high_bit",
            "valid_min": np.int16(-128),
            "valid_max": np.int16(127),
        },
        variable_name="signed_flag",
    )
    np.testing.assert_array_equal(signed.flag("high_bit"), [False, True])

    combined = decode_quality_flags(
        np.asarray([1, 2, 3, 5, 7], dtype=np.uint8),
        {
            "flag_masks": np.asarray([3, 3, 4], dtype=np.uint8),
            "flag_values": np.asarray([1, 2, 4], dtype=np.uint8),
            "flag_meanings": "state_one state_two independent",
            "valid_min": np.uint8(0),
            "valid_max": np.uint8(7),
        },
        variable_name="shared_mask",
    )
    np.testing.assert_array_equal(
        combined.unknown_bits_mask,
        [False, False, True, False, True],
    )

    with pytest.raises(QualityMetadataError, match="outside its paired mask"):
        decode_quality_flags(
            np.asarray([0, 1], dtype=np.uint8),
            {
                "flag_masks": np.asarray([1], dtype=np.uint8),
                "flag_values": np.asarray([2], dtype=np.uint8),
                "flag_meanings": "impossible",
            },
        )


def test_raw_profile_keeps_every_exact_aoi_pixel_and_method_delegates(
    tmp_path: Path,
) -> None:
    observation = _open_qc(
        _write_qc_pixc(
            tmp_path / "raw.nc",
            overrides={"classification": [1, 4, 7, UINT8_FILL]},
        )
    )

    result = apply_qc(observation, profile="raw")
    delegated = observation.apply_qc(profile="raw")

    assert result.profile.name == "raw"
    np.testing.assert_array_equal(_bool_values(result.mask), np.ones(4, dtype=bool))
    np.testing.assert_array_equal(_bool_values(delegated.mask), result.mask)
    assert result.filtered.sizes["points"] == 4
    assert not result.reason_masks
    assert not result.incremental_reason_masks
    assert not result.reason_counts
    assert not result.incremental_reason_counts
    assert set(result.decoded_flags) == set(QUALITY_DEFINITIONS)
    assert result.summary["pixels_before"] == 4
    assert result.summary["pixels_retained"] == 4
    assert result.summary["pixels_removed"] == 0


def test_default_reader_loads_available_phase3_point_variables(tmp_path: Path) -> None:
    observation = open_pixc(
        _write_qc_pixc(tmp_path / "phase3-defaults.nc"),
        aoi=(-1.0, -1.0, 1.0, 1.0),
    )

    assert set(QC_POINT_VARIABLES) <= set(observation.raw)
    assert "pixc_line_qual" not in observation.raw


def test_legacy_strict_reports_overlapping_and_ordered_rejections(
    tmp_path: Path,
) -> None:
    observation = _open_qc(
        _write_qc_pixc(
            tmp_path / "legacy.nc",
            overrides={
                "classification": [4, 3, 4, 3],
                "water_frac": [0.95, 0.50, 0.50, 0.95],
            },
        )
    )

    result = apply_qc(observation, profile="bo_legacy_strict")
    reason_names = list(result.reason_masks)

    assert result.profile.name == "bo_legacy_strict"
    assert len(result.rule_outcomes) == 11
    assert len(reason_names) == 11
    np.testing.assert_array_equal(
        _bool_values(result.reason_masks[reason_names[0]]),
        [False, True, False, True],
    )
    np.testing.assert_array_equal(
        _bool_values(result.reason_masks[reason_names[1]]),
        [False, True, True, False],
    )
    np.testing.assert_array_equal(
        _bool_values(result.incremental_reason_masks[reason_names[0]]),
        [False, True, False, True],
    )
    np.testing.assert_array_equal(
        _bool_values(result.incremental_reason_masks[reason_names[1]]),
        [False, False, True, False],
    )
    assert list(result.reason_counts.values())[:2] == [2, 2]
    assert list(result.incremental_reason_counts.values())[:2] == [2, 1]
    assert sum(result.incremental_reason_counts.values()) == 3
    np.testing.assert_array_equal(
        _bool_values(result.mask), [True, False, False, False]
    )
    assert result.filtered.sizes["points"] == 1


def test_missing_legacy_variables_are_reported_as_skipped(tmp_path: Path) -> None:
    omitted = {
        "water_frac",
        "water_frac_uncert",
        "bright_land_flag",
        "false_detection_rate",
        "phase_noise_std",
        "inc",
    }
    observation = _open_qc(
        _write_qc_pixc(tmp_path / "missing.nc", omitted=omitted),
        omitted=omitted,
    )

    result = apply_qc(observation, profile="bo_legacy_strict")

    assert len(result.rule_outcomes) == 11
    assert sum(outcome.status == "skipped" for outcome in result.rule_outcomes) == 6
    assert len(result.summary["skipped_rules"]) == 6
    assert result.filtered.sizes["points"] == 4


def test_channel_candidate_filters_only_documented_geometry_conditions(
    tmp_path: Path,
) -> None:
    count = 12
    classification_qual = np.zeros(count, dtype=np.uint32)
    classification_qual[2] = np.uint32(16)  # suspect, intentionally retained
    classification_qual[3] = np.uint32(262144)  # in-air degraded
    classification_qual[5] = np.uint32(134217728)  # overlaps class 1 rejection
    classification_qual[7] = np.uint32(134217728)  # coherent_power_bad
    geolocation_qual = np.zeros(count, dtype=np.uint32)
    geolocation_qual[8] = np.uint32(134217728)  # no_geolocation_bad
    interferogram_qual = np.zeros(count, dtype=np.uint32)
    interferogram_qual[4] = np.uint32(262144)  # in-air is an extent rule
    interferogram_qual[9] = np.uint32(134217728)  # diagnostic, not an extent rule
    sig0_qual = np.zeros(count, dtype=np.uint32)
    sig0_qual[10] = np.uint32(33554432)  # irrelevant to extent candidate
    observation = _open_qc(
        _write_qc_pixc(
            tmp_path / "candidate.nc",
            count=count,
            overrides={
                "classification": [3, 4, 5, 6, 7, 1, 2, 4, 4, 4, 4, UINT8_FILL],
                "water_frac": [0.1] * count,
                "classification_qual": classification_qual,
                "geolocation_qual": geolocation_qual,
                "interferogram_qual": interferogram_qual,
                "sig0_qual": sig0_qual,
            },
        )
    )

    result = apply_qc(observation, profile="channel_extent_candidate")

    assert result.profile.name == "channel_extent_candidate"
    np.testing.assert_array_equal(
        observation.raw["source_point_index"].values[_bool_values(result.mask)],
        np.asarray([0, 1, 2, 9, 10], dtype=np.int64),
    )
    assert result.filtered["classification"].values.tolist() == [3, 4, 5, 4, 4]
    assert result.filtered["water_frac"].values.tolist() == [
        np.float32(0.1),
        np.float32(0.1),
        np.float32(0.1),
        np.float32(0.1),
        np.float32(0.1),
    ]
    assert result.filtered["sig0_qual"].values[-1] == np.uint32(33554432)
    assert result.filtered["interferogram_qual"].values[3] == np.uint32(134217728)
    assert sum(result.incremental_reason_counts.values()) == 7
    assert (
        result.decoded_flags["classification_qual"].frequencies()[
            "water_false_detection_rate_suspect"
        ]
        == 1
    )


def test_candidate_rejects_undocumented_quality_bits(tmp_path: Path) -> None:
    observation = _open_qc(
        _write_qc_pixc(
            tmp_path / "unknown-bit.nc",
            count=2,
            overrides={
                "classification": [4, 4],
                # Bit 5 is within valid_max but absent from synthetic metadata.
                "classification_qual": [0, 32],
            },
        )
    )

    result = apply_qc(observation, profile="channel_extent_candidate")

    np.testing.assert_array_equal(result.mask, [True, False])
    assert result.reason_counts["classification_qual_valid"] == 1


def test_qc_does_not_mutate_raw_and_filtered_cannot_alias_it(tmp_path: Path) -> None:
    observation = _open_qc(_write_qc_pixc(tmp_path / "immutable.nc"))
    original_values = {
        name: variable.values.copy()
        for name, variable in observation.raw.data_vars.items()
    }
    original_attributes = {
        name: copy.deepcopy(variable.attrs)
        for name, variable in observation.raw.data_vars.items()
    }

    result = apply_qc(observation, profile="channel_extent_candidate")
    filtered = result.filtered
    filtered["height"].values[0] = np.float32(-9999.0)

    for name, expected in original_values.items():
        np.testing.assert_array_equal(observation.raw[name].values, expected)
        assert observation.raw[name].attrs.keys() == original_attributes[name].keys()
        for attribute, value in original_attributes[name].items():
            np.testing.assert_array_equal(observation.raw[name].attrs[attribute], value)
    assert observation.raw["classification_qual"].dtype == np.dtype("uint32")


def test_provenance_survives_retention_and_rejection_across_tiles(
    tmp_path: Path,
) -> None:
    first = _write_qc_pixc(
        tmp_path / "107L.nc",
        count=2,
        tile="107L",
        overrides={"classification": [4, 1]},
    )
    second = _write_qc_pixc(
        tmp_path / "108L.nc",
        count=2,
        tile="108L",
        overrides={"classification": [5, 2]},
    )
    observation = open_pixc(
        [first, second],
        aoi=(-1.0, -1.0, 1.0, 1.0),
        variables=QC_POINT_VARIABLES,
    )

    result = apply_qc(observation, profile="channel_extent_candidate")
    keep = _bool_values(result.mask)

    np.testing.assert_array_equal(result.filtered["source_index"].values, [0, 1])
    np.testing.assert_array_equal(result.filtered["source_point_index"].values, [0, 0])
    np.testing.assert_array_equal(observation.raw["source_index"].values[~keep], [0, 1])
    np.testing.assert_array_equal(
        observation.raw["source_point_index"].values[~keep], [1, 1]
    )
    assert [source.tile for source in observation.sources] == ["107L", "108L"]


def test_quality_metadata_difference_across_tiles_stops_decoding(
    tmp_path: Path,
) -> None:
    first = _write_qc_pixc(tmp_path / "first-flags.nc", tile="107L")
    second = _write_qc_pixc(tmp_path / "second-flags.nc", tile="108L")
    with netCDF4.Dataset(second, mode="a") as root:
        variable = root["pixel_cloud"]["classification_qual"]
        masks = np.asarray(variable.flag_masks, dtype=np.uint32)
        masks[0] = np.uint32(32)
        variable.setncattr("flag_masks", masks)

    observation = open_pixc(
        [first, second],
        aoi=(-1.0, -1.0, 1.0, 1.0),
        variables=QC_POINT_VARIABLES,
    )

    with pytest.raises(QualityMetadataError, match="differs across source granules"):
        apply_qc(observation, profile="raw")


def test_height_egm2008_is_derived_with_fill_handling_and_raw_is_unchanged(
    tmp_path: Path,
) -> None:
    observation = _open_qc(
        _write_qc_pixc(
            tmp_path / "egm2008.nc",
            count=3,
            overrides={
                "height": [100.0, FLOAT_FILL, 110.0],
                "geoid": [40.0, 41.0, 42.0],
            },
        )
    )

    result = apply_qc(observation, profile="raw")

    assert "height_egm2008" in result.filtered
    np.testing.assert_allclose(
        result.filtered["height_egm2008"].values[[0, 2]],
        np.asarray([60.0, 68.0], dtype=np.float32),
    )
    assert np.isnan(result.filtered["height_egm2008"].values[1])
    assert result.filtered["height_egm2008"].attrs["formula"] == "height - geoid"
    assert observation.raw["height"].values[1] == FLOAT_FILL
    assert "height_egm2008" not in observation.raw


def test_unverified_geoid_uses_safe_derived_name(tmp_path: Path) -> None:
    observation = _open_qc(
        _write_qc_pixc(
            tmp_path / "unknown-geoid.nc",
            count=2,
            geoid_source=None,
            overrides={"height": [100.0, 110.0], "geoid": [40.0, 42.0]},
        )
    )

    result = apply_qc(observation, profile="raw")

    assert "height_minus_geoid" in result.filtered
    assert "height_egm2008" not in result.filtered
    np.testing.assert_allclose(
        result.filtered["height_minus_geoid"].values,
        np.asarray([60.0, 68.0], dtype=np.float32),
    )


def test_line_quality_is_metadata_only_and_never_broadcast_to_points(
    tmp_path: Path,
) -> None:
    observation = _open_qc(_write_qc_pixc(tmp_path / "line-quality.nc"))
    result = apply_qc(observation, profile="raw")

    assert "pixc_line_qual" not in observation.raw
    assert "pixc_line_qual" not in result.filtered
    assert "pixc_line_qual" not in result.decoded_flags
    metadata = observation.sources[0].variables["pixc_line_qual"]
    assert metadata.dimensions == ("num_pixc_lines",)
    assert metadata.shape == (2,)
