"""Phase-3 quality-control reality check for the upper Koshi River.

This script opens only the two already-downloaded Phase-2 NetCDF files. It
never searches for or downloads data. Run it from the repository root::

    python examples/koshi_phase3_reality_check.py
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from pprint import pprint
from typing import Any

import numpy as np
import xarray as xr

import swot_pixc_lab as spl

KOSHI_AOI = (86.87, 26.49, 87.20, 26.90)
KOSHI_PIXC_FILES = (
    Path("data/koshi_phase1")
    / "SWOT_L2_HR_PIXC_009_286_107L_20240114T081432_20240114T081443_PGD0_01.nc",
    Path("data/koshi_phase1")
    / "SWOT_L2_HR_PIXC_009_286_108L_20240114T081442_20240114T081453_PGD0_01.nc",
)

# Use the package-owned selection when available. The explicit fallback keeps
# this standalone reality check readable in older Phase-2 environments.
PHASE3_POINT_VARIABLES = getattr(
    spl,
    "PHASE3_POINT_VARIABLES",
    (
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
        "ancillary_surface_classification_flag",
    ),
)

EXPECTED_SOURCE_COUNTS = {
    "107L": {"points_before": 3_969_693, "points_after": 0},
    "108L": {"points_before": 5_889_878, "points_after": 2_230_522},
}
EXPECTED_PROFILE_COUNTS = {
    "raw": 2_230_522,
    "bo_legacy_strict": 0,
    "channel_extent_candidate": 280_198,
}
QUALITY_VARIABLES = (
    "classification_qual",
    "geolocation_qual",
    "interferogram_qual",
    "sig0_qual",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Phase-3 reality-check invariant failed: {message}")


def _valid_values(dataset: xr.Dataset, name: str) -> np.ndarray[Any, Any]:
    values = np.asarray(dataset[name].values)
    valid = np.ones(values.shape, dtype=np.bool_)
    if np.issubdtype(values.dtype, np.floating):
        valid &= np.isfinite(values)
    fill = dataset[name].attrs.get("_FillValue")
    if fill is not None:
        valid &= values != fill
    return values[valid]


def _classification_counts(dataset: xr.Dataset) -> dict[int, int]:
    values = _valid_values(dataset, "classification")
    labels, counts = np.unique(values, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts, strict=True)}


def _numeric_summary(dataset: xr.Dataset, name: str) -> dict[str, float | int | None]:
    if name not in dataset:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
        }
    values = _valid_values(dataset, name).astype(np.float64, copy=False)
    if not values.size:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
        }
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


def _validate_sources(observation: spl.PixcObservation) -> None:
    _require(
        observation.pixel_count == EXPECTED_PROFILE_COUNTS["raw"], "raw count changed"
    )
    for source in observation.sources:
        expected = EXPECTED_SOURCE_COUNTS.get(source.tile or "")
        _require(expected is not None, f"unexpected source tile {source.tile!r}")
        _require(
            source.points_before == expected["points_before"],
            f"{source.tile} source count changed",
        )
        _require(
            source.points_after == expected["points_after"],
            f"{source.tile} AOI count changed",
        )


def _validate_result(result: Any) -> None:
    name = result.profile.name
    expected = EXPECTED_PROFILE_COUNTS[name]
    retained = int(np.count_nonzero(np.asarray(result.mask)))
    _require(
        retained == expected, f"{name} mask retained {retained}, expected {expected}"
    )
    _require(
        result.filtered.sizes["points"] == expected,
        f"{name} filtered count differs from mask",
    )
    for provenance_name in ("source_index", "source_point_index"):
        _require(provenance_name in result.filtered, f"{name} lost {provenance_name}")


def _print_profile(result: Any, observation: spl.PixcObservation) -> None:
    name = result.profile.name
    print(f"\n=== {name} ===")
    pprint(result.summary, sort_dicts=False)
    print("Classification before:", observation.classification_counts)
    print("Classification after: ", _classification_counts(result.filtered))
    print("Independent rejection counts:")
    pprint(dict(result.reason_counts), sort_dicts=False)
    print("Incremental rejection counts:")
    pprint(dict(result.incremental_reason_counts), sort_dicts=False)

    source_indices = np.asarray(result.filtered["source_index"].values)
    source_counts = {
        source.tile or "unknown": int(
            np.count_nonzero(source_indices == source.source_index)
        )
        for source in observation.sources
    }
    print("Retained source/tile counts:", source_counts)
    print("Raw ellipsoidal height:", _numeric_summary(result.filtered, "height"))

    height_reference = result.height_reference
    print(
        "Height reference:",
        {
            "status": height_reference.status,
            "variable": height_reference.variable_name,
            "formula": height_reference.formula,
        },
    )
    if height_reference.variable_name:
        print(
            f"{height_reference.variable_name}:",
            _numeric_summary(result.filtered, height_reference.variable_name),
        )


def _print_quality_frequencies(result: Any) -> None:
    print("\nDecoded quality-bit frequencies in the raw exact-AOI observation:")
    for variable_name in QUALITY_VARIABLES:
        frequencies: Mapping[str, int] = result.decoded_flags[
            variable_name
        ].frequencies()
        nonzero = {name: int(count) for name, count in frequencies.items() if count}
        print(f"  {variable_name}: {nonzero or 'all decoded flags are zero'}")
    print("  (Zero-frequency named flags are omitted from this compact display.)")


def _print_water_fraction_comparison(dataset: xr.Dataset) -> None:
    classification = np.asarray(dataset["classification"].values)
    water_fraction = np.asarray(dataset["water_frac"].values)
    class_valid = classification != dataset["classification"].attrs.get(
        "_FillValue", 255
    )
    water_valid = np.isfinite(water_fraction)
    water_fill = dataset["water_frac"].attrs.get("_FillValue")
    if water_fill is not None:
        water_valid &= water_fraction != water_fill

    class_four = class_valid & (classification == 4)
    high_fraction = water_valid & (water_fraction >= 0.90)
    named_water = class_valid & np.isin(classification, (3, 4, 5, 6, 7))
    potentially_useful_outside_four = named_water & ~class_four
    print("\nRaw class / water-fraction comparison:")
    print(f"  classification == 4: {int(np.count_nonzero(class_four)):,}")
    print(f"  water_frac >= 0.90: {int(np.count_nonzero(high_fraction)):,}")
    print(f"  both: {int(np.count_nonzero(class_four & high_fraction)):,}")
    print(
        "  named water classes outside class 4: "
        f"{int(np.count_nonzero(potentially_useful_outside_four)):,}"
    )
    print("  water_frac distribution by classification:")
    print("    class      n       min       p05    median       p95       max  >=0.90")
    for label in range(1, 8):
        selected = class_valid & water_valid & (classification == label)
        values = water_fraction[selected].astype(np.float64, copy=False)
        quantiles = np.quantile(values, (0.05, 0.50, 0.95))
        print(
            f"    {label:>5} {values.size:>7,} {np.min(values):>9.4f} "
            f"{quantiles[0]:>9.4f} {quantiles[1]:>9.4f} "
            f"{quantiles[2]:>9.4f} {np.max(values):>9.4f} "
            f"{np.count_nonzero(values >= 0.90):>7,}"
        )


def main() -> None:
    missing = [path for path in KOSHI_PIXC_FILES if not path.is_file()]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "Required local Koshi PIXC file(s) are missing:\n"
            f"{missing_list}\n"
            "This script never downloads data. Restore the verified files under "
            "data/koshi_phase1, then rerun."
        )

    print("PHASE 3 LOCAL REALITY CHECK: no discovery, download, or corrected WSE.")
    observation = spl.open_pixc(
        KOSHI_PIXC_FILES,
        aoi=KOSHI_AOI,
        variables=PHASE3_POINT_VARIABLES,
    )
    _validate_sources(observation)

    results = [
        spl.apply_qc(observation, profile=profile)
        for profile in EXPECTED_PROFILE_COUNTS
    ]
    for result in results:
        _validate_result(result)
        _print_profile(result, observation)

    _print_quality_frequencies(results[0])
    _print_water_fraction_comparison(observation.raw)
    print("\nAll established source, profile, and provenance regressions passed.")


if __name__ == "__main__":
    main()
