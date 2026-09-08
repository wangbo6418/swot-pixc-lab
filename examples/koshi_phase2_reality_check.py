"""Phase-2 raw PIXC ingestion reality check for the upper Koshi River.

This script opens two already-downloaded Phase-1 cache files, clips their raw
``/pixel_cloud`` points to the exact project AOI, and combines them without QC,
additional height correction or WSE transformation, averaging, or
deduplication. It never downloads data.

Run from the repository root::

    python examples/koshi_phase2_reality_check.py
"""

from __future__ import annotations

from pathlib import Path
from pprint import pprint

import numpy as np

from swot_pixc_lab import PixcObservation, open_pixc

KOSHI_AOI = (86.87, 26.49, 87.20, 26.90)
KOSHI_PIXC_FILES = (
    Path("data/koshi_phase1")
    / "SWOT_L2_HR_PIXC_009_286_107L_20240114T081432_20240114T081443_PGD0_01.nc",
    Path("data/koshi_phase1")
    / "SWOT_L2_HR_PIXC_009_286_108L_20240114T081442_20240114T081453_PGD0_01.nc",
)

# Reference counts from these exact PGD0_01 files. They make this script a
# regression check against unintended filtering, rather than just a demo that
# can print superficially plausible results.
EXPECTED_SOURCE_COUNTS = {
    "107L": {"points_before": 3_969_693, "points_after": 0},
    "108L": {"points_before": 5_889_878, "points_after": 2_230_522},
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Phase-2 reality-check invariant failed: {message}")


def _validate_provenance(observation: PixcObservation) -> None:
    """Verify exact-file counts and lossless source-order combination."""

    source_indices = np.asarray(observation.raw["source_index"].values)
    source_point_indices = np.asarray(observation.raw["source_point_index"].values)
    retained_total = sum(source.points_after for source in observation.sources)
    input_total = sum(source.points_before for source in observation.sources)

    _require(observation.pixel_count == retained_total, "combined count changed")
    _require(source_indices.size == retained_total, "source_index is incomplete")
    _require(
        source_point_indices.size == retained_total,
        "source_point_index is incomplete",
    )
    expected_input_total = sum(
        item["points_before"] for item in EXPECTED_SOURCE_COUNTS.values()
    )
    _require(input_total == expected_input_total, "unexpected total source-point count")

    for source in observation.sources:
        expected = EXPECTED_SOURCE_COUNTS.get(source.tile or "")
        _require(expected is not None, f"unexpected source tile {source.tile!r}")
        _require(
            source.points_before == expected["points_before"],
            f"{source.tile} source count changed",
        )
        _require(
            source.points_after == expected["points_after"],
            f"{source.tile} exact-AOI count changed",
        )

        belongs_to_source = source_indices == source.source_index
        original_indices = source_point_indices[belongs_to_source]
        _require(
            int(np.count_nonzero(belongs_to_source)) == source.points_after,
            f"{source.tile} provenance count does not match retained points",
        )
        _require(
            bool(np.all(np.diff(original_indices) > 0)),
            f"{source.tile} source point order or uniqueness was lost",
        )
        if original_indices.size:
            _require(
                int(original_indices[0]) >= 0
                and int(original_indices[-1]) < source.points_before,
                f"{source.tile} contains an invalid source point index",
            )

    for name, variable in observation.raw.data_vars.items():
        _require(
            variable.sizes.get("points") == retained_total,
            f"{name} does not contain one value per retained point",
        )

    longitude = np.asarray(observation.raw["longitude"].values)
    latitude = np.asarray(observation.raw["latitude"].values)
    west, south, east, north = KOSHI_AOI
    _require(
        bool(
            np.all(
                (longitude >= west)
                & (longitude <= east)
                & (latitude >= south)
                & (latitude <= north)
            )
        ),
        "a retained coordinate lies outside the exact AOI",
    )

    schemas = [
        tuple(
            (name, variable.dimensions, variable.dtype)
            for name, variable in source.variables.items()
        )
        for source in observation.sources
    ]
    _require(schemas[0] == schemas[1], "107L and 108L point schemas differ")


def _print_loaded_schema(observation: PixcObservation) -> None:
    """Print compact metadata copied from the first source file."""

    source = observation.sources[0]
    print(
        f"\nNetCDF structure: groups={source.groups}, "
        f"/pixel_cloud variables={len(source.variables)}, "
        f"dimensions={dict(source.dimensions)}"
    )
    print("Loaded /pixel_cloud fields (original dtype, fill, units):")
    for name in source.loaded_variables:
        variable = source.variables[name]
        print(
            f"  {name:22} dtype={variable.dtype:>3} "
            f"fill={variable.fill_value!r} "
            f"units={variable.attributes.get('units')!r}"
        )


def main() -> None:
    """Open, validate, and summarize the two cached Koshi PIXC tiles."""

    missing = [path for path in KOSHI_PIXC_FILES if not path.is_file()]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "Required Phase-1 cache file(s) are missing:\n"
            f"{missing_list}\n"
            "This script never downloads data. Restore these files under "
            "data/koshi_phase1 using the guarded Phase-1 workflow, then rerun."
        )

    print(
        "RAW PHASE-2 INGESTION ONLY: no QC, no additional height correction "
        "or WSE transformation; raw `height` is not WSE."
    )
    observation = open_pixc(KOSHI_PIXC_FILES, aoi=KOSHI_AOI)
    _validate_provenance(observation)

    print("\nSource table:")
    print(observation.source_table.to_string(index=False))
    _print_loaded_schema(observation)
    print("\nRaw exact-clip summary:")
    pprint(observation.summary(), sort_dicts=False)
    print("\nProvenance and no implicit QC/averaging/deduplication invariants: passed")


if __name__ == "__main__":
    main()
