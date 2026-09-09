"""Create the Phase-4 Koshi PIXC classification comparison figure.

This script opens only the two existing, verified local PIXC files. It never
searches for or downloads data, and it deliberately does not choose a real
Koshi transect. Run it from the repository root::

    python examples/koshi_phase4_visual_review.py
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt

import swot_pixc_lab as spl

KOSHI_AOI = (86.87, 26.49, 87.20, 26.90)
KOSHI_PIXC_FILES = (
    Path("data/koshi_phase1")
    / "SWOT_L2_HR_PIXC_009_286_107L_20240114T081432_20240114T081443_PGD0_01.nc",
    Path("data/koshi_phase1")
    / "SWOT_L2_HR_PIXC_009_286_108L_20240114T081442_20240114T081453_PGD0_01.nc",
)
OUTPUT_PATH = Path("docs/phase4_koshi_water_classes.png")

EXPECTED_RAW_COUNT = 2_230_522
EXPECTED_DOCUMENTED_WATER_COUNT = 280_201
EXPECTED_CANDIDATE_COUNT = 280_198
DOCUMENTED_WATER_CLASSES = frozenset(range(3, 8))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Phase-4 visual regression failed: {message}")


def main() -> None:
    """Open cached data, verify Phase-3 counts, render, and save the figure."""

    missing = [path for path in KOSHI_PIXC_FILES if not path.is_file()]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "Required local Koshi PIXC file(s) are missing:\n"
            f"{missing_list}\n"
            "This script never discovers or downloads data. Restore the verified "
            "files under data/koshi_phase1, then rerun."
        )

    print("PHASE 4 VISUAL REVIEW: no discovery, download, transect, or width.")
    total_started = perf_counter()

    opened_at = perf_counter()
    observation = spl.open_pixc(
        KOSHI_PIXC_FILES,
        aoi=KOSHI_AOI,
        variables=spl.PHASE3_POINT_VARIABLES,
    )
    open_seconds = perf_counter() - opened_at
    _require(observation.pixel_count == EXPECTED_RAW_COUNT, "raw count changed")

    raw_class_counts = observation.classification_counts
    documented_water_count = sum(
        raw_class_counts.get(classification, 0)
        for classification in DOCUMENTED_WATER_CLASSES
    )
    _require(
        documented_water_count == EXPECTED_DOCUMENTED_WATER_COUNT,
        "documented classes 3-7 count changed",
    )

    qc_started = perf_counter()
    candidate = spl.apply_qc(observation, profile="channel_extent_candidate")
    qc_seconds = perf_counter() - qc_started
    candidate_count = int(candidate.filtered.sizes["points"])
    _require(
        candidate_count == EXPECTED_CANDIDATE_COUNT,
        "channel_extent_candidate count changed",
    )
    _require(
        candidate.profile.status == "experimental / unvalidated",
        "candidate profile is not visibly marked experimental",
    )

    render_started = perf_counter()
    figure, _axes = spl.plot_classification_comparison(
        observation,
        candidate=candidate,
        title="Koshi PIXC classification review — 2024-01-14",
        extent=KOSHI_AOI,
    )
    figure.canvas.draw()
    render_seconds = perf_counter() - render_started

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_started = perf_counter()
    figure.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    save_seconds = perf_counter() - save_started
    plt.close(figure)

    _require(OUTPUT_PATH.is_file(), "output image was not created")
    output_size = OUTPUT_PATH.stat().st_size
    _require(output_size > 0, "output image is empty")
    _require(output_size < 5 * 1024**2, "output image exceeds 5 MiB")

    ranges = observation.coordinate_ranges
    longitude_range = ranges["longitude"]
    latitude_range = ranges["latitude"]
    total_seconds = perf_counter() - total_started

    print("\nPlotted counts:")
    print(f"  raw classes 1-7:               {observation.pixel_count:,}")
    print(f"  documented water classes 3-7: {documented_water_count:,}")
    print(f"  channel_extent_candidate:      {candidate_count:,}")
    print(f"  raw classification counts:     {raw_class_counts}")
    print("\nImage coordinate extent (EPSG:4326):")
    print(f"  west/south/east/north: {KOSHI_AOI}")
    print("Raw point coordinate ranges:")
    print(f"  longitude: {longitude_range}")
    print(f"  latitude:  {latitude_range}")
    print("\nPerformance:")
    print(f"  open and exact AOI clip: {open_seconds:.3f} s")
    print(f"  Phase-3 candidate QC:    {qc_seconds:.3f} s")
    print(f"  figure render:           {render_seconds:.3f} s")
    print(f"  PNG save:                {save_seconds:.3f} s")
    print(f"  total:                   {total_seconds:.3f} s")
    print(f"\nSaved: {OUTPUT_PATH} ({output_size / (1024**2):.2f} MiB)")
    print(
        "No real transect was selected and no bank, branch, island, or channel "
        "width was calculated."
    )


if __name__ == "__main__":
    main()
