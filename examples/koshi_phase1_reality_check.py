"""Phase-1 real-data reality check for the upper Koshi River, Nepal.

Running this script without arguments performs anonymous CMR discovery only.
Downloading requires both an explicit one-day selection and ``--download``;
the broad two-year collection is never passed to the downloader.

Examples
--------
Discover and inspect all Version D matches::

    python examples/koshi_phase1_reality_check.py

Inspect one day's small result without downloading::

    python examples/koshi_phase1_reality_check.py \
        --observation-date 2024-01-01

Download and verify only that explicitly selected day::

    python examples/koshi_phase1_reality_check.py \
        --observation-date 2024-01-01 --download
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from swot_pixc_lab import (
    DEFAULT_PIXC_COLLECTION_CONCEPT_ID,
    GranuleRecord,
    PixcCollection,
)

KOSHI_AOI = (86.87, 26.49, 87.20, 26.90)
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"
PREFERRED_PASS_TILE = (286, "108L")
DEFAULT_CACHE_DIR = Path("data/koshi_phase1")
DEFAULT_MAX_DAILY_GRANULES = 8


def _observation_date(record: GranuleRecord) -> date | None:
    return record.observation_start.date() if record.observation_start else None


def _format_values(values: Sequence[object]) -> str:
    return ", ".join(str(value) for value in values) if values else "none"


def _format_size(value: object) -> str:
    if value is None or value is pd.NA:
        return "unknown"
    try:
        size_bytes = int(value)
    except (TypeError, ValueError):
        return "unknown"
    return f"{size_bytes / (1024 * 1024):.1f} MiB"


def _print_granule_table(collection: PixcCollection) -> None:
    if not len(collection):
        print("Granule table: none")
        return

    table = collection.table.sort_values(
        ["observation_start", "cycle", "pass", "tile", "filename"],
        na_position="last",
    )
    compact = pd.DataFrame(
        {
            "date": table["observation_start"].dt.strftime("%Y-%m-%d"),
            "cycle": table["cycle"],
            "pass": table["pass"],
            "tile": table["tile"],
            "filename": table["filename"],
            "size": table["size_bytes"].map(_format_size),
            "revision": table["revision_id"],
        }
    )
    compact = compact.astype(object).where(compact.notna(), "unknown")
    print("Granule table (CMR revision ID is shown as revision):")
    print(compact.to_string(index=False))


def _logical_observation_groups(
    records: Sequence[GranuleRecord],
) -> list[tuple[tuple[object, ...], list[GranuleRecord]]]:
    """Find distinct granules sharing the same acquisition/track metadata."""

    grouped: dict[tuple[object, ...], list[GranuleRecord]] = defaultdict(list)
    for record in records:
        if (
            record.observation_start is None
            or record.cycle is None
            or record.pass_number is None
            or record.tile is None
        ):
            continue
        key = (
            record.observation_start,
            record.observation_end,
            record.cycle,
            record.pass_number,
            record.tile,
        )
        grouped[key].append(record)
    return [(key, group) for key, group in grouped.items() if len(group) > 1]


def _print_duplicate_candidates(records: Sequence[GranuleRecord]) -> None:
    groups = _logical_observation_groups(records)
    print("Duplicate/reprocessed observation candidates:")
    if not groups:
        print("  none in the deduplicated CMR response")
        return

    for key, group in groups:
        start, end, cycle, pass_number, tile = key
        print(
            "  "
            f"start={start}, end={end}, cycle={cycle}, pass={pass_number}, "
            f"tile={tile}"
        )
        for record in group:
            print(
                f"    {record.filename} "
                f"(concept={record.concept_id}, revision={record.revision_id})"
            )


def _print_metadata_warnings(collection: PixcCollection) -> None:
    by_warning: dict[str, list[str]] = defaultdict(list)
    for record in collection.records:
        for warning in record.metadata_warnings:
            by_warning[warning].append(record.filename or record.identity)

    print("Metadata warnings:")
    if not by_warning and not collection.search_notes:
        print("  none")
        return

    for note in collection.search_notes:
        print(f"  AOI normalization: {note}")
    for warning, filenames in sorted(by_warning.items()):
        preview = ", ".join(filenames[:3])
        if len(filenames) > 3:
            preview += f", ... (+{len(filenames) - 3})"
        print(f"  {warning} [{len(filenames)} granule(s): {preview}]")


def _print_summary(collection: PixcCollection, *, label: str) -> None:
    records = collection.records
    dates = sorted(
        value for value in {_observation_date(record) for record in records} if value
    )
    cycles = sorted(
        value for value in {record.cycle for record in records} if value is not None
    )
    passes = sorted(
        value
        for value in {record.pass_number for record in records}
        if value is not None
    )
    tiles = sorted(value for value in {record.tile for record in records} if value)
    pass_tiles = {
        (record.pass_number, (record.tile or "").upper()) for record in records
    }

    print(f"\n=== {label} ===")
    print(f"Collection: {collection.collection_concept_id}")
    print(f"Total granules: {len(collection)}")
    print(f"Unique observation dates ({len(dates)}): {_format_values(dates)}")
    print(f"Unique cycles ({len(cycles)}): {_format_values(cycles)}")
    print(f"Unique passes ({len(passes)}): {_format_values(passes)}")
    print(f"Unique tiles ({len(tiles)}): {_format_values(tiles)}")
    for pass_number, tile in (PREFERRED_PASS_TILE, (564, "107R")):
        occurrence = "yes" if (pass_number, tile) in pass_tiles else "no"
        print(f"Contains pass/tile {pass_number}/{tile}: {occurrence}")
    _print_granule_table(collection)
    _print_duplicate_candidates(records)
    _print_metadata_warnings(collection)


def _suggest_observation(records: Sequence[GranuleRecord]) -> GranuleRecord | None:
    dated = [record for record in records if _observation_date(record) is not None]
    if not dated:
        return None
    preferred = [
        record
        for record in dated
        if (record.pass_number, (record.tile or "").upper()) == PREFERRED_PASS_TILE
    ]
    candidates = preferred or dated
    date_counts = Counter(_observation_date(record) for record in dated)
    return min(
        candidates,
        key=lambda record: (
            date_counts[_observation_date(record)],
            _observation_date(record),
            record.filename or record.identity,
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observation-date",
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="re-run discovery for exactly this date; does not download by itself",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="download only the result of --observation-date",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"download cache (default: {DEFAULT_CACHE_DIR.as_posix()})",
    )
    parser.add_argument(
        "--max-daily-granules",
        type=int,
        default=DEFAULT_MAX_DAILY_GRANULES,
        help="refuse a larger one-date download unless explicitly raised",
    )
    args = parser.parse_args()
    if args.download and args.observation_date is None:
        parser.error("--download requires an explicit --observation-date")
    if args.max_daily_granules < 1:
        parser.error("--max-daily-granules must be positive")
    return args


def main() -> None:
    args = _parse_args()
    broad = PixcCollection.search(
        aoi=KOSHI_AOI,
        start_date=START_DATE,
        end_date=END_DATE,
        collection_concept_id=DEFAULT_PIXC_COLLECTION_CONCEPT_ID,
    )
    _print_summary(
        broad,
        label=f"Koshi broad discovery: {START_DATE} through {END_DATE}",
    )

    suggestion = _suggest_observation(broad.records)
    if suggestion is not None:
        suggested_date = _observation_date(suggestion)
        print(
            "Suggested metadata-based one-date check: "
            f"{suggested_date} (pass={suggestion.pass_number}, "
            f"tile={suggestion.tile}, filename={suggestion.filename})"
        )
        print("This is not a PIXC science-quality assessment.")

    if args.observation_date is None:
        print(
            "No download requested. Re-run with --observation-date YYYY-MM-DD "
            "to inspect one day; add --download only after reviewing that result."
        )
        return

    broad_dates = {
        value for record in broad.records if (value := _observation_date(record))
    }
    if args.observation_date not in broad_dates:
        raise SystemExit(
            f"Refusing one-date operation: {args.observation_date} is not an "
            "observation-start date in the broad discovery result."
        )

    selected_date = args.observation_date.isoformat()
    daily = PixcCollection.search(
        aoi=KOSHI_AOI,
        start_date=selected_date,
        end_date=selected_date,
        collection_concept_id=DEFAULT_PIXC_COLLECTION_CONCEPT_ID,
    )
    _print_summary(daily, label=f"Koshi one-date discovery: {selected_date}")

    if not args.download:
        print("One-date result inspected; no download requested.")
        return
    if not len(daily):
        raise SystemExit("Refusing download because the one-date search is empty.")
    if len(daily) > args.max_daily_granules:
        raise SystemExit(
            f"Refusing to download {len(daily)} granules; the safety limit is "
            f"{args.max_daily_granules}. Inspect the table, then explicitly raise "
            "--max-daily-granules if appropriate."
        )

    print(f"Downloading only the {len(daily)} one-date granule(s) listed above.")
    daily.download(
        args.cache_dir,
        verify="auto",
        persist_credentials=False,
    )
    local = daily.resolve_local(args.cache_dir, verify="auto")
    print("Verified local NetCDF file(s):")
    for path in local.paths:
        print(f"  {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
