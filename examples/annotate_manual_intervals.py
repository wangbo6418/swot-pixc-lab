"""Interactively record independent Phase 5A.1 wet-interval boundaries.

Every boundary is placed by the analyst on the station axis. The helper does
not infer or snap boundaries and does not display Phase 5A.2 candidate output.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button, TextBox

from swot_pixc_lab import (
    PHASE3_POINT_VARIABLES,
    apply_qc,
    load_benchmark_config,
    load_transect_manifest,
    open_pixc,
    plot_transect_classification,
    sample_transect,
    save_manual_annotation,
)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("koshi_phase5a3b_pilot_config.json")
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--transect-id", required=True)
    parser.add_argument("--analyst-id", required=True)
    parser.add_argument("--annotation-id", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace an annotation with the same output filename",
    )
    return parser.parse_args()


def _ordered_intervals(boundaries: list[float]) -> tuple[tuple[float, float], ...]:
    if len(boundaries) % 2:
        raise ValueError("Add an end boundary for the final interval.")
    intervals = tuple(
        (boundaries[index], boundaries[index + 1])
        for index in range(0, len(boundaries), 2)
    )
    for index, (start, end) in enumerate(intervals):
        if end <= start:
            raise ValueError("Each end boundary must be greater than its start.")
        if index and start <= intervals[index - 1][1]:
            raise ValueError(
                "Wet intervals must be ordered with a positive dry gap; combine "
                "touching intervals."
            )
    return intervals


def _safe_identifier(value: str, *, label: str) -> str:
    if not _SAFE_IDENTIFIER.fullmatch(value) or value in {".", ".."}:
        raise SystemExit(
            f"{label} must be a safe filename atom containing only letters, "
            "numbers, dot, underscore, or hyphen."
        )
    return value


def main() -> None:
    """Open one review packet's station frame and collect analyst click pairs."""

    args = _parse_args()
    if not args.annotation_id or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in args.annotation_id
    ):
        raise SystemExit(
            "--annotation-id must use only letters, digits, hyphens, and underscores."
        )
    transect_id = _safe_identifier(args.transect_id, label="--transect-id")
    analyst_id = _safe_identifier(args.analyst_id, label="--analyst-id")
    annotation_id = _safe_identifier(args.annotation_id, label="--annotation-id")
    config = load_benchmark_config(args.config)
    manifest = load_transect_manifest(config.transect_manifest_path)
    matches = [item for item in manifest.transects if item.transect_id == transect_id]
    if len(matches) != 1:
        raise SystemExit(
            f"Transect {transect_id!r} was not found exactly once in "
            f"{config.transect_manifest_path}."
        )
    transect = matches[0]
    if not transect.approved:
        raise SystemExit(
            f"Transect {transect.transect_id} is {transect.review_status!r}. "
            "A scientist must approve its geometry before annotation."
        )

    missing = [path for path in config.observation.local_files if not path.is_file()]
    if missing:
        raise SystemExit(
            "Required local PIXC file(s) are missing; this helper never downloads: "
            + ", ".join(str(path) for path in missing)
        )
    template = (
        config.output_directory
        / "annotation_packets"
        / transect.transect_id
        / "annotation_template.json"
    )
    if not template.is_file():
        raise SystemExit(
            f"Annotation template is missing: {template}. Run the pilot packet "
            "builder after approving the transect, then retry."
        )
    try:
        template_payload = json.loads(template.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"Cannot read annotation template {template}: {error}"
        ) from error
    expected_template_fields = {
        "benchmark_id": config.benchmark_id,
        "transect_id": transect.transect_id,
        "review_status": transect.review_status,
    }
    stale_fields = {
        name: (template_payload.get(name), expected)
        for name, expected in expected_template_fields.items()
        if template_payload.get(name) != expected
    }
    if stale_fields:
        raise SystemExit(
            f"Annotation template is stale for the approved manifest: {stale_fields}. "
            "Regenerate packets before annotating."
        )
    output = config.annotation_directory / f"{annotation_id}.json"
    if output.exists() and not args.overwrite:
        raise SystemExit(
            f"Annotation already exists: {output}. Choose a new annotation ID or "
            "pass --overwrite intentionally."
        )

    observation = open_pixc(
        config.observation.local_files,
        aoi=config.observation.aoi,
        variables=PHASE3_POINT_VARIABLES,
    )
    qc_result = apply_qc(observation, profile=config.qc_profile)
    sample = sample_transect(
        qc_result,
        transect.geometry,
        corridor_half_width_m=transect.corridor_half_width_m,
    )

    axes = plot_transect_classification(
        sample,
        title=(
            f"{transect.transect_id}: analyst-selected wet intervals\n"
            "CLICK EVERY BOUNDARY — NO INFERENCE OR SNAPPING"
        ),
    )
    axes.set_xlim(0.0, sample.transect_length_m)
    if sample.selected_pixel_count == 0:
        axes.text(
            0.5,
            0.5,
            "NO PIXC CLASSIFICATION EVIDENCE IN THIS CORRIDOR",
            ha="center",
            va="center",
            transform=axes.transAxes,
            fontweight="bold",
        )
    figure = axes.figure
    figure.subplots_adjust(bottom=0.30)
    boundaries: list[float] = []
    boundary_artists: list[object] = []
    interval_artists: list[object] = []
    previewed: tuple[tuple[float, float], ...] | None = None
    pending_empty = False

    status = figure.text(
        0.05,
        0.255,
        "Click ordered start/end pairs. Candidate inference is intentionally hidden.",
        fontsize=9,
    )
    imagery = config.reference_imagery
    figure.text(
        0.05,
        0.225,
        "Independent imagery: "
        f"{imagery.status}; provider={imagery.source_provider or 'pending'}; "
        f"date={imagery.acquisition_date or 'pending'}",
        fontsize=8,
    )
    confidence_box = TextBox(
        figure.add_axes((0.10, 0.14, 0.22, 0.04)),
        "Confidence ",
        initial="",
    )
    notes_box = TextBox(
        figure.add_axes((0.44, 0.14, 0.50, 0.04)),
        "Notes ",
        initial="",
    )
    undo_button = Button(figure.add_axes((0.05, 0.06, 0.12, 0.05)), "Undo")
    reset_button = Button(figure.add_axes((0.19, 0.06, 0.12, 0.05)), "Reset")
    empty_button = Button(figure.add_axes((0.33, 0.06, 0.18, 0.05)), "Declare empty")
    confirm_button = Button(figure.add_axes((0.55, 0.06, 0.39, 0.05)), "Preview")

    def redraw() -> None:
        nonlocal boundary_artists, interval_artists
        for artist in (*boundary_artists, *interval_artists):
            artist.remove()  # type: ignore[attr-defined]
        boundary_artists = [
            axes.axvline(value, color="black", linestyle="--", linewidth=1.2)
            for value in boundaries
        ]
        interval_artists = []
        if len(boundaries) % 2 == 0:
            for start, end in _ordered_intervals(boundaries):
                interval_artists.append(
                    axes.axvspan(start, end, color="#2ca02c", alpha=0.18)
                )
        figure.canvas.draw_idle()

    def invalidate_preview() -> None:
        nonlocal previewed, pending_empty
        previewed = None
        pending_empty = False
        confirm_button.label.set_text("Preview")

    def on_click(event: object) -> None:
        if (
            getattr(event, "inaxes", None) is not axes
            or getattr(event, "button", None) is not MouseButton.LEFT
        ):
            return
        station = getattr(event, "xdata", None)
        if station is None or not math.isfinite(station):
            return
        station = float(station)
        if not 0.0 <= station <= sample.transect_length_m:
            status.set_text(
                "Boundary is outside the finite transect; click inside [0, L]. "
                "Nothing was snapped."
            )
            figure.canvas.draw_idle()
            return
        trial = [*boundaries, station]
        try:
            if len(trial) % 2 == 0:
                _ordered_intervals(trial)
            elif len(trial) > 1 and station <= trial[-2]:
                raise ValueError("An end boundary must be right of its start.")
        except ValueError as error:
            status.set_text(str(error))
            figure.canvas.draw_idle()
            return
        boundaries.append(station)
        invalidate_preview()
        status.set_text(f"Recorded boundary at {station:.3f} m.")
        redraw()

    def on_undo(_event: object) -> None:
        if boundaries:
            removed = boundaries.pop()
            status.set_text(f"Removed boundary at {removed:.3f} m.")
        else:
            status.set_text("No boundary to undo.")
        invalidate_preview()
        redraw()

    def on_reset(_event: object) -> None:
        boundaries.clear()
        invalidate_preview()
        status.set_text("All draft boundaries cleared.")
        redraw()

    def on_empty(_event: object) -> None:
        nonlocal previewed, pending_empty
        boundaries.clear()
        redraw()
        previewed = ()
        pending_empty = True
        confirm_button.label.set_text("SAVE CONFIRMED EMPTY")
        status.set_text(
            "Preview: explicit empty wet set. Review, then click SAVE CONFIRMED EMPTY."
        )
        figure.canvas.draw_idle()

    def on_confirm(_event: object) -> None:
        nonlocal previewed, pending_empty
        try:
            intervals = _ordered_intervals(boundaries)
        except ValueError as error:
            status.set_text(str(error))
            figure.canvas.draw_idle()
            return
        if not intervals and not pending_empty:
            status.set_text(
                "No intervals are drawn. Use Declare empty to confirm an empty "
                "reference."
            )
            figure.canvas.draw_idle()
            return
        if previewed != intervals:
            previewed = intervals
            pending_empty = not intervals
            confirm_button.label.set_text("SAVE CONFIRMED")
            interval_text = (
                ", ".join(f"[{start:.3f}, {end:.3f}]" for start, end in intervals)
                or "explicit empty wet set"
            )
            status.set_text(
                f"Preview: {interval_text}. Review, then click SAVE CONFIRMED."
            )
            figure.canvas.draw_idle()
            return
        save_manual_annotation(
            sample,
            template,
            output,
            annotation_id=annotation_id,
            analyst_id=analyst_id,
            manual_intervals=intervals,
            confidence=confidence_box.text.strip() or None,
            notes=notes_box.text.strip() or None,
            reference_imagery=config.reference_imagery,
            confirmed=True,
        )
        print(f"Saved independent manual annotation: {output}")
        plt.close(figure)

    figure.canvas.mpl_connect("button_press_event", on_click)
    undo_button.on_clicked(on_undo)
    reset_button.on_clicked(on_reset)
    empty_button.on_clicked(on_empty)
    confirm_button.on_clicked(on_confirm)
    plt.show()


if __name__ == "__main__":
    main()
