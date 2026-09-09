"""Annotate approved transects from independent, georeferenced RGB imagery.

The default view is blind reference mode: Sentinel RGB, the fixed transect,
and station ticks only.  PIXC context is available solely through an explicit
button, and Phase 5A.2 candidate output is never loaded or displayed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Button, TextBox
from pyproj import CRS, Transformer
from shapely.geometry import LineString, box, shape
from shapely.ops import substring, transform

from swot_pixc_lab import (
    PHASE3_POINT_VARIABLES,
    PixcLabError,
    ReferenceImageryMetadata,
    apply_qc,
    load_benchmark_config,
    load_transect_manifest,
    open_pixc,
    sample_transect,
    station_frame_hash,
)
from swot_pixc_lab.reference_annotation import (
    BlindReferenceAnnotationSession,
    ReferenceAnnotationError,
)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("koshi_phase5a3b_pilot_config.json")
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_WGS84 = CRS.from_epsg(4326)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--transect-manifest",
        type=Path,
        required=True,
        help="explicit owner-approved manifest; the proposed F manifest is refused",
    )
    parser.add_argument("--reference-packet", type=Path, required=True)
    parser.add_argument("--transect-id", required=True)
    parser.add_argument("--analyst-id", required=True)
    parser.add_argument("--annotation-id", required=True)
    parser.add_argument(
        "--candidate-index",
        type=int,
        help="zero-based candidate index; omit to use the recorded primary candidate",
    )
    parser.add_argument("--template", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _safe_identifier(value: str, *, label: str) -> str:
    if not _SAFE_IDENTIFIER.fullmatch(value) or value in {".", ".."}:
        raise SystemExit(
            f"{label} must be a safe filename atom containing only letters, "
            "numbers, dot, underscore, or hyphen."
        )
    return value


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    source = path.expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read {label} {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label.capitalize()} must be a JSON object: {source}")
    return value


def _selected_imagery(
    packet_path: Path,
    *,
    transect_id: str,
    candidate_index: int | None,
) -> tuple[ReferenceImageryMetadata, dict[str, Any]]:
    """Resolve either a Phase 5A.3c packet or compact legacy imagery mapping."""

    source = packet_path.expanduser().resolve()
    payload = _read_json(source, label="reference-imagery packet")
    packet_transect = payload.get("transect_id")
    if packet_transect is not None and packet_transect != transect_id:
        raise SystemExit(
            f"Reference packet belongs to {packet_transect!r}, not {transect_id!r}."
        )

    if "primary_candidate" in payload:
        if candidate_index is None:
            raw = payload.get("primary_candidate")
        else:
            candidates = payload.get("candidates")
            if not isinstance(candidates, list):
                raise SystemExit("Reference packet has no candidate list.")
            if not 0 <= candidate_index < len(candidates):
                raise SystemExit(
                    f"--candidate-index must be in [0, {len(candidates) - 1}]."
                )
            raw = candidates[candidate_index]
        if not isinstance(raw, dict):
            raise SystemExit(
                "The selected reference candidate is missing or malformed."
            )
        image_value = raw.get("rgb_geotiff_path")
        if not isinstance(image_value, str) or not image_value.strip():
            raise SystemExit(
                "The selected candidate has no georeferenced rgb_geotiff_path."
            )
        image_path = _resolve_relative(source.parent, image_value)
        item_ids = raw.get("item_ids")
        if (
            not isinstance(item_ids, list)
            or not item_ids
            or not all(isinstance(item, str) and item.strip() for item in item_ids)
        ):
            raise SystemExit("The selected candidate must record non-empty item_ids.")
        provider = _required_text(raw.get("provider"), "provider")
        collection = _required_text(raw.get("collection"), "collection")
        acquisition = _required_text(
            raw.get("acquisition_datetime"), "acquisition_datetime"
        )
        offset = _optional_finite_number(
            raw.get("signed_temporal_offset_hours", raw.get("signed_offset_hours")),
            "signed_temporal_offset_hours",
        )
        if offset is None:
            raise SystemExit(
                "The selected Phase 5A.3c candidate must record a finite signed "
                "temporal offset."
            )
        local_quality = raw.get("local_quality") or raw.get("local_quality_diagnostics")
        quality_note = (
            json.dumps(local_quality, sort_keys=True)
            if isinstance(local_quality, dict)
            else None
        )
        chip_hash = raw.get("chip_sha256")
        if not isinstance(chip_hash, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", chip_hash
        ):
            raise SystemExit(
                "The selected candidate must record a 64-character chip_sha256."
            )
        digest = hashlib.sha256()
        try:
            with image_path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise SystemExit(f"Cannot verify selected reference image: {exc}") from exc
        if digest.hexdigest().lower() != chip_hash.lower():
            raise SystemExit(
                "Selected reference image SHA-256 does not match the preparation "
                "packet. Restore or regenerate the chip before annotation."
            )
        hash_note = f"Selected chip SHA-256 verified: {chip_hash.lower()}."
        imagery = ReferenceImageryMetadata(
            status="selected independent Sentinel-2 L2A reference imagery",
            source_provider=provider,
            acquisition_date=acquisition,
            temporal_offset_hours=offset,
            cloud_quality_note=quality_note,
            local_image_path=image_path,
            source_identifier=f"{collection}:{','.join(item_ids)}",
            analyst_notes=hash_note,
        )
        return imagery, payload

    compact = payload.get("reference_imagery", payload)
    if not isinstance(compact, dict):
        raise SystemExit("Compact reference_imagery metadata is malformed.")
    local_value = compact.get("local_image_path")
    if not isinstance(local_value, str) or not local_value.strip():
        raise SystemExit("Compact imagery metadata has no local_image_path.")
    offset = _optional_finite_number(
        compact.get("temporal_offset_hours"), "temporal_offset_hours"
    )
    imagery = ReferenceImageryMetadata(
        status=_required_text(compact.get("status"), "status"),
        source_provider=_optional_text(compact.get("source_provider")),
        acquisition_date=_optional_text(compact.get("acquisition_date")),
        temporal_offset_hours=offset,
        cloud_quality_note=_optional_text(compact.get("cloud_quality_note")),
        local_image_path=_resolve_relative(source.parent, local_value),
        source_identifier=_optional_text(compact.get("source_identifier")),
        analyst_notes=_optional_text(compact.get("analyst_notes")),
    )
    return imagery, payload


def _resolve_relative(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"Selected reference candidate must record {label}.")
    return value


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SystemExit("Compact reference-imagery text fields must be text or null.")
    return value


def _optional_finite_number(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise SystemExit(f"{label} must be a finite number or null.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{label} must be a finite number or null.") from exc
    if not math.isfinite(number):
        raise SystemExit(f"{label} must be a finite number or null.")
    return number


def _open_rgb(path: Path) -> tuple[np.ndarray, CRS, tuple[float, float, float, float]]:
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - optional environment
        raise SystemExit(
            "Georeferenced annotation requires rasterio. Install the Phase 5A.3c "
            "reference-imagery optional dependencies."
        ) from exc
    try:
        with rasterio.open(path) as dataset:
            if dataset.crs is None:
                raise SystemExit(f"Reference RGB has no CRS: {path}")
            if dataset.count < 3:
                raise SystemExit(
                    f"Reference RGB must contain at least three bands: {path}"
                )
            rgb = np.moveaxis(dataset.read((1, 2, 3), masked=True), 0, -1)
            bounds = dataset.bounds
            crs = CRS.from_user_input(dataset.crs)
    except OSError as exc:
        raise SystemExit(
            f"Cannot open georeferenced reference RGB {path}: {exc}"
        ) from exc
    if np.issubdtype(rgb.dtype, np.integer) and rgb.dtype.itemsize == 1:
        display = np.ma.filled(rgb, 0)
    else:
        display = _display_stretch(rgb)
    return display, crs, (bounds.left, bounds.right, bounds.bottom, bounds.top)


def _display_stretch(rgb: np.ndarray) -> np.ndarray:
    """Create a display-only stretch; it is never used to select boundaries."""

    values = np.ma.asarray(rgb, dtype=np.float64)
    output = np.zeros(values.shape, dtype=np.float64)
    for band in range(3):
        current = values[..., band]
        finite = np.asarray(current.compressed())
        if finite.size == 0:
            continue
        low, high = np.percentile(finite, (2.0, 98.0))
        if high <= low:
            continue
        output[..., band] = np.clip((current.filled(low) - low) / (high - low), 0, 1)
    return output


def _line_in_crs(line: LineString, target_crs: CRS) -> LineString:
    transformer = Transformer.from_crs(_WGS84, target_crs, always_xy=True)
    return transform(transformer.transform, line)


def _plot_station_ticks(axes: Any, sample: Any, image_crs: CRS) -> None:
    to_metric = Transformer.from_crs(_WGS84, sample.local_crs, always_xy=True)
    to_image = Transformer.from_crs(sample.local_crs, image_crs, always_xy=True)
    metric_line = transform(to_metric.transform, sample.transect)
    length = sample.transect_length_m
    interval = 1_000.0 if length >= 4_000.0 else 500.0
    stations = list(np.arange(0.0, length, interval))
    if not stations or not math.isclose(stations[-1], length):
        stations.append(length)
    for station in stations:
        point = metric_line.interpolate(station)
        x, y = to_image.transform(point.x, point.y)
        axes.plot(x, y, marker="o", color="white", markeredgecolor="black", ms=4)
        axes.annotate(
            f"{station / 1000.0:g} km",
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            color="white",
            fontsize=7,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1},
        )


def _plot_aoi(axes: Any, aoi: tuple[float, ...], image_crs: CRS) -> None:
    line = _line_in_crs(box(*aoi).boundary, image_crs)
    x, y = line.xy
    axes.plot(x, y, color="#ffcc00", linestyle=":", linewidth=1.3, label="SWOT AOI")


def _format_preview(session: BlindReferenceAnnotationSession) -> str:
    preview = session.current_preview
    if preview is None:
        return ""
    measurement = preview.measurement
    interval_text = (
        "; ".join(
            f"{item.interval_id}: {item.start_station_m:.2f}–"
            f"{item.end_station_m:.2f} m (width {item.width_m:.2f} m)"
            for item in measurement.intervals
        )
        or "explicitly declared no visible wet interval"
    )
    imagery = preview.reference_imagery
    return (
        f"Intervals: {interval_text}\n"
        f"Total wetted width: {measurement.total_wetted_width_m:.2f} m; "
        f"outer span: {measurement.outer_wetted_span_m}; "
        f"internal dry gap: {measurement.total_internal_dry_gap_m:.2f} m\n"
        f"Reference: {imagery.source_provider}; {imagery.acquisition_date}; "
        f"signed offset {imagery.temporal_offset_hours} h; "
        f"PIXC context viewed: {preview.pixc_context_viewed}"
    )


def main() -> None:
    """Collect manual boundaries without showing PIXC/candidates by default."""

    args = _parse_args()
    transect_id = _safe_identifier(args.transect_id, label="--transect-id")
    analyst_id = _safe_identifier(args.analyst_id, label="--analyst-id")
    annotation_id = _safe_identifier(args.annotation_id, label="--annotation-id")
    config = load_benchmark_config(args.config)

    # Approval is checked from the explicit manifest before imagery or PIXC opens.
    manifest = load_transect_manifest(args.transect_manifest)
    matches = [item for item in manifest.transects if item.transect_id == transect_id]
    if len(matches) != 1:
        raise SystemExit(
            f"Transect {transect_id!r} was not found exactly once in "
            f"{args.transect_manifest}."
        )
    transect = matches[0]
    if not transect.approved:
        raise SystemExit(
            f"Transect {transect_id} is {transect.review_status!r}. A scientist "
            "must explicitly approve its geometry in a separate manifest first."
        )

    imagery, packet = _selected_imagery(
        args.reference_packet,
        transect_id=transect_id,
        candidate_index=args.candidate_index,
    )
    packet_geometry = packet.get("geometry")
    if packet_geometry is not None:
        try:
            imagery_line = shape(packet_geometry)
        except (TypeError, ValueError) as exc:
            raise SystemExit("Reference packet geometry is malformed.") from exc
        if (
            not isinstance(imagery_line, LineString)
            or imagery_line.wkt != transect.geometry.wkt
        ):
            raise SystemExit(
                "Reference packet geometry does not exactly match the approved "
                "manifest geometry. Regenerate imagery before annotation."
            )

    template = (
        args.template.expanduser().resolve()
        if args.template is not None
        else config.output_directory
        / "annotation_packets"
        / transect_id
        / "annotation_template.json"
    )
    if not template.is_file():
        raise SystemExit(
            f"Approved annotation template is missing: {template}. Regenerate "
            "packets from the approved manifest before annotating."
        )
    template_payload = _read_json(template, label="annotation template")
    expected_template_fields = {
        "benchmark_id": config.benchmark_id,
        "transect_id": transect_id,
        "review_status": transect.review_status,
    }
    stale_template_fields = {
        name: (template_payload.get(name), expected)
        for name, expected in expected_template_fields.items()
        if template_payload.get(name) != expected
    }
    if stale_template_fields:
        raise SystemExit(
            "Annotation template is stale for the approved manifest: "
            f"{stale_template_fields}. Regenerate packets before annotating."
        )
    output = config.annotation_directory / f"{annotation_id}.json"
    if output.exists() and not args.overwrite:
        raise SystemExit(
            f"Annotation already exists: {output}. Choose another annotation ID "
            "or pass --overwrite intentionally."
        )
    missing = [path for path in config.observation.local_files if not path.is_file()]
    if missing:
        raise SystemExit(
            "Required local PIXC file(s) are missing; this helper never downloads: "
            + ", ".join(str(path) for path in missing)
        )

    # PIXC is reconstructed only for the immutable Phase 4/5A.1 station frame.
    # It is not drawn until the analyst explicitly presses the reveal button.
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
    if template_payload.get("station_frame_hash") != station_frame_hash(sample):
        raise SystemExit(
            "Annotation template station frame is stale. Regenerate packets from "
            "this approved manifest and QC configuration before annotating."
        )
    try:
        session = BlindReferenceAnnotationSession(sample, transect, imagery)
    except ReferenceAnnotationError as exc:
        raise SystemExit(str(exc)) from exc

    rgb, image_crs, extent = _open_rgb(Path(imagery.local_image_path))
    figure, axes = plt.subplots(figsize=(12, 8), constrained_layout=False)
    figure.subplots_adjust(bottom=0.31)
    axes.imshow(rgb, extent=extent, origin="upper")
    image_line = _line_in_crs(transect.geometry, image_crs)
    line_x, line_y = image_line.xy
    axes.add_patch(
        FancyArrowPatch(
            (line_x[0], line_y[0]),
            (line_x[-1], line_y[-1]),
            arrowstyle="-|>",
            mutation_scale=15,
            color="#ff2d55",
            linewidth=2,
            shrinkA=0,
            shrinkB=0,
            label="approved transect: station 0 → L",
            zorder=3,
        )
    )
    _plot_station_ticks(axes, sample, image_crs)
    _plot_aoi(axes, config.observation.aoi, image_crs)
    # Keep the finite chip as the view domain even when the full AOI boundary
    # extends far outside it. This is especially important for T001F.
    axes.set_xlim(extent[0], extent[1])
    axes.set_ylim(extent[2], extent[3])
    axes.set_aspect("equal")
    blind_title = (
        f"{transect_id}: BLIND REFERENCE MODE\n"
        "Independent RGB + approved transect; no PIXC/candidate shown"
    )
    axes.set_title(blind_title)
    axes.legend(loc="upper right")

    status = figure.text(
        0.05,
        0.265,
        "Click ordered wet start/end boundaries near the transect. No edge snapping.",
        fontsize=9,
    )
    preview_text = figure.text(0.05, 0.19, "", fontsize=8, va="top")
    confidence_box = TextBox(
        figure.add_axes((0.10, 0.115, 0.22, 0.035)), "Confidence ", initial=""
    )
    notes_box = TextBox(
        figure.add_axes((0.44, 0.115, 0.50, 0.035)), "Notes ", initial=""
    )
    undo_button = Button(figure.add_axes((0.05, 0.04, 0.10, 0.045)), "Undo")
    reset_button = Button(figure.add_axes((0.17, 0.04, 0.10, 0.045)), "Reset")
    empty_button = Button(figure.add_axes((0.29, 0.04, 0.16, 0.045)), "Declare empty")
    pixc_button = Button(figure.add_axes((0.47, 0.04, 0.18, 0.045)), "Reveal PIXC")
    confirm_button = Button(figure.add_axes((0.67, 0.04, 0.27, 0.045)), "Preview")
    draft_artists: list[Any] = []
    pixc_artist: Any | None = None

    to_metric = Transformer.from_crs(_WGS84, sample.local_crs, always_xy=True)
    to_image = Transformer.from_crs(sample.local_crs, image_crs, always_xy=True)
    metric_line = transform(to_metric.transform, sample.transect)

    def current_text() -> tuple[str | None, str | None]:
        confidence = confidence_box.text.strip() or None
        notes = notes_box.text.strip() or None
        return confidence, notes

    def invalidate_label() -> None:
        confirm_button.label.set_text("Preview")
        preview_text.set_text("")

    def redraw_draft() -> None:
        for artist in draft_artists:
            artist.remove()
        draft_artists.clear()
        for pick in session.picks:
            point = metric_line.interpolate(pick.station_m)
            x, y = to_image.transform(point.x, point.y)
            draft_artists.extend(
                axes.plot(
                    x,
                    y,
                    marker="|",
                    color="black",
                    markersize=13,
                    markeredgewidth=2,
                    linestyle="none",
                )
            )
            axes.annotate(
                f"{pick.role}: {pick.station_m:.1f} m",
                (x, y),
                xytext=(4, -12),
                textcoords="offset points",
                color="black",
                fontsize=7,
            )
            draft_artists.append(axes.texts[-1])
        for start, end in session.completed_intervals:
            part = substring(metric_line, start, end)
            x, y = to_image.transform(*part.xy)
            draft_artists.extend(
                axes.plot(x, y, color="#00ff7f", linewidth=5, alpha=0.75)
            )
        figure.canvas.draw_idle()

    def on_click(event: Any) -> None:
        if event.inaxes is not axes or event.button is not MouseButton.LEFT:
            return
        if event.xdata is None or event.ydata is None:
            return
        try:
            pick = session.add_click(x=event.xdata, y=event.ydata, click_crs=image_crs)
        except ReferenceAnnotationError as exc:
            status.set_text(str(exc))
            figure.canvas.draw_idle()
            return
        invalidate_label()
        status.set_text(
            f"Recorded {pick.role} at {pick.station_m:.2f} m; click-to-line "
            f"offset {pick.distance_to_transect_m:.1f} m."
        )
        redraw_draft()

    def on_undo(_event: Any) -> None:
        removed = session.undo()
        invalidate_label()
        status.set_text(
            f"Removed boundary at {removed.station_m:.2f} m."
            if removed is not None
            else "No boundary to undo."
        )
        redraw_draft()

    def on_reset(_event: Any) -> None:
        session.reset()
        invalidate_label()
        status.set_text("All draft boundaries and empty declaration cleared.")
        redraw_draft()

    def on_empty(_event: Any) -> None:
        session.declare_empty()
        invalidate_label()
        status.set_text(
            "Explicit empty wet set drafted. Press Preview, review it, then confirm."
        )
        redraw_draft()

    def on_pixc(_event: Any) -> None:
        nonlocal pixc_artist
        show = not session.pixc_context_visible
        session.set_pixc_context_visible(show)
        invalidate_label()
        if show:
            longitudes = np.asarray(sample.pixels["longitude"].values, dtype=float)
            latitudes = np.asarray(sample.pixels["latitude"].values, dtype=float)
            valid = np.isfinite(longitudes) & np.isfinite(latitudes)
            transformer = Transformer.from_crs(_WGS84, image_crs, always_xy=True)
            x, y = transformer.transform(longitudes[valid], latitudes[valid])
            pixc_artist = axes.scatter(
                x,
                y,
                s=4,
                color="#00ffff",
                alpha=0.55,
                label="PIXC context (non-reference)",
                zorder=2,
            )
            pixc_button.label.set_text("Hide PIXC")
            axes.set_title(
                f"{transect_id}: PIXC CONTEXT REVEALED — NON-REFERENCE\n"
                "Independent RGB remains the reference; Phase 5A.2 candidate is "
                "never shown"
            )
            axes.legend(loc="upper right")
            status.set_text(
                "PIXC context explicitly revealed; it is not reference truth."
            )
        else:
            if pixc_artist is not None:
                pixc_artist.remove()
                pixc_artist = None
            pixc_button.label.set_text("Reveal PIXC")
            axes.set_title(blind_title)
            axes.legend(loc="upper right")
            status.set_text("Returned to blind reference view.")
        figure.canvas.draw_idle()

    def on_confirm(_event: Any) -> None:
        confidence, notes = current_text()
        if session.current_preview is None:
            try:
                session.preview(confidence=confidence, notes=notes)
            except PixcLabError as exc:
                status.set_text(str(exc))
                figure.canvas.draw_idle()
                return
            preview_text.set_text(_format_preview(session))
            confirm_button.label.set_text("CONFIRM AND SAVE")
            status.set_text(
                "Review all Phase 5A.1 values and imagery identity, then explicitly "
                "confirm."
            )
            figure.canvas.draw_idle()
            return
        try:
            session.confirm_and_save(
                template,
                output,
                annotation_id=annotation_id,
                analyst_id=analyst_id,
                confidence=confidence,
                notes=notes,
                confirmed=True,
            )
        except PixcLabError as exc:
            invalidate_label()
            status.set_text(str(exc))
            figure.canvas.draw_idle()
            return
        print(f"Saved confirmed independent manual annotation: {output}")
        plt.close(figure)

    figure.canvas.mpl_connect("button_press_event", on_click)
    undo_button.on_clicked(on_undo)
    reset_button.on_clicked(on_reset)
    empty_button.on_clicked(on_empty)
    pixc_button.on_clicked(on_pixc)
    confirm_button.on_clicked(on_confirm)
    plt.show()


if __name__ == "__main__":
    main()
