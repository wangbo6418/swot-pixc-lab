"""Memory-conscious scientific plots for SWOT PIXC point observations.

The functions in this module are diagnostic views only.  They do not alter
PIXC values, infer river boundaries, interpolate samples, or calculate width.
Matplotlib is imported lazily so the non-visualization package remains usable
without the optional visualization dependency.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from ._pixel_input import (
    PixelData,
    ResolvedPixelInput,
    provenance_reference_indices,
    resolve_pixel_input,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .transect import TransectSample
    from .width import ExplicitIntervalWidthResult


CLASSIFICATION_LABELS: Mapping[int, str] = {
    1: "land",
    2: "land near water",
    3: "water near land",
    4: "open water",
    5: "dark water",
    6: "low-coherence water near land",
    7: "open low-coherence water",
}
"""Human-readable display labels for the official PIXC enumeration."""

CLASSIFICATION_COLORS: Mapping[int, str] = {
    1: "#7f7f7f",
    2: "#bcbd22",
    3: "#17becf",
    4: "#1f77b4",
    5: "#542788",
    6: "#ff7f0e",
    7: "#e377c2",
}
"""Stable, explicit colors for official PIXC classifications 1--7."""

SUPPORTED_COLOR_VARIABLES = frozenset(
    {"classification", "height", "height_egm2008", "water_frac", "source_index"}
)


def plot_pixc_map(
    data: PixelData,
    *,
    color_by: str = "classification",
    ax: Axes | None = None,
    title: str | None = None,
    point_size: float = 0.35,
    alpha: float = 0.85,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    extent: Sequence[float] | None = None,
    show_legend: bool = True,
    show_colorbar: bool = True,
) -> Axes:
    """Plot raw, QC-filtered, or dataset PIXC points in geographic plan view.

    Parameters
    ----------
    data
        A :class:`~swot_pixc_lab.PixcObservation`,
        :class:`~swot_pixc_lab.QCResult`, or one-dimensional ``xarray.Dataset``.
    color_by
        One of ``classification``, ``height``, ``height_egm2008``,
        ``water_frac``, or ``source_index``.  A decoded QC flag from a
        ``QCResult`` can be selected as ``"quality_variable:flag_meaning"``.
    extent
        Optional ``(west, south, east, north)`` limits in EPSG:4326.

    Notes
    -----
    Every valid point is submitted to one rasterized Matplotlib collection;
    this function never silently samples or constructs GeoPandas geometries.
    Rasterization keeps PDF/SVG output compact but does not change the data.
    Decoded flags on a ``QCResult`` are shown only for that result's retained
    pixels; use a raw-profile ``QCResult`` to inspect the full raw domain.
    """

    _validate_style(point_size=point_size, alpha=alpha)
    resolved = resolve_pixel_input(data)
    axes = _axes_or_new(ax)
    _plot_map_layer(
        resolved,
        color_by=color_by,
        ax=axes,
        point_size=point_size,
        alpha=alpha,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        show_legend=show_legend,
        show_colorbar=show_colorbar,
    )
    _configure_geographic_axes(axes, resolved.dataset, extent=extent)
    axes.set_title(_plot_title(resolved, color_by=color_by, requested=title))
    return axes


def plot_classification_comparison(
    observation: PixelData,
    candidate: PixelData | None = None,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (15.0, 5.5),
    point_size: float = 0.25,
    alpha: float = 0.85,
    extent: Sequence[float] | None = None,
) -> tuple[Figure, NDArray[Any]]:
    """Compare raw classes, documented water classes, and the QC candidate.

    The middle panel selects official water classifications 3--7.  It is a
    visual diagnostic and is not a validated river-boundary definition.
    ``candidate`` defaults to applying ``channel_extent_candidate`` to a
    supplied ``PixcObservation``.
    """

    _validate_style(point_size=point_size, alpha=alpha)
    raw_input, candidate_input = _comparison_inputs(observation, candidate)
    raw = raw_input.dataset
    if "classification" not in raw:
        raise KeyError("Classification comparison requires 'classification'.")

    classification = np.asarray(raw["classification"].values)
    raw_selection = _valid_data_mask(raw["longitude"])
    raw_selection &= _valid_data_mask(raw["latitude"])
    raw_selection &= _valid_data_mask(raw["classification"])
    raw_selection &= np.isin(classification, tuple(CLASSIFICATION_LABELS))
    water_selection = raw_selection & np.isin(classification, tuple(range(3, 8)))
    candidate_dataset = candidate_input.dataset
    candidate_selection = _valid_data_mask(candidate_dataset["longitude"])
    candidate_selection &= _valid_data_mask(candidate_dataset["latitude"])
    candidate_selection &= _valid_data_mask(candidate_dataset["classification"])
    candidate_selection &= np.isin(
        np.asarray(candidate_dataset["classification"].values),
        tuple(CLASSIFICATION_LABELS),
    )
    candidate_count = int(np.count_nonzero(candidate_selection))
    water_count = int(np.count_nonzero(water_selection))
    raw_count = int(np.count_nonzero(raw_selection))

    plt, _, _ = _import_matplotlib()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_array = np.asarray(axes, dtype=object)

    _plot_map_layer(
        raw_input,
        color_by="classification",
        ax=axes_array[0],
        point_size=point_size,
        alpha=alpha,
        show_legend=False,
        show_colorbar=False,
    )
    water_input = ResolvedPixelInput(
        dataset=raw[["longitude", "latitude", "classification"]].isel(
            points=np.flatnonzero(water_selection)
        ),
        sources=raw_input.sources,
        profile_name="documented_water_classes_3_7",
        profile_label="Documented water classes 3--7",
        profile_status="visual diagnostic / not a validated boundary",
        qc_result=None,
    )
    _plot_map_layer(
        water_input,
        color_by="classification",
        ax=axes_array[1],
        point_size=point_size,
        alpha=alpha,
        show_legend=False,
        show_colorbar=False,
    )
    _plot_map_layer(
        candidate_input,
        color_by="classification",
        ax=axes_array[2],
        point_size=point_size,
        alpha=alpha,
        show_legend=False,
        show_colorbar=False,
    )

    candidate_status = candidate_input.profile_status or "experimental / unvalidated"
    if "experimental" not in candidate_status.lower():
        candidate_status = f"{candidate_status}; experimental / unvalidated"
    panel_titles = (
        f"A. Raw context: classes 1--7\n{raw_count:,} pixels",
        "B. Documented water classes 3--7\n"
        f"{water_count:,} pixels; not a validated boundary",
        f"C. channel_extent_candidate\n{candidate_count:,} pixels; {candidate_status}",
    )
    shared_extent = _normalize_extent(extent) or _coordinate_extent(raw)
    for axis, panel_title in zip(axes_array, panel_titles, strict=True):
        _configure_geographic_axes(axis, raw, extent=shared_extent)
        axis.set_title(panel_title)

    figure.legend(
        handles=_classification_handles(),
        title="Official PIXC classification",
        loc="outside lower center",
        ncols=4,
    )
    heading = title or "PIXC classification review"
    disclaimer = "Visual inspection only; no boundary or width inference"
    if disclaimer.lower() not in heading.lower():
        heading = f"{heading}\n{disclaimer}"
    figure.suptitle(heading)
    return figure, axes_array


def plot_transect_corridor(
    sample: TransectSample,
    *,
    color_by: str = "classification",
    ax: Axes | None = None,
    title: str | None = None,
    point_size: float = 2.0,
    alpha: float = 0.9,
) -> Axes:
    """Plot selected PIXC samples and a user-supplied transect corridor."""

    pixels = _sample_pixels(sample)
    half_width = _sample_half_width(sample)
    requested_title = _sample_title(
        sample,
        title,
        "Manual transect corridor",
        "sampling corridor only; no width inference",
    )
    axes = plot_pixc_map(
        pixels,
        color_by=color_by,
        ax=ax,
        title=requested_title,
        point_size=point_size,
        alpha=alpha,
        show_legend=True,
    )

    transect = getattr(sample, "transect", None)
    corridor = getattr(sample, "corridor", None)
    if transect is None or corridor is None:
        raise TypeError("TransectSample must expose 'transect' and 'corridor'.")

    previous_legend = axes.get_legend()
    corridor_lines = []
    for index, ring in enumerate(_geometry_rings(corridor)):
        x, y = ring.xy
        (corridor_line,) = axes.plot(
            x,
            y,
            color="#d95f02",
            linestyle="--",
            linewidth=1.2,
            label=f"±{half_width:g} m corridor" if index == 0 else None,
            zorder=3,
        )
        if index == 0:
            corridor_lines.append(corridor_line)

    transect_x, transect_y = transect.xy
    (transect_line,) = axes.plot(
        transect_x,
        transect_y,
        color="#111111",
        linewidth=1.5,
        label="user-supplied transect",
        zorder=4,
    )
    start = transect.coords[0]
    end = transect.coords[-1]
    (start_marker,) = axes.plot(
        start[0],
        start[1],
        marker="o",
        markersize=5,
        linestyle="none",
        color="#009e73",
        label="start",
        zorder=5,
    )
    (end_marker,) = axes.plot(
        end[0],
        end[1],
        marker="s",
        markersize=5,
        linestyle="none",
        color="#d55e00",
        label="end",
        zorder=5,
    )
    if previous_legend is not None:
        axes.add_artist(previous_legend)
    axes.legend(
        handles=[*corridor_lines, transect_line, start_marker, end_marker],
        title="Manual sampling geometry",
        loc="upper right",
    )
    _expand_limits_to_geometry(axes, corridor)
    return axes


def plot_transect_classification(
    sample: TransectSample,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    point_size: float = 4.0,
    alpha: float = 0.8,
) -> Axes:
    """Plot categorical PIXC samples against projected transect station.

    Points are shown as samples at their class row.  No gaps are filled and no
    branch or bank interpretation is made.
    """

    _validate_style(point_size=point_size, alpha=alpha)
    pixels = _sample_pixels(sample)
    _require_variables(pixels, "station_m", "classification")
    station = np.asarray(pixels["station_m"].values)
    classification = np.asarray(pixels["classification"].values)
    valid = _valid_data_mask(pixels["station_m"])
    valid &= _valid_data_mask(pixels["classification"])
    valid &= np.isin(classification, tuple(CLASSIFICATION_LABELS))

    axes = _axes_or_new(ax)
    _, colors, _ = _import_matplotlib()
    collection = axes.scatter(
        station[valid],
        classification[valid],
        c=classification[valid],
        cmap=colors.ListedColormap(list(CLASSIFICATION_COLORS.values())),
        norm=colors.BoundaryNorm(np.arange(0.5, 8.5), 7),
        s=point_size,
        alpha=alpha,
        linewidths=0,
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )
    collection.set_gid("swot-pixc-lab:transect-classification")
    axes.set_xlabel("Station along user-supplied transect (m)")
    axes.set_ylabel("PIXC classification")
    axes.set_yticks(
        tuple(CLASSIFICATION_LABELS),
        [f"{value}: {label}" for value, label in CLASSIFICATION_LABELS.items()],
    )
    axes.set_ylim(0.5, 7.5)
    axes.set_axisbelow(True)
    axes.grid(True, color="#d9d9d9", linewidth=0.6)
    axes.legend(
        handles=_classification_handles(),
        title="Official PIXC classification",
        loc="best",
    )
    axes.set_title(
        _sample_title(
            sample,
            title,
            "PIXC classification samples",
            "sampling corridor only; no width inference",
        )
    )
    return axes


def plot_transect_height(
    sample: TransectSample,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    include_egm2008: bool = True,
    point_size: float = 4.0,
    alpha: float = 0.7,
) -> Axes:
    """Plot discrete height samples against station without interpolation."""

    _validate_style(point_size=point_size, alpha=alpha)
    pixels = _sample_pixels(sample)
    _require_variables(pixels, "station_m", "height")
    station = np.asarray(pixels["station_m"].values)
    station_valid = _valid_data_mask(pixels["station_m"])
    axes = _axes_or_new(ax)

    layers = [("height", "Raw ellipsoidal height (not corrected WSE)", "#0072b2")]
    if include_egm2008 and "height_egm2008" in pixels:
        layers.append(
            (
                "height_egm2008",
                "height_egm2008 = height - geoid (not corrected WSE)",
                "#d55e00",
            )
        )
    for name, label, color in layers:
        values = np.asarray(pixels[name].values)
        valid = station_valid & _valid_data_mask(pixels[name])
        collection = axes.scatter(
            station[valid],
            values[valid],
            s=point_size,
            alpha=alpha,
            color=color,
            linewidths=0,
            edgecolors="none",
            rasterized=True,
            label=label,
            zorder=2,
        )
        collection.set_gid(f"swot-pixc-lab:transect-{name}")

    axes.set_xlabel("Station along user-supplied transect (m)")
    axes.set_ylabel("PIXC pixel height sample (m)")
    axes.set_axisbelow(True)
    axes.grid(True, color="#d9d9d9", linewidth=0.6)
    axes.legend(title="Discrete pixel samples", loc="best")
    axes.set_title(
        _sample_title(
            sample,
            title,
            "PIXC discrete height samples",
            "pixel samples only; no interpolation or width inference",
        )
    )
    return axes


def plot_wet_interval_summary(
    result: ExplicitIntervalWidthResult,
    *,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot only the explicit wet intervals and dry gaps in a width result.

    This one-dimensional schematic is a review aid for measurements already
    supplied by an analyst. It does not inspect pixels or infer bank locations.
    """

    from .width import ExplicitIntervalWidthResult

    if not isinstance(result, ExplicitIntervalWidthResult):
        raise TypeError("result must be an ExplicitIntervalWidthResult.")

    axes = _axes_or_new(ax)
    _, _, patches = _import_matplotlib()
    axes.axhline(0.4, color="#666666", linewidth=0.8, zorder=1)

    for interval in result.intervals:
        rectangle = patches.Rectangle(
            (interval.start_station_m, 0.2),
            interval.width_m,
            0.4,
            facecolor="#1f78b4",
            edgecolor="#0b3c5d",
            linewidth=0.8,
            zorder=3,
        )
        rectangle.set_gid(f"swot-pixc-lab:explicit-wet-interval-{interval.interval_id}")
        axes.add_patch(rectangle)
        axes.text(
            0.5 * (interval.start_station_m + interval.end_station_m),
            0.4,
            f"I{interval.interval_id}\n{interval.width_m:g} m",
            ha="center",
            va="center",
            color="white",
            fontsize="small",
            zorder=4,
        )

    for gap in result.dry_gaps:
        rectangle = patches.Rectangle(
            (gap.start_station_m, 0.2),
            gap.width_m,
            0.4,
            facecolor="#f2e6c9",
            edgecolor="#8c6d31",
            hatch="///",
            linewidth=0.8,
            zorder=2,
        )
        rectangle.set_gid(f"swot-pixc-lab:explicit-dry-gap-{gap.gap_id}")
        axes.add_patch(rectangle)
        axes.text(
            0.5 * (gap.start_station_m + gap.end_station_m),
            0.04,
            f"G{gap.gap_id}: {gap.width_m:g} m",
            ha="center",
            va="bottom",
            fontsize="small",
            color="#5d4a1f",
            zorder=4,
        )

    if not result.intervals:
        axes.text(
            0.5,
            0.5,
            "No wet intervals supplied",
            transform=axes.transAxes,
            ha="center",
            va="center",
        )

    span_text = (
        "not defined (no wet intervals)"
        if result.outer_wetted_span_m is None
        else f"{result.outer_wetted_span_m:g} m"
    )
    axes.text(
        0.5,
        0.9,
        f"Total wetted width: {result.total_wetted_width_m:g} m; "
        f"outer wetted span: {span_text}; "
        f"internal dry gaps: {result.total_internal_dry_gap_m:g} m",
        transform=axes.transAxes,
        ha="center",
        va="center",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    heading = title or "Explicit analyst-supplied wet intervals"
    axes.set_title(
        f"{heading}\n{result.method_status}; analyst-supplied measurements only; "
        "no bank inference"
    )
    axes.set_xlabel("Station along user-supplied transect (m)")
    axes.set_xlim(0.0, result.transect_length_m)
    axes.set_ylim(0.0, 1.0)
    axes.set_yticks([])
    axes.set_axisbelow(True)
    axes.grid(axis="x", color="#d9d9d9", linewidth=0.6)

    handles = []
    if result.intervals:
        handles.append(
            patches.Patch(
                facecolor="#1f78b4",
                edgecolor="#0b3c5d",
                label="supplied wet interval",
            )
        )
    if result.dry_gaps:
        handles.append(
            patches.Patch(
                facecolor="#f2e6c9",
                edgecolor="#8c6d31",
                hatch="///",
                label="explicit internal dry gap",
            )
        )
    if handles:
        axes.legend(handles=handles, loc="lower right")
    return axes


def _plot_map_layer(
    resolved: ResolvedPixelInput,
    *,
    color_by: str,
    ax: Axes,
    point_size: float,
    alpha: float,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    show_legend: bool,
    show_colorbar: bool,
) -> object:
    dataset = resolved.dataset
    _require_variables(dataset, "longitude", "latitude")
    longitude = np.asarray(dataset["longitude"].values)
    latitude = np.asarray(dataset["latitude"].values)
    coordinate_valid = _valid_data_mask(dataset["longitude"])
    coordinate_valid &= _valid_data_mask(dataset["latitude"])
    values, value_valid, kind, color_label = _resolve_color_values(resolved, color_by)
    valid = coordinate_valid & value_valid
    plt, colors, _ = _import_matplotlib()

    scatter_kwargs: dict[str, Any] = {
        "s": point_size,
        "alpha": alpha,
        "linewidths": 0,
        "edgecolors": "none",
        "rasterized": True,
        "zorder": 2,
    }
    if kind == "classification":
        scatter_kwargs.update(
            c=values[valid],
            cmap=colors.ListedColormap(list(CLASSIFICATION_COLORS.values())),
            norm=colors.BoundaryNorm(np.arange(0.5, 8.5), 7),
        )
    elif kind == "source":
        unique_sources = np.unique(values[valid])
        ranks = np.searchsorted(unique_sources, values[valid])
        count = max(int(unique_sources.size), 1)
        scatter_kwargs.update(
            c=ranks,
            cmap=plt.get_cmap(cmap or "tab20", count),
            norm=colors.BoundaryNorm(np.arange(-0.5, count + 0.5), count),
        )
    elif kind == "flag":
        scatter_kwargs.update(
            c=values[valid].astype(np.uint8, copy=False),
            cmap=colors.ListedColormap(["#d9d9d9", "#d73027"]),
            norm=colors.BoundaryNorm((-0.5, 0.5, 1.5), 2),
        )
    else:
        scatter_kwargs.update(
            c=values[valid],
            cmap=cmap or "viridis",
            vmin=vmin,
            vmax=vmax,
        )

    collection = ax.scatter(longitude[valid], latitude[valid], **scatter_kwargs)
    collection.set_gid(f"swot-pixc-lab:{color_by}")
    if kind == "classification" and show_legend:
        ax.legend(
            handles=_classification_handles(),
            title="Official PIXC classification",
            loc="best",
        )
    elif kind == "source" and show_legend:
        ax.legend(
            handles=_source_handles(resolved, unique_sources, collection.cmap),
            title="Source granule/tile index",
            loc="best",
        )
    elif kind == "flag" and show_legend:
        ax.legend(
            handles=_flag_handles(color_label),
            title="Decoded quality flag",
            loc="best",
        )
    elif kind == "continuous" and show_colorbar:
        colorbar = ax.figure.colorbar(collection, ax=ax)
        colorbar.set_label(color_label)
    return collection


def _resolve_color_values(
    resolved: ResolvedPixelInput,
    color_by: str,
) -> tuple[NDArray[Any], NDArray[np.bool_], str, str]:
    dataset = resolved.dataset
    if ":" in color_by:
        variable_name, meaning = color_by.split(":", 1)
        if not variable_name or not meaning:
            raise ValueError(
                "Decoded flag selectors use 'quality_variable:flag_meaning'."
            )
        qc_result = resolved.qc_result
        if qc_result is None:
            raise ValueError("Decoded QC flag colors require a QCResult input.")
        if variable_name not in qc_result.decoded_flags:
            available = ", ".join(qc_result.decoded_flags)
            raise KeyError(
                f"Decoded quality variable {variable_name!r} is unavailable; "
                f"choose from: {available}."
            )
        decoded = qc_result.decoded_flags[variable_name]
        values = decoded.flag(meaning)
        valid = np.asarray(decoded.valid_mask, dtype=np.bool_)
        selection = np.asarray(qc_result.mask, dtype=np.bool_)
        if values.shape != selection.shape:
            raise ValueError("Decoded QC flag and QC retention mask lengths differ.")
        matched = provenance_reference_indices(
            qc_result.observation.raw,
            dataset,
            context="QCResult.filtered",
        )
        expected = np.flatnonzero(selection)
        if not np.array_equal(np.sort(matched), expected):
            raise ValueError(
                "QCResult.filtered provenance does not match its retention mask."
            )
        _validate_reference_values(
            qc_result.observation.raw,
            dataset,
            matched,
            names=("longitude", "latitude"),
            context="QCResult.filtered",
        )
        values = values[matched]
        valid = valid[matched]
        return values, valid, "flag", f"{variable_name}: {meaning}"

    if color_by not in SUPPORTED_COLOR_VARIABLES:
        choices = ", ".join(sorted(SUPPORTED_COLOR_VARIABLES))
        raise ValueError(
            f"Unsupported color variable {color_by!r}; choose from {choices}, "
            "or use 'quality_variable:flag_meaning' with a QCResult."
        )
    if color_by not in dataset:
        note = (
            " Pass observation.apply_qc(profile='raw') to derive it without "
            "changing raw height."
            if color_by == "height_egm2008"
            else ""
        )
        raise KeyError(f"PIXC variable {color_by!r} is unavailable.{note}")
    variable = dataset[color_by]
    if color_by in {"classification", "source_index"}:
        if not np.issubdtype(variable.dtype, np.integer):
            raise ValueError(
                f"PIXC {color_by!r} must be an integer point variable; "
                f"got {variable.dtype}."
            )
    elif not np.issubdtype(variable.dtype, np.number) or np.issubdtype(
        variable.dtype, np.complexfloating
    ):
        raise ValueError(
            f"PIXC {color_by!r} must be a real numeric point variable; "
            f"got {variable.dtype}."
        )
    values = np.asarray(variable.values)
    valid = _valid_data_mask(variable)
    if color_by == "classification":
        valid &= np.isin(values, tuple(CLASSIFICATION_LABELS))
        return values, valid, "classification", "PIXC classification"
    if color_by == "source_index":
        return values, valid, "source", "Source granule/tile index"
    units = variable.attrs.get("units")
    unit_text = "" if not units else f" ({units})"
    if color_by == "height":
        label = f"raw ellipsoidal height{unit_text}; not corrected WSE"
    elif color_by == "height_egm2008":
        label = f"height_egm2008 = height - geoid{unit_text}; not corrected WSE"
    else:
        label = f"{color_by}{unit_text}"
    return values, valid, "continuous", label


def _valid_data_mask(variable: xr.DataArray) -> NDArray[np.bool_]:
    values = np.asarray(variable.values)
    if values.ndim != 1:
        raise ValueError(
            f"{variable.name or 'plot variable'} must be one-dimensional over points."
        )
    valid = np.ones(values.shape, dtype=np.bool_)
    if np.issubdtype(values.dtype, np.inexact):
        valid &= np.isfinite(values)
    for attribute in ("_FillValue", "missing_value"):
        fill = variable.attrs.get(attribute)
        if fill is not None:
            try:
                valid &= values != np.asarray(fill).reshape(-1)[0]
            except (TypeError, ValueError, IndexError):
                continue
    valid_min = variable.attrs.get("valid_min")
    valid_max = variable.attrs.get("valid_max")
    valid_range = variable.attrs.get("valid_range")
    if valid_range is not None:
        bounds = np.asarray(valid_range).reshape(-1)
        if bounds.size == 2:
            valid_min, valid_max = bounds
    try:
        if valid_min is not None:
            valid &= values >= np.asarray(valid_min).reshape(-1)[0]
        if valid_max is not None:
            valid &= values <= np.asarray(valid_max).reshape(-1)[0]
    except (TypeError, ValueError, IndexError):
        pass
    return valid


def _comparison_inputs(
    observation: object,
    candidate: object | None,
) -> tuple[ResolvedPixelInput, ResolvedPixelInput]:
    initial = resolve_pixel_input(observation)
    qc_result = initial.qc_result
    if qc_result is not None:
        raw_object: object = qc_result.observation
    else:
        raw_object = observation
    raw = resolve_pixel_input(raw_object)

    if candidate is None:
        if (
            qc_result is not None
            and qc_result.profile.name == "channel_extent_candidate"
        ):
            candidate = qc_result
        elif hasattr(raw_object, "apply_qc"):
            candidate = raw_object.apply_qc(profile="channel_extent_candidate")
        else:
            raise TypeError(
                "candidate is required when observation is a standalone Dataset."
            )
    resolved_candidate = resolve_pixel_input(candidate)
    if resolved_candidate.profile_name != "channel_extent_candidate":
        raise ValueError(
            "Classification comparison requires the Phase-3 "
            "'channel_extent_candidate' result."
        )
    if (
        resolved_candidate.qc_result is not None
        and not isinstance(raw_object, xr.Dataset)
        and resolved_candidate.qc_result.observation.raw is not raw.dataset
    ):
        raise ValueError(
            "The candidate QCResult belongs to a different observation than "
            "the raw comparison input."
        )
    matched = provenance_reference_indices(
        raw.dataset,
        resolved_candidate.dataset,
        context="channel_extent_candidate",
    )
    _validate_reference_values(
        raw.dataset,
        resolved_candidate.dataset,
        matched,
        names=("longitude", "latitude", "classification"),
        context="channel_extent_candidate",
    )
    return raw, resolved_candidate


def _validate_reference_values(
    reference: xr.Dataset,
    subset: xr.Dataset,
    matched: NDArray[np.int64],
    *,
    names: Sequence[str],
    context: str,
) -> None:
    for name in names:
        if name not in reference or name not in subset:
            raise ValueError(
                f"{context} validation requires {name!r} in both datasets."
            )
        expected = np.asarray(reference[name].values)[matched]
        actual = np.asarray(subset[name].values)
        if np.issubdtype(expected.dtype, np.inexact):
            equal = np.array_equal(expected, actual, equal_nan=True)
        else:
            equal = np.array_equal(expected, actual)
        if not equal:
            raise ValueError(
                f"{context} {name!r} values do not match its provenance in "
                "the reference observation."
            )


def _plot_title(
    resolved: ResolvedPixelInput,
    *,
    color_by: str,
    requested: str | None,
) -> str:
    label = resolved.profile_label or "PIXC points"
    title = requested or f"{label}: {color_by}"
    status = resolved.profile_status or ""
    if "experimental" in status.lower() and status.lower() not in title.lower():
        title = f"{title}\n{status}"
    return title


def _configure_geographic_axes(
    ax: Axes,
    dataset: xr.Dataset,
    *,
    extent: Sequence[float] | None,
) -> None:
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude (degrees north)")
    ax.set_axisbelow(True)
    ax.grid(True, color="#d9d9d9", linewidth=0.6)
    limits = _normalize_extent(extent) or _coordinate_extent(dataset)
    if limits is not None:
        west, south, east, north = limits
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        mean_latitude = 0.5 * (south + north)
        cosine = max(abs(math.cos(math.radians(mean_latitude))), 0.05)
        ax.set_aspect(1.0 / cosine, adjustable="box")


def _coordinate_extent(dataset: xr.Dataset) -> tuple[float, float, float, float] | None:
    if "longitude" not in dataset or "latitude" not in dataset:
        return None
    longitude = np.asarray(dataset["longitude"].values)
    latitude = np.asarray(dataset["latitude"].values)
    valid = _valid_data_mask(dataset["longitude"])
    valid &= _valid_data_mask(dataset["latitude"])
    if not np.any(valid):
        return None
    west = float(np.min(longitude[valid]))
    east = float(np.max(longitude[valid]))
    south = float(np.min(latitude[valid]))
    north = float(np.max(latitude[valid]))
    if west == east:
        west, east = west - 1e-6, east + 1e-6
    if south == north:
        south, north = south - 1e-6, north + 1e-6
    return west, south, east, north


def _normalize_extent(
    extent: Sequence[float] | None,
) -> tuple[float, float, float, float] | None:
    if extent is None:
        return None
    if isinstance(extent, (str, bytes)) or len(extent) != 4:
        raise ValueError("extent must be (west, south, east, north).")
    west, south, east, north = (float(value) for value in extent)
    if not all(math.isfinite(value) for value in (west, south, east, north)):
        raise ValueError("extent coordinates must be finite.")
    if west >= east or south >= north:
        raise ValueError("extent must satisfy west < east and south < north.")
    return west, south, east, north


def _classification_handles() -> list[object]:
    _, _, patches = _import_matplotlib()
    return [
        patches.Patch(
            facecolor=CLASSIFICATION_COLORS[value],
            edgecolor="none",
            label=f"{value}: {label}",
        )
        for value, label in CLASSIFICATION_LABELS.items()
    ]


def _source_handles(
    resolved: ResolvedPixelInput,
    sources: NDArray[Any],
    cmap: object,
) -> list[object]:
    _, _, patches = _import_matplotlib()
    metadata = {
        int(source.source_index): source.tile or source.filename
        for source in resolved.sources
    }
    denominator = max(int(sources.size) - 1, 1)
    return [
        patches.Patch(
            facecolor=cmap(rank / denominator),
            edgecolor="none",
            label=f"{int(value)}: {metadata.get(int(value), 'source')}",
        )
        for rank, value in enumerate(sources)
    ]


def _flag_handles(label: str) -> list[object]:
    _, _, patches = _import_matplotlib()
    return [
        patches.Patch(facecolor="#d9d9d9", edgecolor="none", label="not set"),
        patches.Patch(facecolor="#d73027", edgecolor="none", label=label),
    ]


def _sample_pixels(sample: object) -> xr.Dataset:
    pixels = getattr(sample, "pixels", None)
    if not isinstance(pixels, xr.Dataset):
        raise TypeError(
            "sample must be a TransectSample with an xarray 'pixels' dataset."
        )
    return pixels


def _sample_half_width(sample: object) -> float:
    value = getattr(sample, "corridor_half_width_m", None)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise TypeError("TransectSample must expose a finite corridor_half_width_m.")
    return float(value)


def _sample_profile(sample: object) -> tuple[str, str]:
    label = str(getattr(sample, "profile_label", None) or "Unspecified PIXC input")
    status = str(getattr(sample, "profile_status", None) or "status unspecified")
    return label, status


def _sample_title(
    sample: object,
    requested: str | None,
    default_heading: str,
    scientific_caution: str,
) -> str:
    half_width = _sample_half_width(sample)
    label, status = _sample_profile(sample)
    heading = requested or default_heading
    return (
        f"{heading}\n±{half_width:g} m corridor; {label}; {status}; "
        f"{scientific_caution}"
    )


def _geometry_rings(geometry: object) -> list[object]:
    if hasattr(geometry, "exterior"):
        rings = [geometry.exterior]
        rings.extend(getattr(geometry, "interiors", ()))
        return rings
    if hasattr(geometry, "geoms"):
        return [ring for part in geometry.geoms for ring in _geometry_rings(part)]
    raise TypeError("TransectSample corridor must be a Polygon or MultiPolygon.")


def _expand_limits_to_geometry(ax: Axes, geometry: object) -> None:
    west, south, east, north = geometry.bounds
    current_west, current_east = ax.get_xlim()
    current_south, current_north = ax.get_ylim()
    west = min(float(west), current_west)
    east = max(float(east), current_east)
    south = min(float(south), current_south)
    north = max(float(north), current_north)
    if west == east:
        west, east = west - 1e-6, east + 1e-6
    if south == north:
        south, north = south - 1e-6, north + 1e-6
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    mean_latitude = 0.5 * (south + north)
    cosine = max(abs(math.cos(math.radians(mean_latitude))), 0.05)
    ax.set_aspect(1.0 / cosine, adjustable="box")


def _require_variables(dataset: xr.Dataset, *names: str) -> None:
    missing = [name for name in names if name not in dataset]
    if missing:
        raise KeyError("Required PIXC variable(s) unavailable: " + ", ".join(missing))
    point_count = dataset.sizes.get("points")
    if point_count is None:
        raise ValueError("PIXC plotting requires a one-dimensional 'points' dimension.")
    for name in names:
        if dataset[name].dims != ("points",):
            raise ValueError(f"{name!r} must use exactly the 'points' dimension.")


def _validate_style(*, point_size: float, alpha: float) -> None:
    if not math.isfinite(point_size) or point_size <= 0:
        raise ValueError("point_size must be a finite positive number.")
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")


def _axes_or_new(ax: Axes | None) -> Axes:
    if ax is not None:
        return ax
    plt, _, _ = _import_matplotlib()
    _, axes = plt.subplots()
    return axes


def _import_matplotlib() -> tuple[Any, Any, Any]:
    try:
        import matplotlib.colors as colors
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name is not None and exc.name.startswith("matplotlib"):
            raise ImportError(
                "PIXC visualization requires Matplotlib. Install "
                "swot-pixc-lab with the 'visualization' extra."
            ) from exc
        raise
    return plt, colors, patches


__all__ = [
    "CLASSIFICATION_COLORS",
    "CLASSIFICATION_LABELS",
    "SUPPORTED_COLOR_VARIABLES",
    "plot_classification_comparison",
    "plot_pixc_map",
    "plot_transect_classification",
    "plot_transect_corridor",
    "plot_transect_height",
    "plot_wet_interval_summary",
]
