"""Manual, metric sampling of user-supplied transects through PIXC points.

This module deliberately does not infer banks, branches, islands, or widths.
The corridor is only an explicit point-sampling choice.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from pyproj import CRS, Geod, Transformer
from shapely import distance, line_locate_point, points
from shapely.geometry import LineString, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from ._pixel_input import PixelData, resolve_pixel_input
from .exceptions import TransectError
from .reader import PixcSourceMetadata

type Endpoint = Sequence[float]
type TransectInput = LineString | Mapping[str, Any] | Sequence[Endpoint]

_WGS84 = CRS.from_epsg(4326)
_GEOD = Geod(ellps="WGS84")
_BOUNDARY_TOLERANCE_M = 1.0e-6


@dataclass(frozen=True, slots=True)
class TransectSample:
    """Auditable PIXC point samples from an explicit transect corridor.

    ``pixels`` is detached from the input and retains every selected point in
    input order. It includes all original point variables plus ``station_m``
    and ``distance_to_transect_m``. These samples are not an inferred width or
    a continuous cross section.
    """

    pixels: xr.Dataset
    transect: LineString
    corridor: BaseGeometry
    corridor_half_width_m: float
    local_crs: CRS
    projection_center: tuple[float, float]
    profile_name: str | None
    profile_label: str | None
    profile_status: str | None
    sources: tuple[PixcSourceMetadata, ...]
    input_source_indices: tuple[int, ...]
    input_pixel_count: int
    invalid_coordinate_count: int

    @property
    def selected_pixel_count(self) -> int:
        """Return the number of PIXC pixels inside the sampling corridor."""

        return int(self.pixels.sizes["points"])

    @property
    def station_m(self) -> xr.DataArray:
        """Planar distance along the projected line from its first endpoint."""

        return self.pixels["station_m"]

    @property
    def distance_to_transect_m(self) -> xr.DataArray:
        """Shortest planar distance to the line in the recorded local CRS."""

        return self.pixels["distance_to_transect_m"]

    @property
    def transect_length_m(self) -> float:
        """Return projected line length in the recorded local metric CRS."""

        forward = Transformer.from_crs(_WGS84, self.local_crs, always_xy=True)
        projected_line = transform(forward.transform, self.transect)
        length = float(projected_line.length)
        if projected_line.is_empty or not math.isfinite(length) or length <= 0.0:
            raise TransectError(
                "Transect could not be represented with finite positive length "
                "in its recorded local metric CRS."
            )
        return length

    @property
    def classification_counts(self) -> dict[int, int]:
        """Count selected classification values, excluding defined fill data."""

        if "classification" not in self.pixels:
            return {}
        values = np.asarray(self.pixels["classification"].values)
        valid = np.ones(values.shape, dtype=np.bool_)
        if np.issubdtype(values.dtype, np.floating):
            valid &= np.isfinite(values)

        source_indices = np.asarray(self.pixels["source_index"].values)
        covered = np.zeros(values.shape, dtype=np.bool_)
        for source in self.sources:
            source_points = source_indices == source.source_index
            covered |= source_points
            metadata = source.variables.get("classification")
            if metadata is not None and metadata.fill_value is not None:
                valid[source_points] &= values[source_points] != metadata.fill_value

        fallback_fill = self.pixels["classification"].attrs.get("_FillValue")
        if fallback_fill is not None:
            valid[~covered] &= values[~covered] != fallback_fill

        labels, counts = np.unique(values[valid], return_counts=True)
        return {
            int(label): int(count) for label, count in zip(labels, counts, strict=True)
        }

    @property
    def source_counts(self) -> dict[int, int]:
        """Count selected points by numeric ``source_index`` provenance."""

        values = np.asarray(self.pixels["source_index"].values)
        counts = {source_index: 0 for source_index in self.input_source_indices}
        counts.update({source.source_index: 0 for source in self.sources})
        labels, frequencies = np.unique(values, return_counts=True)
        counts.update(
            {
                int(label): int(frequency)
                for label, frequency in zip(labels, frequencies, strict=True)
            }
        )
        return counts

    @property
    def source_tile_counts(self) -> dict[str, int]:
        """Count samples by source tile when source metadata is available."""

        counts_by_source = self.source_counts
        if not self.sources:
            return {
                f"source_{source_index}": count
                for source_index, count in counts_by_source.items()
            }
        result: dict[str, int] = {}
        known_indices: set[int] = set()
        for source in self.sources:
            known_indices.add(source.source_index)
            label = source.tile or source.filename
            result[label] = result.get(label, 0) + counts_by_source.get(
                source.source_index, 0
            )
        for source_index, count in counts_by_source.items():
            if source_index not in known_indices:
                result[f"source_{source_index}"] = count
        return result

    @property
    def outside_corridor_count(self) -> int:
        """Return valid input pixels not selected by the corridor."""

        return (
            self.input_pixel_count
            - self.invalid_coordinate_count
            - self.selected_pixel_count
        )

    def to_geodataframe(self) -> Any:
        """Materialize selected points as an optional EPSG:4326 GeoDataFrame.

        GeoPandas is imported only when this convenience method is called, so
        core transect calculations never create point geometry objects for all
        input pixels. The frame repeats corridor, QC-profile, and known source
        descriptors in columns and carries the complete audit context in
        ``GeoDataFrame.attrs``; not every file format preserves those attrs.
        """

        try:
            import geopandas as gpd
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "GeoDataFrame export requires the optional 'geo' dependencies: "
                "install swot-pixc-lab[geo]."
            ) from exc

        point_variables = [
            name
            for name, variable in self.pixels.data_vars.items()
            if variable.dims == ("points",)
        ]
        frame = self.pixels[point_variables].to_dataframe().reset_index()
        frame["corridor_half_width_m"] = self.corridor_half_width_m
        frame["qc_profile"] = self.profile_name
        frame["qc_profile_status"] = self.profile_status
        source_lookup = {source.source_index: source for source in self.sources}
        if source_lookup:
            frame["source_granule_id"] = frame["source_index"].map(
                {
                    source_index: source.granule_id
                    for source_index, source in source_lookup.items()
                }
            )
            frame["source_tile"] = frame["source_index"].map(
                {
                    source_index: source.tile
                    for source_index, source in source_lookup.items()
                }
            )
            frame["source_filename"] = frame["source_index"].map(
                {
                    source_index: source.filename
                    for source_index, source in source_lookup.items()
                }
            )
        geometry = gpd.points_from_xy(
            frame["longitude"],
            frame["latitude"],
            crs="EPSG:4326",
        )
        result = gpd.GeoDataFrame(frame, geometry=geometry, crs="EPSG:4326")
        result.attrs.update(
            {
                "swot_pixc_lab_transect_wkt": self.transect.wkt,
                "swot_pixc_lab_corridor_half_width_m": self.corridor_half_width_m,
                "swot_pixc_lab_metric_crs_wkt": self.local_crs.to_wkt(),
                "swot_pixc_lab_projection_center": self.projection_center,
                "swot_pixc_lab_qc_profile": self.profile_name,
                "swot_pixc_lab_qc_label": self.profile_label,
                "swot_pixc_lab_qc_status": self.profile_status,
                "swot_pixc_lab_sources": tuple(
                    {
                        "source_index": source.source_index,
                        "granule_id": source.granule_id,
                        "tile": source.tile,
                        "filename": source.filename,
                    }
                    for source in self.sources
                ),
            }
        )
        return result


def sample_transect(
    data: PixelData,
    transect: TransectInput,
    *,
    corridor_half_width_m: float,
    chunk_size: int = 250_000,
) -> TransectSample:
    """Select PIXC points near a user-supplied EPSG:4326 transect.

    Distances are planar measurements of the projected polyline in a local
    WGS84-ellipsoid azimuthal-equidistant CRS centered on the transect's
    geodesic midpoint.  They are not guaranteed to equal arbitrary geodesic
    along-line or offset distances, especially for long or multi-vertex lines.
    Points are selected when their shortest projected distance to the finite
    line is less than or equal to the explicit half-width; this gives the
    corridor round endpoint caps.

    No PIXC values are interpolated, averaged, or modified, and this function
    does not calculate a river or channel width.
    """

    resolved = resolve_pixel_input(data)
    dataset = resolved.dataset
    _require_provenance(dataset)
    line = _normalize_transect(transect)
    half_width = _validate_half_width(corridor_half_width_m)
    resolved_chunk_size = _validate_chunk_size(chunk_size)

    center, segment_lengths = _geodesic_midpoint(line)
    local_crs = _local_aeqd_crs(center)
    forward = Transformer.from_crs(_WGS84, local_crs, always_xy=True)
    reverse = Transformer.from_crs(local_crs, _WGS84, always_xy=True)
    projected_line = transform(forward.transform, line)
    if projected_line.is_empty or not math.isfinite(projected_line.length):
        raise TransectError(
            "Transect could not be represented in its local metric CRS."
        )
    if projected_line.length <= 0.0 or not all(
        length > 0.0 for length in segment_lengths
    ):
        raise TransectError("Transect must have nonzero metric length.")

    metric_corridor = projected_line.buffer(half_width, cap_style="round")
    corridor = transform(reverse.transform, metric_corridor)

    selected_indices: list[NDArray[np.int64]] = []
    selected_stations: list[NDArray[np.float64]] = []
    selected_distances: list[NDArray[np.float64]] = []
    invalid_coordinate_count = 0
    point_count = int(dataset.sizes["points"])

    for start in range(0, point_count, resolved_chunk_size):
        stop = min(start + resolved_chunk_size, point_count)
        longitudes, latitudes, valid = _coordinate_chunk(dataset, start, stop)
        invalid_coordinate_count += int(np.count_nonzero(~valid))
        if not np.any(valid):
            continue

        normalized_longitudes = np.array(longitudes[valid], copy=True)
        needs_normalization = (normalized_longitudes < -180.0) | (
            normalized_longitudes >= 180.0
        )
        normalized_longitudes[needs_normalization] = (
            (normalized_longitudes[needs_normalization] + 180.0) % 360.0
        ) - 180.0
        x, y = forward.transform(normalized_longitudes, latitudes[valid])
        x_values = np.asarray(x, dtype=np.float64)
        y_values = np.asarray(y, dtype=np.float64)
        transform_valid = np.isfinite(x_values) & np.isfinite(y_values)
        if not np.all(transform_valid):
            invalid_coordinate_count += int(np.count_nonzero(~transform_valid))
        if not np.any(transform_valid):
            continue

        chunk_points = points(x_values[transform_valid], y_values[transform_valid])
        metric_distances = np.asarray(
            distance(projected_line, chunk_points), dtype=np.float64
        )
        # Forward/inverse projection roundoff can put a mathematical boundary
        # point fractions of a micrometre outside. The tolerance is many orders
        # smaller than a meaningful PIXC sampling distance.
        inside = metric_distances <= half_width + _BOUNDARY_TOLERANCE_M
        if not np.any(inside):
            continue

        metric_stations = np.asarray(
            line_locate_point(projected_line, chunk_points[inside]),
            dtype=np.float64,
        )

        valid_offsets = np.flatnonzero(valid)
        projected_offsets = valid_offsets[transform_valid]
        selected_indices.append((start + projected_offsets[inside]).astype(np.int64))
        selected_stations.append(metric_stations)
        selected_distances.append(metric_distances[inside])

    indices = _concatenate_or_empty(selected_indices, np.int64)
    stations = _concatenate_or_empty(selected_stations, np.float64)
    distances = _concatenate_or_empty(selected_distances, np.float64)
    selected = dataset.isel(points=indices).copy(deep=True)
    selected["station_m"] = xr.DataArray(
        stations,
        dims=("points",),
        attrs={
            "long_name": "distance along the user-supplied transect",
            "units": "m",
            "comment": (
                "Distance from the first endpoint to the nearest projected point "
                "on the finite transect; no pixel interpolation was performed."
            ),
        },
    )
    selected["distance_to_transect_m"] = xr.DataArray(
        distances,
        dims=("points",),
        attrs={
            "long_name": "shortest distance to the user-supplied transect",
            "units": "m",
            "comment": (
                "Metric distance in the recorded local AEQD projection; the "
                "corridor is a sampling choice, not a width measurement."
            ),
        },
    )
    selected.attrs.update(
        {
            "swot_pixc_lab_transect_wkt": line.wkt,
            "swot_pixc_lab_corridor_half_width_m": half_width,
            "swot_pixc_lab_metric_crs_wkt": local_crs.to_wkt(),
            "swot_pixc_lab_projection_center_longitude": center[0],
            "swot_pixc_lab_projection_center_latitude": center[1],
            "swot_pixc_lab_transect_length_m": float(projected_line.length),
            "swot_pixc_lab_transect_note": (
                "User-supplied corridor sample only; no bank, branch, or width "
                "was inferred."
            ),
        }
    )

    return TransectSample(
        pixels=selected,
        transect=line,
        corridor=corridor,
        corridor_half_width_m=half_width,
        local_crs=local_crs,
        projection_center=center,
        profile_name=resolved.profile_name,
        profile_label=resolved.profile_label,
        profile_status=resolved.profile_status,
        sources=resolved.sources,
        input_source_indices=_input_source_indices(dataset),
        input_pixel_count=point_count,
        invalid_coordinate_count=invalid_coordinate_count,
    )


def _normalize_transect(value: TransectInput) -> LineString:
    if isinstance(value, LineString):
        line = value
    elif isinstance(value, Mapping):
        if value.get("type") != "LineString":
            raise TransectError("GeoJSON transect type must be 'LineString'.")
        try:
            geometry = shape(value)
        except (TypeError, ValueError) as exc:
            raise TransectError("GeoJSON transect coordinates are invalid.") from exc
        if not isinstance(geometry, LineString):  # pragma: no cover - shape invariant
            raise TransectError("GeoJSON transect must resolve to a LineString.")
        line = geometry
    else:
        if isinstance(value, (str, bytes, bytearray)):
            raise TransectError(
                "Transect endpoints must be two (longitude, latitude) pairs."
            )
        try:
            endpoints = tuple(value)
        except TypeError as exc:
            raise TransectError(
                "Transect must be a LineString, GeoJSON LineString, or two endpoints."
            ) from exc
        if len(endpoints) != 2:
            raise TransectError(
                "Endpoint input must contain exactly two coordinate pairs."
            )
        try:
            line = LineString(endpoints)
        except (TypeError, ValueError) as exc:
            raise TransectError(
                "Transect endpoints must be numeric (longitude, latitude) pairs."
            ) from exc

    if line.is_empty:
        raise TransectError("Transect LineString must not be empty.")
    if line.has_z:
        raise TransectError("Transect must contain only 2-D EPSG:4326 coordinates.")
    coordinates = np.asarray(line.coords, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[0] < 2 or coordinates.shape[1] != 2:
        raise TransectError("Transect must contain at least two 2-D coordinates.")
    if not np.all(np.isfinite(coordinates)):
        raise TransectError("Transect coordinates must be finite.")
    if np.any(np.all(np.diff(coordinates, axis=0) == 0.0, axis=1)):
        raise TransectError("Transect must have finite, non-repeated vertices.")
    if np.any((coordinates[:, 0] < -180.0) | (coordinates[:, 0] > 180.0)):
        raise TransectError("Transect longitudes must lie within [-180, 180].")
    if np.any((coordinates[:, 1] < -90.0) | (coordinates[:, 1] > 90.0)):
        raise TransectError("Transect latitudes must lie within [-90, 90].")
    if np.any(np.abs(np.diff(coordinates[:, 0])) > 180.0):
        raise TransectError(
            "Transects crossing the antimeridian are not supported; split the "
            "inspection into non-crossing transects."
        )
    if not line.is_valid:
        raise TransectError("Transect LineString geometry is invalid.")
    if not line.is_simple:
        raise TransectError("Transect LineString must not self-intersect.")
    if line.is_closed:
        raise TransectError(
            "Transect LineString must have distinct start and end points."
        )
    return line


def _geodesic_midpoint(
    line: LineString,
) -> tuple[tuple[float, float], NDArray[np.float64]]:
    coordinates = np.asarray(line.coords, dtype=np.float64)
    forward_azimuth, _, lengths = _GEOD.inv(
        coordinates[:-1, 0],
        coordinates[:-1, 1],
        coordinates[1:, 0],
        coordinates[1:, 1],
    )
    segment_lengths = np.asarray(lengths, dtype=np.float64)
    if (
        not np.all(np.isfinite(segment_lengths))
        or np.any(segment_lengths <= 0.0)
        or not math.isfinite(float(segment_lengths.sum()))
        or float(segment_lengths.sum()) <= 0.0
    ):
        raise TransectError(
            "Transect must have finite, non-repeated vertices and nonzero length."
        )

    halfway = float(segment_lengths.sum()) / 2.0
    cumulative = np.cumsum(segment_lengths)
    segment = int(np.searchsorted(cumulative, halfway, side="left"))
    before = 0.0 if segment == 0 else float(cumulative[segment - 1])
    center_lon, center_lat, _ = _GEOD.fwd(
        coordinates[segment, 0],
        coordinates[segment, 1],
        float(np.asarray(forward_azimuth)[segment]),
        halfway - before,
    )
    center_lon = ((float(center_lon) + 180.0) % 360.0) - 180.0
    return (center_lon, float(center_lat)), segment_lengths


def _local_aeqd_crs(center: tuple[float, float]) -> CRS:
    longitude, latitude = center
    return CRS.from_proj4(
        f"+proj=aeqd +lat_0={latitude:.15g} +lon_0={longitude:.15g} "
        "+ellps=WGS84 +units=m +no_defs +type=crs"
    )


def _coordinate_chunk(
    dataset: xr.Dataset,
    start: int,
    stop: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    selection = {"points": slice(start, stop)}
    try:
        longitude = np.asarray(
            dataset["longitude"].isel(selection).values, dtype=np.float64
        )
        latitude = np.asarray(
            dataset["latitude"].isel(selection).values, dtype=np.float64
        )
    except (TypeError, ValueError) as exc:
        raise TransectError("PIXC longitude and latitude must be numeric.") from exc

    valid = np.isfinite(longitude) & np.isfinite(latitude)
    valid &= (latitude >= -90.0) & (latitude <= 90.0)
    longitude_fill = dataset["longitude"].attrs.get("_FillValue")
    latitude_fill = dataset["latitude"].attrs.get("_FillValue")
    if longitude_fill is not None:
        valid &= longitude != longitude_fill
    if latitude_fill is not None:
        valid &= latitude != latitude_fill
    return longitude, latitude, valid


def _require_provenance(dataset: xr.Dataset) -> None:
    for name in ("source_index", "source_point_index"):
        if name not in dataset:
            raise TransectError(
                f"PIXC transect sampling requires the {name!r} provenance variable."
            )
        if dataset[name].dims != ("points",):
            raise TransectError(
                f"PIXC provenance variable {name!r} must have the ('points',) "
                f"dimension; got {dataset[name].dims}."
            )
        if not np.issubdtype(dataset[name].dtype, np.integer):
            raise TransectError(f"PIXC provenance variable {name!r} must be integer.")


def _input_source_indices(dataset: xr.Dataset) -> tuple[int, ...]:
    values = np.asarray(dataset["source_index"].values)
    return tuple(int(value) for value in np.unique(values))


def _validate_half_width(value: float) -> float:
    if isinstance(value, bool):
        raise TransectError("corridor_half_width_m must be a positive finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TransectError(
            "corridor_half_width_m must be a positive finite number."
        ) from exc
    if not math.isfinite(result) or result <= 0.0:
        raise TransectError("corridor_half_width_m must be a positive finite number.")
    return result


def _validate_chunk_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TransectError("chunk_size must be a positive integer.")
    return value


def _concatenate_or_empty(
    arrays: Sequence[NDArray[Any]],
    dtype: np.dtype[Any] | type[np.generic],
) -> NDArray[Any]:
    if not arrays:
        return np.empty(0, dtype=dtype)
    return np.asarray(np.concatenate(arrays), dtype=dtype)
