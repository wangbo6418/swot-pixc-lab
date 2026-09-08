"""Exact spatial clipping for raw SWOT PIXC point coordinates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from shapely import intersects_xy
from shapely.geometry import MultiPolygon, Polygon, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .aoi import AoiInput
from .exceptions import AoiError


@dataclass(frozen=True, slots=True)
class ExactAoi:
    """Validated WGS 84 geometry used for exact point-level clipping."""

    geometry: BaseGeometry
    kind: Literal["bounding_box", "geometry"]
    bounding_box: tuple[float, float, float, float] | None = None

    @property
    def wkt(self) -> str:
        """Return the normalized AOI as Well-Known Text."""

        return self.geometry.wkt


@dataclass(frozen=True, slots=True)
class ClipMask:
    """A point-selection mask plus explicit coordinate diagnostics."""

    values: NDArray[np.bool_]
    invalid_coordinate_count: int
    normalized_longitude_count: int


def normalize_exact_aoi(aoi: AoiInput) -> ExactAoi:
    """Normalize an AOI without dropping polygon holes or feature parts.

    Bounding boxes and GeoJSON coordinates are interpreted as EPSG:4326.
    GeoDataFrame-like objects are transformed explicitly to EPSG:4326. A
    bounding box with ``west > east`` represents an antimeridian crossing.
    Polygon boundaries are retained by :func:`clip_points`.
    """

    if _looks_like_bbox(aoi):
        return _normalize_exact_bbox(aoi)

    candidate = aoi
    if _looks_like_geodataframe(candidate):
        if getattr(candidate, "crs", None) is None:
            raise AoiError(
                "GeoDataFrame-like AOIs must declare a CRS before exact clipping."
            )
        try:
            candidate = candidate.to_crs(epsg=4326)
        except Exception as exc:  # pragma: no cover - backend-specific details
            raise AoiError(
                "Could not transform the clipping AOI to EPSG:4326 with to_crs()."
            ) from exc

    geo_interface = getattr(candidate, "__geo_interface__", None)
    if geo_interface is not None:
        candidate = geo_interface
    if not isinstance(candidate, Mapping):
        raise AoiError(
            "Clipping AOI must be a four-value bounding box, GeoJSON Polygon/"
            "MultiPolygon, or a GeoDataFrame-like object."
        )

    polygons = _polygons_from_geojson(candidate)
    if not polygons:
        raise AoiError("Clipping AOI contains no non-empty polygon geometry.")
    for polygon in polygons:
        _validate_polygon(polygon)
        if _crosses_antimeridian(polygon):
            raise AoiError(
                "A clipping polygon edge crosses the antimeridian. Split it into "
                "an explicit MultiPolygon on either side of +/-180 degrees."
            )

    geometry = unary_union(polygons)
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise AoiError("Clipping AOI must resolve to Polygon or MultiPolygon geometry.")
    return ExactAoi(geometry=geometry, kind="geometry")


def clip_points(
    longitude: NDArray[np.floating[Any]],
    latitude: NDArray[np.floating[Any]],
    aoi: ExactAoi,
    *,
    longitude_fill_value: object | None = None,
    latitude_fill_value: object | None = None,
) -> ClipMask:
    """Return a boundary-inclusive mask for point coordinates inside ``aoi``.

    Longitude is normalized to ``[-180, 180)`` only for comparison. The caller's
    original coordinate arrays are never changed. Missing/non-finite coordinates
    cannot be located spatially and are counted explicitly rather than treated as
    a quality-control decision.
    """

    longitudes = np.asarray(longitude)
    latitudes = np.asarray(latitude)
    if longitudes.ndim != 1 or latitudes.ndim != 1:
        raise PixcCoordinateError("PIXC latitude and longitude must be 1-D arrays.")
    if longitudes.shape != latitudes.shape:
        raise PixcCoordinateError(
            "PIXC latitude and longitude must have identical point dimensions."
        )

    valid = np.isfinite(longitudes) & np.isfinite(latitudes)
    valid &= ~_equals_fill(longitudes, longitude_fill_value)
    valid &= ~_equals_fill(latitudes, latitude_fill_value)

    normalized = longitudes.copy()
    needs_normalization = valid & ((longitudes < -180.0) | (longitudes >= 180.0))
    normalized[needs_normalization] = (
        (longitudes[needs_normalization] + 180.0) % 360.0
    ) - 180.0
    selected = np.zeros(longitudes.shape, dtype=np.bool_)
    if aoi.kind == "bounding_box":
        if aoi.bounding_box is None:  # pragma: no cover - dataclass invariant
            raise RuntimeError("Bounding-box AOI is missing its numeric bounds.")
        west, south, east, north = aoi.bounding_box
        latitude_match = (latitudes >= south) & (latitudes <= north)
        if west < east:
            longitude_match = (normalized >= west) & (normalized <= east)
        else:
            longitude_match = (normalized >= west) | (normalized <= east)
        selected = valid & latitude_match & longitude_match
    elif np.any(valid):
        selected[valid] = intersects_xy(
            aoi.geometry,
            normalized[valid],
            latitudes[valid],
        )

    return ClipMask(
        values=selected,
        invalid_coordinate_count=int((~valid).sum()),
        normalized_longitude_count=int(needs_normalization.sum()),
    )


class PixcCoordinateError(AoiError):
    """Raised when PIXC coordinate arrays cannot be clipped consistently."""


def _looks_like_bbox(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray, Mapping))
        and len(value) == 4
    )


def _normalize_exact_bbox(values: Sequence[float]) -> ExactAoi:
    coordinates: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise AoiError("Bounding-box coordinates must be real numbers, not bool.")
        try:
            coordinate = float(value)
        except (TypeError, ValueError) as exc:
            raise AoiError("Bounding-box coordinates must be numeric.") from exc
        if not math.isfinite(coordinate):
            raise AoiError("Bounding-box coordinates must be finite.")
        coordinates.append(coordinate)

    west, south, east, north = coordinates
    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise AoiError("Bounding-box longitudes must be between -180 and 180.")
    if not (-90 <= south <= 90 and -90 <= north <= 90):
        raise AoiError("Bounding-box latitudes must be between -90 and 90.")
    if south >= north:
        raise AoiError("Bounding-box south latitude must be less than north latitude.")
    if west == east:
        raise AoiError("Bounding-box west and east longitudes must differ.")

    bounds = (west, south, east, north)
    if west < east:
        geometry: BaseGeometry = box(west, south, east, north)
    else:
        geometry = MultiPolygon(
            [box(west, south, 180.0, north), box(-180.0, south, east, north)]
        )
    return ExactAoi(
        geometry=geometry,
        kind="bounding_box",
        bounding_box=bounds,
    )


def _looks_like_geodataframe(value: object) -> bool:
    return all(hasattr(value, attribute) for attribute in ("geometry", "crs", "to_crs"))


def _polygons_from_geojson(value: Mapping[str, Any]) -> list[Polygon]:
    geo_type = value.get("type")
    if geo_type == "Feature":
        geometry = value.get("geometry")
        if geometry is None:
            return []
        if not isinstance(geometry, Mapping):
            raise AoiError("GeoJSON feature geometry must be a mapping.")
        return _polygons_from_geojson(geometry)
    if geo_type == "FeatureCollection":
        polygons: list[Polygon] = []
        for feature in value.get("features", []):
            if not isinstance(feature, Mapping):
                raise AoiError("GeoJSON features must be mapping objects.")
            polygons.extend(_polygons_from_geojson(feature))
        return polygons

    try:
        geometry = shape(value)
    except Exception as exc:
        raise AoiError("Clipping AOI is not valid GeoJSON geometry.") from exc
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    raise AoiError("Clipping AOI must contain only Polygon or MultiPolygon geometry.")


def _validate_polygon(polygon: Polygon) -> None:
    if polygon.is_empty or polygon.area == 0:
        raise AoiError("Clipping polygons must be non-empty and have non-zero area.")
    if not polygon.is_valid:
        raise AoiError(
            "Clipping polygon is invalid. Repair it explicitly; the library will "
            "not silently alter scientific geometry."
        )
    for ring in (polygon.exterior, *polygon.interiors):
        for longitude, latitude in ring.coords:
            if not (math.isfinite(longitude) and math.isfinite(latitude)):
                raise AoiError("Clipping AOI coordinates must be finite.")
            if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
                raise AoiError("Clipping AOI must use EPSG:4326 longitude/latitude.")


def _crosses_antimeridian(polygon: Polygon) -> bool:
    for ring in (polygon.exterior, *polygon.interiors):
        longitudes = [coordinate[0] for coordinate in ring.coords]
        if any(
            abs(right - left) > 180
            for left, right in zip(longitudes, longitudes[1:], strict=False)
        ):
            return True
    return False


def _equals_fill(values: NDArray[Any], fill_value: object | None) -> NDArray[np.bool_]:
    if fill_value is None:
        return np.zeros(values.shape, dtype=np.bool_)
    try:
        return np.asarray(values == fill_value, dtype=np.bool_)
    except (TypeError, ValueError):
        return np.zeros(values.shape, dtype=np.bool_)
