"""AOI and temporal normalization for NASA CMR granule searches."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from typing import Any, Literal, Protocol

from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient

from .exceptions import AoiError, DateRangeError

type DateLike = str | date | datetime


class GeoInterface(Protocol):
    """Object exposing an RFC-7946-like geometry mapping."""

    @property
    def __geo_interface__(self) -> Mapping[str, Any]:
        """Return this object's GeoJSON-compatible representation."""


class GeoDataFrameLike(GeoInterface, Protocol):
    """Minimal GeoDataFrame behavior needed for explicit CRS conversion."""

    crs: object
    geometry: object

    def to_crs(self, *, epsg: int) -> GeoInterface:
        """Return a copy transformed to the requested EPSG code."""


type AoiInput = Sequence[float] | Mapping[str, Any] | GeoInterface


@dataclass(frozen=True, slots=True)
class SpatialQuery:
    """One spatial filter accepted by ``earthaccess.search_data``."""

    kind: Literal["bounding_box", "polygon"]
    coordinates: tuple[float, ...] | tuple[tuple[float, float], ...]

    def as_kwargs(self) -> dict[str, object]:
        """Return this query in earthaccess keyword form."""

        if self.kind == "bounding_box":
            return {"bounding_box": self.coordinates}
        return {"polygon": list(self.coordinates)}


@dataclass(frozen=True, slots=True)
class NormalizedAoi:
    """Normalized CMR searches plus notes about conservative broadening."""

    queries: tuple[SpatialQuery, ...]
    notes: tuple[str, ...] = ()


def normalize_aoi(aoi: AoiInput) -> NormalizedAoi:
    """Normalize a bounding box, GeoJSON, or GeoDataFrame-like AOI.

    Coordinates sent to CMR are WGS 84 longitude/latitude. A bounding box with
    ``west > east`` is interpreted as crossing the antimeridian and is split
    into two queries. GeoDataFrame-like inputs must declare a CRS and provide
    ``to_crs`` so projected inputs can be transformed explicitly.

    Polygon holes cannot be represented by earthaccess' polygon parameter.
    Their exterior rings are searched conservatively and a warning/note records
    that the search may include extra tiles; exact pixel clipping belongs to
    Phase 2.
    """

    if _looks_like_bbox(aoi):
        return _normalize_bbox(aoi)

    candidate = aoi
    if _looks_like_geodataframe(candidate):
        if getattr(candidate, "crs", None) is None:
            raise AoiError(
                "GeoDataFrame-like AOIs must declare a CRS; set EPSG:4326 or "
                "provide a CRS that can be transformed to EPSG:4326."
            )
        try:
            candidate = candidate.to_crs(epsg=4326)
        except Exception as exc:  # pragma: no cover - backend-specific details
            raise AoiError(
                "Could not transform the AOI to EPSG:4326 with to_crs()."
            ) from exc

    geo_interface = getattr(candidate, "__geo_interface__", None)
    if geo_interface is not None:
        candidate = geo_interface

    if not isinstance(candidate, Mapping):
        raise AoiError(
            "AOI must be a four-value bounding box, GeoJSON Polygon/"
            "MultiPolygon, or a GeoDataFrame-like object."
        )

    geometries = _geometries_from_geojson(candidate)
    if not geometries:
        raise AoiError("AOI contains no non-empty polygon geometry.")

    polygons: list[Polygon] = []
    for geometry in geometries:
        if not isinstance(geometry, (Polygon, MultiPolygon)):
            raise AoiError("AOI must contain only Polygon or MultiPolygon geometry.")
        if not geometry.is_valid:
            raise AoiError(
                "AOI geometry is invalid. Repair it explicitly before searching; "
                "the library will not silently alter scientific geometry."
            )
        polygons.extend(_extract_polygons(geometry))
    if not polygons:
        raise AoiError("AOI must contain Polygon or MultiPolygon geometry.")

    queries: list[SpatialQuery] = []
    notes: list[str] = []
    omitted_holes = sum(len(polygon.interiors) for polygon in polygons)
    if omitted_holes:
        note = (
            f"CMR polygon search cannot encode {omitted_holes} AOI hole(s); "
            "exterior rings were searched conservatively."
        )
        notes.append(note)
        warnings.warn(note, UserWarning, stacklevel=2)

    for polygon in polygons:
        _validate_polygon(polygon)
        if _crosses_antimeridian(polygon):
            raise AoiError(
                "A polygon edge crosses the antimeridian. Split it into an "
                "explicit GeoJSON MultiPolygon on either side of ±180°, or use "
                "a bounding box with west > east."
            )
        ccw = orient(polygon, sign=1.0)
        coordinates = tuple((float(x), float(y)) for x, y in ccw.exterior.coords)
        queries.append(SpatialQuery("polygon", coordinates))

    return NormalizedAoi(tuple(queries), tuple(notes))


def normalize_temporal_bounds(
    start_date: DateLike, end_date: DateLike
) -> tuple[str, str]:
    """Return explicit inclusive UTC bounds suitable for CMR.

    Date-only values cover the complete end date. Naive datetimes are treated
    as UTC, while timezone-aware values are converted to UTC.
    """

    start = _parse_date_like(start_date, is_end=False)
    end = _parse_date_like(end_date, is_end=True)
    if start > end:
        raise DateRangeError(
            f"start_date ({start.isoformat()}) must not be after "
            f"end_date ({end.isoformat()})."
        )
    return _format_utc(start), _format_utc(end)


def _looks_like_bbox(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray, Mapping))
        and len(value) == 4
    )


def _normalize_bbox(values: Sequence[float]) -> NormalizedAoi:
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

    if west < east:
        query = SpatialQuery("bounding_box", (west, south, east, north))
        return NormalizedAoi((query,))

    note = "Antimeridian-crossing bounding box was split into two CMR searches."
    return NormalizedAoi(
        (
            SpatialQuery("bounding_box", (west, south, 180.0, north)),
            SpatialQuery("bounding_box", (-180.0, south, east, north)),
        ),
        (note,),
    )


def _looks_like_geodataframe(value: object) -> bool:
    return all(hasattr(value, attribute) for attribute in ("geometry", "crs", "to_crs"))


def _geometries_from_geojson(value: Mapping[str, Any]) -> list[BaseGeometry]:
    geo_type = value.get("type")
    try:
        if geo_type == "Feature":
            geometry = value.get("geometry")
            if geometry is None:
                return []
            return _geometries_from_geojson(geometry)
        if geo_type == "FeatureCollection":
            geometries: list[BaseGeometry] = []
            for feature in value.get("features", []):
                if not isinstance(feature, Mapping):
                    raise AoiError("GeoJSON features must be mapping objects.")
                geometries.extend(_geometries_from_geojson(feature))
            return geometries
        geometry = shape(value)
    except AoiError:
        raise
    except Exception as exc:
        raise AoiError("AOI is not valid GeoJSON geometry.") from exc

    return [] if geometry.is_empty else [geometry]


def _extract_polygons(geometry: BaseGeometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    return []


def _validate_polygon(polygon: Polygon) -> None:
    if polygon.is_empty or polygon.area == 0:
        raise AoiError("AOI polygons must be non-empty and have non-zero area.")
    if not polygon.is_valid:
        raise AoiError(
            "AOI polygon is invalid. Repair it explicitly before searching; "
            "the library will not silently alter scientific geometry."
        )
    for ring in (polygon.exterior, *polygon.interiors):
        for longitude, latitude in ring.coords:
            if not (math.isfinite(longitude) and math.isfinite(latitude)):
                raise AoiError("AOI coordinates must be finite.")
            if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
                raise AoiError(
                    "AOI coordinates must use EPSG:4326 longitude/latitude ranges."
                )


def _crosses_antimeridian(polygon: Polygon) -> bool:
    for ring in (polygon.exterior, *polygon.interiors):
        longitudes = [coordinate[0] for coordinate in ring.coords]
        if any(
            abs(right - left) > 180
            for left, right in zip(longitudes, longitudes[1:], strict=False)
        ):
            return True
    return False


def _parse_date_like(value: DateLike, *, is_end: bool) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.max if is_end else time.min)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise DateRangeError("Date values must not be empty.")
        date_only = len(text) == 10 and text[4:5] == "-" and text[7:8] == "-"
        try:
            if date_only:
                parsed_date = date.fromisoformat(text)
                parsed = datetime.combine(parsed_date, time.max if is_end else time.min)
            else:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise DateRangeError(
                f"Could not parse date value {value!r}; use ISO 8601."
            ) from exc
    else:
        raise DateRangeError("Date values must be strings, dates, or datetimes.")

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _format_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
