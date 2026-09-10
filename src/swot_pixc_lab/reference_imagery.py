"""Independent Sentinel-2 reference imagery preparation utilities.

This module deliberately stops at imagery preparation. Sentinel-2 RGB, NDWI,
and SCL are annotation aids and usability diagnostics; none of them create a
wet/dry classification, bank location, or Phase 5A.1 interval.

Network and raster dependencies are imported only inside the functions that
need them so the core SWOT PIXC package remains usable without imagery extras.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import LineString, mapping, shape
from shapely.ops import transform

from .exceptions import ReferenceImageryError

EARTH_SEARCH_ENDPOINT = "https://earth-search.aws.element84.com/v1"
EARTH_SEARCH_SENTINEL2_L2A_COLLECTION = "sentinel-2-l2a"
REQUIRED_S2_ASSETS = ("red", "green", "blue", "nir", "scl")
S2_ASSET_BAND_NAMES = {
    "red": "B04",
    "green": "B03",
    "blue": "B02",
    "nir": "B08",
    "scl": "SCL",
}

# Official Sentinel-2 L2A Scene Classification Layer codes. These categorical
# values are never interpolated and class 6 is never interpreted as truth here.
SCL_CLASS_NAMES = {
    0: "no_data",
    1: "saturated_or_defective",
    2: "dark_area_pixels_or_cast_shadows",
    3: "cloud_shadows",
    4: "vegetation",
    5: "bare_soils",
    6: "water",
    7: "clouds_low_probability_or_unclassified",
    8: "clouds_medium_probability",
    9: "clouds_high_probability",
    10: "cirrus",
    11: "snow_or_ice",
}
SCL_CLOUD_CLASSES = frozenset({8, 9})
SCL_CLOUD_SHADOW_CLASSES = frozenset({3})
SCL_CIRRUS_CLASSES = frozenset({10})
SCL_SNOW_ICE_CLASSES = frozenset({11})
SCL_NO_DATA_CLASSES = frozenset({0})
SCL_SATURATED_DEFECTIVE_CLASSES = frozenset({1})
SCL_NON_OBSCURED_DIAGNOSTIC_CLASSES = frozenset({4, 5, 6})
SCL_OBSCURED_DIAGNOSTIC_CLASSES = frozenset({1, 2, 3, 7, 8, 9, 10, 11})

NDWI_FORMULA = "(B03 - B08) / (B03 + B08)"
NDWI_INTERPRETATION = "continuous visualization aid only; not wet/dry truth"


def _require_rasterio() -> Any:
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise ReferenceImageryError(
            "Sentinel chip preparation requires the optional imagery dependencies. "
            "Install with `pip install -e '.[reference-imagery]'`."
        ) from exc
    return rasterio


def _parse_datetime(value: str | datetime, *, field_name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ReferenceImageryError(
                f"{field_name} must be an ISO-8601 datetime with a timezone."
            ) from exc
    else:
        raise ReferenceImageryError(
            f"{field_name} must be an ISO-8601 datetime with a timezone."
        )
    if parsed.tzinfo is None:
        raise ReferenceImageryError(f"{field_name} must include a timezone.")
    return parsed.astimezone(UTC)


def _isoformat_z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def stable_source_url(href: str) -> str:
    """Return a durable asset identifier without credentials or signed query data."""

    parts = urlsplit(href)
    if parts.scheme not in {"http", "https", "s3"}:
        raise ReferenceImageryError(
            f"Unsupported STAC asset URL scheme {parts.scheme!r}."
        )
    if parts.username is not None or parts.password is not None:
        raise ReferenceImageryError(
            "STAC asset URLs containing embedded user information are refused."
        )
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _redacted_url_for_message(value: str) -> str:
    parts = urlsplit(value)
    hostname = parts.hostname or "configured-host"
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    try:
        port = parts.port
    except ValueError:
        port = None
    netloc = f"{hostname}:{port}" if port is not None else hostname
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


@dataclass(frozen=True)
class StacAsset:
    """One provider asset mapped to a canonical Sentinel-2 band role."""

    key: str
    band_name: str
    href: str = field(repr=False)
    source_identifier: str
    media_type: str | None
    scale: float
    offset: float
    nodata: int | float | None
    spatial_resolution_m: float | None

    def to_dict(self, *, include_access_href: bool = False) -> dict[str, Any]:
        """Serialize durable provenance, omitting signed access URLs by default."""

        result = asdict(self)
        if not include_access_href:
            result.pop("href")
        return result


@dataclass(frozen=True)
class SentinelScene:
    """Provider-neutral metadata for one Sentinel-2 L2A STAC item."""

    provider: str
    stac_endpoint: str
    collection: str
    item_id: str
    acquisition_datetime: datetime
    processing_level: str | None
    processing_baseline: str | None
    scene_cloud_cover_percent: float | None
    mgrs_tile: str | None
    epsg: int | None
    bbox: tuple[float, float, float, float]
    geometry: Mapping[str, Any]
    assets: Mapping[str, StacAsset]
    query_time: datetime
    search_window_days: int
    datatake_id: str | None = None
    platform: str | None = None
    cloud_metadata: Mapping[str, float] = field(default_factory=dict)

    @property
    def acquisition_key(self) -> str:
        """Stable grouping key for same-datatake MGRS tiles."""

        if self.datatake_id:
            return self.datatake_id
        minute = self.acquisition_datetime.strftime("%Y%m%dT%H%M")
        return f"{self.platform or 'sentinel-2'}_{minute}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize scene metadata without expiring/signed query strings."""

        return {
            "provider": self.provider,
            "stac_endpoint": self.stac_endpoint,
            "collection": self.collection,
            "item_id": self.item_id,
            "acquisition_datetime": _isoformat_z(self.acquisition_datetime),
            "processing_level": self.processing_level,
            "processing_baseline": self.processing_baseline,
            "scene_cloud_cover_percent": self.scene_cloud_cover_percent,
            "mgrs_tile": self.mgrs_tile,
            "epsg": self.epsg,
            "bbox": list(self.bbox),
            "geometry": dict(self.geometry),
            "assets": {key: asset.to_dict() for key, asset in self.assets.items()},
            "query_time": _isoformat_z(self.query_time),
            "search_window_days": self.search_window_days,
            "datatake_id": self.datatake_id,
            "platform": self.platform,
            "cloud_metadata": dict(self.cloud_metadata),
        }


def _asset_raster_metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    bands = payload.get("raster:bands")
    if isinstance(bands, Sequence) and bands and isinstance(bands[0], Mapping):
        return bands[0]
    return {}


def parse_stac_item(
    item: Mapping[str, Any],
    *,
    provider: str,
    stac_endpoint: str,
    collection: str,
    query_time: str | datetime,
    search_window_days: int,
    asset_keys: Mapping[str, str] | None = None,
) -> SentinelScene:
    """Parse and validate a Sentinel-2 L2A STAC item.

    ``asset_keys`` maps canonical names (``red``, ``green``, ``blue``, ``nir``,
    ``scl``) to provider-specific item asset keys.
    """

    if search_window_days not in {7, 14, 30}:
        raise ReferenceImageryError("search_window_days must be 7, 14, or 30.")
    item_id = item.get("id")
    properties = item.get("properties")
    raw_assets = item.get("assets")
    geometry = item.get("geometry")
    if not isinstance(item_id, str) or not item_id:
        raise ReferenceImageryError("STAC item is missing a non-empty id.")
    if not isinstance(properties, Mapping):
        raise ReferenceImageryError(f"STAC item {item_id!r} has no properties mapping.")
    if not isinstance(raw_assets, Mapping):
        raise ReferenceImageryError(f"STAC item {item_id!r} has no assets mapping.")
    if not isinstance(geometry, Mapping):
        raise ReferenceImageryError(f"STAC item {item_id!r} has no geometry.")

    acquired = _parse_datetime(
        properties.get("datetime"), field_name=f"STAC item {item_id} datetime"
    )
    bbox_value = item.get("bbox")
    if not isinstance(bbox_value, Sequence) or len(bbox_value) < 4:
        try:
            bbox_value = shape(geometry).bounds
        except Exception as exc:
            raise ReferenceImageryError(
                f"STAC item {item_id!r} has invalid geometry/bbox."
            ) from exc
    bbox_tuple = tuple(float(value) for value in bbox_value[:4])
    if not all(math.isfinite(value) for value in bbox_tuple):
        raise ReferenceImageryError(f"STAC item {item_id!r} has a non-finite bbox.")

    provider_keys = dict(asset_keys or {key: key for key in REQUIRED_S2_ASSETS})
    missing_roles = set(REQUIRED_S2_ASSETS).difference(provider_keys)
    if missing_roles:
        raise ReferenceImageryError(
            "Provider asset mapping is missing canonical roles: "
            + ", ".join(sorted(missing_roles))
        )

    assets: dict[str, StacAsset] = {}
    for canonical in REQUIRED_S2_ASSETS:
        provider_key = provider_keys[canonical]
        payload = raw_assets.get(provider_key)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("href"), str):
            raise ReferenceImageryError(
                f"STAC item {item_id!r} is missing required asset {provider_key!r} "
                f"for {canonical}."
            )
        raster = _asset_raster_metadata(payload)
        scale = float(raster.get("scale", 1.0))
        offset = float(raster.get("offset", 0.0))
        nodata = raster.get("nodata")
        resolution = raster.get("spatial_resolution")
        href = str(payload["href"])
        assets[canonical] = StacAsset(
            key=provider_key,
            band_name=S2_ASSET_BAND_NAMES[canonical],
            href=href,
            source_identifier=stable_source_url(href),
            media_type=(
                str(payload["type"]) if payload.get("type") is not None else None
            ),
            scale=scale,
            offset=offset,
            nodata=nodata if isinstance(nodata, (int, float)) else None,
            spatial_resolution_m=(
                float(resolution) if resolution is not None else None
            ),
        )

    cloud_keys = (
        "eo:cloud_cover",
        "s2:cloud_shadow_percentage",
        "s2:high_proba_clouds_percentage",
        "s2:medium_proba_clouds_percentage",
        "s2:thin_cirrus_percentage",
        "s2:snow_ice_percentage",
        "s2:nodata_pixel_percentage",
    )
    cloud_metadata = {
        key: float(properties[key])
        for key in cloud_keys
        if isinstance(properties.get(key), (int, float))
    }
    scene_cloud = cloud_metadata.get("eo:cloud_cover")
    epsg = properties.get("proj:epsg")
    if epsg is not None and not isinstance(epsg, int):
        try:
            epsg = int(epsg)
        except (TypeError, ValueError) as exc:
            raise ReferenceImageryError(
                f"STAC item {item_id!r} has invalid proj:epsg metadata."
            ) from exc
    mgrs = properties.get("grid:code") or properties.get("s2:mgrs_tile")
    if isinstance(mgrs, str) and mgrs.startswith("MGRS-"):
        mgrs = mgrs.removeprefix("MGRS-")
    product_type = properties.get("s2:product_type")
    processing_level = str(product_type) if product_type is not None else None

    return SentinelScene(
        provider=provider,
        stac_endpoint=stac_endpoint.rstrip("/"),
        collection=collection,
        item_id=item_id,
        acquisition_datetime=acquired,
        processing_level=processing_level,
        processing_baseline=(
            str(properties["s2:processing_baseline"])
            if properties.get("s2:processing_baseline") is not None
            else None
        ),
        scene_cloud_cover_percent=scene_cloud,
        mgrs_tile=str(mgrs) if mgrs is not None else None,
        epsg=epsg,
        bbox=bbox_tuple,
        geometry=dict(geometry),
        assets=assets,
        query_time=_parse_datetime(query_time, field_name="query_time"),
        search_window_days=search_window_days,
        datatake_id=(
            str(properties["s2:datatake_id"])
            if properties.get("s2:datatake_id") is not None
            else None
        ),
        platform=(
            str(properties["platform"])
            if properties.get("platform") is not None
            else None
        ),
        cloud_metadata=cloud_metadata,
    )


class SentinelStacProvider(Protocol):
    """Protocol for an isolated Sentinel-2 STAC search adapter."""

    provider_name: str
    endpoint: str
    collection: str

    def search(
        self,
        *,
        bbox_wgs84: tuple[float, float, float, float],
        start: datetime,
        end: datetime,
        search_window_days: int,
    ) -> tuple[SentinelScene, ...]:
        """Return parsed scenes intersecting a space/time query."""


JsonTransport = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


def _post_json(url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    request = Request(
        url,
        data=json.dumps(dict(payload)).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "swot-pixc-lab/reference-imagery",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed adapter URL
            result = json.load(response)
    except Exception:
        raise ReferenceImageryError(
            f"Public STAC search failed at {_redacted_url_for_message(url)}. "
            "Retry later or select a "
            "different configured Sentinel provider."
        ) from None
    if not isinstance(result, Mapping):
        raise ReferenceImageryError("STAC search returned a non-object response.")
    return result


@dataclass(frozen=True)
class EarthSearchSentinelProvider:
    """Element84 Earth Search v1 Sentinel-2 L2A adapter."""

    endpoint: str = EARTH_SEARCH_ENDPOINT
    collection: str = EARTH_SEARCH_SENTINEL2_L2A_COLLECTION
    provider_name: str = "Element84 Earth Search"
    transport: JsonTransport = field(default=_post_json, repr=False, compare=False)

    def search(
        self,
        *,
        bbox_wgs84: tuple[float, float, float, float],
        start: datetime,
        end: datetime,
        search_window_days: int,
    ) -> tuple[SentinelScene, ...]:
        """Search public Earth Search without credentials or signed URLs."""

        start = _parse_datetime(start, field_name="search start")
        end = _parse_datetime(end, field_name="search end")
        if start > end:
            raise ReferenceImageryError("Sentinel search start must not exceed end.")
        if len(bbox_wgs84) != 4 or not all(
            math.isfinite(float(value)) for value in bbox_wgs84
        ):
            raise ReferenceImageryError(
                "Sentinel search bbox must contain four values."
            )
        query_time = datetime.now(UTC)
        payload = {
            "collections": [self.collection],
            "bbox": [float(value) for value in bbox_wgs84],
            "datetime": f"{_isoformat_z(start)}/{_isoformat_z(end)}",
            "limit": 100,
        }
        response = self.transport(f"{self.endpoint.rstrip('/')}/search", payload)
        features = response.get("features")
        if not isinstance(features, Sequence):
            raise ReferenceImageryError("STAC search response has no features array.")
        parsed = [
            parse_stac_item(
                feature,
                provider=self.provider_name,
                stac_endpoint=self.endpoint,
                collection=self.collection,
                query_time=query_time,
                search_window_days=search_window_days,
            )
            for feature in features
            if isinstance(feature, Mapping)
        ]
        unique = {scene.item_id: scene for scene in parsed}
        return tuple(
            sorted(
                unique.values(),
                key=lambda scene: (scene.acquisition_datetime, scene.item_id),
            )
        )


@dataclass(frozen=True)
class SearchResult:
    """Scenes and the explicitly attempted progressive search windows."""

    scenes: tuple[SentinelScene, ...]
    windows_queried_days: tuple[int, ...]


def search_progressive_windows(
    provider: SentinelStacProvider,
    *,
    bbox_wgs84: tuple[float, float, float, float],
    swot_datetime: str | datetime,
    windows_days: Sequence[int] = (7, 14, 30),
    is_sufficient: Callable[[tuple[SentinelScene, ...]], bool] | None = None,
) -> SearchResult:
    """Search ±7, then ±14, then ±30 days without silently exceeding 30.

    The callback is a preparation policy hook, not a scientific classifier. It
    receives all deduplicated candidates found so far. The default stops after
    the first window that returns any scenes.
    """

    if tuple(windows_days) != tuple(sorted(set(windows_days))):
        raise ReferenceImageryError("Search windows must be unique and increasing.")
    if any(days not in {7, 14, 30} for days in windows_days):
        raise ReferenceImageryError("Search windows are limited to 7, 14, or 30 days.")
    swot = _parse_datetime(swot_datetime, field_name="swot_datetime")
    sufficient = is_sufficient or bool
    scenes: dict[str, SentinelScene] = {}
    attempted: list[int] = []
    for days in windows_days:
        attempted.append(days)
        found = provider.search(
            bbox_wgs84=bbox_wgs84,
            start=swot - timedelta(days=days),
            end=swot + timedelta(days=days),
            search_window_days=days,
        )
        for scene in found:
            existing = scenes.get(scene.item_id)
            if (
                existing is None
                or scene.search_window_days < existing.search_window_days
            ):
                scenes[scene.item_id] = scene
        ordered = tuple(
            sorted(
                scenes.values(),
                key=lambda scene: (scene.acquisition_datetime, scene.item_id),
            )
        )
        if sufficient(ordered):
            return SearchResult(ordered, tuple(attempted))
    return SearchResult(
        tuple(
            sorted(
                scenes.values(),
                key=lambda scene: (scene.acquisition_datetime, scene.item_id),
            )
        ),
        tuple(attempted),
    )


@dataclass(frozen=True)
class TemporalProvenance:
    """Sentinel timing relative to an explicit SWOT reference instant."""

    sentinel_datetime: str
    swot_reference_datetime: str
    signed_offset_hours: float
    absolute_offset_hours: float
    descriptive_flag: str


@dataclass(frozen=True)
class SwotTemporalContext:
    """Observed SWOT time coverage and the declared comparison instant."""

    coverage_start: str
    coverage_end: str
    reference_datetime: str
    reference_basis: str
    source_files: tuple[str, ...]
    source_time_coverage: tuple[Mapping[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible temporal provenance."""

        return asdict(self)


def read_swot_temporal_context(paths: Sequence[str | Path]) -> SwotTemporalContext:
    """Read exact root time coverage from local SWOT NetCDF granules.

    The comparison instant is the explicitly recorded midpoint of the combined
    coverage interval; both measured endpoints and all source files are retained.
    """

    if not paths:
        raise ReferenceImageryError("At least one SWOT NetCDF file is required.")
    try:
        from netCDF4 import Dataset
    except ImportError as exc:  # pragma: no cover - core dependency is declared
        raise ReferenceImageryError(
            "Reading SWOT timing requires the declared netCDF4 dependency."
        ) from exc
    starts: list[datetime] = []
    ends: list[datetime] = []
    records: list[Mapping[str, str]] = []
    sources: list[str] = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise ReferenceImageryError(f"SWOT NetCDF file does not exist: {path}")
        try:
            with Dataset(path, mode="r") as dataset:
                start_value = dataset.getncattr("time_coverage_start")
                end_value = dataset.getncattr("time_coverage_end")
        except Exception as exc:
            raise ReferenceImageryError(
                f"Could not read time_coverage_start/end from {path.name}: {exc}"
            ) from exc
        start = _parse_datetime(
            str(start_value), field_name=f"{path.name} time_coverage_start"
        )
        end = _parse_datetime(
            str(end_value), field_name=f"{path.name} time_coverage_end"
        )
        if start > end:
            raise ReferenceImageryError(
                f"SWOT time coverage is reversed in {path.name}."
            )
        starts.append(start)
        ends.append(end)
        sources.append(str(path))
        records.append(
            {
                "filename": path.name,
                "time_coverage_start": _isoformat_z(start),
                "time_coverage_end": _isoformat_z(end),
            }
        )
    combined_start = min(starts)
    combined_end = max(ends)
    midpoint = combined_start + (combined_end - combined_start) / 2
    return SwotTemporalContext(
        coverage_start=_isoformat_z(combined_start),
        coverage_end=_isoformat_z(combined_end),
        reference_datetime=_isoformat_z(midpoint),
        reference_basis="midpoint_of_combined_netcdf_time_coverage",
        source_files=tuple(sources),
        source_time_coverage=tuple(records),
    )


def temporal_provenance(
    sentinel_datetime: str | datetime,
    swot_datetime: str | datetime,
) -> TemporalProvenance:
    """Calculate signed Sentinel-minus-SWOT timing and a descriptive flag."""

    sentinel = _parse_datetime(
        sentinel_datetime, field_name="sentinel acquisition datetime"
    )
    swot = _parse_datetime(swot_datetime, field_name="SWOT reference datetime")
    hours = (sentinel - swot).total_seconds() / 3600.0
    absolute = abs(hours)
    if sentinel.date() == swot.date():
        flag = "SAME_DAY"
    elif absolute <= 3 * 24:
        flag = "WITHIN_3_DAYS"
    elif absolute <= 7 * 24:
        flag = "WITHIN_7_DAYS"
    elif absolute <= 14 * 24:
        flag = "WITHIN_14_DAYS"
    elif absolute <= 30 * 24:
        flag = "WITHIN_30_DAYS"
    else:
        flag = "BEYOND_30_DAYS"
    return TemporalProvenance(
        sentinel_datetime=_isoformat_z(sentinel),
        swot_reference_datetime=_isoformat_z(swot),
        signed_offset_hours=hours,
        absolute_offset_hours=absolute,
        descriptive_flag=flag,
    )


@dataclass(frozen=True)
class ReviewRegion:
    """A full finite transect plus a metric imagery-context buffer."""

    bounds_wgs84: tuple[float, float, float, float]
    polygon_wgs84: Mapping[str, Any]
    context_margin_m: float
    construction_crs_wkt: str


def make_review_region(
    transect: LineString,
    *,
    context_margin_m: float = 1500.0,
) -> ReviewRegion:
    """Buffer a WGS84 finite line in a local metric CRS without AOI clipping."""

    if not isinstance(transect, LineString) or transect.is_empty:
        raise ReferenceImageryError("transect must be a non-empty LineString.")
    if len(transect.coords) < 2 or not transect.is_valid:
        raise ReferenceImageryError("transect must contain a valid finite line.")
    coords = np.asarray(transect.coords, dtype=float)
    if coords.shape[1] < 2 or not np.isfinite(coords[:, :2]).all():
        raise ReferenceImageryError("transect coordinates must be finite WGS84 values.")
    if not math.isfinite(context_margin_m) or context_margin_m <= 0:
        raise ReferenceImageryError("context_margin_m must be positive and finite.")
    center = transect.centroid
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center.y:.12f} +lon_0={center.x:.12f} "
        "+datum=WGS84 +units=m +no_defs"
    )
    forward = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True).transform
    inverse = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True).transform
    metric_line = transform(forward, transect)
    buffered = transform(inverse, metric_line.buffer(context_margin_m))
    bounds = tuple(float(value) for value in buffered.bounds)
    return ReviewRegion(
        bounds_wgs84=bounds,
        polygon_wgs84=mapping(buffered),
        context_margin_m=float(context_margin_m),
        construction_crs_wkt=local_crs.to_wkt(),
    )


def scenes_intersecting_region(
    scenes: Iterable[SentinelScene],
    region: ReviewRegion,
) -> tuple[SentinelScene, ...]:
    """Return scenes whose STAC footprints intersect the review region."""

    region_shape = shape(region.polygon_wgs84)
    result = []
    for scene in scenes:
        try:
            footprint = shape(scene.geometry)
        except Exception as exc:
            raise ReferenceImageryError(
                f"STAC item {scene.item_id!r} has invalid footprint geometry."
            ) from exc
        if footprint.intersects(region_shape):
            result.append(scene)
    return tuple(
        sorted(result, key=lambda value: (value.acquisition_datetime, value.item_id))
    )


def group_same_acquisition(
    scenes: Iterable[SentinelScene],
) -> dict[str, tuple[SentinelScene, ...]]:
    """Group spatial tiles from one Sentinel datatake; never group different dates."""

    groups: dict[str, list[SentinelScene]] = {}
    for scene in scenes:
        groups.setdefault(scene.acquisition_key, []).append(scene)
    return {
        key: tuple(
            sorted(values, key=lambda item: (item.acquisition_datetime, item.item_id))
        )
        for key, values in sorted(groups.items())
    }


@dataclass(frozen=True)
class SclDiagnostics:
    """Auditable local SCL fractions used only to assess image usability."""

    total_pixel_count: int
    valid_pixel_count: int
    class_counts: Mapping[str, int]
    class_fractions: Mapping[str, float]
    valid_fraction: float
    cloud_fraction: float
    cloud_shadow_fraction: float
    cirrus_fraction: float
    snow_ice_fraction: float
    no_data_fraction: float
    saturated_defective_fraction: float
    non_obscured_diagnostic_fraction: float
    obscured_diagnostic_fraction: float
    unexpected_value_count: int
    interpretation: str = (
        "SCL image-usability diagnostics only; SCL class 6 is not reference truth"
    )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible diagnostic metadata."""

        return asdict(self)


def calculate_scl_diagnostics(scl: np.ndarray) -> SclDiagnostics:
    """Compute exact local SCL class fractions without deriving wet intervals."""

    values = np.asarray(scl)
    if values.size == 0:
        raise ReferenceImageryError("SCL diagnostic array must not be empty.")
    if np.issubdtype(values.dtype, np.floating):
        finite = np.isfinite(values)
        integer_like = finite & (values == np.floor(values))
        numeric = values[integer_like].astype(np.int16)
        invalid_numeric = int(values.size - integer_like.sum())
    else:
        numeric = values.astype(np.int64, copy=False).ravel()
        invalid_numeric = 0
    numeric = np.asarray(numeric).ravel()
    total = int(values.size)
    raw_counts = {code: int(np.count_nonzero(numeric == code)) for code in range(12)}
    known_count = sum(raw_counts.values())
    unexpected = total - known_count + invalid_numeric
    # ``invalid_numeric`` is already outside ``numeric`` for floating inputs.
    if np.issubdtype(values.dtype, np.floating):
        unexpected = total - known_count
    class_counts = {SCL_CLASS_NAMES[code]: raw_counts[code] for code in range(12)}
    class_fractions = {name: count / total for name, count in class_counts.items()}

    def fraction(codes: Iterable[int]) -> float:
        return sum(raw_counts[code] for code in codes) / total

    no_data = fraction(SCL_NO_DATA_CLASSES)
    return SclDiagnostics(
        total_pixel_count=total,
        valid_pixel_count=total - raw_counts[0] - unexpected,
        class_counts=class_counts,
        class_fractions=class_fractions,
        valid_fraction=(total - raw_counts[0] - unexpected) / total,
        cloud_fraction=fraction(SCL_CLOUD_CLASSES),
        cloud_shadow_fraction=fraction(SCL_CLOUD_SHADOW_CLASSES),
        cirrus_fraction=fraction(SCL_CIRRUS_CLASSES),
        snow_ice_fraction=fraction(SCL_SNOW_ICE_CLASSES),
        no_data_fraction=no_data,
        saturated_defective_fraction=fraction(SCL_SATURATED_DEFECTIVE_CLASSES),
        non_obscured_diagnostic_fraction=fraction(SCL_NON_OBSCURED_DIAGNOSTIC_CLASSES),
        obscured_diagnostic_fraction=fraction(SCL_OBSCURED_DIAGNOSTIC_CLASSES),
        unexpected_value_count=unexpected,
    )


def apply_asset_scale(
    values: np.ndarray,
    *,
    scale: float,
    offset: float,
    nodata: int | float | None,
) -> np.ndarray:
    """Convert raw asset values using STAC raster scale/offset and retain nodata."""

    raw = np.asarray(values)
    result = raw.astype(np.float32) * np.float32(scale) + np.float32(offset)
    if nodata is not None:
        result = np.where(raw == nodata, np.nan, result)
    return result.astype(np.float32, copy=False)


def calculate_ndwi(
    green_reflectance: np.ndarray, nir_reflectance: np.ndarray
) -> np.ndarray:
    """Return continuous NDWI with NaN at nodata or a zero denominator.

    This function intentionally has no threshold argument and returns no mask or
    intervals. The result is a visualization diagnostic only.
    """

    green = np.asarray(green_reflectance, dtype=np.float32)
    nir = np.asarray(nir_reflectance, dtype=np.float32)
    if green.shape != nir.shape:
        raise ReferenceImageryError("B03 and B08 arrays must have identical shapes.")
    denominator = green + nir
    valid = np.isfinite(green) & np.isfinite(nir) & (denominator != 0)
    result = np.full(green.shape, np.nan, dtype=np.float32)
    np.divide(green - nir, denominator, out=result, where=valid)
    return result


def scale_rgb(
    red_reflectance: np.ndarray,
    green_reflectance: np.ndarray,
    blue_reflectance: np.ndarray,
    *,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Create a deterministic uint8 review RGB using per-band percentile stretch."""

    bands = [
        np.asarray(red_reflectance, dtype=np.float32),
        np.asarray(green_reflectance, dtype=np.float32),
        np.asarray(blue_reflectance, dtype=np.float32),
    ]
    if len({band.shape for band in bands}) != 1:
        raise ReferenceImageryError("B04, B03, and B02 arrays must share one grid.")
    if not (0 <= lower_percentile < upper_percentile <= 100):
        raise ReferenceImageryError(
            "RGB percentiles must satisfy 0 <= low < high <= 100."
        )
    if not math.isfinite(gamma) or gamma <= 0:
        raise ReferenceImageryError("RGB gamma must be positive and finite.")
    output = np.zeros((*bands[0].shape, 3), dtype=np.uint8)
    for index, band in enumerate(bands):
        finite = np.isfinite(band)
        if not finite.any():
            continue
        low, high = np.nanpercentile(
            band,
            [lower_percentile, upper_percentile],
        )
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            scaled = np.where(finite, 0.0, 0.0)
        else:
            scaled = np.clip((band - low) / (high - low), 0.0, 1.0)
        scaled = np.power(scaled, 1.0 / gamma, where=finite, out=np.zeros_like(scaled))
        output[..., index] = np.where(finite, np.round(scaled * 255), 0).astype(
            np.uint8
        )
    return output


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a local file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class AssetChipRecord:
    """Reproducibility record for one local windowed COG asset chip."""

    schema_version: str
    provider: str
    collection: str
    stac_item_id: str
    acquisition_key: str
    acquisition_datetime: str
    mgrs_tile: str | None
    source_asset_key: str
    canonical_asset_role: str
    source_band_name: str
    source_identifier: str
    request_bounds_wgs84: tuple[float, float, float, float]
    raster_bounds: tuple[float, float, float, float]
    crs: str
    resolution: tuple[float, float]
    width: int
    height: int
    transform: tuple[float, ...]
    dtype: str
    stac_nodata: int | float | None
    nodata: int | float | None
    scale: float
    offset: float
    local_path: str
    sha256: str
    prepared_at: str
    cache_status: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible chip metadata."""

        return asdict(self)


def _commit_file_no_clobber(source: Path, target: Path) -> None:
    """Copy into a newly created target without hard-link ACL inheritance."""

    copied_identity: tuple[int, int, int, int, int, int] | None = None
    try:
        with source.open("rb") as source_stream:
            try:
                target_stream = target.open("xb", buffering=0)
            except FileExistsError as exc:
                raise ReferenceImageryError(
                    "Cache target appeared concurrently and was not overwritten: "
                    f"{target}"
                ) from exc
            with target_stream:
                stat = os.fstat(target_stream.fileno())
                copied_identity = (
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_mode,
                    stat.st_size,
                    stat.st_mtime_ns,
                    getattr(stat, "st_birthtime_ns", stat.st_ctime_ns),
                )
                try:
                    shutil.copyfileobj(source_stream, target_stream, length=1024 * 1024)
                    target_stream.flush()
                    os.fsync(target_stream.fileno())
                finally:
                    stat = os.fstat(target_stream.fileno())
                    copied_identity = (
                        stat.st_dev,
                        stat.st_ino,
                        stat.st_mode,
                        stat.st_size,
                        stat.st_mtime_ns,
                        getattr(stat, "st_birthtime_ns", stat.st_ctime_ns),
                    )
    except FileExistsError as exc:
        raise ReferenceImageryError(
            f"Cache target appeared concurrently and was not overwritten: {target}"
        ) from exc
    except Exception:
        # Roll back only if every recorded identity field still matches. This
        # protects an externally replaced path even when an inode is reused.
        try:
            stat = target.stat(follow_symlinks=False)
            current_identity = (
                stat.st_dev,
                stat.st_ino,
                stat.st_mode,
                stat.st_size,
                stat.st_mtime_ns,
                getattr(stat, "st_birthtime_ns", stat.st_ctime_ns),
            )
            if copied_identity is not None and current_identity == copied_identity:
                target.unlink()
        except OSError:
            pass
        raise


def _chip_sidecar_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.json")


def _load_cached_chip(
    target: Path,
    *,
    scene: SentinelScene,
    asset: StacAsset,
    asset_role: str,
    bounds_wgs84: tuple[float, float, float, float],
) -> AssetChipRecord | None:
    sidecar = _chip_sidecar_path(target)
    if not target.exists() and not sidecar.exists():
        return None
    if not target.exists() or not sidecar.exists():
        raise ReferenceImageryError(
            f"Incomplete imagery cache entry at {target}; remove the chip and its "
            "sidecar, then rerun."
        )
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        for tuple_field in (
            "request_bounds_wgs84",
            "raster_bounds",
            "resolution",
            "transform",
        ):
            payload[tuple_field] = tuple(payload[tuple_field])
        record = AssetChipRecord(**payload)
    except Exception as exc:
        raise ReferenceImageryError(
            f"Imagery cache metadata is invalid at {sidecar}."
        ) from exc
    expected_bounds = tuple(float(value) for value in bounds_wgs84)
    if (
        record.provider != scene.provider
        or record.collection != scene.collection
        or record.stac_item_id != scene.item_id
        or record.acquisition_key != scene.acquisition_key
        or record.source_asset_key != asset.key
        or record.canonical_asset_role != asset_role
        or tuple(record.request_bounds_wgs84) != expected_bounds
        or record.source_identifier != asset.source_identifier
        or not math.isclose(record.scale, asset.scale, rel_tol=0, abs_tol=1e-15)
        or not math.isclose(record.offset, asset.offset, rel_tol=0, abs_tol=1e-15)
        or record.stac_nodata != asset.nodata
    ):
        raise ReferenceImageryError(
            f"Existing cache target {target} belongs to a different request; it was "
            "not overwritten."
        )
    current_hash = sha256_file(target)
    if current_hash != record.sha256:
        raise ReferenceImageryError(
            f"Cached imagery hash mismatch at {target}; remove it and rerun."
        )
    return AssetChipRecord(**{**record.to_dict(), "cache_status": "hit"})


def read_windowed_cog_asset(
    scene: SentinelScene,
    asset_role: str,
    *,
    bounds_wgs84: tuple[float, float, float, float],
    output_path: str | Path,
) -> AssetChipRecord:
    """Read and cache only the intersecting window of one public COG asset."""

    rasterio = _require_rasterio()
    from rasterio.coords import BoundingBox
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds

    if asset_role not in scene.assets:
        raise ReferenceImageryError(
            f"Scene {scene.item_id!r} has no canonical {asset_role!r} asset."
        )
    asset = scene.assets[asset_role]
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    cached = _load_cached_chip(
        target,
        scene=scene,
        asset=asset,
        asset_role=asset_role,
        bounds_wgs84=bounds_wgs84,
    )
    if cached is not None:
        return cached

    temp_directory = Path(
        tempfile.mkdtemp(prefix=f".{target.stem}-", dir=target.parent)
    )
    temporary = temp_directory / target.name
    try:
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
                with rasterio.open(asset.href) as source:
                    if source.crs is None:
                        raise ReferenceImageryError(
                            f"Asset {asset.source_identifier} has no raster CRS."
                        )
                    projected = transform_bounds(
                        "EPSG:4326",
                        source.crs,
                        *bounds_wgs84,
                        densify_pts=21,
                    )
                    requested = BoundingBox(*projected)
                    intersection = BoundingBox(
                        max(requested.left, source.bounds.left),
                        max(requested.bottom, source.bounds.bottom),
                        min(requested.right, source.bounds.right),
                        min(requested.top, source.bounds.top),
                    )
                    if (
                        intersection.left >= intersection.right
                        or intersection.bottom >= intersection.top
                    ):
                        raise ReferenceImageryError(
                            f"STAC item {scene.item_id!r} does not cover the requested "
                            f"chip bounds for {asset_role}."
                        )
                    window = from_bounds(*intersection, transform=source.transform)
                    window = window.round_offsets().round_lengths()
                    window = window.intersection(
                        Window(0, 0, source.width, source.height)
                    )
                    values = source.read(1, window=window)
                    if values.size == 0:
                        raise ReferenceImageryError(
                            "Window read returned no pixels for "
                            f"{scene.item_id} {asset_role}."
                        )
                    chip_transform = source.window_transform(window)
                    profile = source.profile.copy()
                    profile.update(
                        driver="GTiff",
                        count=1,
                        width=values.shape[1],
                        height=values.shape[0],
                        transform=chip_transform,
                        compress="deflate",
                        tiled=False,
                    )
                    with rasterio.open(temporary, "w", **profile) as destination:
                        destination.write(values, 1)
                        destination.update_tags(
                            stac_item_id=scene.item_id,
                            source_asset_key=asset.key,
                            source_identifier=asset.source_identifier,
                            provider=scene.provider,
                        )
                    raster_bounds = rasterio.transform.array_bounds(
                        values.shape[0], values.shape[1], chip_transform
                    )
                    crs_text = source.crs.to_string()
                    resolution = (
                        abs(float(chip_transform.a)),
                        abs(float(chip_transform.e)),
                    )
                    dtype = str(values.dtype)
                    nodata = (
                        source.nodata if source.nodata is not None else asset.nodata
                    )
        except ReferenceImageryError:
            raise
        except Exception:
            raise ReferenceImageryError(
                f"Windowed COG read failed for {scene.item_id} {asset_role}. "
                "Check public asset availability, network range-read support, and "
                "the recorded stable source identifier."
            ) from None

        digest = sha256_file(temporary)
        prepared_at = _isoformat_z(datetime.now(UTC))
        record = AssetChipRecord(
            schema_version="1.1",
            provider=scene.provider,
            collection=scene.collection,
            stac_item_id=scene.item_id,
            acquisition_key=scene.acquisition_key,
            acquisition_datetime=_isoformat_z(scene.acquisition_datetime),
            mgrs_tile=scene.mgrs_tile,
            source_asset_key=asset.key,
            canonical_asset_role=asset_role,
            source_band_name=asset.band_name,
            source_identifier=asset.source_identifier,
            request_bounds_wgs84=tuple(float(value) for value in bounds_wgs84),
            raster_bounds=tuple(float(value) for value in raster_bounds),
            crs=crs_text,
            resolution=resolution,
            width=int(values.shape[1]),
            height=int(values.shape[0]),
            transform=tuple(float(value) for value in chip_transform),
            dtype=dtype,
            stac_nodata=asset.nodata,
            nodata=nodata,
            scale=asset.scale,
            offset=asset.offset,
            local_path=str(target),
            sha256=digest,
            prepared_at=prepared_at,
            cache_status="created",
        )
        _commit_file_no_clobber(temporary, target)
        sidecar = _chip_sidecar_path(target)
        sidecar_temp = temp_directory / sidecar.name
        sidecar_temp.write_text(
            json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            _commit_file_no_clobber(sidecar_temp, sidecar)
        except Exception:
            # The committed raster is deliberately retained rather than silently
            # overwritten. A rerun reports the incomplete transaction explicitly.
            raise
        return record
    finally:
        shutil.rmtree(temp_directory, ignore_errors=True)


@dataclass(frozen=True)
class MosaicRecord:
    """A same-acquisition, non-averaged mosaic of one or more spatial tile chips."""

    asset_role: str
    source_chip_records: tuple[AssetChipRecord, ...]
    local_path: str
    sha256: str
    crs: str
    bounds: tuple[float, float, float, float]
    resolution: tuple[float, float]
    transform: tuple[float, ...]
    width: int
    height: int
    dtype: str
    nodata: int | float | None
    scale: float
    offset: float
    method: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible mosaic provenance."""

        return {
            **asdict(self),
            "source_chip_records": [
                record.to_dict() for record in self.source_chip_records
            ],
        }


def mosaic_same_acquisition_chips(
    records: Sequence[AssetChipRecord],
    *,
    asset_role: str,
    output_path: str | Path,
) -> MosaicRecord:
    """Mosaic spatial tiles from one acquisition using first-valid, never averaging."""

    rasterio = _require_rasterio()
    from rasterio.merge import merge

    if not records:
        raise ReferenceImageryError("At least one chip is required for a mosaic.")
    acquisition_keys = {record.acquisition_key for record in records}
    if len(acquisition_keys) != 1:
        raise ReferenceImageryError(
            "Refusing to mosaic chips from different Sentinel datatakes."
        )
    canonical_roles = {record.canonical_asset_role for record in records}
    expected_band_name = S2_ASSET_BAND_NAMES.get(asset_role)
    band_names = {record.source_band_name for record in records}
    if canonical_roles != {asset_role} or band_names != {expected_band_name}:
        raise ReferenceImageryError(
            "Refusing to mosaic chips from different spectral asset roles."
        )
    acquisition_times = [
        _parse_datetime(record.acquisition_datetime, field_name="chip acquisition")
        for record in records
    ]
    if max(acquisition_times) - min(acquisition_times) > timedelta(minutes=2):
        raise ReferenceImageryError(
            "Refusing to mosaic chips that are not spatial tiles from the same "
            "near-simultaneous acquisition."
        )
    scales = {(record.scale, record.offset) for record in records}
    if len(scales) != 1:
        raise ReferenceImageryError(
            "Cannot mosaic asset chips with different scale/offset metadata."
        )
    raster_signatures = {
        (record.dtype, record.nodata, record.resolution) for record in records
    }
    if len(raster_signatures) != 1:
        raise ReferenceImageryError(
            "Cannot mosaic asset chips with different dtype, nodata, or resolution."
        )
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(record.local_path) for record in records]
    if len(records) == 1:
        record = records[0]
        return MosaicRecord(
            asset_role=asset_role,
            source_chip_records=(record,),
            local_path=record.local_path,
            sha256=record.sha256,
            crs=record.crs,
            bounds=record.raster_bounds,
            resolution=record.resolution,
            transform=record.transform,
            width=record.width,
            height=record.height,
            dtype=record.dtype,
            nodata=record.nodata,
            scale=record.scale,
            offset=record.offset,
            method="single_spatial_tile",
        )
    sources = [rasterio.open(path) for path in source_paths]
    try:
        crs_values = {
            source.crs.to_string() if source.crs else None for source in sources
        }
        if len(crs_values) != 1 or None in crs_values:
            raise ReferenceImageryError(
                "Spatial tile chips do not share one known CRS."
            )
        nodata = records[0].nodata
        mosaic, transform_value = merge(sources, nodata=nodata, method="first")
        profile = sources[0].profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            width=mosaic.shape[2],
            height=mosaic.shape[1],
            transform=transform_value,
            compress="deflate",
            tiled=False,
        )
        temp_directory = Path(
            tempfile.mkdtemp(prefix=f".{target.stem}-", dir=target.parent)
        )
        temporary = temp_directory / target.name
        try:
            with rasterio.open(temporary, "w", **profile) as destination:
                destination.write(mosaic[0], 1)
                destination.update_tags(
                    mosaic_method="first-valid-no-averaging",
                    source_items=",".join(record.stac_item_id for record in records),
                )
            digest = sha256_file(temporary)
            if target.exists():
                if sha256_file(target) != digest:
                    raise ReferenceImageryError(
                        "Existing mosaic cache target differs and was not "
                        f"overwritten: {target}"
                    )
            else:
                _commit_file_no_clobber(temporary, target)
        finally:
            shutil.rmtree(temp_directory, ignore_errors=True)
        bounds = rasterio.transform.array_bounds(
            mosaic.shape[1], mosaic.shape[2], transform_value
        )
        return MosaicRecord(
            asset_role=asset_role,
            source_chip_records=tuple(records),
            local_path=str(target),
            sha256=sha256_file(target),
            crs=str(next(iter(crs_values))),
            bounds=tuple(float(value) for value in bounds),
            resolution=(abs(float(transform_value.a)), abs(float(transform_value.e))),
            transform=tuple(float(value) for value in transform_value),
            width=int(mosaic.shape[2]),
            height=int(mosaic.shape[1]),
            dtype=str(mosaic.dtype),
            nodata=nodata,
            scale=records[0].scale,
            offset=records[0].offset,
            method="first_valid_spatial_tile_no_averaging",
        )
    finally:
        for source in sources:
            source.close()


def read_scaled_mosaic(record: MosaicRecord) -> np.ndarray:
    """Read a cached mosaic and apply its recorded scale/offset."""

    rasterio = _require_rasterio()
    try:
        with rasterio.open(record.local_path) as source:
            values = source.read(1)
    except Exception as exc:
        raise ReferenceImageryError(
            f"Could not read cached imagery mosaic {record.local_path}: {exc}"
        ) from exc
    return apply_asset_scale(
        values,
        scale=record.scale,
        offset=record.offset,
        nodata=record.nodata,
    )


def read_raw_mosaic(record: MosaicRecord) -> np.ndarray:
    """Read native categorical/raw values from a cached mosaic."""

    rasterio = _require_rasterio()
    try:
        with rasterio.open(record.local_path) as source:
            return source.read(1)
    except Exception as exc:
        raise ReferenceImageryError(
            f"Could not read cached imagery mosaic {record.local_path}: {exc}"
        ) from exc


def review_region_pixel_mask(
    record: MosaicRecord,
    region: ReviewRegion,
) -> np.ndarray:
    """Select raster cells whose centers fall inside the exact review polygon.

    The WGS84 review polygon is transformed to the mosaic CRS. This keeps the
    SCL diagnostic denominator tied to the finite transect's metric context
    buffer, rather than its larger rectangular COG read window. The explicit
    pixel-center rule (``all_touched=False``) is deterministic.
    """

    rasterio = _require_rasterio()
    from rasterio.features import geometry_mask

    if not isinstance(record, MosaicRecord):
        raise TypeError("record must be a MosaicRecord.")
    if not isinstance(region, ReviewRegion):
        raise TypeError("region must be a ReviewRegion.")
    try:
        polygon = shape(region.polygon_wgs84)
        projector = Transformer.from_crs(
            "EPSG:4326", record.crs, always_xy=True
        ).transform
        projected = transform(projector, polygon)
        included = geometry_mask(
            [mapping(projected)],
            out_shape=(record.height, record.width),
            transform=rasterio.Affine(*record.transform[:6]),
            all_touched=False,
            invert=True,
        )
    except Exception as exc:
        raise ReferenceImageryError(
            "Could not map the transect review polygon onto the imagery raster."
        ) from exc
    if not np.any(included):
        raise ReferenceImageryError(
            "The transect review polygon contains no imagery pixel centers."
        )
    return included


def shortlist_acquisitions(
    candidates: Sequence[Mapping[str, Any]],
    *,
    swot_datetime: str | datetime,
    maximum_count: int = 3,
) -> tuple[Mapping[str, Any], ...]:
    """Select transparent before/after/clearest acquisition candidates.

    Candidate mappings must include ``acquisition_datetime``, a unique
    ``candidate_id``, and ``local_obscured_fraction``. This is scene review
    ordering only; it does not create reference truth.
    """

    if maximum_count < 1 or maximum_count > 3:
        raise ReferenceImageryError("maximum_count must be between 1 and 3.")
    swot = _parse_datetime(swot_datetime, field_name="swot_datetime")
    normalized: list[tuple[Mapping[str, Any], datetime, float]] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ReferenceImageryError("Every candidate needs a unique candidate_id.")
        if candidate_id in seen:
            raise ReferenceImageryError(f"Duplicate candidate_id {candidate_id!r}.")
        seen.add(candidate_id)
        acquired = _parse_datetime(
            candidate.get("acquisition_datetime"),
            field_name=f"candidate {candidate_id} acquisition_datetime",
        )
        obscured = candidate.get("local_obscured_fraction")
        if not isinstance(obscured, (int, float)) or not math.isfinite(obscured):
            raise ReferenceImageryError(
                f"Candidate {candidate_id!r} needs finite local_obscured_fraction."
            )
        normalized.append((candidate, acquired, float(obscured)))

    selected: list[Mapping[str, Any]] = []

    def add(value: Mapping[str, Any] | None) -> None:
        if (
            value is not None
            and value not in selected
            and len(selected) < maximum_count
        ):
            selected.append(value)

    before = [entry for entry in normalized if entry[1] <= swot]
    after = [entry for entry in normalized if entry[1] > swot]
    before.sort(key=lambda entry: (swot - entry[1], entry[2], entry[0]["candidate_id"]))
    after.sort(key=lambda entry: (entry[1] - swot, entry[2], entry[0]["candidate_id"]))
    add(before[0][0] if before else None)
    add(after[0][0] if after else None)
    remaining = [entry for entry in normalized if entry[0] not in selected]
    remaining.sort(
        key=lambda entry: (
            entry[2],
            abs((entry[1] - swot).total_seconds()),
            entry[0]["candidate_id"],
        )
    )
    for entry in remaining:
        add(entry[0])
    return tuple(selected)


def json_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a mapping using stable canonical JSON serialization."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AssetChipRecord",
    "EARTH_SEARCH_ENDPOINT",
    "EARTH_SEARCH_SENTINEL2_L2A_COLLECTION",
    "EarthSearchSentinelProvider",
    "MosaicRecord",
    "NDWI_FORMULA",
    "NDWI_INTERPRETATION",
    "REQUIRED_S2_ASSETS",
    "ReviewRegion",
    "SCL_CLASS_NAMES",
    "SCL_NON_OBSCURED_DIAGNOSTIC_CLASSES",
    "SCL_OBSCURED_DIAGNOSTIC_CLASSES",
    "SearchResult",
    "SentinelScene",
    "SentinelStacProvider",
    "SclDiagnostics",
    "StacAsset",
    "SwotTemporalContext",
    "TemporalProvenance",
    "apply_asset_scale",
    "calculate_ndwi",
    "calculate_scl_diagnostics",
    "group_same_acquisition",
    "json_sha256",
    "make_review_region",
    "mosaic_same_acquisition_chips",
    "parse_stac_item",
    "read_raw_mosaic",
    "read_scaled_mosaic",
    "read_swot_temporal_context",
    "read_windowed_cog_asset",
    "review_region_pixel_mask",
    "scale_rgb",
    "scenes_intersecting_region",
    "search_progressive_windows",
    "sha256_file",
    "shortlist_acquisitions",
    "stable_source_url",
    "temporal_provenance",
]
