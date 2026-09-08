"""NASA CMR discovery backend and transparent PIXC metadata parsing."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

from .exceptions import DiscoveryError

_PIXC_FILENAME = re.compile(
    r"^SWOT_L2_HR_PIXC_(?P<cycle>\d{3})_(?P<pass>\d{3})_"
    r"(?P<tile>\d{3}[LRF])_.+\.nc(?:4)?$",
    re.IGNORECASE,
)


class EarthdataBackend(Protocol):
    """Injectable boundary around the external Earthdata client."""

    def search_data(self, *, count: int = -1, **kwargs: Any) -> Sequence[Any]:
        """Search CMR and return Earthdata granule objects."""

    def download(
        self,
        granules: Sequence[Any],
        destination: Path,
        *,
        auth_strategy: str | None,
        persist_credentials: bool,
        threads: int,
        show_progress: bool | None,
    ) -> Sequence[Path]:
        """Download granules to a staging directory."""


class EarthaccessBackend:
    """Official NASA ``earthaccess`` implementation of the backend protocol."""

    def search_data(self, *, count: int = -1, **kwargs: Any) -> Sequence[Any]:
        """Search NASA's Common Metadata Repository without requiring login."""

        earthaccess = _import_earthaccess()
        return earthaccess.search_data(count=count, **kwargs)

    def download(
        self,
        granules: Sequence[Any],
        destination: Path,
        *,
        auth_strategy: str | None,
        persist_credentials: bool,
        threads: int,
        show_progress: bool | None,
    ) -> Sequence[Path]:
        """Authenticate using normal Earthdata mechanisms and download files."""

        earthaccess = _import_earthaccess()
        if auth_strategy is not None:
            earthaccess.login(
                strategy=auth_strategy,
                persist=persist_credentials,
            )
        return earthaccess.download(
            list(granules),
            destination,
            threads=threads,
            show_progress=show_progress,
        )


@dataclass(frozen=True, slots=True)
class GranuleRecord:
    """Stable Phase-1 metadata for one CMR granule."""

    identity: str
    observation_start: datetime | None
    observation_end: datetime | None
    cycle: int | None
    pass_number: int | None
    tile: str | None
    granule_id: str | None
    native_id: str | None
    filename: str | None
    product_short_name: str | None
    product_version: str | None
    concept_id: str | None
    collection_concept_id: str | None
    provider: str | None
    revision_id: int | None
    revision_date: datetime | None
    metadata_specification_version: str | None
    source_url: str | None
    s3_url: str | None
    size_bytes: int | None
    checksum: str | None
    checksum_algorithm: str | None
    pge_version: str | None
    production_datetime: datetime | None
    granule_bbox: tuple[float, float, float, float] | None
    local_path: Path | None = None
    metadata_warnings: tuple[str, ...] = ()

    def with_local_path(self, path: Path) -> GranuleRecord:
        """Return a copy carrying an absolute, verified cache path."""

        return replace(self, local_path=path.resolve())

    def as_row(self) -> dict[str, object]:
        """Return the public metadata-table row for this granule."""

        return {
            "observation_datetime": self.observation_start,
            "observation_start": self.observation_start,
            "observation_end": self.observation_end,
            "cycle": self.cycle,
            "pass": self.pass_number,
            "tile": self.tile,
            "granule_id": self.granule_id,
            "native_id": self.native_id,
            "filename": self.filename,
            "product_short_name": self.product_short_name,
            "product_version": self.product_version,
            "concept_id": self.concept_id,
            "collection_concept_id": self.collection_concept_id,
            "provider": self.provider,
            "revision_id": self.revision_id,
            "revision_date": self.revision_date,
            "metadata_specification_version": self.metadata_specification_version,
            "source_url": self.source_url,
            "s3_url": self.s3_url,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "checksum_algorithm": self.checksum_algorithm,
            "pge_version": self.pge_version,
            "production_datetime": self.production_datetime,
            "granule_bbox": self.granule_bbox,
            "local_path": str(self.local_path) if self.local_path else None,
            "metadata_warnings": self.metadata_warnings,
        }


def granule_identity(raw: Mapping[str, Any]) -> str:
    """Build a conservative identity key without merging product versions."""

    meta = _mapping(raw.get("meta"))
    concept_id = _text(meta.get("concept-id"))
    if concept_id:
        return f"concept:{concept_id}"

    umm = _mapping(raw.get("umm"))
    reference = _mapping(umm.get("CollectionReference"))
    granule_ur = _text(umm.get("GranuleUR"))
    if granule_ur:
        return "umm:{short}:{version}:{granule}".format(
            short=_text(reference.get("ShortName")) or "?",
            version=_text(reference.get("Version")) or "?",
            granule=granule_ur,
        )

    source_url, _ = _select_data_urls(umm)
    if source_url:
        return f"url:{source_url}"
    raise DiscoveryError(
        "CMR returned a granule without concept-id, GranuleUR, or a data URL; "
        "it cannot be deduplicated safely."
    )


def parse_granule(
    raw: Mapping[str, Any], *, fallback_collection_concept_id: str | None = None
) -> GranuleRecord:
    """Parse a CMR UMM-G mapping without inventing missing metadata."""

    identity = granule_identity(raw)
    meta = _mapping(raw.get("meta"))
    umm = _mapping(raw.get("umm"))
    warnings_: list[str] = []

    reference = _mapping(umm.get("CollectionReference"))
    temporal = _mapping(umm.get("TemporalExtent"))
    range_time = _mapping(temporal.get("RangeDateTime"))
    start = _parse_datetime(range_time.get("BeginningDateTime"), warnings_, "start")
    end = _parse_datetime(range_time.get("EndingDateTime"), warnings_, "end")

    source_url, s3_url = _select_data_urls(umm)
    filename = _producer_filename(umm, source_url)
    cycle, pass_number, tile = _track_fields(umm, filename, warnings_)
    size_bytes, checksum, checksum_algorithm = _archive_metadata(umm, filename)

    production = _mapping(umm.get("DataGranule"))
    production_datetime = _parse_datetime(
        production.get("ProductionDateTime"), warnings_, "production"
    )
    pge = _mapping(umm.get("PGEVersionClass"))
    specification = _mapping(umm.get("MetadataSpecification"))
    revision_date = _parse_datetime(meta.get("revision-date"), warnings_, "revision")

    return GranuleRecord(
        identity=identity,
        observation_start=start,
        observation_end=end,
        cycle=cycle,
        pass_number=pass_number,
        tile=tile,
        granule_id=_text(umm.get("GranuleUR")),
        native_id=_text(meta.get("native-id")),
        filename=filename,
        product_short_name=_text(reference.get("ShortName")),
        product_version=_text(reference.get("Version")),
        concept_id=_text(meta.get("concept-id")),
        collection_concept_id=(
            _text(meta.get("collection-concept-id")) or fallback_collection_concept_id
        ),
        provider=_text(meta.get("provider-id")),
        revision_id=_integer(meta.get("revision-id")),
        revision_date=revision_date,
        metadata_specification_version=_text(specification.get("Version")),
        source_url=source_url,
        s3_url=s3_url,
        size_bytes=size_bytes,
        checksum=checksum,
        checksum_algorithm=checksum_algorithm,
        pge_version=_text(pge.get("PGEVersion")),
        production_datetime=production_datetime,
        granule_bbox=_granule_bbox(umm),
        metadata_warnings=tuple(warnings_),
    )


def sort_records(records: Sequence[GranuleRecord]) -> list[GranuleRecord]:
    """Return records in deterministic observation/track/name order."""

    latest = datetime.max.replace(tzinfo=UTC)
    return sorted(
        records,
        key=lambda record: (
            record.observation_start or latest,
            record.cycle if record.cycle is not None else 10**9,
            record.pass_number if record.pass_number is not None else 10**9,
            record.tile or "",
            record.granule_id or record.identity,
        ),
    )


def safe_filename(filename: str | None) -> str:
    """Validate that a CMR-provided filename cannot escape a cache folder."""

    if not filename:
        raise DiscoveryError(
            "Granule metadata does not contain a downloadable filename."
        )
    decoded = unquote(filename)
    if decoded in {"", ".", ".."} or Path(decoded).name != decoded:
        raise DiscoveryError(f"Unsafe granule filename in CMR metadata: {filename!r}")
    if "/" in decoded or "\\" in decoded or "\x00" in decoded:
        raise DiscoveryError(f"Unsafe granule filename in CMR metadata: {filename!r}")
    return decoded


def _import_earthaccess() -> Any:
    try:
        import earthaccess
    except ImportError as exc:  # pragma: no cover - declared project dependency
        raise DiscoveryError(
            "earthaccess is required for NASA CMR discovery and download. "
            "Install the project with `pip install -e .`."
        ) from exc
    return earthaccess


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _parse_datetime(
    value: object, warnings_: list[str], field_name: str
) -> datetime | None:
    text = _text(value)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        warnings_.append(f"Invalid CMR {field_name} datetime: {text!r}")
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _select_data_urls(umm: Mapping[str, Any]) -> tuple[str | None, str | None]:
    https: list[str] = []
    s3: list[str] = []
    for related in umm.get("RelatedUrls", []):
        item = _mapping(related)
        url = _text(item.get("URL"))
        link_type = _text(item.get("Type"))
        if not url:
            continue
        is_netcdf = _url_suffix(url) in {".nc", ".nc4"}
        if link_type == "GET DATA" and url.startswith(("https://", "http://")):
            if is_netcdf:
                https.append(url)
        elif link_type == "GET DATA VIA DIRECT ACCESS" and url.startswith("s3://"):
            if is_netcdf:
                s3.append(url)
    return (min(https) if https else None, min(s3) if s3 else None)


def _url_suffix(url: str) -> str:
    return Path(urlparse(url).path).suffix.lower()


def _producer_filename(umm: Mapping[str, Any], source_url: str | None) -> str | None:
    data_granule = _mapping(umm.get("DataGranule"))
    identifiers = data_granule.get("Identifiers", [])
    candidates: list[str] = []
    for identifier in identifiers:
        item = _mapping(identifier)
        if item.get("IdentifierType") == "ProducerGranuleId":
            value = _text(item.get("Identifier"))
            if value and Path(value).suffix.lower() in {".nc", ".nc4"}:
                candidates.append(value)
    if candidates:
        return safe_filename(min(candidates))
    if source_url:
        basename = unquote(Path(urlparse(source_url).path).name)
        return safe_filename(basename)
    return None


def _track_fields(
    umm: Mapping[str, Any], filename: str | None, warnings_: list[str]
) -> tuple[int | None, int | None, str | None]:
    spatial = _mapping(umm.get("SpatialExtent"))
    horizontal = _mapping(spatial.get("HorizontalSpatialDomain"))
    track = _mapping(horizontal.get("Track"))
    cycle = _integer(track.get("Cycle"))

    pass_numbers: list[int] = []
    tiles: list[str] = []
    for pass_item in track.get("Passes", []):
        pass_mapping = _mapping(pass_item)
        pass_number = _integer(pass_mapping.get("Pass"))
        if pass_number is not None:
            pass_numbers.append(pass_number)
        for tile_value in pass_mapping.get("Tiles", []):
            tile = _text(tile_value)
            if tile:
                tiles.append(tile)

    unique_passes = list(dict.fromkeys(pass_numbers))
    unique_tiles = list(dict.fromkeys(tiles))
    if len(unique_passes) > 1:
        warnings_.append(
            f"CMR track contains multiple passes: {tuple(unique_passes)!r}"
        )
    if len(unique_tiles) > 1:
        warnings_.append(f"CMR track contains multiple tiles: {tuple(unique_tiles)!r}")
    pass_number = unique_passes[0] if unique_passes else None
    tile = unique_tiles[0] if unique_tiles else _additional_tile(umm)

    match = _PIXC_FILENAME.match(filename or "")
    if match:
        if cycle is None:
            cycle = int(match.group("cycle"))
            warnings_.append("Cycle was recovered from the official PIXC filename.")
        if pass_number is None:
            pass_number = int(match.group("pass"))
            warnings_.append("Pass was recovered from the official PIXC filename.")
        if tile is None:
            tile = match.group("tile").upper()
            warnings_.append("Tile was recovered from the official PIXC filename.")
    return cycle, pass_number, tile


def _additional_tile(umm: Mapping[str, Any]) -> str | None:
    for attribute in umm.get("AdditionalAttributes", []):
        item = _mapping(attribute)
        if item.get("Name") == "TILE":
            values = item.get("Values", [])
            if (
                isinstance(values, Sequence)
                and not isinstance(values, (str, bytes, bytearray))
                and values
            ):
                return _text(values[0])
    return None


def _integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _archive_metadata(
    umm: Mapping[str, Any], filename: str | None
) -> tuple[int | None, str | None, str | None]:
    data_granule = _mapping(umm.get("DataGranule"))
    entries = [
        _mapping(entry)
        for entry in data_granule.get("ArchiveAndDistributionInformation", [])
    ]
    matching = [entry for entry in entries if _text(entry.get("Name")) == filename]
    if not matching:
        matching = [
            entry
            for entry in entries
            if _text(entry.get("Name"))
            and Path(str(entry["Name"])).suffix.lower() in {".nc", ".nc4"}
        ]
    if len(matching) != 1:
        return None, None, None
    entry = matching[0]
    size_bytes = _integer(entry.get("SizeInBytes"))
    checksum_mapping = _mapping(entry.get("Checksum"))
    return (
        size_bytes,
        _text(checksum_mapping.get("Value")),
        _text(checksum_mapping.get("Algorithm")),
    )


def _granule_bbox(
    umm: Mapping[str, Any],
) -> tuple[float, float, float, float] | None:
    spatial = _mapping(umm.get("SpatialExtent"))
    horizontal = _mapping(spatial.get("HorizontalSpatialDomain"))
    geometry = _mapping(horizontal.get("Geometry"))
    rectangles = geometry.get("BoundingRectangles", [])
    if not isinstance(rectangles, Sequence) or len(rectangles) != 1:
        return None
    rectangle = _mapping(rectangles[0])
    names = (
        "WestBoundingCoordinate",
        "SouthBoundingCoordinate",
        "EastBoundingCoordinate",
        "NorthBoundingCoordinate",
    )
    try:
        values = tuple(float(rectangle[name]) for name in names)
    except (KeyError, TypeError, ValueError):
        return None
    return values  # type: ignore[return-value]
