"""Public discovery, local-resolution, and PIXC-opening APIs."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import pandas as pd

from .aoi import (
    AoiInput,
    DateLike,
    SpatialQuery,
    normalize_aoi,
    normalize_temporal_bounds,
)
from .constants import DEFAULT_PIXC_COLLECTION_CONCEPT_ID, METADATA_COLUMNS
from .discovery import (
    EarthaccessBackend,
    EarthdataBackend,
    GranuleRecord,
    granule_identity,
    parse_granule,
    sort_records,
)
from .download import (
    VerificationMode,
    download_records,
    resolve_local_records,
)
from .exceptions import DiscoveryError
from .mosaic import PixcObservation, _validate_record_selection, open_pixc

LOGGER = logging.getLogger(__name__)
_COLLECTION_ID = re.compile(r"^C\d+-[A-Za-z0-9_]+$")
_PIXC_SHORT_NAME = re.compile(r"^SWOT_L2_HR_PIXC(?:_[A-Za-z0-9.]+)?$")


@dataclass(frozen=True, slots=True)
class SearchProvenance:
    """Immutable description of the CMR query that produced a collection."""

    collection_concept_id: str
    temporal: tuple[str, str]
    spatial_queries: tuple[SpatialQuery, ...]
    notes: tuple[str, ...] = ()
    count: int = -1

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable provenance mapping."""

        return {
            "collection_concept_id": self.collection_concept_id,
            "temporal": list(self.temporal),
            "spatial_queries": [query.as_kwargs() for query in self.spatial_queries],
            "notes": list(self.notes),
            "count": self.count,
        }


class PixcCollection:
    """A discovered set of SWOT L2 HR PIXC granules.

    Discovery metadata and opaque Earthdata handles remain separate from the
    raw point data. :meth:`open` resolves verified local files and performs the
    Phase-2 raw read, exact clip, and no-loss tile combination.
    """

    def __init__(
        self,
        records: Sequence[GranuleRecord],
        *,
        handles: Mapping[str, Any] | None = None,
        backend: EarthdataBackend | None = None,
        collection_concept_id: str = DEFAULT_PIXC_COLLECTION_CONCEPT_ID,
        search_notes: Sequence[str] = (),
        provenance: SearchProvenance | None = None,
    ) -> None:
        self._records = tuple(records)
        self._handles = dict(handles or {})
        self._backend = backend or EarthaccessBackend()
        self.collection_concept_id = collection_concept_id
        self.search_notes = tuple(search_notes)
        self.provenance = provenance

    @classmethod
    def search(
        cls,
        aoi: AoiInput,
        start_date: DateLike,
        end_date: DateLike,
        *,
        collection_concept_id: str = DEFAULT_PIXC_COLLECTION_CONCEPT_ID,
        backend: EarthdataBackend | None = None,
    ) -> Self:
        """Discover PIXC granules intersecting an AOI and inclusive date range.

        Parameters
        ----------
        aoi:
            WGS 84 bounding box ``(west, south, east, north)``, GeoJSON
            Polygon/MultiPolygon (including Feature wrappers), or a
            GeoDataFrame-like object with a declared CRS.
        start_date, end_date:
            ISO 8601 strings, ``date`` objects, or ``datetime`` objects. A
            date-only ``end_date`` includes the complete day.
        collection_concept_id:
            Exact NASA CMR collection concept ID. The default pins the current
            PO.DAAC Version D PIXC collection for reproducibility.
        backend:
            Advanced injectable Earthdata backend, mainly for testing.
        """

        if not isinstance(collection_concept_id, str) or not _COLLECTION_ID.fullmatch(
            collection_concept_id
        ):
            raise ValueError(
                "collection_concept_id must look like a NASA CMR collection ID "
                "(for example, C3233944986-POCLOUD)."
            )
        normalized_aoi = normalize_aoi(aoi)
        temporal = normalize_temporal_bounds(start_date, end_date)
        selected_backend = backend or EarthaccessBackend()

        raw_by_identity: dict[str, Mapping[str, Any]] = {}
        handles: dict[str, Any] = {}
        for index, spatial in enumerate(normalized_aoi.queries, start=1):
            kwargs: dict[str, Any] = {
                "concept_id": collection_concept_id,
                "temporal": temporal,
                **spatial.as_kwargs(),
            }
            try:
                results = selected_backend.search_data(count=-1, **kwargs)
            except Exception as exc:
                raise DiscoveryError(
                    f"NASA CMR search {index} of {len(normalized_aoi.queries)} "
                    "failed; no partial collection was returned."
                ) from exc

            for raw in results:
                if not isinstance(raw, Mapping):
                    raise DiscoveryError(
                        "earthaccess returned a granule that is not mapping-like."
                    )
                identity = granule_identity(raw)
                if identity not in raw_by_identity:
                    raw_by_identity[identity] = raw
                    handles[identity] = raw

        records = [
            parse_granule(
                raw,
                fallback_collection_concept_id=collection_concept_id,
            )
            for raw in raw_by_identity.values()
        ]
        for record in records:
            if record.product_short_name and not _PIXC_SHORT_NAME.fullmatch(
                record.product_short_name
            ):
                raise DiscoveryError(
                    "The requested collection returned a non-PIXC product: "
                    f"{record.product_short_name!r}. Supply a SWOT L2 HR PIXC "
                    "collection concept ID."
                )
        ordered = sort_records(records)
        LOGGER.info(
            "Discovered %d unique PIXC granules with %d CMR spatial query/queries.",
            len(ordered),
            len(normalized_aoi.queries),
        )
        return cls(
            ordered,
            handles=handles,
            backend=selected_backend,
            collection_concept_id=collection_concept_id,
            search_notes=normalized_aoi.notes,
            provenance=SearchProvenance(
                collection_concept_id=collection_concept_id,
                temporal=temporal,
                spatial_queries=normalized_aoi.queries,
                notes=normalized_aoi.notes,
            ),
        )

    @property
    def records(self) -> tuple[GranuleRecord, ...]:
        """Return immutable granule metadata records."""

        return self._records

    @property
    def table(self) -> pd.DataFrame:
        """Return a defensive copy of the stable discovery metadata table."""

        return _records_to_frame(self._records)

    @property
    def metadata(self) -> pd.DataFrame:
        """Alias for :attr:`table`."""

        return self.table

    @property
    def local_files(self) -> tuple[Path, ...]:
        """Return currently resolved local paths, without performing I/O."""

        return tuple(
            record.local_path
            for record in self._records
            if record.local_path is not None
        )

    def download(
        self,
        destination: str | os.PathLike[str],
        *,
        auth_strategy: str | None = "all",
        persist_credentials: bool = False,
        threads: int = 8,
        verify: VerificationMode = "auto",
        show_progress: bool | None = None,
    ) -> Self:
        """Download missing granules into a verified local cache.

        Existing files are reused only after verification. New downloads first
        land in a temporary staging directory and are committed to the cache
        only after all files pass validation. Credentials are delegated to
        ``earthaccess`` and are never stored by this package.

        The collection is updated in place and returned so both the documented
        two-line workflow and method chaining are supported.
        """

        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer.")
        self._records = download_records(
            self._records,
            self._handles,
            self._backend,
            destination,
            auth_strategy=auth_strategy,
            persist_credentials=persist_credentials,
            threads=threads,
            verify=verify,
            show_progress=show_progress,
        )
        return self

    def resolve_local(
        self,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        verify: VerificationMode = "auto",
    ) -> LocalPixcCollection:
        """Resolve and verify a Phase-1 local manifest.

        This method performs local verification only and never downloads data.
        It does not open or parse NetCDF content. The ``open`` name is reserved
        for the Phase-2 PIXC NetCDF parsing API.
        """

        resolved = resolve_local_records(
            self._records,
            cache_dir=cache_dir,
            verify=verify,
        )
        self._records = resolved
        return LocalPixcCollection(resolved, provenance=self.provenance)

    def open(
        self,
        *,
        aoi: AoiInput,
        cache_dir: str | os.PathLike[str] | None = None,
        verify: VerificationMode = "auto",
        variables: Sequence[str] | None = None,
    ) -> PixcObservation:
        """Open verified local PIXC files as one raw clipped observation.

        This method never downloads implicitly. It preserves documented raw
        values and quality bitfields, clips exactly to ``aoi``, and concatenates
        source tiles without filtering, averaging, or deduplication.
        """

        _validate_record_selection(self._records)
        local = self.resolve_local(cache_dir, verify=verify)
        return local.open(aoi=aoi, variables=variables)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[GranuleRecord]:
        return iter(self._records)

    def __repr__(self) -> str:
        downloaded = sum(record.local_path is not None for record in self._records)
        return (
            f"PixcCollection(granules={len(self)}, local={downloaded}, "
            f"collection_concept_id={self.collection_concept_id!r})"
        )


@dataclass(frozen=True, slots=True)
class LocalPixcCollection:
    """Verified local PIXC file manifest returned by ``resolve_local``."""

    records: tuple[GranuleRecord, ...]
    provenance: SearchProvenance | None = None

    @property
    def paths(self) -> tuple[Path, ...]:
        """Return verified local PIXC paths in metadata-table order."""

        return tuple(record.local_path for record in self.records if record.local_path)

    @property
    def table(self) -> pd.DataFrame:
        """Return metadata and provenance for the local files."""

        return _records_to_frame(self.records)

    def open(
        self,
        *,
        aoi: AoiInput,
        variables: Sequence[str] | None = None,
    ) -> PixcObservation:
        """Read, exactly clip, and concatenate this verified local manifest."""

        return open_pixc(
            self.paths,
            aoi=aoi,
            records=self.records,
            variables=variables,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[Path]:
        return iter(self.paths)


def _records_to_frame(records: Sequence[GranuleRecord]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [record.as_row() for record in records],
        columns=METADATA_COLUMNS,
    )
    for column in (
        "observation_datetime",
        "observation_start",
        "observation_end",
        "production_datetime",
        "revision_date",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    for column in ("cycle", "pass", "size_bytes", "revision_id"):
        frame[column] = pd.array(frame[column], dtype="Int64")
    for column in (
        "tile",
        "granule_id",
        "native_id",
        "filename",
        "product_short_name",
        "product_version",
        "concept_id",
        "collection_concept_id",
        "provider",
        "metadata_specification_version",
        "source_url",
        "s3_url",
        "checksum",
        "checksum_algorithm",
        "pge_version",
        "local_path",
    ):
        frame[column] = pd.array(frame[column], dtype="string")
    return frame
