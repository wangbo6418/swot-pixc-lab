"""No-loss combination of exactly clipped SWOT PIXC tile point arrays."""

from __future__ import annotations

import copy
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from .aoi import AoiInput
from .discovery import GranuleRecord
from .exceptions import ObservationMismatchError, PixcOpenError, PixcSchemaError
from .reader import LoadedPixcSource, PixcSourceMetadata, read_pixc_source
from .subset import ExactAoi, normalize_exact_aoi

if TYPE_CHECKING:
    from .qc import QCResult


@dataclass(frozen=True, slots=True)
class PixcObservation:
    """Raw PIXC points clipped and concatenated without QC or deduplication.

    ``pixels`` contains original point-variable names and raw values. The
    integer ``source_index`` variable maps every retained point to one entry in
    ``sources``; ``source_point_index`` is that sample's original zero-based
    index within the source file's ``/pixel_cloud/points`` dimension.
    """

    pixels: xr.Dataset
    sources: tuple[PixcSourceMetadata, ...]
    aoi: ExactAoi
    notes: tuple[str, ...] = ()

    @property
    def raw(self) -> xr.Dataset:
        """Return the raw, unfiltered clipped pixel dataset."""

        return self.pixels

    @property
    def source_table(self) -> pd.DataFrame:
        """Return one provenance/count row for every source, including zeros."""

        return pd.DataFrame(
            [
                {
                    "source_index": source.source_index,
                    "granule_id": source.granule_id,
                    "filename": source.filename,
                    "path": str(source.path),
                    "cycle": source.cycle,
                    "pass": source.pass_number,
                    "tile": source.tile,
                    "crid": source.crid,
                    "netcdf_product_version": source.netcdf_product_version,
                    "netcdf_pge_version": source.netcdf_pge_version,
                    "collection_version": (
                        source.cmr_record.product_version
                        if source.cmr_record is not None
                        else None
                    ),
                    "cmr_concept_id": (
                        source.cmr_record.concept_id
                        if source.cmr_record is not None
                        else None
                    ),
                    "cmr_revision_id": (
                        source.cmr_record.revision_id
                        if source.cmr_record is not None
                        else None
                    ),
                    "points_before": source.points_before,
                    "points_after": source.points_after,
                    "invalid_coordinates": source.invalid_coordinate_count,
                    "normalized_longitudes": source.normalized_longitude_count,
                }
                for source in self.sources
            ]
        )

    @property
    def pixel_count(self) -> int:
        """Return the number of retained pixels across all source tiles."""

        return int(self.pixels.sizes["points"])

    @property
    def classification_counts(self) -> dict[int, int]:
        """Count raw classification values, excluding only their fill value."""

        values = self._valid_values("classification")
        if values is None:
            return {}
        labels, counts = np.unique(values, return_counts=True)
        return {
            int(label): int(count) for label, count in zip(labels, counts, strict=True)
        }

    @property
    def coordinate_ranges(
        self,
    ) -> dict[str, tuple[float, float] | None]:
        """Return raw clipped longitude and latitude ranges."""

        return {
            name: _numeric_range(self._valid_values(name))
            for name in ("longitude", "latitude")
        }

    @property
    def height_summary(self) -> dict[str, float | int | None]:
        """Describe raw ellipsoidal ``height`` values without applying QC."""

        values = self._valid_values("height")
        if values is None or not values.size:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "std": None,
            }
        numeric = values.astype(np.float64, copy=False)
        return {
            "count": int(numeric.size),
            "min": float(np.min(numeric)),
            "max": float(np.max(numeric)),
            "mean": float(np.mean(numeric)),
            "median": float(np.median(numeric)),
            "std": float(np.std(numeric)),
        }

    @property
    def source_tile_counts(self) -> dict[str, int]:
        """Return retained-pixel counts aggregated by original source tile."""

        counts: dict[str, int] = {}
        for source in self.sources:
            label = source.tile or "unknown"
            counts[label] = counts.get(label, 0) + source.points_after
        return counts

    def summary(self) -> dict[str, object]:
        """Return Phase-2 ingestion and clipping diagnostics."""

        return {
            "pixels_before": sum(source.points_before for source in self.sources),
            "pixels_after": self.pixel_count,
            "invalid_coordinates": sum(
                source.invalid_coordinate_count for source in self.sources
            ),
            "classification_counts": self.classification_counts,
            "coordinate_ranges": self.coordinate_ranges,
            "height": self.height_summary,
            "source_tile_counts": self.source_tile_counts,
        }

    def apply_qc(
        self,
        profile: str = "raw",
    ) -> QCResult:
        """Apply a named Phase-3 QC profile without modifying raw PIXC values."""

        from .qc import apply_qc

        return apply_qc(self, profile=profile)

    def _valid_values(self, name: str) -> NDArray[Any] | None:
        if name not in self.pixels:
            return None
        values = np.asarray(self.pixels[name].values)
        valid = np.ones(values.shape, dtype=np.bool_)
        if np.issubdtype(values.dtype, np.floating):
            valid &= np.isfinite(values)
        source_indices = np.asarray(self.pixels["source_index"].values)
        for source in self.sources:
            variable = source.variables.get(name)
            if variable is None or variable.fill_value is None:
                continue
            source_points = source_indices == source.source_index
            valid[source_points] &= values[source_points] != variable.fill_value
        return values[valid]

    def __len__(self) -> int:
        return self.pixel_count


def open_pixc(
    paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    *,
    aoi: AoiInput,
    records: Sequence[GranuleRecord] | None = None,
    variables: Sequence[str] | None = None,
) -> PixcObservation:
    """Open, exactly clip, and combine one logical PIXC observation.

    Files are processed in the supplied order. Every retained point is kept;
    this function never averages, deduplicates, or applies quality filtering.
    All files must describe the same SWOT cycle and pass.
    """

    if isinstance(paths, (str, os.PathLike)):
        source_paths = (Path(paths),)
    else:
        source_paths = tuple(Path(path) for path in paths)
    if not source_paths:
        raise PixcOpenError("At least one local PIXC file is required.")
    if isinstance(variables, (str, bytes, bytearray)):
        raise TypeError(
            "variables must be a sequence of complete /pixel_cloud variable names, "
            "not one string."
        )

    resolved_paths = tuple(path.expanduser().resolve() for path in source_paths)
    duplicate_paths = sorted(
        {str(path) for path in resolved_paths if resolved_paths.count(path) > 1}
    )
    if duplicate_paths:
        raise PixcOpenError(
            "Each PIXC source path may be opened only once; duplicate input(s): "
            + ", ".join(duplicate_paths)
        )
    source_paths = resolved_paths
    if records is None:
        source_records: tuple[GranuleRecord | None, ...] = (None,) * len(source_paths)
    else:
        if len(records) != len(source_paths):
            raise ValueError("records must correspond one-to-one with paths.")
        source_records = tuple(records)
        _validate_record_selection(source_records)

    exact_aoi = normalize_exact_aoi(aoi)
    loaded: list[LoadedPixcSource] = []
    for index, (path, record) in enumerate(
        zip(source_paths, source_records, strict=True)
    ):
        source = read_pixc_source(
            path,
            aoi=exact_aoi,
            source_index=index,
            record=record,
            variables=variables,
        )
        _validate_same_observation((*loaded, source))
        loaded.append(source)
    return combine_pixc_sources(loaded, aoi=exact_aoi)


def _validate_record_selection(records: Sequence[GranuleRecord]) -> None:
    if len(records) < 2:
        return
    known_observations = {
        (record.cycle, record.pass_number)
        for record in records
        if record.cycle is not None and record.pass_number is not None
    }
    if len(known_observations) > 1:
        raise ObservationMismatchError(
            "Cannot open a manifest containing different SWOT cycle/pass "
            "observations; select one observation first."
        )
    known_tiles = [record.tile for record in records if record.tile is not None]
    duplicate_tiles = sorted(
        {tile for tile in known_tiles if known_tiles.count(tile) > 1}
    )
    if duplicate_tiles:
        raise ObservationMismatchError(
            "Cannot open multiple CMR granules for the same tile without an "
            "explicit revision-selection policy; select one source for tile(s): "
            + ", ".join(duplicate_tiles)
        )


def combine_pixc_sources(
    sources: Sequence[LoadedPixcSource],
    *,
    aoi: ExactAoi,
) -> PixcObservation:
    """Concatenate clipped point arrays while retaining source provenance."""

    if not sources:
        raise PixcOpenError("At least one clipped PIXC source is required.")
    _validate_same_observation(sources)
    variable_names = sources[0].metadata.loaded_variables
    for source in sources[1:]:
        if source.metadata.loaded_variables != variable_names:
            raise PixcSchemaError(
                "PIXC sources expose different point-variable sets; no variables "
                "were silently dropped. Select an explicit compatible set."
            )

    data_vars: dict[str, xr.DataArray] = {}
    notes = [note for source in sources for note in source.metadata.notes]
    notes.extend(_processing_generation_notes(sources))
    for name in variable_names:
        dtype = np.dtype(sources[0].arrays[name].dtype)
        for source in sources[1:]:
            if np.dtype(source.arrays[name].dtype) != dtype:
                raise PixcSchemaError(
                    f"/pixel_cloud/{name} has inconsistent dtypes across sources."
                )
        values = np.concatenate([source.arrays[name] for source in sources])
        attributes, attributes_vary = _common_variable_attributes(sources, name)
        if attributes_vary:
            notes.append(
                f"Attributes for /pixel_cloud/{name} vary by source; exact "
                "definitions remain in observation.sources."
            )
        data_vars[name] = xr.DataArray(values, dims=("points",), attrs=attributes)

    source_index = np.concatenate(
        [
            np.full(
                source.metadata.points_after,
                source.metadata.source_index,
                dtype=np.int32,
            )
            for source in sources
        ]
    )
    source_point_index = np.concatenate(
        [source.source_point_index for source in sources]
    )
    data_vars["source_index"] = xr.DataArray(
        source_index,
        dims=("points",),
        attrs={
            "long_name": "swot_pixc_lab source table index",
            "comment": (
                "Maps each point to PixcObservation.sources; not a SWOT variable."
            ),
        },
    )
    data_vars["source_point_index"] = xr.DataArray(
        source_point_index,
        dims=("points",),
        attrs={
            "long_name": "zero-based point index in the source PIXC granule",
            "comment": "Provenance added by swot-pixc-lab; not a SWOT variable.",
        },
    )
    dataset = xr.Dataset(
        data_vars=data_vars,
        attrs={
            "swot_pixc_lab_aoi_wkt": aoi.wkt,
            "swot_pixc_lab_processing": (
                "exact spatial clip and stable source-order concatenation; "
                "no QC, additional height correction, WSE transformation, "
                "averaging, or deduplication"
            ),
        },
    )
    return PixcObservation(
        pixels=dataset,
        sources=tuple(source.metadata for source in sources),
        aoi=aoi,
        notes=tuple(dict.fromkeys(notes)),
    )


def _validate_same_observation(sources: Sequence[LoadedPixcSource]) -> None:
    if len(sources) == 1:
        return
    missing_identity = [
        source.metadata.filename
        for source in sources
        if source.metadata.cycle is None
        or source.metadata.pass_number is None
        or source.metadata.tile is None
    ]
    if missing_identity:
        raise ObservationMismatchError(
            "Cannot combine multiple PIXC files without a resolved cycle and pass "
            "and tile for every source; missing identity for: "
            + ", ".join(missing_identity)
        )

    identities = {
        (source.metadata.cycle, source.metadata.pass_number) for source in sources
    }
    if len(identities) > 1:
        descriptions = ", ".join(
            f"{source.metadata.filename}="
            f"cycle {source.metadata.cycle}/pass {source.metadata.pass_number}"
            for source in sources
        )
        raise ObservationMismatchError(
            "Cannot combine different SWOT cycle/pass observations: " + descriptions
        )

    tiles = [source.metadata.tile for source in sources if source.metadata.tile]
    duplicate_tiles = sorted({tile for tile in tiles if tiles.count(tile) > 1})
    if duplicate_tiles:
        raise ObservationMismatchError(
            "Cannot combine multiple PIXC granules for the same tile without an "
            "explicit revision-selection policy; select one source for tile(s): "
            + ", ".join(duplicate_tiles)
        )


def _common_variable_attributes(
    sources: Sequence[LoadedPixcSource],
    name: str,
) -> tuple[dict[str, Any], bool]:
    definitions = [source.metadata.variables[name].attributes for source in sources]
    first = definitions[0]
    common: dict[str, Any] = {}
    varied = False
    all_keys = set().union(*(definition.keys() for definition in definitions))
    for key in all_keys:
        if key not in first:
            varied = True
            continue
        value = first[key]
        if all(
            key in definition and _attribute_equal(value, definition[key])
            for definition in definitions[1:]
        ):
            common[key] = copy.deepcopy(value)
        else:
            varied = True
    return common, varied


def _processing_generation_notes(
    sources: Sequence[LoadedPixcSource],
) -> list[str]:
    fields: tuple[tuple[str, list[object]], ...] = (
        ("NetCDF CRID", [source.metadata.crid for source in sources]),
        (
            "NetCDF product_version",
            [source.metadata.netcdf_product_version for source in sources],
        ),
        (
            "NetCDF PGE version",
            [source.metadata.netcdf_pge_version for source in sources],
        ),
        (
            "CMR collection version",
            [
                source.metadata.cmr_record.product_version
                if source.metadata.cmr_record is not None
                else None
                for source in sources
            ],
        ),
    )
    notes: list[str] = []
    for label, values in fields:
        if any(value is not None for value in values) and len(set(values)) > 1:
            notes.append(
                f"{label} differs or is missing across source tiles: "
                + ", ".join(
                    "missing" if value is None else str(value) for value in values
                )
                + "."
            )
    return notes


def _attribute_equal(left: object, right: object) -> bool:
    try:
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        if left_array.shape != right_array.shape:
            return False
        try:
            return bool(np.array_equal(left_array, right_array, equal_nan=True))
        except TypeError:
            return bool(np.array_equal(left_array, right_array))
    except (TypeError, ValueError):
        return left == right


def _numeric_range(values: NDArray[Any] | None) -> tuple[float, float] | None:
    if values is None or not values.size:
        return None
    return float(np.min(values)), float(np.max(values))
