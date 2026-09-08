"""Raw, metadata-preserving reader for SWOT L2 HR PIXC NetCDF files."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import netCDF4
import numpy as np
from numpy.typing import NDArray

from .discovery import GranuleRecord
from .exceptions import PixcOpenError, PixcSchemaError
from .subset import ExactAoi, clip_points

REQUIRED_COORDINATES = ("latitude", "longitude")
CORE_POINT_VARIABLES = (
    "azimuth_index",
    "range_index",
    "latitude",
    "longitude",
    "height",
    "classification",
    "water_frac",
    "pixel_area",
    "sig0",
    "cross_track",
)
PHASE3_POINT_VARIABLES = (
    "classification",
    "water_frac",
    "water_frac_uncert",
    "height",
    "geoid",
    "inc",
    "phase_noise_std",
    "bright_land_flag",
    "false_detection_rate",
    "missed_detection_rate",
    "prior_water_prob",
    "prior_water_change",
    "classification_qual",
    "geolocation_qual",
    "interferogram_qual",
    "sig0_qual",
    "ancillary_surface_classification_flag",
)
DEFAULT_POINT_VARIABLES = tuple(
    dict.fromkeys((*CORE_POINT_VARIABLES, *PHASE3_POINT_VARIABLES))
)
PROVENANCE_VARIABLES = frozenset({"source_index", "source_point_index"})
_TILE_NAME = re.compile(r"(?P<pass>\d{3})_(?P<tile>\d{3}[LRF])$")


@dataclass(frozen=True, slots=True)
class PixcVariableMetadata:
    """Original NetCDF definition for one ``/pixel_cloud`` variable."""

    name: str
    dimensions: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    attributes: Mapping[str, Any]

    @property
    def fill_value(self) -> object | None:
        """Return the original ``_FillValue``, if one is defined."""

        return self.attributes.get("_FillValue")


@dataclass(frozen=True, slots=True)
class PixcSourceMetadata:
    """File, tile, schema, and clipping provenance for one PIXC source."""

    source_index: int
    path: Path
    filename: str
    granule_id: str
    cycle: int | None
    pass_number: int | None
    tile: str | None
    crid: str | None
    netcdf_product_version: str | None
    netcdf_pge_version: str | None
    cmr_record: GranuleRecord | None
    data_model: str
    groups: tuple[str, ...]
    dimensions: Mapping[str, int]
    root_attributes: Mapping[str, Any]
    pixel_cloud_attributes: Mapping[str, Any]
    variables: Mapping[str, PixcVariableMetadata]
    loaded_variables: tuple[str, ...]
    points_before: int
    points_after: int
    invalid_coordinate_count: int
    normalized_longitude_count: int
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class LoadedPixcSource:
    """Internal clipped arrays and immutable source metadata."""

    arrays: dict[str, NDArray[Any]]
    source_point_index: NDArray[np.int64]
    metadata: PixcSourceMetadata


def read_pixc_source(
    path: str | Path,
    *,
    aoi: ExactAoi,
    source_index: int,
    record: GranuleRecord | None = None,
    variables: Sequence[str] | None = None,
) -> LoadedPixcSource:
    """Read and exactly clip one PIXC file without CF masking or scaling.

    The returned point arrays retain their original NetCDF dtypes and variable
    attributes. Metadata for every variable in ``/pixel_cloud`` is captured,
    including variables such as ``pixc_line_qual`` that do not use the point
    dimension and therefore cannot be added to the clipped point dataset.
    """

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise PixcOpenError(f"PIXC source is not a readable file: {source_path}")

    try:
        root = netCDF4.Dataset(source_path, mode="r")
    except (OSError, RuntimeError) as exc:
        raise PixcOpenError(f"Could not open PIXC NetCDF file: {source_path}") from exc

    try:
        if "pixel_cloud" not in root.groups:
            raise PixcSchemaError(
                f"PIXC file has no /pixel_cloud group: {source_path.name}"
            )
        group = root.groups["pixel_cloud"]
        _disable_automatic_decoding(group)
        variable_metadata = {
            name: _variable_metadata(name, variable)
            for name, variable in group.variables.items()
        }
        _validate_coordinates(group, source_path)
        selected_names, selection_notes = _select_point_variables(
            group,
            variables=variables,
        )
        root_attributes = _attributes(root)
        group_attributes = _attributes(group)
        cycle = _metadata_integer_attribute(
            root_attributes,
            "cycle_number",
            source_path=source_path,
        )
        pass_number = _metadata_integer_attribute(
            root_attributes,
            "pass_number",
            source_path=source_path,
        )
        tile, tile_name_pass = _source_tile(root_attributes, source_path=source_path)
        if pass_number is not None and tile_name_pass is not None:
            if pass_number != tile_name_pass:
                raise PixcSchemaError(
                    f"NetCDF pass_number {pass_number} conflicts with tile_name pass "
                    f"{tile_name_pass} in {source_path.name}."
                )
        elif pass_number is None:
            pass_number = tile_name_pass
        _validate_record_provenance(
            record,
            source_path=source_path,
            cycle=cycle,
            pass_number=pass_number,
            tile=tile,
        )
        cycle = cycle if cycle is not None else (record.cycle if record else None)
        pass_number = (
            pass_number
            if pass_number is not None
            else (record.pass_number if record else None)
        )
        tile = tile if tile is not None else (record.tile if record else None)

        latitude_variable = group.variables["latitude"]
        longitude_variable = group.variables["longitude"]
        latitude = _read_one_dimensional(latitude_variable, source_path)
        longitude = _read_one_dimensional(longitude_variable, source_path)
        if latitude.shape != longitude.shape:
            raise PixcSchemaError(
                f"latitude and longitude lengths differ in {source_path.name}."
            )

        clip = clip_points(
            longitude,
            latitude,
            aoi,
            longitude_fill_value=variable_metadata["longitude"].fill_value,
            latitude_fill_value=variable_metadata["latitude"].fill_value,
        )
        source_point_index = np.flatnonzero(clip.values).astype(np.int64, copy=False)
        arrays: dict[str, NDArray[Any]] = {}
        for name in selected_names:
            variable = group.variables[name]
            if name == "latitude":
                raw = latitude
            elif name == "longitude":
                raw = longitude
            elif source_point_index.size:
                raw = _read_one_dimensional(variable, source_path)
            else:
                arrays[name] = np.empty(0, dtype=variable.dtype)
                continue
            arrays[name] = np.asarray(raw[clip.values], dtype=variable.dtype)

        metadata = PixcSourceMetadata(
            source_index=source_index,
            path=source_path,
            filename=source_path.name,
            granule_id=_granule_id(record, source_path),
            cycle=cycle,
            pass_number=pass_number,
            tile=tile,
            crid=_metadata_string(root_attributes.get("crid")),
            netcdf_product_version=_metadata_string(
                root_attributes.get("product_version")
            ),
            netcdf_pge_version=_metadata_string(root_attributes.get("pge_version")),
            cmr_record=record,
            data_model=str(root.data_model),
            groups=tuple(root.groups),
            dimensions=MappingProxyType(
                {name: len(dimension) for name, dimension in group.dimensions.items()}
            ),
            root_attributes=MappingProxyType(root_attributes),
            pixel_cloud_attributes=MappingProxyType(group_attributes),
            variables=MappingProxyType(variable_metadata),
            loaded_variables=selected_names,
            points_before=int(latitude.size),
            points_after=int(source_point_index.size),
            invalid_coordinate_count=clip.invalid_coordinate_count,
            normalized_longitude_count=clip.normalized_longitude_count,
            notes=selection_notes,
        )
        return LoadedPixcSource(
            arrays=arrays,
            source_point_index=source_point_index,
            metadata=metadata,
        )
    except (PixcOpenError, PixcSchemaError):
        raise
    except Exception as exc:
        raise PixcOpenError(
            f"Failed while reading /pixel_cloud from {source_path.name}."
        ) from exc
    finally:
        root.close()


def _disable_automatic_decoding(group: netCDF4.Group) -> None:
    for variable in group.variables.values():
        variable.set_auto_maskandscale(False)
        variable.set_auto_chartostring(False)


def _variable_metadata(
    name: str,
    variable: netCDF4.Variable,
) -> PixcVariableMetadata:
    return PixcVariableMetadata(
        name=name,
        dimensions=tuple(variable.dimensions),
        shape=tuple(variable.shape),
        dtype=np.dtype(variable.dtype).str,
        attributes=MappingProxyType(_attributes(variable)),
    )


def _attributes(value: Any) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    for name in value.ncattrs():
        attribute = copy.deepcopy(value.getncattr(name))
        if isinstance(attribute, np.ndarray):
            attribute.setflags(write=False)
        attributes[name] = attribute
    return attributes


def _validate_coordinates(group: netCDF4.Group, source_path: Path) -> None:
    for name in REQUIRED_COORDINATES:
        if name not in group.variables:
            raise PixcSchemaError(
                f"/pixel_cloud/{name} is required in {source_path.name}."
            )
        if tuple(group.variables[name].dimensions) != ("points",):
            raise PixcSchemaError(
                f"/pixel_cloud/{name} must use only the points dimension in "
                f"{source_path.name}."
            )


def _select_point_variables(
    group: netCDF4.Group,
    *,
    variables: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if isinstance(variables, (str, bytes, bytearray)):
        raise TypeError(
            "variables must be a sequence of complete /pixel_cloud variable names, "
            "not one string."
        )
    if variables is None:
        selected = [name for name in DEFAULT_POINT_VARIABLES if name in group.variables]
        selected.extend(
            name
            for name, variable in group.variables.items()
            if name.endswith("_qual")
            and tuple(variable.dimensions) == ("points",)
            and name not in selected
        )
        missing = [
            name for name in DEFAULT_POINT_VARIABLES if name not in group.variables
        ]
        notes = (
            ("Optional documented point variables not present: " + ", ".join(missing),)
            if missing
            else ()
        )
    else:
        selected = []
        for name in (*REQUIRED_COORDINATES, *variables):
            if not isinstance(name, str) or not name:
                raise ValueError("PIXC variable names must be non-empty strings.")
            if name in PROVENANCE_VARIABLES:
                raise ValueError(f"{name!r} is reserved for source provenance.")
            if name not in selected:
                selected.append(name)
        missing = [name for name in selected if name not in group.variables]
        if missing:
            raise PixcSchemaError(
                "Requested /pixel_cloud variable(s) are missing: " + ", ".join(missing)
            )
        notes = ()

    for name in selected:
        dimensions = tuple(group.variables[name].dimensions)
        if dimensions != ("points",):
            raise PixcSchemaError(
                f"/pixel_cloud/{name} uses dimensions {dimensions!r}; only "
                "1-D point variables can be loaded into the clipped dataset."
            )
    return tuple(selected), notes


def _read_one_dimensional(
    variable: netCDF4.Variable,
    source_path: Path,
) -> NDArray[Any]:
    values = np.asarray(variable[:])
    if values.ndim != 1:
        raise PixcSchemaError(
            f"/pixel_cloud/{variable.name} is not 1-D in {source_path.name}."
        )
    return values


def _source_tile(
    attributes: Mapping[str, Any],
    *,
    source_path: Path,
) -> tuple[str | None, int | None]:
    tile_name = _metadata_string(attributes.get("tile_name"))
    tile_from_name: str | None = None
    pass_from_name: int | None = None
    if tile_name:
        match = _TILE_NAME.fullmatch(tile_name.upper())
        if not match:
            raise PixcSchemaError(
                f"NetCDF tile_name {tile_name!r} does not use official PPP_TTTS "
                f"format in {source_path.name}."
            )
        tile_from_name = match.group("tile")
        pass_from_name = int(match.group("pass"))

    tile_number = _metadata_integer_attribute(
        attributes,
        "tile_number",
        source_path=source_path,
    )
    swath_side = _metadata_string(attributes.get("swath_side"))
    tile_from_parts: str | None = None
    if (tile_number is None) != (swath_side is None):
        raise PixcSchemaError(
            f"NetCDF tile_number and swath_side must both be present in "
            f"{source_path.name}."
        )
    if tile_number is not None and swath_side is not None:
        normalized_side = swath_side.upper()
        if normalized_side not in {"L", "R", "F"}:
            raise PixcSchemaError(
                f"NetCDF swath_side {swath_side!r} is invalid in {source_path.name}."
            )
        tile_from_parts = f"{tile_number:03d}{normalized_side}"

    if (
        tile_from_name is not None
        and tile_from_parts is not None
        and tile_from_name != tile_from_parts
    ):
        raise PixcSchemaError(
            f"NetCDF tile_name {tile_name!r} conflicts with tile_number/"
            f"swath_side {tile_from_parts!r} in {source_path.name}."
        )
    return tile_from_name or tile_from_parts, pass_from_name


def _validate_record_provenance(
    record: GranuleRecord | None,
    *,
    source_path: Path,
    cycle: int | None,
    pass_number: int | None,
    tile: str | None,
) -> None:
    if record is None:
        return
    if record.filename is not None and record.filename != source_path.name:
        raise PixcSchemaError(
            f"CMR filename {record.filename!r} conflicts with local filename "
            f"{source_path.name!r}."
        )
    if record.local_path is not None:
        expected_path = Path(record.local_path).expanduser().resolve()
        if expected_path != source_path:
            raise PixcSchemaError(
                f"CMR local path {expected_path} conflicts with opened path "
                f"{source_path}."
            )
    comparisons = (
        ("cycle", record.cycle, cycle),
        ("pass", record.pass_number, pass_number),
        ("tile", record.tile, tile),
    )
    for label, cmr_value, file_value in comparisons:
        if cmr_value is None or file_value is None:
            continue
        if str(cmr_value).upper() != str(file_value).upper():
            raise PixcSchemaError(
                f"CMR {label} {cmr_value!r} conflicts with NetCDF {label} "
                f"{file_value!r} for {source_path.name}."
            )


def _granule_id(record: GranuleRecord | None, source_path: Path) -> str:
    if record is not None:
        return record.granule_id or record.native_id or source_path.stem
    return source_path.stem


def _metadata_integer(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return int(numeric) if np.isfinite(numeric) and numeric.is_integer() else None
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value)
    return None


def _metadata_integer_attribute(
    attributes: Mapping[str, Any],
    name: str,
    *,
    source_path: Path,
) -> int | None:
    value = attributes.get(name)
    parsed = _metadata_integer(value)
    if value is not None and parsed is None:
        raise PixcSchemaError(
            f"NetCDF {name} {value!r} is not an integer in {source_path.name}."
        )
    return parsed


def _metadata_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
