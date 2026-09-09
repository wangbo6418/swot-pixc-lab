"""Shared, read-only normalization of Phase-2/3 pixel data inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .mosaic import PixcObservation
from .qc import QC_PROFILES, QCResult
from .reader import PixcSourceMetadata

type PixelData = PixcObservation | QCResult | xr.Dataset


@dataclass(frozen=True, slots=True)
class ResolvedPixelInput:
    """A borrowed pixel dataset plus any context retained by its owner.

    The dataset is deliberately not copied here. Consumers must treat it as
    read-only and detach only the much smaller result they create.
    """

    dataset: xr.Dataset
    sources: tuple[PixcSourceMetadata, ...]
    profile_name: str | None
    profile_label: str | None
    profile_status: str | None
    qc_result: QCResult | None


def resolve_pixel_input(data: PixelData) -> ResolvedPixelInput:
    """Resolve supported Phase-2/3 objects without changing or copying them."""

    if isinstance(data, QCResult):
        resolved = ResolvedPixelInput(
            dataset=data.filtered,
            sources=data.observation.sources,
            profile_name=data.profile.name,
            profile_label=data.profile.label,
            profile_status=data.profile.status,
            qc_result=data,
        )
    elif isinstance(data, PixcObservation):
        profile = QC_PROFILES["raw"]
        resolved = ResolvedPixelInput(
            dataset=data.raw,
            sources=data.sources,
            profile_name=profile.name,
            profile_label=profile.label,
            profile_status=profile.status,
            qc_result=None,
        )
    elif isinstance(data, xr.Dataset):
        resolved = ResolvedPixelInput(
            dataset=data,
            sources=(),
            profile_name=_optional_text(data.attrs.get("swot_pixc_lab_qc_profile")),
            profile_label=_optional_text(data.attrs.get("swot_pixc_lab_qc_label")),
            profile_status=_optional_text(data.attrs.get("swot_pixc_lab_qc_status")),
            qc_result=None,
        )
    else:
        raise TypeError(
            "data must be a PixcObservation, QCResult, or xarray.Dataset; "
            f"got {type(data).__name__}."
        )

    _validate_point_dataset(resolved.dataset)
    return resolved


def provenance_reference_indices(
    reference: xr.Dataset,
    subset: xr.Dataset,
    *,
    context: str,
) -> NDArray[np.int64]:
    """Map each subset provenance pair to one unique reference point."""

    for label, dataset in (("reference", reference), (context, subset)):
        for name in ("source_index", "source_point_index"):
            if name not in dataset:
                raise ValueError(
                    f"{context} validation requires {name!r} in the {label} data."
                )
            variable = dataset[name]
            if variable.dims != ("points",) or not np.issubdtype(
                variable.dtype, np.integer
            ):
                raise ValueError(
                    f"{context} {name!r} must be a one-dimensional integer "
                    f"provenance variable in the {label} data."
                )

    reference_count = int(reference.sizes.get("points", 0))
    subset_count = int(subset.sizes.get("points", 0))
    if subset_count > reference_count:
        raise ValueError(
            f"{context} has more points than its proposed reference observation."
        )
    result = np.empty(subset_count, dtype=np.int64)
    if subset_count == 0:
        return result

    reference_sources = np.asarray(reference["source_index"].values)
    reference_points = np.asarray(reference["source_point_index"].values)
    subset_sources = np.asarray(subset["source_index"].values)
    subset_points = np.asarray(subset["source_point_index"].values)

    for source in np.unique(subset_sources):
        reference_offsets = np.flatnonzero(reference_sources == source)
        subset_offsets = np.flatnonzero(subset_sources == source)
        if reference_offsets.size == 0:
            raise ValueError(
                f"{context} contains source_index {int(source)} that is absent "
                "from the reference observation."
            )

        reference_values = reference_points[reference_offsets]
        if reference_values.size > 1 and np.all(
            reference_values[1:] > reference_values[:-1]
        ):
            ordered_values = reference_values
            ordered_offsets = reference_offsets
        else:
            order = np.argsort(reference_values, kind="stable")
            ordered_values = reference_values[order]
            ordered_offsets = reference_offsets[order]
            if ordered_values.size > 1 and np.any(
                ordered_values[1:] == ordered_values[:-1]
            ):
                raise ValueError(
                    f"Reference observation has duplicate provenance for "
                    f"source_index {int(source)}."
                )

        requested = subset_points[subset_offsets]
        if np.unique(requested).size != requested.size:
            raise ValueError(
                f"{context} repeats a source_index/source_point_index pair."
            )
        positions = np.searchsorted(ordered_values, requested)
        within = positions < ordered_values.size
        found = np.zeros(requested.shape, dtype=np.bool_)
        found[within] = ordered_values[positions[within]] == requested[within]
        if not np.all(found):
            missing = requested[np.flatnonzero(~found)[0]]
            raise ValueError(
                f"{context} provenance pair ({int(source)}, {int(missing)}) is "
                "absent from the reference observation."
            )
        result[subset_offsets] = ordered_offsets[positions]
    return result


def _validate_point_dataset(dataset: xr.Dataset) -> None:
    if "points" not in dataset.sizes:
        raise ValueError("PIXC data must have a 'points' dimension.")
    for name in ("longitude", "latitude"):
        if name not in dataset:
            raise ValueError(f"PIXC data must contain a {name!r} point variable.")
        if dataset[name].dims != ("points",):
            raise ValueError(
                f"PIXC {name!r} must have exactly the ('points',) dimension; "
                f"got {dataset[name].dims}."
            )
        dtype = dataset[name].dtype
        if not np.issubdtype(dtype, np.number) or np.issubdtype(
            dtype, np.complexfloating
        ):
            raise ValueError(
                f"PIXC {name!r} must contain real numeric coordinates; got {dtype}."
            )


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
