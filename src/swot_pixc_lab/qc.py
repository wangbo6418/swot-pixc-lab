"""Transparent, metadata-driven quality control for raw SWOT PIXC points.

This module never changes :class:`~swot_pixc_lab.PixcObservation.raw`.  A QC
result contains full-length masks aligned with that raw dataset and a detached
filtered copy.  Numeric flag meanings come from each source NetCDF file rather
than from an undocumented package lookup table.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .exceptions import QualityMetadataError
from .mosaic import PixcObservation

PDD_REFERENCE = (
    "SWOT L2 HR PIXC Product Description Document, JPL D-56411 Revision C "
    "(2025-02-24), sections 4.1.2.2-4.1.2.4 and Appendix B Table 15"
)
ATBD_CLASS_REFERENCE = (
    "SWOT L2 HR PIXC Algorithm Theoretical Basis Document, JPL D-105504, "
    "section 3.3.8.3"
)

ProfileName = Literal["raw", "bo_legacy_strict", "channel_extent_candidate"]


@dataclass(frozen=True, slots=True)
class FlagDefinition:
    """One official CF flag definition copied from NetCDF metadata."""

    meaning: str
    mask: int | None
    value: int | None


@dataclass(frozen=True, slots=True)
class DecodedQualityFlags:
    """Lazy named views of one unchanged integer flag variable.

    ``flag()`` returns a new, read-only Boolean mask.  Fill and out-of-range
    samples are false in every named mask and are exposed separately.  This is
    important for PIXC's unsigned-32 fill value, whose bits are all set.
    """

    variable_name: str
    raw: NDArray[np.integer[Any]]
    definitions: tuple[FlagDefinition, ...]
    fill_mask: NDArray[np.bool_]
    out_of_range_mask: NDArray[np.bool_]
    valid_mask: NDArray[np.bool_]
    unknown_bits_mask: NDArray[np.bool_]
    attributes: Mapping[str, Any]

    @property
    def meanings(self) -> tuple[str, ...]:
        """Return official flag meanings in NetCDF metadata order."""

        return tuple(definition.meaning for definition in self.definitions)

    def flag(self, meaning: str) -> NDArray[np.bool_]:
        """Decode one named flag as a read-only Boolean mask."""

        matches = [item for item in self.definitions if item.meaning == meaning]
        if not matches:
            raise KeyError(
                f"{meaning!r} is not defined by {self.variable_name!r}; "
                f"available meanings: {', '.join(self.meanings)}"
            )
        definition = matches[0]
        if definition.mask is None:
            condition = self.raw == definition.value
        elif definition.value is None:
            condition = _integer_bitwise_and(self.raw, definition.mask) != 0
        else:
            condition = (
                _integer_bitwise_and(self.raw, definition.mask) == definition.value
            )
        result = np.asarray(condition & self.valid_mask, dtype=np.bool_)
        result.setflags(write=False)
        return result

    @property
    def flags(self) -> Mapping[str, NDArray[np.bool_]]:
        """Materialize all named masks.

        For large pixel clouds, prefer :meth:`flag` or :meth:`frequencies` so
        masks are produced one at a time.
        """

        return MappingProxyType({name: self.flag(name) for name in self.meanings})

    def frequencies(self) -> dict[str, int]:
        """Count each named condition independently, excluding invalid data."""

        return {name: int(np.count_nonzero(self.flag(name))) for name in self.meanings}


@dataclass(frozen=True, slots=True)
class QCProfile:
    """Scientific status and intent of one named QC profile."""

    name: ProfileName
    label: str
    status: str
    description: str


@dataclass(frozen=True, slots=True)
class QCRule:
    """Auditable definition of one ordered pixel-rejection rule."""

    id: str
    variable: str
    condition: str
    rationale: str
    definition_source: str


@dataclass(frozen=True, slots=True)
class QCRuleOutcome:
    """Application status and counts for one rule."""

    rule: QCRule
    status: Literal["applied", "skipped"]
    independent_failure_count: int
    incremental_removal_count: int
    invalid_value_count: int
    remaining_count: int
    skip_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly record for reports."""

        return {
            "id": self.rule.id,
            "variable": self.rule.variable,
            "condition": self.rule.condition,
            "rationale": self.rule.rationale,
            "definition_source": self.rule.definition_source,
            "status": self.status,
            "independent_failure_count": self.independent_failure_count,
            "incremental_removal_count": self.incremental_removal_count,
            "invalid_value_count": self.invalid_value_count,
            "remaining_count": self.remaining_count,
            "skip_reason": self.skip_reason,
        }


@dataclass(frozen=True, slots=True)
class HeightReferenceResult:
    """Metadata for a derived ``height - geoid`` variable."""

    status: Literal["derived", "skipped"]
    variable_name: str | None
    formula: str
    egm2008_confirmed: bool
    valid_count: int
    invalid_count: int
    definition_source: str
    skipped_reason: str | None = None
    count_scope: str = "raw exact-AOI input pairs before profile filtering"

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly height-reference record."""

        return {
            "status": self.status,
            "variable_name": self.variable_name,
            "formula": self.formula,
            "egm2008_confirmed": self.egm2008_confirmed,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "definition_source": self.definition_source,
            "skipped_reason": self.skipped_reason,
            "count_scope": self.count_scope,
        }


@dataclass(frozen=True, slots=True)
class QCResult:
    """A transparent mask and detached filtered view of one raw observation."""

    profile: QCProfile
    observation: PixcObservation
    mask: NDArray[np.bool_]
    filtered: xr.Dataset
    reason_masks: Mapping[str, NDArray[np.bool_]]
    rule_outcomes: tuple[QCRuleOutcome, ...]
    decoded_flags: Mapping[str, DecodedQualityFlags]
    height_reference: HeightReferenceResult

    @property
    def raw(self) -> xr.Dataset:
        """Return the original, unchanged Phase-2 dataset."""

        return self.observation.raw

    @property
    def reason_counts(self) -> Mapping[str, int]:
        """Return independent rule failures; these counts can overlap."""

        return MappingProxyType(
            {
                outcome.rule.id: outcome.independent_failure_count
                for outcome in self.rule_outcomes
                if outcome.status == "applied"
            }
        )

    @property
    def incremental_reason_counts(self) -> Mapping[str, int]:
        """Return ordered, non-overlapping removals for applied rules."""

        return MappingProxyType(
            {
                outcome.rule.id: outcome.incremental_removal_count
                for outcome in self.rule_outcomes
                if outcome.status == "applied"
            }
        )

    @property
    def incremental_reason_masks(self) -> Mapping[str, NDArray[np.bool_]]:
        """Materialize ordered, non-overlapping full-length removal masks."""

        remaining = np.ones(self.mask.shape, dtype=np.bool_)
        masks: dict[str, NDArray[np.bool_]] = {}
        for outcome in self.rule_outcomes:
            if outcome.status != "applied":
                continue
            incremental = remaining & self.reason_masks[outcome.rule.id]
            remaining &= ~self.reason_masks[outcome.rule.id]
            incremental.setflags(write=False)
            masks[outcome.rule.id] = incremental
        return MappingProxyType(masks)

    @property
    def summary(self) -> dict[str, object]:
        """Return profile, rejection, provenance, and height diagnostics."""

        before = int(self.mask.size)
        retained = int(np.count_nonzero(self.mask))
        return {
            "profile": self.profile.name,
            "profile_label": self.profile.label,
            "profile_status": self.profile.status,
            "pixels_before": before,
            "pixels_removed": before - retained,
            "pixels_retained": retained,
            "percent_retained": (100.0 * retained / before) if before else 0.0,
            "classification_before": _classification_counts(
                self.observation, np.ones(before, dtype=np.bool_)
            ),
            "classification_after": _classification_counts(self.observation, self.mask),
            "source_counts_before": _source_counts(
                self.observation, np.ones(before, dtype=np.bool_)
            ),
            "source_counts_after": _source_counts(self.observation, self.mask),
            "raw_height": self.observation.height_summary,
            "filtered_height": _filtered_numeric_summary(self.filtered, "height"),
            "derived_height": (
                _filtered_numeric_summary(
                    self.filtered, self.height_reference.variable_name
                )
                if self.height_reference.variable_name is not None
                else None
            ),
            "height_reference": self.height_reference.as_dict(),
            "independent_reason_counts": dict(self.reason_counts),
            "incremental_reason_counts": dict(self.incremental_reason_counts),
            "rules": [outcome.as_dict() for outcome in self.rule_outcomes],
            "skipped_rules": [
                outcome.rule.id
                for outcome in self.rule_outcomes
                if outcome.status == "skipped"
            ],
        }


QC_PROFILES: Mapping[str, QCProfile] = MappingProxyType(
    {
        "raw": QCProfile(
            name="raw",
            label="Raw exact-AOI PIXC",
            status="baseline / no scientific filtering",
            description="The Phase-2 pixels are retained without a QC rule.",
        ),
        "bo_legacy_strict": QCProfile(
            name="bo_legacy_strict",
            label="Bo legacy strict",
            status="legacy / conservative / not NASA recommended",
            description=(
                "Reproduces the project owner's historical WSE-oriented screen "
                "when each required variable is available."
            ),
        ),
        "channel_extent_candidate": QCProfile(
            name="channel_extent_candidate",
            label="Channel extent candidate",
            status="experimental / unvalidated",
            description=(
                "High-recall geometry candidate retaining documented water edge, "
                "dark-water, and low-coherence water classes while excluding "
                "documented land, invalid classification/geolocation quality, "
                "in-air pixels, and named bad classification/geolocation states."
            ),
        ),
    }
)


@dataclass(slots=True)
class _RuleEvaluation:
    rule: QCRule
    failure_mask: NDArray[np.bool_] | None
    invalid_value_count: int = 0
    skip_reason: str | None = None


@dataclass(slots=True)
class _HeightDerivation:
    result: HeightReferenceResult
    values: NDArray[np.floating[Any]] | None
    attributes: dict[str, Any]


def decode_quality_flags(
    values: xr.DataArray | NDArray[Any] | Sequence[int],
    attributes: Mapping[str, Any] | None = None,
    *,
    variable_name: str | None = None,
) -> DecodedQualityFlags:
    """Decode CF ``flag_masks``/``flag_values`` metadata without changing raw data.

    CF semantics are used exactly: masks alone mean ``value & mask != 0``;
    values alone mean equality; and masks plus values mean
    ``value & mask == flag_value``.  Fill and metadata-out-of-range samples are
    excluded before any named condition is evaluated.
    """

    if isinstance(values, xr.DataArray):
        raw_values = np.asarray(values.values)
        source_attributes = values.attrs if attributes is None else attributes
        resolved_name = variable_name or values.name or "quality_flag"
    else:
        raw_values = np.asarray(values)
        source_attributes = attributes
        resolved_name = variable_name or "quality_flag"
    if source_attributes is None:
        raise QualityMetadataError(
            f"{resolved_name!r} requires NetCDF flag metadata for decoding."
        )
    if not np.issubdtype(raw_values.dtype, np.integer):
        raise QualityMetadataError(
            f"{resolved_name!r} must retain an integer dtype; got {raw_values.dtype}."
        )

    meanings = _parse_flag_meanings(source_attributes.get("flag_meanings"))
    masks = _parse_integer_attribute(source_attributes.get("flag_masks"), "flag_masks")
    flag_values = _parse_integer_attribute(
        source_attributes.get("flag_values"), "flag_values"
    )
    if masks is None and flag_values is None:
        raise QualityMetadataError(
            f"{resolved_name!r} has flag_meanings but no flag_masks or flag_values."
        )
    for attribute_name, items in (("flag_masks", masks), ("flag_values", flag_values)):
        if items is not None and len(items) != len(meanings):
            raise QualityMetadataError(
                f"{resolved_name!r} has {len(meanings)} flag_meanings but "
                f"{len(items)} {attribute_name} entries."
            )
    if len(set(meanings)) != len(meanings):
        raise QualityMetadataError(
            f"{resolved_name!r} contains duplicate flag_meanings."
        )

    dtype_bits = raw_values.dtype.itemsize * 8
    unsigned_max = (1 << dtype_bits) - 1
    if masks is not None:
        for mask in masks:
            if mask <= 0 or mask > unsigned_max:
                raise QualityMetadataError(
                    f"{resolved_name!r} has flag mask {mask} outside its "
                    f"{dtype_bits}-bit integer representation."
                )
    if flag_values is not None:
        dtype_info = np.iinfo(raw_values.dtype)
        for value in flag_values:
            if value < int(dtype_info.min) or value > int(dtype_info.max):
                raise QualityMetadataError(
                    f"{resolved_name!r} has flag value {value} outside dtype "
                    f"{raw_values.dtype}."
                )
    if masks is not None and flag_values is not None:
        for mask, value in zip(masks, flag_values, strict=True):
            if value < 0 or value & mask != value:
                raise QualityMetadataError(
                    f"{resolved_name!r} has flag value {value} containing bits "
                    f"outside its paired mask {mask}."
                )

    raw = np.array(raw_values, copy=True)
    raw.setflags(write=False)
    fill_value = _parse_scalar_integer(
        source_attributes.get("_FillValue"), "_FillValue", allow_none=True
    )
    valid_min = _parse_scalar_integer(
        source_attributes.get("valid_min"), "valid_min", allow_none=True
    )
    valid_max = _parse_scalar_integer(
        source_attributes.get("valid_max"), "valid_max", allow_none=True
    )
    if valid_min is not None and valid_max is not None and valid_min > valid_max:
        raise QualityMetadataError(
            f"{resolved_name!r} has valid_min greater than valid_max."
        )

    fill_mask = (
        np.zeros(raw.shape, dtype=np.bool_)
        if fill_value is None
        else np.asarray(raw == fill_value, dtype=np.bool_)
    )
    out_of_range = np.zeros(raw.shape, dtype=np.bool_)
    if valid_min is not None:
        out_of_range |= raw < valid_min
    if valid_max is not None:
        out_of_range |= raw > valid_max
    out_of_range &= ~fill_mask
    valid = ~(fill_mask | out_of_range)

    definitions = tuple(
        FlagDefinition(
            meaning=meaning,
            mask=masks[index] if masks is not None else None,
            value=flag_values[index] if flag_values is not None else None,
        )
        for index, meaning in enumerate(meanings)
    )
    unknown = _unknown_condition_mask(raw, valid, definitions, unsigned_max)
    for mask in (fill_mask, out_of_range, valid, unknown):
        mask.setflags(write=False)

    return DecodedQualityFlags(
        variable_name=resolved_name,
        raw=raw,
        definitions=definitions,
        fill_mask=fill_mask,
        out_of_range_mask=out_of_range,
        valid_mask=valid,
        unknown_bits_mask=unknown,
        attributes=MappingProxyType(_copy_attributes(source_attributes)),
    )


def apply_qc(
    observation: PixcObservation,
    *,
    profile: ProfileName | str = "raw",
) -> QCResult:
    """Apply one named Phase-3 profile as a derived mask and filtered copy.

    Missing variables cause only the dependent rule to be skipped and reported.
    Metadata that is present but internally inconsistent raises
    :class:`QualityMetadataError` rather than being guessed or reinterpreted.
    """

    if profile not in QC_PROFILES:
        raise ValueError(
            f"Unknown QC profile {profile!r}; choose from "
            + ", ".join(QC_PROFILES)
            + "."
        )
    if "points" not in observation.raw.sizes:
        raise QualityMetadataError("The raw observation has no points dimension.")

    profile_definition = QC_PROFILES[profile]
    decoded = _decode_observation_flags(observation)
    if profile == "raw":
        evaluations: list[_RuleEvaluation] = []
    elif profile == "bo_legacy_strict":
        evaluations = _legacy_evaluations(observation)
    else:
        evaluations = _candidate_evaluations(observation, decoded)

    point_count = int(observation.raw.sizes["points"])
    keep = np.ones(point_count, dtype=np.bool_)
    reason_masks: dict[str, NDArray[np.bool_]] = {}
    outcomes: list[QCRuleOutcome] = []
    for evaluation in evaluations:
        if evaluation.failure_mask is None:
            outcomes.append(
                QCRuleOutcome(
                    rule=evaluation.rule,
                    status="skipped",
                    independent_failure_count=0,
                    incremental_removal_count=0,
                    invalid_value_count=evaluation.invalid_value_count,
                    remaining_count=int(np.count_nonzero(keep)),
                    skip_reason=evaluation.skip_reason,
                )
            )
            continue
        failure = np.asarray(evaluation.failure_mask, dtype=np.bool_)
        if failure.shape != keep.shape:
            raise QualityMetadataError(
                f"QC rule {evaluation.rule.id!r} produced shape {failure.shape}; "
                f"expected {keep.shape}."
            )
        independent_count = int(np.count_nonzero(failure))
        incremental_count = int(np.count_nonzero(keep & failure))
        keep &= ~failure
        failure.setflags(write=False)
        reason_masks[evaluation.rule.id] = failure
        outcomes.append(
            QCRuleOutcome(
                rule=evaluation.rule,
                status="applied",
                independent_failure_count=independent_count,
                incremental_removal_count=incremental_count,
                invalid_value_count=evaluation.invalid_value_count,
                remaining_count=int(np.count_nonzero(keep)),
            )
        )

    keep.setflags(write=False)
    selected = np.flatnonzero(keep)
    filtered = observation.raw.isel(points=selected).copy(deep=True)
    filtered.attrs["swot_pixc_lab_qc_profile"] = profile_definition.name
    filtered.attrs["swot_pixc_lab_qc_status"] = profile_definition.status
    filtered.attrs["swot_pixc_lab_qc_note"] = (
        "Derived view; PixcObservation.raw remains unchanged."
    )

    height = _derive_height_reference(observation)
    if height.values is not None and height.result.variable_name is not None:
        values = np.array(height.values[keep], copy=True)
        filtered[height.result.variable_name] = xr.DataArray(
            values,
            dims=("points",),
            attrs=height.attributes,
        )

    return QCResult(
        profile=profile_definition,
        observation=observation,
        mask=keep,
        filtered=filtered,
        reason_masks=MappingProxyType(reason_masks),
        rule_outcomes=tuple(outcomes),
        decoded_flags=MappingProxyType(decoded),
        height_reference=height.result,
    )


def _legacy_evaluations(observation: PixcObservation) -> list[_RuleEvaluation]:
    source = "Project-owner legacy workflow; thresholds are not NASA recommendations"
    specifications: tuple[
        tuple[QCRule, Callable[[NDArray[Any]], NDArray[np.bool_]], str | None], ...
    ] = (
        (
            QCRule(
                "classification_eq_4",
                "classification",
                "classification == 4",
                "Historical interior/open-water selection for WSE reproducibility.",
                source,
            ),
            lambda values: values == 4,
            None,
        ),
        (
            QCRule(
                "water_frac_ge_0_90",
                "water_frac",
                "water_frac >= 0.90",
                "Historical high water-fraction threshold.",
                source,
            ),
            lambda values: values >= 0.90,
            "1",
        ),
        (
            QCRule(
                "water_frac_uncert_le_0_15",
                "water_frac_uncert",
                "water_frac_uncert <= 0.15",
                "Historical water-fraction uncertainty threshold.",
                source,
            ),
            lambda values: values <= 0.15,
            "1",
        ),
        (
            QCRule(
                "bright_land_flag_eq_0",
                "bright_land_flag",
                "bright_land_flag == 0",
                "Historical exclusion of all nonzero bright-land prior states.",
                source,
            ),
            lambda values: values == 0,
            None,
        ),
        (
            QCRule(
                "false_detection_rate_le_0_10",
                "false_detection_rate",
                "false_detection_rate <= 0.10",
                "Historical maximum estimated false-detection probability.",
                source,
            ),
            lambda values: values <= 0.10,
            "1",
        ),
        *tuple(
            (
                QCRule(
                    f"{name}_eq_0",
                    name,
                    f"{name} == 0",
                    "Historical requirement that no documented quality bit is set.",
                    source,
                ),
                lambda values: values == 0,
                None,
            )
            for name in (
                "classification_qual",
                "geolocation_qual",
                "interferogram_qual",
                "sig0_qual",
            )
        ),
        (
            QCRule(
                "phase_noise_std_le_1",
                "phase_noise_std",
                "phase_noise_std <= 1.0 radians",
                "Historical maximum estimated phase-noise standard deviation.",
                source,
            ),
            lambda values: values <= 1.0,
            "radians",
        ),
        (
            QCRule(
                "inc_between_0_5_and_5",
                "inc",
                "0.5 <= inc <= 5.0 degrees",
                "Historical inclusive incidence-angle range.",
                source,
            ),
            lambda values: (values >= 0.5) & (values <= 5.0),
            "degrees",
        ),
    )
    return [
        _numeric_evaluation(observation, rule, predicate, expected_units=units)
        for rule, predicate, units in specifications
    ]


def _candidate_evaluations(
    observation: PixcObservation,
    decoded: Mapping[str, DecodedQualityFlags],
) -> list[_RuleEvaluation]:
    evaluations = [
        _numeric_evaluation(
            observation,
            QCRule(
                "classification_water_classes_3_7",
                "classification",
                "classification in {3, 4, 5, 6, 7}",
                (
                    "Classes 1 and 2 are documented land classes; classes 3-7 "
                    "are detected edge, open, dark, or low-coherence water."
                ),
                f"{PDD_REFERENCE}; {ATBD_CLASS_REFERENCE}",
            ),
            lambda values: np.isin(values, (3, 4, 5, 6, 7)),
        )
    ]

    for variable in ("classification_qual", "geolocation_qual"):
        evaluations.append(_quality_validity_evaluation(variable, decoded))

    flag_rules: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "classification_qual",
            (
                "in_air_pixel_degraded",
                "coherent_power_bad",
                "tvp_bad",
                "sc_event_bad",
                "large_karin_gap",
            ),
        ),
        (
            "geolocation_qual",
            (
                "no_geolocation_bad",
                "medium_phase_bad",
                "tvp_bad",
                "sc_event_bad",
                "large_karin_gap",
            ),
        ),
        ("interferogram_qual", ("in_air_pixel_degraded",)),
    )
    for variable, meanings in flag_rules:
        for meaning in meanings:
            evaluations.append(_named_flag_evaluation(variable, meaning, decoded))
    return evaluations


def _numeric_evaluation(
    observation: PixcObservation,
    rule: QCRule,
    predicate: Callable[[NDArray[Any]], NDArray[np.bool_]],
    *,
    expected_units: str | None = None,
) -> _RuleEvaluation:
    if rule.variable not in observation.raw:
        return _RuleEvaluation(
            rule,
            None,
            skip_reason=f"/{rule.variable} was not loaded or is not present.",
        )
    compatibility_issue = _threshold_metadata_issue(
        observation, rule.variable, expected_units
    )
    if compatibility_issue is not None:
        return _RuleEvaluation(rule, None, skip_reason=compatibility_issue)
    values = np.asarray(observation.raw[rule.variable].values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.number):
        raise QualityMetadataError(
            f"QC variable {rule.variable!r} must be a one-dimensional numeric array."
        )
    valid = _valid_data_mask(observation, rule.variable)
    try:
        passes = np.asarray(predicate(values), dtype=np.bool_)
    except (TypeError, ValueError) as exc:
        raise QualityMetadataError(
            f"Could not evaluate QC rule {rule.id!r} on {rule.variable!r}."
        ) from exc
    if passes.shape != values.shape:
        raise QualityMetadataError(
            f"QC rule {rule.id!r} returned an incompatible mask shape."
        )
    passes &= valid
    return _RuleEvaluation(
        rule,
        ~passes,
        invalid_value_count=int(np.count_nonzero(~valid)),
    )


def _quality_validity_evaluation(
    variable: str,
    decoded: Mapping[str, DecodedQualityFlags],
) -> _RuleEvaluation:
    rule = QCRule(
        id=f"{variable}_valid",
        variable=variable,
        condition=(
            f"{variable} is not fill, is within valid_min/valid_max, and has "
            "no undocumented set bits"
        ),
        rationale=(
            "A geometry candidate cannot establish the absence of documented bad "
            "classification/geolocation conditions from undefined or unknown "
            "flag metadata."
        ),
        definition_source=PDD_REFERENCE,
    )
    item = decoded.get(variable)
    if item is None:
        return _RuleEvaluation(
            rule,
            None,
            skip_reason=(
                f"/{variable} was not loaded or has no decodable flag metadata."
            ),
        )
    invalid = ~item.valid_mask
    failure = invalid | item.unknown_bits_mask
    return _RuleEvaluation(
        rule,
        failure,
        invalid_value_count=int(np.count_nonzero(invalid)),
    )


def _named_flag_evaluation(
    variable: str,
    meaning: str,
    decoded: Mapping[str, DecodedQualityFlags],
) -> _RuleEvaluation:
    is_in_air = meaning == "in_air_pixel_degraded"
    rule = QCRule(
        id=f"{variable}__{meaning}",
        variable=variable,
        condition=f"decoded flag {meaning} is false",
        rationale=(
            "The PDD says an in-air pixel does not intersect Earth's surface."
            if is_in_air
            else "The PDD explicitly labels this condition bad or says no usable "
            "measurement is computed."
        ),
        definition_source=PDD_REFERENCE,
    )
    item = decoded.get(variable)
    if item is None:
        return _RuleEvaluation(
            rule,
            None,
            skip_reason=(
                f"/{variable} was not loaded or has no decodable flag metadata."
            ),
        )
    if meaning not in item.meanings:
        return _RuleEvaluation(
            rule,
            None,
            invalid_value_count=int(np.count_nonzero(~item.valid_mask)),
            skip_reason=(
                f"The NetCDF metadata for /{variable} does not define {meaning!r}; "
                "the bit meaning was not guessed."
            ),
        )
    return _RuleEvaluation(
        rule,
        item.flag(meaning),
        invalid_value_count=int(np.count_nonzero(~item.valid_mask)),
    )


def _decode_observation_flags(
    observation: PixcObservation,
) -> dict[str, DecodedQualityFlags]:
    decoded: dict[str, DecodedQualityFlags] = {}
    for name, data_array in observation.raw.data_vars.items():
        if name in {"source_index", "source_point_index"}:
            continue
        if not np.issubdtype(data_array.dtype, np.integer):
            continue
        metadata = _consistent_flag_metadata(observation, name)
        if metadata is None or "flag_masks" not in metadata:
            continue
        decoded[name] = decode_quality_flags(
            data_array,
            metadata,
            variable_name=name,
        )
    return decoded


def _consistent_flag_metadata(
    observation: PixcObservation,
    name: str,
) -> Mapping[str, Any] | None:
    definitions = []
    for source in observation.sources:
        metadata = source.variables.get(name)
        if metadata is None:
            raise QualityMetadataError(
                f"Raw variable {name!r} lacks source metadata in {source.filename}."
            )
        definitions.append(metadata)
    if not definitions:
        return None
    first = definitions[0]
    attributes = first.attributes
    if "flag_meanings" not in attributes:
        return None
    if "flag_masks" not in attributes and "flag_values" not in attributes:
        return None
    relevant = (
        "flag_meanings",
        "flag_masks",
        "flag_values",
        "_FillValue",
        "valid_min",
        "valid_max",
    )
    for source, metadata in zip(observation.sources[1:], definitions[1:], strict=True):
        if np.dtype(metadata.dtype) != np.dtype(first.dtype):
            raise QualityMetadataError(
                f"/{name} dtype differs across source granules; cannot decode "
                f"{source.filename} without guessing."
            )
        for key in relevant:
            if (key in attributes) != (key in metadata.attributes) or (
                key in attributes
                and not _attribute_equal(attributes[key], metadata.attributes[key])
            ):
                raise QualityMetadataError(
                    f"/{name} attribute {key!r} differs across source granules; "
                    "quality decoding stopped rather than guessing."
                )
    return attributes


def _valid_data_mask(
    observation: PixcObservation,
    name: str,
) -> NDArray[np.bool_]:
    values = np.asarray(observation.raw[name].values)
    valid = np.ones(values.shape, dtype=np.bool_)
    if np.issubdtype(values.dtype, np.floating):
        valid &= np.isfinite(values)
    source_indices = np.asarray(observation.raw["source_index"].values)
    covered = np.zeros(values.shape, dtype=np.bool_)
    for source in observation.sources:
        selected = source_indices == source.source_index
        covered |= selected
        metadata = source.variables.get(name)
        if metadata is None:
            raise QualityMetadataError(
                f"Raw variable {name!r} lacks source metadata in {source.filename}."
            )
        attributes = metadata.attributes
        fill = attributes.get("_FillValue")
        if fill is not None:
            valid[selected] &= values[selected] != fill
        valid_min = attributes.get("valid_min")
        if valid_min is not None:
            valid[selected] &= values[selected] >= valid_min
        valid_max = attributes.get("valid_max")
        if valid_max is not None:
            valid[selected] &= values[selected] <= valid_max
    if not bool(np.all(covered)):
        unknown = np.unique(source_indices[~covered]).tolist()
        raise QualityMetadataError(
            f"Raw points reference unknown source_index values: {unknown}."
        )
    return valid


def _threshold_metadata_issue(
    observation: PixcObservation,
    name: str,
    expected_units: str | None,
) -> str | None:
    units: list[str | None] = []
    for source in observation.sources:
        metadata = source.variables.get(name)
        if metadata is None:
            return f"/{name} lacks source metadata in {source.filename}."
        attributes = metadata.attributes
        scale = attributes.get("scale_factor")
        offset = attributes.get("add_offset")
        if (scale is not None and float(scale) != 1.0) or (
            offset is not None and float(offset) != 0.0
        ):
            return (
                f"/{name} uses scale_factor/add_offset while raw packed values are "
                "preserved; the threshold was skipped rather than reinterpreted."
            )
        units.append(_normalized_unit(attributes.get("units")))
    if len(set(units)) > 1:
        return f"/{name} units differ across source granules: {units!r}."
    if expected_units is not None:
        expected = _normalized_unit(expected_units)
        if not units or units[0] != expected:
            return (
                f"/{name} threshold expects units {expected_units!r}, but source "
                f"metadata reports {units[0] if units else None!r}."
            )
    return None


def _derive_height_reference(observation: PixcObservation) -> _HeightDerivation:
    formula = "height - geoid"
    if "height" not in observation.raw or "geoid" not in observation.raw:
        missing = [name for name in ("height", "geoid") if name not in observation.raw]
        result = HeightReferenceResult(
            status="skipped",
            variable_name=None,
            formula=formula,
            egm2008_confirmed=False,
            valid_count=0,
            invalid_count=int(observation.raw.sizes["points"]),
            definition_source=PDD_REFERENCE,
            skipped_reason="Missing point variable(s): " + ", ".join(missing),
        )
        return _HeightDerivation(result, None, {})

    height_units: list[str | None] = []
    geoid_units: list[str | None] = []
    geoid_sources: list[str] = []
    geoid_standard_names: list[str] = []
    height_long_names: list[str] = []
    for source in observation.sources:
        height_metadata = source.variables.get("height")
        geoid_metadata = source.variables.get("geoid")
        if height_metadata is None or geoid_metadata is None:
            result = HeightReferenceResult(
                status="skipped",
                variable_name=None,
                formula=formula,
                egm2008_confirmed=False,
                valid_count=0,
                invalid_count=int(observation.raw.sizes["points"]),
                definition_source=PDD_REFERENCE,
                skipped_reason=(
                    f"height/geoid source metadata is incomplete in {source.filename}."
                ),
            )
            return _HeightDerivation(result, None, {})
        for name, metadata in (("height", height_metadata), ("geoid", geoid_metadata)):
            attributes = metadata.attributes
            scale = attributes.get("scale_factor")
            offset = attributes.get("add_offset")
            if (scale is not None and float(scale) != 1.0) or (
                offset is not None and float(offset) != 0.0
            ):
                result = HeightReferenceResult(
                    status="skipped",
                    variable_name=None,
                    formula=formula,
                    egm2008_confirmed=False,
                    valid_count=0,
                    invalid_count=int(observation.raw.sizes["points"]),
                    definition_source=PDD_REFERENCE,
                    skipped_reason=(
                        f"/{name} is packed with scale/offset while raw values are "
                        "preserved; subtraction was not guessed."
                    ),
                )
                return _HeightDerivation(result, None, {})
        height_units.append(_normalized_unit(height_metadata.attributes.get("units")))
        geoid_units.append(_normalized_unit(geoid_metadata.attributes.get("units")))
        height_long_names.append(str(height_metadata.attributes.get("long_name", "")))
        geoid_sources.append(str(geoid_metadata.attributes.get("source", "")))
        geoid_standard_names.append(
            str(geoid_metadata.attributes.get("standard_name", ""))
        )

    if set(height_units) != {"m"} or set(geoid_units) != {"m"}:
        result = HeightReferenceResult(
            status="skipped",
            variable_name=None,
            formula=formula,
            egm2008_confirmed=False,
            valid_count=0,
            invalid_count=int(observation.raw.sizes["points"]),
            definition_source=PDD_REFERENCE,
            skipped_reason=(
                "height and geoid must both have consistent metre units; found "
                f"height={height_units!r}, geoid={geoid_units!r}."
            ),
        )
        return _HeightDerivation(result, None, {})

    reference_semantics_confirmed = all(
        "height above reference ellipsoid" in name.lower() for name in height_long_names
    ) and all(
        name == "geoid_height_above_reference_ellipsoid"
        for name in geoid_standard_names
    )
    if not reference_semantics_confirmed:
        result = HeightReferenceResult(
            status="skipped",
            variable_name=None,
            formula=formula,
            egm2008_confirmed=False,
            valid_count=0,
            invalid_count=int(observation.raw.sizes["points"]),
            definition_source=PDD_REFERENCE,
            skipped_reason=(
                "Source metadata does not explicitly establish ellipsoidal height "
                "and geoid height above the same reference ellipsoid."
            ),
        )
        return _HeightDerivation(result, None, {})

    egm2008 = all("egm2008" in source.lower() for source in geoid_sources)
    variable_name = "height_egm2008" if egm2008 else "height_minus_geoid"
    height_values = np.asarray(observation.raw["height"].values)
    geoid_values = np.asarray(observation.raw["geoid"].values)
    valid = _valid_data_mask(observation, "height") & _valid_data_mask(
        observation, "geoid"
    )
    output_dtype = np.result_type(height_values.dtype, geoid_values.dtype)
    if not np.issubdtype(output_dtype, np.floating):
        output_dtype = np.dtype(np.float64)
    derived = np.full(height_values.shape, np.nan, dtype=output_dtype)
    np.subtract(height_values, geoid_values, out=derived, where=valid)
    derived.setflags(write=False)
    valid_count = int(np.count_nonzero(valid))
    result = HeightReferenceResult(
        status="derived",
        variable_name=variable_name,
        formula=formula,
        egm2008_confirmed=egm2008,
        valid_count=valid_count,
        invalid_count=int(valid.size - valid_count),
        definition_source=PDD_REFERENCE,
    )
    attributes = {
        "long_name": (
            "derived height relative to the EGM2008 geoid"
            if egm2008
            else "derived height minus the supplied geoid height"
        ),
        "units": "m",
        "formula": formula,
        "source_variables": "height geoid",
        "comment": (
            "Derived by swot-pixc-lab without applying solid Earth, load, pole, "
            "troposphere, ionosphere, or other geophysical corrections. This "
            "variable is not corrected WSE. Invalid input pairs are NaN."
        ),
        "vertical_reference": (
            "EGM2008 mean-tide geoid" if egm2008 else "supplied geoid model"
        ),
    }
    return _HeightDerivation(result, derived, attributes)


def _classification_counts(
    observation: PixcObservation,
    selection: NDArray[np.bool_],
) -> dict[int, int]:
    if "classification" not in observation.raw:
        return {}
    values = np.asarray(observation.raw["classification"].values)
    valid = _valid_data_mask(observation, "classification") & selection
    labels, counts = np.unique(values[valid], return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts, strict=True)}


def _source_counts(
    observation: PixcObservation,
    selection: NDArray[np.bool_],
) -> dict[str, int]:
    source_indices = np.asarray(observation.raw["source_index"].values)
    return {
        f"{source.source_index}:{source.tile or source.filename}": int(
            np.count_nonzero(selection & (source_indices == source.source_index))
        )
        for source in observation.sources
    }


def _filtered_numeric_summary(
    dataset: xr.Dataset,
    name: str | None,
) -> dict[str, float | int | None] | None:
    if name is None or name not in dataset:
        return None
    values = np.asarray(dataset[name].values)
    valid = np.isfinite(values)
    fill = dataset[name].attrs.get("_FillValue")
    if fill is not None:
        valid &= values != fill
    numeric = values[valid].astype(np.float64, copy=False)
    if not numeric.size:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
        }
    return {
        "count": int(numeric.size),
        "min": float(np.min(numeric)),
        "max": float(np.max(numeric)),
        "mean": float(np.mean(numeric)),
        "median": float(np.median(numeric)),
        "std": float(np.std(numeric)),
    }


def _parse_flag_meanings(value: object) -> tuple[str, ...]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value.strip():
        raise QualityMetadataError("flag_meanings must be a non-empty string.")
    return tuple(value.split())


def _parse_integer_attribute(
    value: object,
    name: str,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise QualityMetadataError(f"{name} must be a scalar or one-dimensional.")
    return tuple(_coerce_integer(item, name) for item in array.tolist())


def _parse_scalar_integer(
    value: object,
    name: str,
    *,
    allow_none: bool,
) -> int | None:
    if value is None and allow_none:
        return None
    array = np.asarray(value)
    if array.size != 1:
        raise QualityMetadataError(f"{name} must be scalar.")
    return _coerce_integer(array.reshape(-1)[0], name)


def _coerce_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise QualityMetadataError(f"{name} must contain integers, not booleans.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        numeric = float(value)
        if numeric.is_integer():
            return int(numeric)
    raise QualityMetadataError(f"{name} contains non-integer value {value!r}.")


def _unknown_condition_mask(
    raw: NDArray[np.integer[Any]],
    valid: NDArray[np.bool_],
    definitions: tuple[FlagDefinition, ...],
    unsigned_max: int,
) -> NDArray[np.bool_]:
    masks = [item.mask for item in definitions if item.mask is not None]
    if masks:
        known_mask = 0
        for mask in masks:
            known_mask |= int(mask)
        unknown_mask = unsigned_max ^ known_mask
        unknown = _integer_bitwise_and(raw, unknown_mask) != 0
        if all(item.value is not None for item in definitions):
            grouped_values: dict[int, list[int]] = {}
            for definition in definitions:
                grouped_values.setdefault(int(definition.mask), []).append(
                    int(definition.value)
                )
            for mask, values in grouped_values.items():
                state = _integer_bitwise_and(raw, mask)
                recognized = state == 0
                for value in values:
                    recognized |= state == value
                unknown |= ~recognized
        return np.asarray(valid & unknown, dtype=np.bool_)
    recognized = np.zeros(raw.shape, dtype=np.bool_)
    for definition in definitions:
        recognized |= raw == definition.value
    return np.asarray(valid & ~recognized, dtype=np.bool_)


def _copy_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for name, value in attributes.items():
        item = copy.deepcopy(value)
        if isinstance(item, np.ndarray):
            item.setflags(write=False)
        copied[name] = item
    return copied


def _integer_bitwise_and(
    values: NDArray[np.integer[Any]],
    mask: int,
) -> NDArray[np.integer[Any]]:
    """Apply a mask safely to signed or unsigned integer storage."""

    if np.issubdtype(values.dtype, np.unsignedinteger):
        return np.bitwise_and(values, mask)
    unsigned = values.astype(np.uint64, copy=False)
    return np.bitwise_and(unsigned, np.uint64(mask))


def _attribute_equal(left: object, right: object) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return False
    try:
        return bool(np.array_equal(left_array, right_array, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(left_array, right_array))


def _normalized_unit(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    aliases = {
        "meter": "m",
        "meters": "m",
        "metre": "m",
        "metres": "m",
        "degree": "degrees",
        "radian": "radians",
    }
    return aliases.get(text, text)
