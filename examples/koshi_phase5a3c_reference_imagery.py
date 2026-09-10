"""Prepare independent Sentinel-2 imagery for proposed Koshi F transects.

This script searches public STAC metadata and reads only small spatial windows
from cloud-optimized Sentinel-2 L2A assets. It creates review material only: it
never approves a geometry, thresholds NDWI, or creates a manual wet interval.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from pyproj import Transformer
from shapely.geometry import LineString, box, shape
from shapely.ops import transform, unary_union

from swot_pixc_lab.reference_imagery import (
    EARTH_SEARCH_ENDPOINT,
    EARTH_SEARCH_SENTINEL2_L2A_COLLECTION,
    NDWI_FORMULA,
    NDWI_INTERPRETATION,
    EarthSearchSentinelProvider,
    MosaicRecord,
    ReferenceImageryError,
    ReviewRegion,
    SclDiagnostics,
    SentinelScene,
    calculate_ndwi,
    calculate_scl_diagnostics,
    group_same_acquisition,
    make_review_region,
    mosaic_same_acquisition_chips,
    read_raw_mosaic,
    read_scaled_mosaic,
    read_swot_temporal_context,
    read_windowed_cog_asset,
    review_region_pixel_mask,
    scale_rgb,
    scenes_intersecting_region,
    sha256_file,
    shortlist_acquisitions,
    temporal_provenance,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_F_MANIFEST = (
    REPOSITORY_ROOT
    / "benchmark_output"
    / "koshi_phase5a3b_pilot"
    / "geometry_review"
    / "refined_transects_proposed.geojson"
)
DEFAULT_OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "benchmark_output"
    / "koshi_phase5a3b_pilot"
    / "reference_imagery_review"
)
DEFAULT_CACHE_DIRECTORY = REPOSITORY_ROOT / "benchmark_cache" / "sentinel2"
DEFAULT_PIXC_DIRECTORY = REPOSITORY_ROOT / "data" / "koshi_phase1"
EXPECTED_PIXC_FILENAMES = (
    "SWOT_L2_HR_PIXC_009_286_107L_20240114T081432_20240114T081443_PGD0_01.nc",
    "SWOT_L2_HR_PIXC_009_286_108L_20240114T081442_20240114T081453_PGD0_01.nc",
)
EXPECTED_TRANSECT_IDS = tuple(f"T{number:03d}F" for number in range(1, 8))
CORE_TRANSECT_IDS = tuple(f"T{number:03d}F" for number in range(2, 8))
KOSHI_AOI = (86.87, 26.49, 87.20, 26.90)
LOCKED_AZIMUTH_DEGREES = 130.0

# These are explicit imagery-preparation guards, not scientific validation
# criteria and never wet/dry thresholds. Values are saved in every manifest.
DEFAULT_MINIMUM_VALID_FRACTION = 0.90
DEFAULT_MAXIMUM_OBSCURED_FRACTION = 0.25
DEFAULT_CONTEXT_MARGIN_M = 1500.0
SEARCH_WINDOWS_DAYS = (7, 14, 30)
_SAFE_PATH_ATOM = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class TransectInput:
    transect_id: str
    geometry: LineString
    properties: dict[str, Any]
    role: str
    region: ReviewRegion


@dataclass
class CandidateWork:
    candidate_id: str
    acquisition_key: str
    acquisition_datetime: str
    temporal_reference_item_id: str
    scenes: tuple[SentinelScene, ...]
    search_window_days: int
    temporal: dict[str, Any]
    local_quality: SclDiagnostics
    technically_usable_for_review: bool
    technical_usability_reason: str
    mosaics: dict[str, MosaicRecord]
    candidate_directory: Path
    rgb_geotiff_path: Path | None = None
    rgb_png_path: Path | None = None
    overlay_png_path: Path | None = None
    ndwi_png_path: Path | None = None
    rgb_geotiff_sha256: str | None = None

    def public_metadata(self) -> dict[str, Any]:
        """Return durable metadata without in-memory objects or access secrets."""

        return {
            "candidate_id": self.candidate_id,
            "acquisition_key": self.acquisition_key,
            "acquisition_datetime": self.acquisition_datetime,
            "temporal_reference_item_id": self.temporal_reference_item_id,
            "provider": self.scenes[0].provider,
            "stac_endpoint": self.scenes[0].stac_endpoint,
            "collection": self.scenes[0].collection,
            "processing_levels": sorted(
                {
                    scene.processing_level
                    for scene in self.scenes
                    if scene.processing_level is not None
                }
            ),
            "processing_baselines": sorted(
                {
                    scene.processing_baseline
                    for scene in self.scenes
                    if scene.processing_baseline is not None
                }
            ),
            "item_ids": [scene.item_id for scene in self.scenes],
            "mgrs_tiles": sorted(
                {scene.mgrs_tile for scene in self.scenes if scene.mgrs_tile}
            ),
            "epsg_values": sorted(
                {scene.epsg for scene in self.scenes if scene.epsg is not None}
            ),
            "scene_cloud_cover_percent": {
                scene.item_id: scene.scene_cloud_cover_percent for scene in self.scenes
            },
            "source_assets": {
                scene.item_id: {
                    role: asset.to_dict() for role, asset in scene.assets.items()
                }
                for scene in self.scenes
            },
            "query_times": sorted(
                {
                    scene.query_time.isoformat().replace("+00:00", "Z")
                    for scene in self.scenes
                }
            ),
            "search_window_days": self.search_window_days,
            **self.temporal,
            "local_quality": self.local_quality.to_dict(),
            "technically_usable_for_review": self.technically_usable_for_review,
            "technical_usability_reason": self.technical_usability_reason,
            "mosaics": {
                role: record.to_dict() for role, record in self.mosaics.items()
            },
            "rgb_geotiff_path": (
                str(self.rgb_geotiff_path) if self.rgb_geotiff_path else None
            ),
            "rgb_png_path": str(self.rgb_png_path) if self.rgb_png_path else None,
            "rgb_with_transect_png_path": (
                str(self.overlay_png_path) if self.overlay_png_path else None
            ),
            "ndwi_png_path": str(self.ndwi_png_path) if self.ndwi_png_path else None,
            "chip_sha256": self.rgb_geotiff_sha256,
            "chip_bounds_wgs84": list(
                self.mosaics["red"].source_chip_records[0].request_bounds_wgs84
            )
            if "red" in self.mosaics
            else list(self.mosaics["scl"].source_chip_records[0].request_bounds_wgs84),
            "chip_crs": (
                self.mosaics["red"].crs
                if "red" in self.mosaics
                else self.mosaics["scl"].crs
            ),
            "chip_transform": list(
                self.mosaics["red"].transform
                if "red" in self.mosaics
                else self.mosaics["scl"].transform
            ),
            "interpretation": (
                "independent imagery review candidate only; analyst must select "
                "every manual boundary"
            ),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transects", type=Path, default=DEFAULT_F_MANIFEST)
    parser.add_argument("--pixc-directory", type=Path, default=DEFAULT_PIXC_DIRECTORY)
    parser.add_argument("--cache-directory", type=Path, default=DEFAULT_CACHE_DIRECTORY)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument(
        "--context-margin-m", type=float, default=DEFAULT_CONTEXT_MARGIN_M
    )
    parser.add_argument(
        "--minimum-valid-fraction",
        type=float,
        default=DEFAULT_MINIMUM_VALID_FRACTION,
        help="technical data-coverage guard only; not a scientific threshold",
    )
    parser.add_argument(
        "--maximum-obscured-fraction",
        type=float,
        default=DEFAULT_MAXIMUM_OBSCURED_FRACTION,
        help="review-preparation guard only; never used to infer water",
    )
    parser.add_argument("--stac-endpoint", default=EARTH_SEARCH_ENDPOINT)
    parser.add_argument(
        "--stac-collection", default=EARTH_SEARCH_SENTINEL2_L2A_COLLECTION
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="write only STAC search metadata; do not read COG windows",
    )
    return parser.parse_args()


def _validate_fraction(value: float, label: str) -> float:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise SystemExit(f"{label} must be finite and between 0 and 1.")
    return value


def _file_sha256(path: Path) -> str:
    return sha256_file(path)


def _safe_atom(value: str) -> str:
    cleaned = _SAFE_PATH_ATOM.sub("_", value).strip("._")
    if not cleaned:
        raise ReferenceImageryError("Candidate identifier cannot form a safe path.")
    return cleaned


def _transect_role(transect_id: str) -> str:
    if transect_id == "T001F":
        return "AOI-EDGE / FAILURE-CONTROL CASE"
    if transect_id == "T007F":
        return "CHALLENGE MORPHOLOGY CASE"
    return "PRIMARY BENCHMARK CANDIDATE"


def _load_proposed_f_transects(
    path: Path,
    *,
    context_margin_m: float,
) -> tuple[dict[str, Any], tuple[TransectInput, ...]]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(
            f"Refined F-geometry proposal is missing: {source}\n"
            "Regenerate or restore the scientist-review output; do not substitute "
            "the older T001-T010 manifest."
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read F-geometry proposal {source}: {exc}") from exc
    features = payload.get("features")
    if not isinstance(features, list):
        raise SystemExit("F-geometry proposal has no GeoJSON features array.")
    identifiers = tuple(
        feature.get("properties", {}).get("transect_id") for feature in features
    )
    if identifiers != EXPECTED_TRANSECT_IDS:
        raise SystemExit(
            "Expected exactly ordered T001F-T007F refined proposals; found "
            f"{identifiers!r}."
        )
    result: list[TransectInput] = []
    for feature in features:
        properties = feature.get("properties")
        if not isinstance(properties, dict):
            raise SystemExit("Every F geometry needs a properties object.")
        transect_id = properties["transect_id"]
        review_status = properties.get("review_status")
        if review_status != "PROPOSED / REQUIRES SCIENTIST REVIEW":
            raise SystemExit(
                f"{transect_id} must remain an unapproved proposal in this input."
            )
        diagnostics = properties.get("proposal_diagnostics")
        azimuth = (
            diagnostics.get("azimuth_deg") if isinstance(diagnostics, dict) else None
        )
        if azimuth != LOCKED_AZIMUTH_DEGREES:
            raise SystemExit(
                f"{transect_id} does not preserve locked 130.0-degree azimuth."
            )
        geometry = shape(feature.get("geometry"))
        if not isinstance(geometry, LineString):
            raise SystemExit(f"{transect_id} is not a LineString.")
        result.append(
            TransectInput(
                transect_id=transect_id,
                geometry=geometry,
                properties=properties,
                role=_transect_role(transect_id),
                region=make_review_region(geometry, context_margin_m=context_margin_m),
            )
        )
    return payload, tuple(result)


def _union_bounds(
    transects: tuple[TransectInput, ...],
) -> tuple[float, float, float, float]:
    return tuple(
        float(value)
        for value in unary_union(
            [box(*transect.region.bounds_wgs84) for transect in transects]
        ).bounds
    )


def _representative_scene(
    scenes: tuple[SentinelScene, ...], transect: LineString
) -> SentinelScene:
    center = transect.centroid
    containing = [scene for scene in scenes if shape(scene.geometry).covers(center)]
    choices = containing or list(scenes)
    return min(
        choices,
        key=lambda scene: (shape(scene.geometry).distance(center), scene.item_id),
    )


def _candidate_time_metadata(
    representative: SentinelScene, swot_datetime: str
) -> dict[str, Any]:
    timing = temporal_provenance(representative.acquisition_datetime, swot_datetime)
    return asdict(timing)


def _prepare_asset_mosaic(
    *,
    transect: TransectInput,
    scenes: tuple[SentinelScene, ...],
    candidate_directory: Path,
    asset_role: str,
) -> MosaicRecord:
    records = []
    for scene in scenes:
        item_directory = candidate_directory / _safe_atom(scene.item_id)
        record = read_windowed_cog_asset(
            scene,
            asset_role,
            bounds_wgs84=transect.region.bounds_wgs84,
            output_path=item_directory / f"{asset_role}.tif",
        )
        records.append(record)
    return mosaic_same_acquisition_chips(
        records,
        asset_role=asset_role,
        output_path=candidate_directory / f"{asset_role}_same_acquisition_mosaic.tif",
    )


def _prepare_scl_candidates(
    *,
    transect: TransectInput,
    scenes: tuple[SentinelScene, ...],
    cache_directory: Path,
    swot_datetime: str,
    minimum_valid_fraction: float,
    maximum_obscured_fraction: float,
    failure_records: list[dict[str, Any]],
) -> list[CandidateWork]:
    intersecting = scenes_intersecting_region(scenes, transect.region)
    groups = group_same_acquisition(intersecting)
    candidates: list[CandidateWork] = []
    for acquisition_key, group in groups.items():
        candidate_directory = (
            cache_directory / transect.transect_id / _safe_atom(acquisition_key)
        )
        try:
            scl = _prepare_asset_mosaic(
                transect=transect,
                scenes=group,
                candidate_directory=candidate_directory,
                asset_role="scl",
            )
            raw_scl = read_raw_mosaic(scl)
            local_mask = review_region_pixel_mask(scl, transect.region)
            diagnostics = calculate_scl_diagnostics(raw_scl[local_mask])
        except ReferenceImageryError as exc:
            failure_records.append(
                {
                    "transect_id": transect.transect_id,
                    "acquisition_key": acquisition_key,
                    "item_ids": [scene.item_id for scene in group],
                    "stage": "windowed_scl_preparation",
                    "error": str(exc),
                }
            )
            print(
                f"  {transect.transect_id} {acquisition_key}: "
                f"SCL chip unavailable: {exc}"
            )
            continue
        usable = (
            diagnostics.valid_fraction >= minimum_valid_fraction
            and diagnostics.obscured_diagnostic_fraction <= maximum_obscured_fraction
        )
        reason = (
            "meets explicit imagery-preparation coverage/obscuration guards"
            if usable
            else "retained for review but does not meet explicit preparation guards"
        )
        representative = _representative_scene(group, transect.geometry)
        candidates.append(
            CandidateWork(
                candidate_id=acquisition_key,
                acquisition_key=acquisition_key,
                acquisition_datetime=representative.acquisition_datetime.isoformat().replace(
                    "+00:00", "Z"
                ),
                temporal_reference_item_id=representative.item_id,
                scenes=group,
                search_window_days=min(scene.search_window_days for scene in group),
                temporal=_candidate_time_metadata(representative, swot_datetime),
                local_quality=diagnostics,
                technically_usable_for_review=usable,
                technical_usability_reason=reason,
                mosaics={"scl": scl},
                candidate_directory=candidate_directory,
            )
        )
    return sorted(
        candidates,
        key=lambda item: (item.acquisition_datetime, item.candidate_id),
    )


def _shortlist(
    candidates: list[CandidateWork], *, swot_datetime: str
) -> list[CandidateWork]:
    eligible = [item for item in candidates if item.technically_usable_for_review]
    pool = eligible if eligible else candidates
    mappings = [
        {
            "candidate_id": item.candidate_id,
            "acquisition_datetime": item.acquisition_datetime,
            "local_obscured_fraction": item.local_quality.obscured_diagnostic_fraction,
        }
        for item in pool
    ]
    selected_ids = {
        item["candidate_id"]
        for item in shortlist_acquisitions(mappings, swot_datetime=swot_datetime)
    }
    return [item for item in candidates if item.candidate_id in selected_ids]


def _ensure_matching_grids(records: list[MosaicRecord]) -> None:
    first = records[0]
    for record in records[1:]:
        if (
            record.crs != first.crs
            or record.width != first.width
            or record.height != first.height
            or not np.allclose(record.transform, first.transform, rtol=0, atol=1e-7)
        ):
            raise ReferenceImageryError(
                "Native 10 m B02/B03/B04/B08 mosaics do not share one grid; "
                "automatic mixed-grid resampling is intentionally not performed."
            )


def _write_rgb_geotiff(
    rgb: np.ndarray, *, reference: MosaicRecord, output_path: Path
) -> str:
    try:
        import rasterio
        from affine import Affine
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ReferenceImageryError(
            "RGB GeoTIFF output requires the reference-imagery extra."
        ) from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with rasterio.open(
            temporary,
            "w",
            driver="GTiff",
            width=reference.width,
            height=reference.height,
            count=3,
            dtype="uint8",
            crs=reference.crs,
            transform=Affine(*reference.transform[:6]),
            compress="deflate",
            photometric="RGB",
        ) as destination:
            destination.write(np.moveaxis(rgb, -1, 0))
            destination.update_tags(
                interpretation="display-only percentile-stretched Sentinel-2 RGB",
                source="B04/B03/B02",
            )
        expected_digest = sha256_file(temporary)
        if output_path.exists():
            if sha256_file(output_path) != expected_digest:
                raise ReferenceImageryError(
                    "Existing RGB cache target differs from the current source "
                    f"arrays and was not overwritten: {output_path}"
                )
            return expected_digest
        try:
            with temporary.open("rb") as source, output_path.open("xb") as target:
                shutil.copyfileobj(source, target)
                target.flush()
        except FileExistsError as exc:
            raise ReferenceImageryError(
                f"RGB cache target appeared and was not overwritten: {output_path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return expected_digest


def _line_in_crs(line: LineString, target_crs: str) -> LineString:
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    return transform(transformer.transform, line)


def _plot_rgb(
    rgb: np.ndarray,
    record: MosaicRecord,
    output_path: Path,
    *,
    title: str,
    transect: LineString | None = None,
    role: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
    left, bottom, right, top = record.bounds
    ax.imshow(rgb, extent=(left, right, bottom, top), origin="upper")
    if transect is not None:
        projected = _line_in_crs(transect, record.crs)
        x, y = projected.xy
        ax.add_patch(
            FancyArrowPatch(
                (x[0], y[0]),
                (x[-1], y[-1]),
                arrowstyle="-|>",
                mutation_scale=15,
                color="#ff2d55",
                linewidth=2.2,
                shrinkA=0,
                shrinkB=0,
                label="fixed 130° transect: station 0 → L",
                zorder=3,
            )
        )
        ax.scatter([x[0]], [y[0]], color="white", edgecolor="black", s=34, zorder=4)
        ax.annotate(
            "station 0",
            (x[0], y[0]),
            xytext=(5, 6),
            textcoords="offset points",
            color="white",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1},
        )
        ax.annotate(
            "station L",
            (x[-1], y[-1]),
            xytext=(5, 6),
            textcoords="offset points",
            color="white",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1},
        )
        ax.legend(loc="lower left", fontsize=8)
    ax.set_title(title + (f"\n{role}" if role else ""))
    ax.set_xlabel(f"Easting ({record.crs})")
    ax.set_ylabel("Northing")
    ax.set_aspect("equal")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_ndwi(
    ndwi: np.ndarray,
    record: MosaicRecord,
    output_path: Path,
    *,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
    left, bottom, right, top = record.bounds
    image = ax.imshow(
        ndwi,
        extent=(left, right, bottom, top),
        origin="upper",
        cmap="BrBG",
        norm=Normalize(vmin=-1.0, vmax=1.0),
    )
    ax.set_title(f"{title}\nNDWI continuous visualization only — no threshold")
    ax.set_xlabel(f"Easting ({record.crs})")
    ax.set_ylabel("Northing")
    ax.set_aspect("equal")
    fig.colorbar(image, ax=ax, label=f"NDWI {NDWI_FORMULA}")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _prepare_spectral_candidate(
    transect: TransectInput,
    candidate: CandidateWork,
    packet_directory: Path,
) -> None:
    for role in ("red", "green", "blue", "nir"):
        candidate.mosaics[role] = _prepare_asset_mosaic(
            transect=transect,
            scenes=candidate.scenes,
            candidate_directory=candidate.candidate_directory,
            asset_role=role,
        )
    ten_m = [candidate.mosaics[role] for role in ("red", "green", "blue", "nir")]
    _ensure_matching_grids(ten_m)
    red, green, blue, nir = [read_scaled_mosaic(record) for record in ten_m]
    rgb = scale_rgb(red, green, blue)
    ndwi = calculate_ndwi(green, nir)

    cache_rgb = candidate.candidate_directory / "rgb_display_B04_B03_B02.tif"
    candidate.rgb_geotiff_sha256 = _write_rgb_geotiff(
        rgb, reference=candidate.mosaics["red"], output_path=cache_rgb
    )
    candidate.rgb_geotiff_path = cache_rgb.resolve()
    candidate_output = (
        packet_directory / "candidates" / _safe_atom(candidate.candidate_id)
    )
    candidate.rgb_png_path = candidate_output / "sentinel_rgb.png"
    candidate.overlay_png_path = candidate_output / "sentinel_rgb_with_transect.png"
    candidate.ndwi_png_path = candidate_output / "sentinel_ndwi.png"
    title = (
        f"{transect.transect_id} — {candidate.acquisition_datetime} — "
        f"offset {candidate.temporal['signed_offset_hours']:+.2f} h"
    )
    _plot_rgb(
        rgb,
        candidate.mosaics["red"],
        candidate.rgb_png_path,
        title=title,
    )
    _plot_rgb(
        rgb,
        candidate.mosaics["red"],
        candidate.overlay_png_path,
        title=title,
        transect=transect.geometry,
        role=transect.role,
    )
    _plot_ndwi(
        ndwi,
        candidate.mosaics["green"],
        candidate.ndwi_png_path,
        title=title,
    )


def _copy_primary_review_files(candidate: CandidateWork, packet: Path) -> None:
    required = {
        "sentinel_rgb_primary.png": candidate.rgb_png_path,
        "sentinel_rgb_with_transect.png": candidate.overlay_png_path,
        "sentinel_ndwi_primary.png": candidate.ndwi_png_path,
    }
    for name, source in required.items():
        if source is None or not source.is_file():
            raise ReferenceImageryError(
                f"Prepared candidate is missing required review image {name}."
            )
        shutil.copyfile(source, packet / name)


def _candidate_csv_row(
    transect: TransectInput,
    candidate: CandidateWork,
    *,
    shortlisted: bool,
    primary: bool,
) -> dict[str, Any]:
    quality = candidate.local_quality
    return {
        "transect_id": transect.transect_id,
        "transect_role": transect.role,
        "candidate_id": candidate.candidate_id,
        "shortlisted_for_image_review": shortlisted,
        "primary_review_image": primary,
        "provider": candidate.scenes[0].provider,
        "collection": candidate.scenes[0].collection,
        "item_ids": "|".join(scene.item_id for scene in candidate.scenes),
        "mgrs_tiles": "|".join(
            sorted({scene.mgrs_tile or "UNKNOWN" for scene in candidate.scenes})
        ),
        "acquisition_datetime": candidate.acquisition_datetime,
        "signed_temporal_offset_hours": candidate.temporal["signed_offset_hours"],
        "absolute_temporal_offset_hours": candidate.temporal["absolute_offset_hours"],
        "temporal_flag": candidate.temporal["descriptive_flag"],
        "search_window_days": candidate.search_window_days,
        "valid_fraction": quality.valid_fraction,
        "cloud_fraction": quality.cloud_fraction,
        "cloud_shadow_fraction": quality.cloud_shadow_fraction,
        "cirrus_fraction": quality.cirrus_fraction,
        "snow_ice_fraction": quality.snow_ice_fraction,
        "no_data_fraction": quality.no_data_fraction,
        "saturated_defective_fraction": quality.saturated_defective_fraction,
        "non_obscured_diagnostic_fraction": quality.non_obscured_diagnostic_fraction,
        "obscured_diagnostic_fraction": quality.obscured_diagnostic_fraction,
        "technically_usable_for_review": candidate.technically_usable_for_review,
        "technical_usability_reason": candidate.technical_usability_reason,
        "rgb_geotiff_path": (
            str(candidate.rgb_geotiff_path) if candidate.rgb_geotiff_path else ""
        ),
        "rgb_geotiff_sha256": candidate.rgb_geotiff_sha256 or "",
        "rgb_review_png_path": (
            str(candidate.rgb_png_path) if candidate.rgb_png_path else ""
        ),
        "ndwi_review_png_path": (
            str(candidate.ndwi_png_path) if candidate.ndwi_png_path else ""
        ),
        "scene_cloud_cover_percent_by_item": json.dumps(
            {
                scene.item_id: scene.scene_cloud_cover_percent
                for scene in candidate.scenes
            },
            sort_keys=True,
        ),
        "interpretation": "imagery usability only; not scientific good/bad or truth",
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ReferenceImageryError(f"Cannot write empty CSV {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _common_primary_id(
    candidates_by_transect: dict[str, list[CandidateWork]],
) -> str | None:
    sets = []
    for transect_id in CORE_TRANSECT_IDS:
        usable = {
            candidate.candidate_id
            for candidate in candidates_by_transect[transect_id]
            if candidate.technically_usable_for_review
        }
        sets.append(usable)
    common = set.intersection(*sets) if sets else set()
    if not common:
        return None
    return min(
        common,
        key=lambda candidate_id: (
            sum(
                abs(
                    next(
                        candidate
                        for candidate in candidates_by_transect[transect_id]
                        if candidate.candidate_id == candidate_id
                    ).temporal["signed_offset_hours"]
                )
                for transect_id in CORE_TRANSECT_IDS
            ),
            sum(
                next(
                    candidate
                    for candidate in candidates_by_transect[transect_id]
                    if candidate.candidate_id == candidate_id
                ).local_quality.obscured_diagnostic_fraction
                for transect_id in CORE_TRANSECT_IDS
            ),
            candidate_id,
        ),
    )


def _choose_primary(
    candidates: list[CandidateWork], *, common_id: str | None
) -> CandidateWork:
    if common_id is not None:
        match = next(
            (
                candidate
                for candidate in candidates
                if candidate.candidate_id == common_id
            ),
            None,
        )
        if match is not None and match.technically_usable_for_review:
            return match
    return min(
        candidates,
        key=lambda candidate: (
            not candidate.technically_usable_for_review,
            abs(candidate.temporal["signed_offset_hours"]),
            candidate.local_quality.obscured_diagnostic_fraction,
            candidate.candidate_id,
        ),
    )


def _write_packet_metadata(
    *,
    transect: TransectInput,
    candidates: list[CandidateWork],
    shortlisted_candidates: list[CandidateWork],
    primary: CandidateWork,
    packet: Path,
    swot_context: dict[str, Any],
    search_windows_queried: list[int],
    manifest_path: Path,
    manifest_sha256: str,
    common_id: str | None,
) -> Path:
    payload = {
        "schema_version": "1.0",
        "phase": "5A.3c",
        "transect_id": transect.transect_id,
        "transect_role": transect.role,
        "geometry_status": transect.properties["review_status"],
        "geometry": {
            "type": "LineString",
            "coordinates": [list(value) for value in transect.geometry.coords],
            "crs": "EPSG:4326",
            "locked_azimuth_degrees": LOCKED_AZIMUTH_DEGREES,
            "source_manifest_path": str(manifest_path),
            "source_manifest_sha256": manifest_sha256,
        },
        "review_region": asdict(transect.region),
        "swot_temporal_context": swot_context,
        "search_windows_queried_days": search_windows_queried,
        "common_core_acquisition_candidate_id": common_id,
        "primary_candidate": primary.public_metadata(),
        "shortlisted_candidate_ids": [
            candidate.candidate_id for candidate in shortlisted_candidates
        ],
        "candidates": [
            candidate.public_metadata() for candidate in shortlisted_candidates
        ],
        "evaluated_candidates": [
            candidate.public_metadata() for candidate in candidates
        ],
        "manual_annotation_status": "NOT CREATED",
        "approval_status": "PROPOSED / OWNER DECISION REQUIRED",
        "prohibitions": {
            "scl_water_used_as_truth": False,
            "ndwi_threshold_applied": False,
            "optical_segmentation_applied": False,
            "phase5a2_candidates_displayed": False,
            "manual_boundaries_created": False,
        },
    }
    if transect.transect_id == "T001F":
        payload["failure_control_warning"] = (
            "Independent imagery may show water beyond the SWOT AOI. Missing SWOT "
            "PIXC outside the AOI is unknown, never dry."
        )
    path = packet / "reference_imagery_metadata.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _write_imagery_review_panel(
    transect: TransectInput,
    primary: CandidateWork,
    packet: Path,
) -> Path:
    paths = [
        packet / "sentinel_rgb_primary.png",
        packet / "sentinel_rgb_with_transect.png",
        packet / "sentinel_ndwi_primary.png",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    labels = ("Independent RGB", "RGB + fixed transect", "Continuous NDWI")
    for axis, path, label in zip(axes, paths, labels, strict=True):
        axis.imshow(plt.imread(path))
        axis.set_title(label)
        axis.axis("off")
    quality = primary.local_quality
    fig.suptitle(
        f"{transect.transect_id} — {transect.role}\n"
        f"{primary.acquisition_datetime}; offset "
        f"{primary.temporal['signed_offset_hours']:+.2f} h; "
        f"local obscured={quality.obscured_diagnostic_fraction:.1%}; "
        "NO MANUAL BOUNDARIES / NO AUTOMATIC TRUTH",
        fontsize=12,
    )
    output = packet / "imagery_review_panel.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def _write_overview(
    transects: tuple[TransectInput, ...],
    primary_by_transect: dict[str, CandidateWork],
    output: Path,
) -> None:
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover
        raise ReferenceImageryError("Overview rendering requires rasterio.") from exc
    target_crs = next(iter(primary_by_transect.values())).mosaics["red"].crs
    acquisition_dates = {
        candidate.acquisition_datetime[:10]
        for candidate in primary_by_transect.values()
    }
    multi_date = len(acquisition_dates) > 1
    fig, ax = plt.subplots(figsize=(10, 12), constrained_layout=True)
    # Same-acquisition chips overlap. Repeated rendering is intentional and does
    # not average pixels; each patch stays traceable to its transect packet.
    for transect in transects:
        primary = primary_by_transect[transect.transect_id]
        with rasterio.open(primary.rgb_geotiff_path) as source:
            rgb = np.moveaxis(source.read((1, 2, 3)), 0, -1)
        left, bottom, right, top = primary.mosaics["red"].bounds
        ax.imshow(rgb, extent=(left, right, bottom, top), origin="upper", zorder=1)
    project = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True).transform
    for transect in transects:
        line = transform(project, transect.geometry)
        x, y = line.xy
        color = "#ffcc00" if transect.transect_id == "T001F" else "#ff2d55"
        ax.plot(x, y, color=color, linewidth=2.0, zorder=3)
        center = line.interpolate(0.5, normalized=True)
        ax.text(
            center.x,
            center.y,
            (
                f"{transect.transect_id}\n{primary_by_transect[transect.transect_id].acquisition_datetime[:10]}"
                if multi_date
                else transect.transect_id
            ),
            color="white",
            weight="bold",
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 1.5},
            zorder=4,
        )
        ax.annotate(
            "",
            xy=(x[-1], y[-1]),
            xytext=(x[0], y[0]),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.3},
            zorder=3,
        )
    aoi = transform(project, box(*KOSHI_AOI))
    ax.plot(*aoi.exterior.xy, color="#00e5ff", linewidth=2, linestyle="--", zorder=4)
    ax.annotate(
        "UPSTREAM / NORTH",
        xy=(0.965, 0.94),
        xytext=(0.965, 0.80),
        xycoords="axes fraction",
        ha="right",
        color="black",
        weight="bold",
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.8},
    )
    ax.text(
        0.965,
        0.06,
        "DOWNSTREAM / SOUTH",
        transform=ax.transAxes,
        ha="right",
        weight="bold",
    )
    acquisition_note = (
        "MULTI-DATE REVIEW CHIPS — NOT A MOSAIC OR ONE OBSERVATION"
        if multi_date
        else f"single common acquisition date {next(iter(acquisition_dates))}"
    )
    ax.set_title(
        "Koshi Phase 5A.3c independent Sentinel-2 reference context\n"
        "T001F control; T002F–T006F primary; T007F challenge — "
        f"arrows show station direction\n{acquisition_note}"
    )
    ax.set_xlabel(f"Easting ({target_crs})")
    ax.set_ylabel("Northing")
    ax.set_aspect("equal")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _write_review_board(
    transects: tuple[TransectInput, ...],
    primary_by_transect: dict[str, CandidateWork],
    output_directory: Path,
) -> Path:
    fig, axes = plt.subplots(
        len(transects), 3, figsize=(18, 4.2 * len(transects)), constrained_layout=True
    )
    for row, transect in enumerate(transects):
        primary = primary_by_transect[transect.transect_id]
        quality = primary.local_quality
        packet = output_directory / transect.transect_id
        paths = (
            packet / "sentinel_rgb_primary.png",
            packet / "sentinel_rgb_with_transect.png",
            packet / "sentinel_ndwi_primary.png",
        )
        for column, path in enumerate(paths):
            axes[row, column].imshow(plt.imread(path))
            axes[row, column].axis("off")
        axes[row, 0].set_title(
            f"{transect.transect_id} — {transect.role}\n"
            f"{primary.acquisition_datetime[:10]}; "
            f"offset {primary.temporal['signed_offset_hours']:+.2f} h "
            f"({primary.temporal['descriptive_flag']})\n"
            f"local valid={quality.valid_fraction:.1%}; "
            f"obscured={quality.obscured_diagnostic_fraction:.1%}\nRGB"
        )
        axes[row, 1].set_title("RGB + 130° fixed proposal")
        axes[row, 2].set_title("NDWI diagnostic only")
    fig.suptitle(
        "Koshi T001F–T007F imagery review board — proposals remain unapproved",
        fontsize=15,
    )
    output = output_directory / "koshi_T001F_T007F_imagery_review_board.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def _write_owner_review(path: Path) -> bool:
    """Create a blank owner sheet or preserve it; return whether it has entries."""

    fieldnames = (
        "transect_id",
        "geometry_decision",
        "imagery_decision",
        "owner_notes",
    )
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        if tuple(reader.fieldnames or ()) != fieldnames:
            raise ReferenceImageryError(
                f"Existing owner-review table has unexpected columns: {path}"
            )
        if tuple(row["transect_id"] for row in rows) != EXPECTED_TRANSECT_IDS:
            raise ReferenceImageryError(
                f"Existing owner-review table has unexpected transect IDs: {path}"
            )
        return any(
            any((row.get(field) or "").strip() for field in fieldnames[1:])
            for row in rows
        )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for transect_id in EXPECTED_TRANSECT_IDS:
            writer.writerow(
                {
                    "transect_id": transect_id,
                    "geometry_decision": "",
                    "imagery_decision": "",
                    "owner_notes": "",
                }
            )
    return False


def _relative_output_hashes(output_directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(output_directory)): sha256_file(path)
        for path in sorted(output_directory.rglob("*"))
        if path.is_file() and path.name != "reference_imagery_preparation_manifest.json"
    }


def main() -> None:
    """Prepare the real Phase 5A.3c imagery-review package and stop."""

    args = _parse_args()
    minimum_valid = _validate_fraction(
        args.minimum_valid_fraction, "--minimum-valid-fraction"
    )
    maximum_obscured = _validate_fraction(
        args.maximum_obscured_fraction, "--maximum-obscured-fraction"
    )
    if (
        not math.isfinite(args.context_margin_m)
        or not 1000 <= args.context_margin_m <= 2000
    ):
        raise SystemExit("--context-margin-m must be in the documented 1–2 km range.")

    geometry_path = args.transects.expanduser().resolve()
    geometry_hash_before = _file_sha256(geometry_path)
    geometry_payload, transects = _load_proposed_f_transects(
        geometry_path,
        context_margin_m=args.context_margin_m,
    )
    pixc_paths = tuple(
        (args.pixc_directory.expanduser().resolve() / name)
        for name in EXPECTED_PIXC_FILENAMES
    )
    missing_pixc = [path for path in pixc_paths if not path.is_file()]
    if missing_pixc:
        raise SystemExit(
            "Exact SWOT observation timing cannot be read because required local "
            "granules are missing:\n"
            + "\n".join(f"  - {path}" for path in missing_pixc)
        )
    swot_context = read_swot_temporal_context(pixc_paths)
    swot_metadata = swot_context.to_dict()
    output_directory = args.output_directory.expanduser().resolve()
    cache_directory = args.cache_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    cache_directory.mkdir(parents=True, exist_ok=True)

    provider = EarthSearchSentinelProvider(
        endpoint=args.stac_endpoint,
        collection=args.stac_collection,
    )
    search_bounds = _union_bounds(transects)
    all_scenes: dict[str, SentinelScene] = {}
    candidates_by_transect: dict[str, list[CandidateWork]] = {
        transect.transect_id: [] for transect in transects
    }
    windows_queried: list[int] = []
    preparation_failures: list[dict[str, Any]] = []

    for window_days in SEARCH_WINDOWS_DAYS:
        windows_queried.append(window_days)
        reference = datetime.fromisoformat(
            swot_context.reference_datetime.replace("Z", "+00:00")
        )
        print(f"Searching public Sentinel-2 L2A STAC +/-{window_days} days...")
        found = provider.search(
            bbox_wgs84=search_bounds,
            start=reference - timedelta(days=window_days),
            end=reference + timedelta(days=window_days),
            search_window_days=window_days,
        )
        for scene in found:
            all_scenes.setdefault(scene.item_id, scene)
        if args.search_only:
            if all_scenes:
                break
            continue
        for transect in transects:
            print(f"Preparing local SCL diagnostics for {transect.transect_id}...")
            candidates_by_transect[transect.transect_id] = _prepare_scl_candidates(
                transect=transect,
                scenes=tuple(all_scenes.values()),
                cache_directory=cache_directory,
                swot_datetime=swot_context.reference_datetime,
                minimum_valid_fraction=minimum_valid,
                maximum_obscured_fraction=maximum_obscured,
                failure_records=preparation_failures,
            )
        if all(
            any(item.technically_usable_for_review for item in candidates)
            for candidates in candidates_by_transect.values()
        ):
            break

    search_record = {
        "schema_version": "1.0",
        "provider": provider.provider_name,
        "stac_endpoint": provider.endpoint,
        "collection": provider.collection,
        "search_bounds_wgs84": search_bounds,
        "windows_queried_days": windows_queried,
        "swot_temporal_context": swot_metadata,
        "scenes": [scene.to_dict() for scene in all_scenes.values()],
        "preparation_failures": preparation_failures,
    }
    (output_directory / "stac_search_record.json").write_text(
        json.dumps(search_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.search_only:
        print(f"Search record: {output_directory / 'stac_search_record.json'}")
        print("Search-only mode stopped before any imagery chips or annotations.")
        return
    if not all_scenes:
        raise SystemExit("No Sentinel-2 L2A candidates were found within +/-30 days.")
    missing_candidates = [
        transect_id
        for transect_id, candidates in candidates_by_transect.items()
        if not candidates
    ]
    if missing_candidates:
        raise SystemExit(
            "No windowed, locally diagnosed imagery candidate could be prepared "
            "within +/-30 days for: " + ", ".join(missing_candidates)
        )

    evaluated_by_transect = candidates_by_transect
    shortlisted_by_transect = {
        transect.transect_id: _shortlist(
            evaluated_by_transect[transect.transect_id],
            swot_datetime=swot_context.reference_datetime,
        )
        for transect in transects
    }
    common_id = _common_primary_id(shortlisted_by_transect)
    primary_by_transect: dict[str, CandidateWork] = {}
    scene_rows: list[dict[str, Any]] = []
    metadata_paths: list[Path] = []

    for transect in transects:
        packet = output_directory / transect.transect_id
        packet.mkdir(parents=True, exist_ok=True)
        candidates = evaluated_by_transect[transect.transect_id]
        shortlisted = shortlisted_by_transect[transect.transect_id]
        primary = _choose_primary(shortlisted, common_id=common_id)
        primary_by_transect[transect.transect_id] = primary
        for candidate in shortlisted:
            print(
                f"Reading windowed B02/B03/B04/B08 for {transect.transect_id} "
                f"{candidate.acquisition_datetime}..."
            )
            _prepare_spectral_candidate(transect, candidate, packet)
        _copy_primary_review_files(primary, packet)
        rows = [
            _candidate_csv_row(
                transect,
                candidate,
                shortlisted=candidate in shortlisted,
                primary=candidate.candidate_id == primary.candidate_id,
            )
            for candidate in candidates
        ]
        _write_rows(packet / "sentinel_candidates.csv", rows)
        scene_rows.extend(rows)
        metadata_paths.append(
            _write_packet_metadata(
                transect=transect,
                candidates=candidates,
                shortlisted_candidates=shortlisted,
                primary=primary,
                packet=packet,
                swot_context=swot_metadata,
                search_windows_queried=windows_queried,
                manifest_path=geometry_path,
                manifest_sha256=geometry_hash_before,
                common_id=common_id,
            )
        )
        _write_imagery_review_panel(transect, primary, packet)

    _write_rows(output_directory / "sentinel_scene_summary.csv", scene_rows)
    owner_review_path = output_directory / "geometry_imagery_owner_review.csv"
    owner_decisions_present = _write_owner_review(owner_review_path)
    _write_overview(
        transects,
        primary_by_transect,
        output_directory / "koshi_reference_imagery_overview.png",
    )
    _write_review_board(transects, primary_by_transect, output_directory)

    geometry_hash_after = _file_sha256(geometry_path)
    if geometry_hash_after != geometry_hash_before:
        raise SystemExit(
            "F-geometry input changed during imagery preparation; refusing to "
            "complete the manifest."
        )
    preparation_manifest = {
        "schema_version": "1.0",
        "phase": "5A.3c",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "provider": provider.provider_name,
        "stac_endpoint": provider.endpoint,
        "stac_collection": provider.collection,
        "search_windows_queried_days": windows_queried,
        "search_bounds_wgs84": search_bounds,
        "context_margin_m": args.context_margin_m,
        "local_usability_policy": {
            "minimum_valid_fraction": minimum_valid,
            "maximum_obscured_fraction": maximum_obscured,
            "interpretation": (
                "operational imagery-preparation guard only; not wet/dry truth, "
                "geometry approval, or scientific validation"
            ),
        },
        "rgb_display": {
            "bands": ["B04", "B03", "B02"],
            "stretch_percentiles": [2.0, 98.0],
            "gamma": 1.0,
        },
        "ndwi": {"formula": NDWI_FORMULA, "interpretation": NDWI_INTERPRETATION},
        "scl_interpretation": (
            "official categorical classes used for local image-usability "
            "diagnostics only; class 6 water is never truth"
        ),
        "swot_temporal_context": swot_metadata,
        "input_f_geometry": {
            "path": str(geometry_path),
            "sha256_before": geometry_hash_before,
            "sha256_after": geometry_hash_after,
            "byte_for_byte_unchanged": geometry_hash_before == geometry_hash_after,
            "feature_count": len(geometry_payload["features"]),
            "review_status": "PROPOSED / REQUIRES SCIENTIST REVIEW",
        },
        "transects": {
            transect.transect_id: {
                "role": transect.role,
                "primary_candidate_id": primary_by_transect[
                    transect.transect_id
                ].candidate_id,
                "candidate_count": len(candidates_by_transect[transect.transect_id]),
                "shortlisted_candidate_count": len(
                    shortlisted_by_transect[transect.transect_id]
                ),
                "metadata_path": str(
                    output_directory
                    / transect.transect_id
                    / "reference_imagery_metadata.json"
                ),
            }
            for transect in transects
        },
        "common_core_acquisition": {
            "candidate_id": common_id,
            "covers_T002F_through_T007F_under_preparation_guards": common_id
            is not None,
        },
        "owner_review": {
            "path": str(owner_review_path),
            "decision_cells_populated": owner_decisions_present,
            "existing_entries_preserved_without_interpretation": True,
            "allowed_geometry_decisions": ["APPROVE", "MODIFY", "HOLD"],
        },
        "annotations_created": 0,
        "manual_boundaries_fabricated": False,
        "phase5a3_sensitivity_validation_started": False,
        "preparation_failures": preparation_failures,
        "output_sha256": _relative_output_hashes(output_directory),
        "cache_directory": str(cache_directory),
    }
    manifest_path = output_directory / "reference_imagery_preparation_manifest.json"
    manifest_path.write_text(
        json.dumps(preparation_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared independent imagery for {len(transects)} F transects.")
    print(
        f"Common core acquisition: {common_id or 'none; per-transect selection used'}"
    )
    owner_status = (
        "existing entries preserved" if owner_decisions_present else "blank decisions"
    )
    print(f"Owner review table ({owner_status}): {owner_review_path}")
    print(f"Preparation manifest: {manifest_path}")
    print(
        "No geometry was approved and no manual boundary or wet interval was created."
    )


if __name__ == "__main__":
    main()
