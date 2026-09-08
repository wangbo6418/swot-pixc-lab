# SWOT PIXC Lab

SWOT PIXC Lab is an early-stage scientific Python toolkit for finding,
retrieving, and opening NASA Surface Water and Ocean Topography (SWOT) Level 2
High Rate Pixel Cloud (PIXC) granules. Its long-term purpose is to support
reproducible, pixel-level work on multiple-channel and anabranching rivers. The
current release implements **Phases 1–3**: AOI/date discovery, metadata
inspection, download/cache handling, verified local-file manifests, raw
`/pixel_cloud` reading, exact AOI clipping, and provenance-preserving
combination of tiles from one cycle/pass observation, followed by explicit,
metadata-driven quality-control views and a documented EGM2008 height
derivation.

The Phase 2 representation remains deliberately raw. Phase 3 never overwrites
it: QC results contain derived masks and filtered views while the original
values, integer bitfields, source indices, and source-point indices remain
available. No profile is presented as a validated universal scientific
standard.

## Why PIXC?

The [PO.DAAC PIXC Version D collection](https://podaac.jpl.nasa.gov/dataset/SWOT_L2_HR_PIXC_D)
contains a geolocated, irregular pixel cloud of water detections plus retained
neighboring/pruning-mask pixels from which higher-level hydrology products are
derived. In contrast, the
[RiverSP product](https://podaac.jpl.nasa.gov/dataset/SWOT_L2_HR_RiverSP_D)
reports measurements for predefined river reaches and nodes. RiverSP is the
appropriate product for many feature-level analyses; PIXC is useful when the
individual detections, classifications, quality information, or spatial
arrangement within a reach must be examined.

That distinction matters for multiple-channel rivers. Several wetted branches
can occupy one reach, and feature-level summaries do not by themselves expose
the evidence needed to separate branches or evaluate gaps, bank edges,
layover, dark water, and classification uncertainty. PIXC retains the
pixel-level evidence needed for that future analysis. This package does **not**
yet calculate channel widths or interpret river morphology.

## Product and API decisions

Discovery uses NASA's Common Metadata Repository (CMR) through
[`earthaccess`](https://earthaccess.readthedocs.io/en/latest/user/). The default
is deliberately pinned to the active PO.DAAC collection:

- short name: `SWOT_L2_HR_PIXC_D`
- PO.DAAC collection version: `D`
- CMR collection concept ID: `C3233944986-POCLOUD`
- dataset DOI: [`10.5067/SWOT-PIXC-D`](https://doi.org/10.5067/SWOT-PIXC-D)

The concept ID selects Version D rather than asking CMR for an ambiguous
"latest" version. It does not freeze the CMR catalog: PO.DAAC can add granules
or revise metadata, so save the returned metadata table with each analysis.
The default will not silently move to a future collection version. A different
verified collection can be requested with `collection_concept_id=...`.

Searches use CMR footprint intersection. Discovery itself does not clip pixels
to the AOI; exact pixel clipping occurs only when verified local files are
opened. Date-only bounds are interpreted as inclusive UTC days. The accepted
AOIs are:

- `(west, south, east, north)` in WGS 84 longitude/latitude;
- GeoJSON Polygon or MultiPolygon, including Feature wrappers; or
- a GeoDataFrame-like object with a declared CRS (install the `geo` extra).

GeoDataFrame-like AOIs are explicitly transformed to EPSG:4326. A bounding
box that crosses the antimeridian is split into two CMR searches. Polygon holes
are omitted from the conservative CMR discovery query and recorded as a search
note. Exact Phase 2 clipping uses the complete polygon, including holes, and
includes points on the AOI boundary. Longitude is normalized to `[-180, 180)`
only for the spatial comparison; the original PIXC longitude array is retained
unchanged. Antimeridian-crossing polygons must be supplied as an explicit
MultiPolygon.

`open()` is local-only and never downloads implicitly. Multiple files can be
combined only when every source has the same resolved SWOT cycle and pass, and
only one granule is supplied for each tile. Duplicate paths and ambiguous
same-tile reprocessings are rejected for explicit source selection. Files are
concatenated in manifest order; every retained pixel remains in the result,
even if two source files contain spatially coincident samples.

Authoritative product details, schema definitions, and processing notes are in
the [PIXC Product Description Document](https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-docs/web-misc/swot_mission_docs/pdd/D-56411_SWOT_Product_Description_L2_HR_PIXC_20250224a_RevC_clean_sig_final.pdf)
and the [Version D KaRIn release notes](https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-docs/web-misc/swot_mission_docs/SWOT_VersionD_KaRIn_Products_Release_Note_20250423b.pdf).

## Installation

Python 3.12 or newer is required by the current package configuration.
Runtime dependencies, including `earthaccess`, `netCDF4`, NumPy, pandas,
Shapely, and xarray, are declared in `pyproject.toml` and are installed with the
package; they do not need to be installed individually.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

For GeoDataFrame AOIs and the demonstration notebook:

```bash
python -m pip install -e ".[geo,notebook]"
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
python -m pytest -m "not integration"
```

## Five-minute Phase 1–3 example

This example uses the project owner's upper Koshi River AOI and one known
cycle/pass observation. Discovery is normally anonymous; protected PO.DAAC
files require a free
[NASA Earthdata Login](https://urs.earthdata.nasa.gov/users/new). A PIXC tile can
be hundreds of MiB, so always inspect the discovery table before opting in to a
download.

```python
from pathlib import Path

from swot_pixc_lab import PixcCollection, apply_qc

# West, south, east, north in EPSG:4326.
KOSHI_AOI = (86.87, 26.49, 87.20, 26.90)

collection = PixcCollection.search(
    aoi=KOSHI_AOI,
    start_date="2024-01-14",
    end_date="2024-01-14",
)

# Confirm that the selected records form one cycle/pass observation.
print(
    collection.table[
        [
            "observation_datetime",
            "cycle",
            "pass",
            "tile",
            "filename",
            "size_bytes",
        ]
    ].to_string(index=False)
)

cache = Path("data/koshi_phase1")

# Explicitly opt in only after reviewing every file and its size:
# collection.download(cache, verify="auto", persist_credentials=False)

# Once the selected granules are present in the verified cache:
local = collection.resolve_local(cache, verify="auto")
observation = local.open(aoi=KOSHI_AOI)

print(observation.source_table.to_string(index=False))
print(observation.summary())
raw_pixels = observation.raw  # xarray.Dataset; no QC or WSE transformation

# Phase 3 creates transparent masks/views; observation.raw is unchanged.
raw = observation.apply_qc(profile="raw")
legacy = apply_qc(observation, profile="bo_legacy_strict")
extent = observation.apply_qc(profile="channel_extent_candidate")

print(legacy.summary)
print(legacy.reason_counts)  # each rule evaluated independently
print(legacy.incremental_reason_counts)  # removal in documented rule order
print(extent.filtered)  # still has per-pixel provenance
```

With the two project-owner Koshi files already cached, run
[`examples/koshi_phase2_reality_check.py`](examples/koshi_phase2_reality_check.py)
from the repository root for the reproducible schema, provenance, and pixel-
count validation used in Phase 2.

The corresponding Phase 3 real-data regression is
[`examples/koshi_phase3_reality_check.py`](examples/koshi_phase3_reality_check.py).
It never downloads files and reports the raw, legacy, and experimental-profile
results for the two existing Koshi tiles. The resulting scientific evidence,
exact rule definitions, and open review questions are recorded in
[`docs/phase3_koshi_qc_report.md`](docs/phase3_koshi_qc_report.md).

`local.open(aoi=...)` reads the real `/pixel_cloud` group, excludes only points
that cannot be spatially located or do not intersect the exact AOI, and then
concatenates the retained point arrays. It does not classify any quality flag
as good or bad. `observation.summary()` reports before/after and invalid-
coordinate counts, raw classification counts, coordinate ranges, an
unfiltered summary of the reported ellipsoidal `height`, and retained counts by
source tile.

The default point variables now include the Phase 2 core fields
`azimuth_index`, `range_index`, `latitude`, `longitude`, `height`,
`classification`, `water_frac`, `pixel_area`, `sig0`, and `cross_track`, plus
the Phase 3 fields `water_frac_uncert`, `geoid`, `inc`, `phase_noise_std`,
`bright_land_flag`, `false_detection_rate`, `missed_detection_rate`,
`prior_water_prob`, `prior_water_change`, and
`ancillary_surface_classification_flag`. All available one-dimensional point
quality variables ending in `_qual` are also loaded, including
`classification_qual`, `geolocation_qual`, `interferogram_qual`, and
`sig0_qual`. Pass an explicit `variables=(...)` sequence to load a smaller
compatible selection; `latitude` and `longitude` are always included because
exact clipping requires them. A QC profile records a skipped rule instead of
inventing a substitute when an optional input is absent.

`collection.table` is a defensive-copy pandas table. Available columns are:

```text
observation_datetime, observation_start, observation_end, cycle, pass, tile,
granule_id, native_id, filename, product_short_name, product_version,
concept_id, collection_concept_id, provider, revision_id, revision_date,
metadata_specification_version, source_url, s3_url, size_bytes, checksum,
checksum_algorithm, pge_version, production_datetime, granule_bbox, local_path,
metadata_warnings
```

Missing CMR fields remain null rather than being guessed. Cycle, pass, and tile
are read from CMR metadata when available; a documented warning accompanies
any fallback to the official PIXC filename convention. Duplicate search hits
are removed conservatively without merging different product versions.

The Phase 1 discovery/download walkthrough is
[`notebooks/01_pixc_aoi_demo.ipynb`](notebooks/01_pixc_aoi_demo.ipynb).

## Earthdata authentication

`PixcCollection.search()` queries CMR without forcing a login.
`PixcCollection.download()` delegates authentication to `earthaccess`; this
project never accepts or stores passwords. With the default
`auth_strategy="all"`, `earthaccess` tries, in order:

1. `EARTHDATA_TOKEN`, or `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`;
2. `.netrc` (`_netrc` on Windows); and
3. an interactive prompt.

The package defaults to `persist_credentials=False`. Set persistence only if
you understand that it writes credentials to a netrc file. See the official
[`earthaccess` authentication guide](https://earthaccess.readthedocs.io/en/latest/user/explanation/authenticate/)
and [NASA CMR Search API documentation](https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html).

## Cache and local-resolution semantics

`download(destination)` treats the destination as a filename-based cache.
Existing files are reused only after validation. Missing files are downloaded
to a temporary staging directory and validated as a set. Each file is committed
with a no-clobber hard link when the filesystem supports one. Otherwise, the
fallback opens the destination in exclusive-create mode, copies the staged
content, and verifies the copy. Either strategy refuses an existing or
concurrently created target, and a commit failure rolls back files created by
the transaction.

The default `verify="auto"` verifies the CMR checksum when one is available and
supported, otherwise falling back to the CMR byte size. `verify="size"` avoids
the checksum scan for faster but weaker repeat access, while
`verify="checksum"` requires checksum metadata. `verify="none"` still requires
a regular, non-empty file. An invalid cache entry raises an actionable error
instead of being deleted or replaced automatically.

`collection.resolve_local(cache_dir)` makes no network request, does not
download anything, and does **not** parse NetCDF variables. It returns a
`LocalPixcCollection`: a verified manifest with `paths` and a provenance
`table`. Phase 2 parsing begins only when `local.open(aoi=...)` is called. As a
convenience, `collection.open(aoi=..., cache_dir=...)` performs the local
resolution and opening steps together, but it still never downloads data.

## Raw PIXC representation and preservation

Opening returns a `PixcObservation`. Its `pixels` dataset, also available as
`raw`, is an in-memory xarray `Dataset` indexed by `points`. NetCDF automatic
masking, scaling, and character conversion are disabled while reading so each
loaded SWOT point variable retains its original name and dtype. Its original
attributes—including units, `_FillValue`, `flag_values`, `flag_masks`, and
`flag_meanings`—are copied into immutable per-source metadata.

Every `PixcSourceMetadata` entry in `observation.sources` preserves:

- the source path, filename, granule identifier, cycle, pass, tile, CRID,
  NetCDF `product_version` (the file product counter), and PGE version;
- the complete immutable Phase 1 `cmr_record`, when opening from a discovered
  collection, including collection Version D, CMR concept/revision IDs,
  timestamps, checksum, and size;
- the root and `/pixel_cloud` group attributes;
- the original dimensions, dtype, shape, fill value, and complete attributes
  for every `/pixel_cloud` variable, including variables not loaded as points;
  and
- the point counts before and after clipping, invalid-coordinate count,
  longitude-normalization count, and any reader notes.

Two library-added integer arrays make provenance explicit for every retained
pixel: `source_index` identifies the corresponding entry in
`observation.sources`, and `source_point_index` records the pixel's original
zero-based position in that file's `/pixel_cloud/points` dimension. They are
clearly marked as library provenance rather than SWOT variables.

When a variable's attributes are identical across all tiles, they are also
attached to the combined xarray variable. If definitions differ, only common
attributes appear on the combined variable and each exact source definition
remains available in `observation.sources`; the difference is recorded in
`observation.notes`. No source-specific metadata is silently overwritten.

Coordinate fill values and non-finite coordinates cannot be located and are
therefore excluded from the spatial result, with their count recorded. All
other pixels outside the AOI are removed by the boundary-inclusive geometric
clip. These are ingestion operations, not quality-control decisions.

## Official variable meanings and cautions

The default Phase 2 fields use their official `/pixel_cloud` names and
meanings:

| Variable | Official meaning and important caution |
|---|---|
| `latitude`, `longitude` | Medium-layer geodetic coordinates relative to the file's reference ellipsoid. Longitude is documented on `[-180, 180)`; pixel-level geolocation can be noisy. |
| `height` | Height above the reference ellipsoid. It is **not renamed to WSE**. Instrument, propagation-delay, and crossover-calibration effects are already embodied in the reported value, while supplied geoid and tide models are not applied. |
| `classification` | Raw values 1–7 mean `land`, `land_near_water`, `water_near_land`, `open_water`, `dark_water`, `low_coh_water_near_land`, and `open_low_coh_water`; 255 is the documented fill value. No class is filtered in Phase 2. |
| `water_frac` | Noisy water-fraction estimate. Values below 0 or above 1 are intentional possibilities and are not clipped; PO.DAAC advises aggregation rather than pixel-level interpretation. |
| `pixel_area` | Horizontal posting area on a flat surface at the reference-DEM height. It does not account for surface slope and is not the instrument's 3 dB resolution footprint. |
| `sig0` | Normalized radar cross section in real linear units, not decibels. It can be negative after noise subtraction. |
| `cross_track` | Approximate signed distance from nadir: positive right and negative left. It uses a local spherical-Earth approximation and the reference-DEM location rather than the computed geolocation. |

The one-value-per-pixel quality fields are `interferogram_qual`,
`classification_qual`, `geolocation_qual`, and `sig0_qual`. They remain raw
unsigned bit fields with official masks and meanings attached. The related
`pixc_line_qual` uses the separate `num_pixc_lines` dimension and is preserved
in source schema metadata rather than incorrectly broadcast into the point
dataset.

## Data provenance

Keep both the discovery manifest and `PixcObservation.source_table` alongside
every downstream analysis. The manifest records the CMR
granule, native, collection, and metadata-revision identifiers; observation,
revision, and production times; cycle/pass/tile identifiers; product, PGE, and
UMM-G specification versions when supplied; source URLs; bounding boxes;
checksums; byte sizes; metadata warnings; and resolved local paths. Also save
`collection.provenance.as_dict()`, which contains the exact normalized temporal
and spatial filters used for discovery. The observation source table adds the
NetCDF-reported identifiers and exact before/after counts, while
`source_index` and `source_point_index` trace each output pixel back to its file
and original array position. Local paths are machine-specific; the NASA
identifiers and source granule filenames are the durable links back to the
archive.

The source table labels CMR `collection_version` (for example, `D`) separately
from the root-attribute `netcdf_product_version` (for these files, the `01`
product counter). Direct `open_pixc(...)` calls have no CMR record, so those CMR
columns remain null instead of being inferred from a filename.

When publishing work, cite the dataset using the guidance on the
[PO.DAAC collection page](https://podaac.jpl.nasa.gov/dataset/SWOT_L2_HR_PIXC_D),
including the data-access date.

## QC philosophy

Phase 3 is a separate layer over the Phase 2 raw observation. Calling
`apply_qc(observation, profile=...)` or `observation.apply_qc(...)` returns a
result with the selected profile, a boolean mask aligned to every raw point, a
detached filtered xarray dataset, named reason masks, rule outcomes, decoded
quality flags, and summary counts. Both retained and rejected pixels remain
traceable through the unchanged `source_index` and `source_point_index` arrays
in the raw observation.

The decoder follows CF-style flag metadata instead of embedding undocumented
integer interpretations. It reads `_FillValue`, `valid_min`, `valid_max`,
`flag_masks`, `flag_values`, and `flag_meanings` from the loaded variable.
Original quality arrays remain `uint32`; each official condition is exposed as
a derived boolean mask. Fill values are handled before bit operations so the
`0xffffffff` fill value cannot be mistaken for every flag being set. The four
PIXC point-quality fields use combinable masks, not mutually exclusive
`good`/`suspect`/`bad` summary codes: zero means no named condition is set,
while a nonzero value can contain one or more suspect, degraded, missing, or
bad conditions.

Three profiles are available:

- `raw` retains every Phase 2 exact-AOI pixel. It performs no additional
  scientific filtering.
- `bo_legacy_strict` reproduces the project owner's conservative, WSE-oriented
  legacy sequence: `classification == 4`, `water_frac >= 0.90`,
  `water_frac_uncert <= 0.15`, `bright_land_flag == 0`,
  `false_detection_rate <= 0.10`, each of the four point `*_qual` integers
  equal to zero, `phase_noise_std <= 1.0`, and inclusive
  `0.5 <= inc <= 5.0` degrees. It is a reproducibility profile, not a NASA
  recommendation. Rules whose variables are absent are reported as skipped.
- `channel_extent_candidate` is experimental. It keeps the documented water
  classes 3–7 rather than only class 4 and imposes no per-pixel water-fraction
  threshold. It requires valid classification and classification/geolocation
  quality values with no undocumented set bits, rejects `classification_qual`
  conditions
  `in_air_pixel_degraded`, `coherent_power_bad`, `tvp_bad`, `sc_event_bad`, and
  `large_karin_gap`, and rejects `geolocation_qual` conditions
  `no_geolocation_bad`, `medium_phase_bad`, `tvp_bad`, `sc_event_bad`, and
  `large_karin_gap`. Other suspect/degraded conditions remain visible
  diagnostics. `interferogram_qual` and `sig0_qual` do not otherwise filter
  this profile, but the independently reported `interferogram_qual`
  `in_air_pixel_degraded` condition is also excluded because it states that
  the range bin does not intersect Earth's surface.

Every rule records its variable, condition, scientific rationale, definition
source, and outcome. `reason_counts` are independent failure counts, so one
pixel can appear under several reasons and those counts must not be summed.
`incremental_reason_counts` report only newly removed pixels as rules are
applied in their documented order. The final removed count is therefore not
generally the sum of independent reason counts.

When the loaded `geoid` metadata explicitly identifies EGM2008 in metres above
the reference ellipsoid, Phase 3 adds the derived variable
`height_egm2008 = height - geoid`. Raw `height` and `geoid` remain unchanged.
If the model identity is not explicit but the units and reference metadata are
compatible, the safer name `height_minus_geoid` is used instead.
The supplied geoid is in the mean-tide system; the derivation changes the
height reference and is **not corrected WSE**. Phase 3 does not apply solid
Earth, load, or pole tides, tropospheric or ionospheric corrections, or any
other geophysical field.

These profiles make filtering inspectable and reproducible; they do not make
the retained pixels automatically analysis-ready. In particular, the
experimental extent profile still requires review by a SWOT/river scientist
before use in any channel-geometry method.

## Current limitations

- CMR footprint intersection can return tiles with no pixels inside the exact
  AOI. Downloads still retrieve complete NetCDF tiles; clipping occurs during
  local opening, and zero-hit source files remain represented in provenance.
- Exact duplicate hits from multiple AOI-component searches are removed by CMR
  granule identity, but distinct reprocessed instances, CRIDs, or product
  counters are retained in discovery. Opening multiple versions of the same
  tile is rejected; choosing one requires an explicit scientist-controlled
  selection because no automated revision policy is implemented.
- Multi-file opening requires one cycle/pass. It does not infer observation
  groups from a broad, multi-date manifest.
- The reader eagerly loads selected one-dimensional `/pixel_cloud/points`
  variables into memory. The `/tvp` and `/noise` groups and non-point variables
  are not combined into the pixel dataset.
- `pixc_line_qual` remains line-level schema metadata; it is not broadcast or
  expanded across points, and Phase 3 does not use it as a point filter.
- Tile combination is stable concatenation. It intentionally does not detect,
  average, or remove coincident or duplicate pixels.
- Longitude normalization is used only for clipping; original longitudes are
  retained. Polygon edges crossing the antimeridian must be split into an
  explicit MultiPolygon.
- The current notebook demonstrates Phase 1 discovery and download. Phase 2
  and Phase 3 real-data scripts are provided, but a combined scientific
  notebook, visualization, and export workflow is not yet implemented.
- The legacy QC profile is intentionally conservative and is retained for
  reproducibility, not endorsed as a universal WSE filter. The
  `channel_extent_candidate` profile is explicitly experimental and has not
  been validated as a channel-boundary or width method.
- `height_egm2008` is a metadata-checked `height - geoid` derivation. No
  corrected WSE, tide adjustment, uncertainty propagation, or research-ready
  export is implemented.
- No multi-channel segmentation, width, WSE comparison, or morphological
  interpretation is implemented.
- The default collection ID and Version D product assumptions must be reviewed
  when PO.DAAC releases a successor or revises collection guidance.
- Network availability, Earthdata authorization, real granule metadata, file
  sizes, and checksums are external conditions. Unit tests mock these services;
  the optional live CMR test is not a substitute for scientist validation.

## Roadmap

1. **Phase 1 (complete):** CMR discovery, metadata, download/cache, and verified
   local manifests.
2. **Phase 2 (complete):** metadata-preserving Version D `/pixel_cloud` reading,
   exact AOI clipping, and no-loss combination of tiles from one observation.
3. **Phase 3 (current, complete):** preserve raw pixels; decode official
   Version D quality flags; provide raw, legacy, and experimental QC profiles
   with auditable removal summaries; and derive metadata-verified EGM2008
   height without claiming corrected WSE.
4. **Phase 4:** scalable pixel visualization and research-ready export.
5. Validate experimental multiple-channel methods with SWOT specialists,
   independent observations, and sensitivity tests.
6. Consider a web interface or AI orchestration only after the scientific API
   is stable. Neither is part of the current implementation.

## Official references

- [PO.DAAC: SWOT L2 HR PIXC, Version D](https://podaac.jpl.nasa.gov/dataset/SWOT_L2_HR_PIXC_D)
- [NASA CMR virtual directory for `C3233944986-POCLOUD`](https://cmr.earthdata.nasa.gov/virtual-directory/collections/C3233944986-POCLOUD)
- [SWOT PIXC Product Description Document](https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-docs/web-misc/swot_mission_docs/pdd/D-56411_SWOT_Product_Description_L2_HR_PIXC_20250224a_RevC_clean_sig_final.pdf)
- [SWOT Version D KaRIn Products Release Note](https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-docs/web-misc/swot_mission_docs/SWOT_VersionD_KaRIn_Products_Release_Note_20250423b.pdf)
- [`earthaccess` user guide](https://earthaccess.readthedocs.io/en/latest/user/)
- [NASA CMR Search API](https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html)
