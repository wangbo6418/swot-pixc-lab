# Phase 5A.3c independent Sentinel-2 reference imagery

## Scope and scientific status

Phase 5A.3c prepares independent, near-date Sentinel-2 context for manual
Phase 5A.1 annotation of the Koshi pilot transects. It does not detect banks,
segment optical water, create wet intervals, approve a transect, tune the
Phase 5A.2 method, or run Phase 5A.3 validation.

The intended evidence path is:

```text
SWOT observation and exact acquisition-time metadata
        ↓
independent Sentinel-2 Level-2A review imagery
        ↓
scientist reviews geometry, imagery, clouds, and timing
        ↓
scientist explicitly clicks every wet-interval boundary
        ↓
Phase 5A.1 measure_explicit_wet_intervals(...)
        ↓
later, after the gates are satisfied: Phase 5A.2/5A.3 validation
```

Sentinel RGB, continuous NDWI, and the Scene Classification Layer (SCL) are
annotation aids or image-usability diagnostics only. None is accepted as
ground truth. In particular, the workflow never turns PIXC, a Phase 5A.2
candidate, an NDWI threshold, SCL class 6, optical segmentation, RiverSP,
SWORD, or machine learning into a manual reference interval.

This page documents the preparation workflow and expected products. It does
not assert that the public service was reached, that a suitable scene was
found, that the real Koshi packets were generated, or that any owner decision
or manual annotation has been made.

## Koshi inputs and immutable roles

The preparation command uses the local refined proposal manifest by default:

```text
benchmark_output/koshi_phase5a3b_pilot/geometry_review/
    refined_transects_proposed.geojson
```

Its scope is deliberately limited to:

| Transects | Pilot role |
|---|---|
| `T002F`–`T006F` | primary benchmark candidates |
| `T007F` | harder/challenge morphology case |
| `T001F` | separate AOI-edge/failure-control case |

`T008`–`T010` are excluded. The F lines retain the scientist-specified
130.0-degree cross-section azimuth, endpoint order, station direction,
corridor, and coordinates. Imagery preparation must not edit them. Their
status remains `PROPOSED / REQUIRES SCIENTIST REVIEW`; downloading a clear
image does not approve a geometry.

The associated SWOT observation is 2024-01-14, cycle 9, pass 286, using the
two cache-only Koshi granules declared in
[`../examples/koshi_phase5a3b_pilot_config.json`](../examples/koshi_phase5a3b_pilot_config.json):

```text
data/koshi_phase1/
    SWOT_L2_HR_PIXC_009_286_107L_20240114T081432_20240114T081443_PGD0_01.nc
    SWOT_L2_HR_PIXC_009_286_108L_20240114T081442_20240114T081453_PGD0_01.nc
```

The program reads `time_coverage_start` and `time_coverage_end` from those
NetCDF files. Filenames and the date alone are not substituted for measured
time metadata.

## Source and optional dependencies

The implemented provider adapter uses the public
[Element84 Earth Search v1 STAC API](https://earth-search.aws.element84.com/v1)
and its authoritative
[`sentinel-2-l2a` collection](https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a).
The collection exposes Sentinel-2 Level-2A assets as cloud-optimized GeoTIFFs,
including 10 m `red`, `green`, `blue`, and `nir` assets and the categorical
20 m `scl` asset.

Provider-specific search and asset naming are isolated in an Earth Search
adapter. The scientific preparation functions consume a provider protocol;
they do not encode Earth Search into manual-width or candidate-inference
logic. Another public STAC provider can therefore be added without changing
the Phase 5 contracts.

Core PIXC use does not require the raster/network extras. Install them only
for this workflow:

```bash
python -m pip install -e ".[reference-imagery]"
```

The imagery code imports raster support only when it is used and reports an
actionable error when an optional dependency is absent. CI uses mock STAC
responses and tiny local rasters; it does not require a live Sentinel service.

## Progressive search and candidate selection

The search is centered on the exact SWOT reference instant and widens in
explicit stages:

1. query ±7 days;
2. if no sufficiently usable local coverage is available, query ±14 days;
3. only if it is still needed, query ±30 days; and
4. stop at 30 days rather than silently widening farther.

The preparation manifest records every attempted window and the smallest
window that produced each deduplicated STAC item. The driver uses two explicit
operational guards when deciding whether it needs to widen the search:

- `valid_fraction >= 0.90`; and
- `obscured_diagnostic_fraction <= 0.25`.

Both values can be overridden on the command line and are recorded in the
manifest. They are conservative preparation/coverage guards, not scientific
good/bad labels, optical water thresholds, or validation criteria. Scenes that
do not meet them remain identified as such for review; when no scene meets the
guards, the workflow widens the search and may retain the available candidates
rather than silently fabricate coverage. Scene-level
`eo:cloud_cover` and related provider metadata are retained, but they are not
used alone to declare a scene suitable. Suitability requires local SCL
diagnostics over the particular transect review region and remains visible for
human review; it is not a wet/dry scientific classifier.

For each transect the transparent shortlist contains at most three distinct
acquisitions, when available:

1. the nearest locally usable acquisition before SWOT;
2. the nearest locally usable acquisition after SWOT; and
3. a different acquisition selected for lower local obscuration, with time
   distance and stable candidate identity used as deterministic tie-breaks.

The candidate CSV retains acquisition identity, a primary-review-image flag,
temporal offset, search window, scene metadata, local diagnostics, and chip
provenance. The shortlist rule above is deterministic even though the retained
rows are not relabeled as scientific ranks. The primary image is a review
choice, not automatic ground truth.
If pre- and post-SWOT boundaries disagree, the analyst must select one image,
flag temporal uncertainty, and describe the mismatch. Boundaries are never
averaged or interpolated in time, and the exact image used is retained with
the annotation.

### Common acquisition and spatial tiles

One locally usable acquisition is preferred for `T002F`–`T007F` so that the
core pilot is viewed under consistent river conditions. That preference must
not force cloudy, shadowed, missing, or otherwise unusable imagery. Separate
per-transect dates are permitted and are recorded independently.

Adjacent MGRS tiles from the same Sentinel datatake may be mosaicked to cover
one review region. A same-datatake mosaic:

- groups by `s2:datatake_id` when available, with platform plus acquisition
  minute retained as a documented fallback key;
- uses one acquisition and one spectral asset at a time;
- requires one known CRS and consistent scale/offset metadata; the B02/B03/
  B04/B08 products must also share the same native 10 m grid;
- retains every contributing STAC item, MGRS tile, asset identifier, and chip
  hash; and
- is a spatial assembly of one observation, not a temporal composite.

Tiles from different datatakes or dates are never mosaicked and presented as
one observation.

## Exact temporal provenance

The combined SWOT coverage begins at the earliest NetCDF
`time_coverage_start` and ends at the latest `time_coverage_end`. The declared
comparison instant is the midpoint of that combined interval. The manifest
records the two endpoints, the midpoint rule, every source filename, and the
per-file coverage values so that the choice can be reconstructed rather than
mistaken for an instrument timestamp copied from a filename.

For every Sentinel candidate, the workflow stores:

- acquisition datetime in UTC;
- the SWOT coverage and reference datetime in UTC;
- signed offset in hours, defined as **Sentinel minus SWOT**;
- absolute offset in hours; and
- one descriptive label: `SAME_DAY`, `WITHIN_3_DAYS`, `WITHIN_7_DAYS`,
  `WITHIN_14_DAYS`, or `WITHIN_30_DAYS`.

Those labels communicate timing; they are not validation thresholds. A large
offset remains visible in tables and review panels.

## Review windows and cache

Each review region contains the complete finite F transect plus a metric
context margin. The default is 1,500 m, within the requested approximate
1–2 km context. The region is constructed in a local metric azimuthal
equidistant CRS and saved with its WGS84 polygon/bounds, construction CRS, and
exact margin. It is not clipped to the SWOT AOI.

Only intersecting windows of the Earth Search cloud-optimized assets are read.
The workflow does not download a full Sentinel SAFE product or silently retain
a whole scene. Georeferenced chip data and sidecar provenance are cached under:

```text
benchmark_cache/sentinel2/T001F/
benchmark_cache/sentinel2/T002F/
...
benchmark_cache/sentinel2/T007F/
```

Every cached chip or same-datatake mosaic records bounds, CRS/EPSG, shape,
resolution, transform, band/asset identity, scale, offset, nodata value,
provider, collection, STAC item IDs, acquisition datetime, local path, cache
status, and SHA-256. Stable item and asset identifiers are saved without
authentication query strings; an expiring signed URL is never the only
provenance. Existing cache entries are accepted only when their request
identity and hash agree.

`benchmark_cache/`, imagery-review output, large GeoTIFFs, and temporary
download/STAC files are local artifacts and must remain ignored by Git.

## Radiometry, RGB, and continuous NDWI

Raw optical digital numbers are converted according to each STAC asset's
`raster:bands` metadata:

```text
scaled value = raw value × scale + offset
```

Nodata becomes `NaN`. The Earth Search asset metadata returned for this pilot
advertises scale `0.0001` and offset `-0.1` for the 10 m reflectance assets,
but the program reads and records the supplied metadata rather than silently
assuming those constants. A same-acquisition mosaic rejects
inconsistent scale/offset metadata.

True color uses scaled B04/B03/B02 as red/green/blue. Review PNGs use a
deterministic per-band 2nd–98th percentile stretch and gamma 1.0; those display
settings do not alter cached reflectance or create classifications.

NDWI uses scaled native 10 m B03 and B08 on a common grid:

```text
NDWI = (B03 - B08) / (B03 + B08)
```

The result is a continuous floating-point visualization. Nodata and a zero
denominator remain `NaN`. The API intentionally has no threshold parameter and
returns no optical mask, bank, or interval. NDWI color is therefore never
manual or automatic truth.

## Local SCL diagnostics

The Level-2A SCL is categorical and remains at its native 20 m resolution for
diagnostics. If grid alignment is required for display, it must use
nearest-neighbor sampling; categorical codes must never be interpolated. The
official values are described in the
[Sentinel-2 Product Specification](https://sentinels.copernicus.eu/documents/d/sentinel/s2-pdgs-cs-di-psd-v15-0)
and the
[Copernicus Sentinel-2 L2A documentation](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html).

The preparation metadata records all class counts and fractions and uses this
explicit diagnostic grouping:

| Code | Recorded class | Diagnostic treatment |
|---:|---|---|
| 0 | no data | no-data fraction; excluded from valid count |
| 1 | saturated or defective | obscured/invalid diagnostic |
| 2 | dark-area pixels; renamed cast shadows in newer processing baselines | obscured diagnostic; baseline-dependent, so inspect |
| 3 | cloud shadows | obscured diagnostic |
| 4 | vegetation | non-obscured diagnostic |
| 5 | bare soil / not vegetated | non-obscured diagnostic |
| 6 | water | non-obscured usability diagnostic only; **never truth** |
| 7 | low-probability cloud or unclassified | obscured diagnostic because interpretation is ambiguous |
| 8 | medium-probability cloud | cloud and obscured diagnostic |
| 9 | high-probability cloud | cloud and obscured diagnostic |
| 10 | thin cirrus | cirrus and obscured diagnostic |
| 11 | snow or ice | snow/ice and obscured diagnostic |

ESA changed the name of code 2 in newer processing baselines, so the raw code,
acquisition identity, and item provenance are more durable than one universal
label. Phase 5A.3c retains the raw counts and does not silently reinterpret an
older scene. The normalized scene model records both processing level and the
provider's `s2:processing_baseline` field when supplied. The code conservatively
includes class 2 in `obscured_diagnostic_fraction`; it does not use the class to
mask imagery or treat it as wet/dry truth.

All fractions use as their denominator the SCL pixel centers inside the exact
metric-buffer review polygon. The larger rectangular COG read window is never
silently included in those local diagnostics.
`valid_fraction` means a known code other than 0; it does not mean cloud-free.
Separate fields report cloud (8+9), cloud shadow (3), cirrus (10), snow/ice
(11), no data (0), saturated/defective (1), the listed non-obscured group, the
listed obscured group, and unexpected values. These are image-usability
descriptions only. No SCL code is converted to a wet interval.

## Prepare the Koshi review package

After installing the optional dependencies, run from the repository root:

```bash
python examples/koshi_phase5a3c_reference_imagery.py
```

The command defaults to the local F proposal manifest, the two cache-only
Koshi PIXC filenames above, a 1,500 m context margin, the local cache at
`benchmark_cache/sentinel2/`, and:

```text
benchmark_output/koshi_phase5a3b_pilot/reference_imagery_review/
```

The exact optional overrides are:

```text
--transects PATH                    F proposal GeoJSON
--pixc-directory PATH               directory containing the two fixed Koshi files
--cache-directory PATH              ignored Sentinel chip cache
--output-directory PATH             imagery-review output
--context-margin-m METRES           1000–2000; default 1500
--minimum-valid-fraction FRACTION    default 0.90 preparation guard
--maximum-obscured-fraction FRACTION default 0.25 preparation guard
--stac-endpoint URL                 default Earth Search v1
--stac-collection ID                default sentinel-2-l2a
--search-only                       save STAC search metadata; read no COG chips
```

`--pixc-directory` changes the parent directory, not the two required
filenames. `--search-only` writes `stac_search_record.json` and stops before
chips, review packets, or annotations. All normal overrides are recorded in
the preparation outputs and must not silently mutate the source F manifest.

This is a live network preparation command, not a normal CI command. A public
STAC, COG, certificate, or network failure must be reported as such; it is not
permission to fabricate a scene, timestamp, chip, or packet.

The review directory is expected to contain:

```text
koshi_reference_imagery_overview.png
koshi_T001F_T007F_imagery_review_board.png
sentinel_scene_summary.csv
geometry_imagery_owner_review.csv
reference_imagery_preparation_manifest.json
stac_search_record.json
T001F/
    sentinel_rgb_primary.png
    sentinel_rgb_with_transect.png
    sentinel_ndwi_primary.png
    sentinel_candidates.csv
    reference_imagery_metadata.json
    imagery_review_panel.png
    candidates/<candidate-id>/
        sentinel_rgb.png
        sentinel_rgb_with_transect.png
        sentinel_ndwi.png
...
T007F/
    ...
```

The metadata and preparation manifest carry machine-readable query windows,
scene/item/asset identities, local diagnostics, temporal provenance, primary
selection, chip paths and hashes, review-region construction, and any honest
missing-imagery/failure state. The small PNGs and CSV/JSON files are review
products; the georeferenced imagery remains in the ignored cache.

`geometry_imagery_owner_review.csv` contains exactly these columns:

```text
transect_id,geometry_decision,imagery_decision,owner_notes
```

Transect IDs may be pre-populated, but all decisions and notes start blank.
Allowed geometry decisions are `APPROVE`, `MODIFY`, and `HOLD`; the software
does not choose one.

## T001F failure-control handling

T001F receives the same imagery products but is visibly labeled
`AOI-EDGE / FAILURE-CONTROL CASE`. Its full review window may show water beyond
the SWOT AOI. Missing PIXC outside that AOI is unknown, not dry. T001F is not
promoted automatically into the primary calibration set, and a discrepancy at
the AOI edge is useful diagnostic evidence rather than a reason to move the
transect or manufacture a bank.

## Approval and blind-reference annotation

Imagery preparation leaves every F geometry proposed. The owner should:

1. inspect the overview, review board, per-transect RGB/NDWI, acquisition time,
   local SCL diagnostics, and temporal offset;
2. enter geometry and imagery decisions in the owner-review CSV;
3. retain the proposal manifest unchanged and create a traceable reviewed
   manifest for approved geometries;
4. if a geometry is modified, rerun imagery preparation and regenerate its
   annotation template so the image, line, and station frame agree; and
5. generate a Phase 5A.3b annotation template from the approved manifest
   before starting annotation.

The annotation helper refuses a proposed geometry, a missing georeferenced
image, a geometry mismatch, a stale station-frame hash, a stale approval
template, or a changed preview. Changing an approved geometry after annotation
requires a new geometry/annotation identity rather than transforming saved
stations.

After approval and packet regeneration, start the image-first tool for one
transect, for example:

```bash
python examples/annotate_from_reference_imagery.py \
  --transect-manifest path/to/owner_approved_transects.geojson \
  --reference-packet benchmark_output/koshi_phase5a3b_pilot/reference_imagery_review/T002F/reference_imagery_metadata.json \
  --transect-id T002F \
  --analyst-id analyst_name \
  --annotation-id T002F_analyst_name
```

The default blind-reference view shows only the independent georeferenced RGB,
the fixed approved transect, its 0→L direction, station ticks, and the SWOT AOI
outline. It neither imports nor displays Phase 5A.2 candidate output. PIXC is
hidden by default and appears only after the analyst deliberately presses the
reveal control; it is labeled non-reference context.

Each analyst click is projected geometrically to the nearest point on the
finite transect in the existing Phase 4 local station frame. Projection to the
line is the only snapping: image values, RGB/NDWI edges, SCL, PIXC pixels,
classifications, and candidate boundaries are not consulted. Clicks alternate
wet start and wet end and must remain in increasing station order. The
interface supports undo, reset, an explicit no-visible-wet declaration,
confidence, notes, distinct analyst/annotation IDs, and an optional explicit
candidate-image index when several Sentinel acquisitions are available.

Before saving, the analyst sees every station pair, interval width, total
wetted width, total internal dry gap, outer span, imagery identity/date,
temporal offset, and whether PIXC context was revealed. Those derived values
come from an in-memory call to the authoritative
`measure_explicit_wet_intervals(...)`; they are not editable primary input.
Saving requires a second explicit confirmation and delegates to the existing
benchmark-record machinery. An explicit empty declaration is distinct from a
pending annotation or missing imagery.

## Quality checks

Normal repository checks remain offline and deterministic:

```bash
ruff check .
ruff format --check .
pytest -m "not integration"
```

Any live Earth Search test must be marked `integration` and excluded from the
normal run.

## Limitations and stop condition

- Optical shorelines can differ from the SWOT-observed state because the
  acquisitions are not simultaneous and river boundaries can move quickly.
- Clouds, haze, cirrus, shadows, snow/ice, turbid water, wet sediment,
  vegetation, and georegistration uncertainty can obscure or displace an
  apparent boundary. SCL diagnostics do not resolve that ambiguity.
- A low scene-cloud percentage does not guarantee a usable local transect
  chip. Conversely, a scene with higher global cloud may have clear local
  coverage.
- NDWI is sensitive to surface and atmospheric conditions and remains only a
  continuous visualization; no threshold is endorsed.
- A common acquisition across `T002F`–`T007F` is preferred but may not exist.
  Per-transect dates are scientifically more honest than a cross-date mosaic.
- T001F is AOI-edge influenced; absent out-of-AOI PIXC cannot be interpreted as
  dry land.
- Earth Search is an external service. Catalog contents, network access, and
  hosted assets can change; saved item identities, query times, chip hashes,
  and preparation metadata are therefore essential.
- The owner-review CSV is an auditable human worksheet, not an automatic
  approval engine. The executable hard gate is an explicit `APPROVED` geometry
  in a matching reviewed manifest and annotation template.

Phase 5A.3c stops when the independent imagery/review package and annotation
tool are ready. Do not begin sensitivity validation until the scientist has
reviewed the imagery, explicitly approved the intended geometries, and created
manual Phase 5A.1 annotations.
