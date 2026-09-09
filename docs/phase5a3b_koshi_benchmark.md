# Phase 5A.3b Koshi pilot benchmark workflow

## Scope and scientific status

Phase 5A.3b coordinates the existing Phase 1–5A.3a APIs so that a real,
independently annotated validation experiment can be prepared and repeated. It
does not add a bank detector, infer branch identity, calculate corrected WSE,
or declare a preferred Phase 5A.2 configuration.

The pilot workflow is:

```text
real local PIXC files
        ↓
exact AOI opening and one explicit QC result
        ↓
approved transects, or clearly labeled review proposals
        ↓
annotation packets
        ↓
INDEPENDENT HUMAN INTERPRETATION
        ↓
Phase 5A.1 explicit wet intervals
        ↓
ordered Phase 5A.2 sensitivity runs
        ↓
Phase 5A.3a interval-set validation
        ↓
individual tables + aggregate descriptions + figures + run manifest
```

The human-annotation step is a hard scientific gate. Classification values,
Phase 5A.2 candidate intervals, NDWI or another optical mask, RiverSP, SWORD,
hulls, and raster masks are never converted into reference wet intervals. If
no completed annotation exists, the real validation stage stops with:

> Annotation packets created. Manual reference annotations are required before
> quantitative validation can run.

That stop is expected, not a failed workaround.

## First pilot observation

The example configuration describes the project-owner observation:

| Field | Value |
|---|---|
| AOI, EPSG:4326 | `(86.87, 26.49, 87.20, 26.90)` |
| Date | `2024-01-14` |
| Cycle / pass | `9 / 286` |
| Tiles | `107L`, `108L` |
| Benchmark subset | `pilot` |

The cache-only files are expected at `data/koshi_phase1/`. The loader retains
both granules in provenance even though the established exact-AOI regression
found zero retained pixels in 107L and useful Koshi pixels in 108L. A CMR tile
footprint intersecting an AOI is not evidence that the file contains an
exact-AOI pixel.

The example never searches for or downloads missing PIXC data. At the time this
workflow was prepared, both files were available locally, but there was no
owner-approved Koshi transect manifest, independent reference image, or real
manual annotation. The existing Phase 4 Koshi PNG is PIXC-derived and is not
independent reference imagery.

## Configuration

[`../examples/koshi_phase5a3b_pilot_config.json`](../examples/koshi_phase5a3b_pilot_config.json)
is a human-readable, versioned example. Relative paths are resolved from the
configuration file's directory. It explicitly records:

- observation identity, date, cycle, pass, tiles, exact AOI, and filenames;
- the local-data directory;
- the upstream QC profile and transect-corridor half-width;
- the transect manifest, annotation directory, and output directory;
- reference-imagery metadata, currently marked `independent imagery pending`;
- the one separately labeled Phase 5A.2 configuration used in review packets;
- every axis of the pilot sensitivity grid; and
- every deterministic transect-proposal parameter.

Load it with `load_benchmark_config(path)`. Unknown, missing, malformed, or
hidden fields are rejected rather than supplied as scientific defaults.

### Exploratory sensitivity grid

The example contains these caller-ordered axes:

- extent classes: `(4,)`, `(3, 4)`, `(3, 4, 5)`, `(3, 4, 5, 6, 7)`;
- station-bin widths: `10`, `25`, `50`, and `100 m`;
- minimum eligible pixels per bin: `1`, `2`, and `3`; and
- maximum bridge gaps: `0`, `25`, `50`, and `100 m`.

Their Cartesian product contains `4 × 4 × 3 × 4 = 192` configurations. These
are exploratory pilot values only. They are not package defaults, optimal
parameters, or a recommended final configuration. Configuration and result
order follows the JSON axis order; the workflow does not rank or automatically
select configurations.

## Running the cache-only pilot builder

From the repository root:

```bash
python examples/koshi_phase5a3b_pilot.py
```

An alternative config path may be supplied with `--config`. The script:

1. validates the configuration with `load_benchmark_config`;
2. refuses to download any missing local granule;
3. opens both local files once with the existing `open_pixc` API;
4. applies the configured QC profile once with `apply_qc`;
5. loads the explicit manifest with `load_transect_manifest`, or, only when the
   file is absent and proposals are enabled, calls
   `propose_candidate_transects` and `write_transect_manifest`;
6. calls `generate_annotation_packets` while reusing that QC result;
7. refuses to validate any annotation whose transect remains
   `PROPOSED / REQUIRES SCIENTIST REVIEW`;
8. loads completed human records with `load_manual_annotations`; and
9. calls `run_benchmark` only after both approval and annotation gates pass.

Missing local data, an unreviewed manifest, and missing annotations each stop
with an actionable message. Generated packet collections, tables, plots, and
large imagery belong in ignored working/output directories, not Git.
Packet preparation writes `preparation_manifest.json` with the code/data/
configuration identity and an explicit zero-evaluation status, so the expected
human-gated stop remains auditable.

## Transect manifest and proposal gate

`load_transect_manifest` and `write_transect_manifest` use an ordered GeoJSON
FeatureCollection in EPSG:4326. Every feature retains:

- `transect_id`;
- the explicit LineString geometry and station direction;
- `corridor_half_width_m`;
- `review_status`;
- optional site/reach and morphology labels;
- selection notes; and
- proposal diagnostics when applicable.

The deterministic proposal helper exists only to reduce manual placement work.
The example requests 10 latitude-stratified, east–west-oriented candidate
lines. Neither that count nor orientation is asserted to be scientifically
optimal. Each generated feature is labeled exactly:

```text
PROPOSED / REQUIRES SCIENTIST REVIEW
```

Proposals may use coarse PIXC evidence to record density, preliminary support,
unsampled fraction, classification diversity, and corridor position. These are
selection diagnostics, not morphology or physical branch truth. A scientist
must inspect the spatial coverage, imagery, orientation, corridor width,
junction proximity, and diversity; edit or replace geometries if needed; then
change the status to `APPROVED`. Proposed transects are never promoted
automatically. Do not alter an approved geometry after annotation; create a
new transect/annotation identity instead.

## Annotation packets

`generate_annotation_packets` creates one review directory per manifest
transect. A packet contains, at minimum:

```text
T001/
    planview_pixc.png
    transect_classification.png
    candidate_diagnostic.png
    station_evidence.csv
    annotation_template.json
    metadata.json
```

The plan view shows the explicit line, sampling corridor, and classification
context. The classification view covers the complete finite station domain, so
blank regions visibly lack plotted classification evidence. Explicit
three-state `candidate_wet` / `sampled_noneligible` / `unsampled` bin labels
appear only in the separate Phase 5A.2 diagnostic and `station_evidence.csv`.
That candidate plot carries an experimental/non-truth warning so it cannot
visually masquerade as the reference. The blank template contains no inferred
wet boundaries or derived width.

Reference-imagery metadata accompanies every packet:

- source/provider;
- acquisition date and temporal offset from SWOT;
- cloud/quality and registration notes;
- local path or source identifier; and
- analyst notes.

Missing imagery is allowed and is visibly marked pending. The software does
not retrieve imagery in CI and does not turn an imagery water mask into truth.
The file format deliberately permits a pending record; it does not invent a
source. Before an annotation is represented as independently image-supported,
the owner must accept the source, timing, cloud/quality, and co-registration.
If an analyst proceeds using another independent basis, the pending imagery
status and that basis must remain explicit in the notes and limitations.

## Manual annotation helper

Run the lightweight helper for one reviewed packet:

```bash
python examples/annotate_manual_intervals.py \
  --config examples/koshi_phase5a3b_pilot_config.json \
  --transect-id T001 \
  --analyst-id analyst_name \
  --annotation-id T001_analyst_name
```

The Matplotlib interface records click pairs on the station axis. Every click
is an analyst choice: there is no automatic guessing, boundary snapping, mask
conversion, or candidate-to-reference copying. Use the on-screen buttons to
undo the last boundary, reset all boundaries, review the proposed intervals,
and confirm before saving. Notes and confidence are descriptive text, not a
numeric accuracy guarantee.

`save_manual_annotation` writes only primary analyst input and required
identity/frame metadata. Derived `width_m`, total wetted width, dry gap, and
outer span are neither accepted nor editable. During a benchmark run, the
software passes the saved boundary pairs to the existing
`measure_explicit_wet_intervals(...)` Phase 5A.1 contract, then creates the
existing Phase 5A.3 benchmark record. An explicitly confirmed empty list means
the analyst observed no wet interval; it is distinct from a pending or missing
annotation.

Use a unique annotation ID for each analyst or repeat interpretation, such as
`T003_bo` and `T003_analyst2`. Records remain independent. The workflow does not
average boundaries, choose an analyst, or create consensus truth.

## Automated validation and outputs

After approved transects and completed annotations exist, `run_benchmark`
reuses each cached `TransectSample`. Phase 5A.2 is evaluated once per transect
and configuration where possible, then each independent annotation is
validated with the existing Phase 5A.3a interval-set evaluator. The tidy output
contains one row per:

```text
manual annotation × sensitivity configuration
```

Each row retains benchmark, transect, annotation, analyst, observation, QC,
subset, and all four configuration fields. Observed-support and
bridge-inclusive results stay in distinct namespaces, including IoU, F1,
precision, recall, FP/FN length, signed/absolute width error, boundary-distance
diagnostics, component-count difference, total bridged gap, bridge overlap with
manual wet and manual nonwet space, and bridge-induced changes in IoU and F1.
No scalar score replaces these diagnostics.

Aggregate tables retain individual rows and describe each configuration in
caller order. They report annotation/transect counts, valid-value counts,
means/medians, width bias and MAE, boundary and component errors, FP/FN extent,
and bridge benefit/harm summaries. Undefined metrics remain missing rather than
being converted to zero. Aggregation does not sort from best to worst.

The runner generates research-review figures for IoU, F1, signed and absolute
width errors, boundary distance, component-count difference, and bridge effect,
plus per-transect comparison plots. Figures show individual distributions
where practical and carry no winner label. The Markdown report may describe
that one configuration had a higher pilot median, but it must preserve the
pilot/non-independent context and must not recommend a configuration.

## Reproducibility manifest

Every completed benchmark run records a machine-readable manifest containing:

- package version, Git commit, and UTC run time;
- observation, granules, provenance, and AOI;
- configuration and QC profile;
- hashes for the transect manifest and every annotation file;
- the complete ordered sensitivity grid;
- evaluation count; and
- output filenames.

The reproducibility identity is:

```text
data + approved transects + independent annotations + config + code commit
```

The runner does not repeatedly open the full granules for every configuration,
does not mutate inputs, and does not use nondeterministic result ordering.

## Development and holdout

The example uses `benchmark_subset: "pilot"`. The supported labels are
`pilot`, `development`, `holdout`, and `unassigned`, and they are supplied by
the owner. The package neither randomly assigns nor manufactures a holdout. If
pilot or development annotations inform parameter selection, they cannot also
serve as independent final evaluation. A future final assessment needs
predeclared, previously unseen annotations.

## Required owner actions

Before real quantitative validation can run:

1. inspect the generated proposal manifest and approve, edit, or replace every
   intended transect;
2. select acceptable independent, near-date, co-registered imagery and fill in
   its provenance, acquisition time, temporal offset, cloud/quality, and
   registration notes;
3. use the annotation helper to create at least one confirmed, independent
   Phase 5A.1-compatible annotation per approved transect; and
4. decide any development/holdout allocation before looking at results used for
   parameter selection.

Until those actions occur, annotation-packet preparation may run, but real
sensitivity validation must not.
