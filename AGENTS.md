# AGENTS.md — SWOT PIXC Lab

## 1. Project mission

Build a research-grade, open-source toolkit for **SWOT L2 HR PIXC (pixel cloud) data**, with a particular focus on **multiple-channel / anabranching rivers**.

The project is not intended to be another general SWOT web portal. Its purpose is to make difficult PIXC workflows easy, reproducible, and scientifically defensible.

Primary use case:

> A river scientist selects an AOI and date range, discovers all relevant SWOT PIXC observations, loads and mosaics the relevant tiles, applies transparent quality control, visualizes pixel-level variables, and prepares the data for multiple-channel analysis.

The first goal is a reliable scientific Python toolkit. AI/agent functionality and a polished web application come later.

---

## 2. Core principles

1. **Scientific correctness before UI.**
2. **Never silently discard data.** All filtering and QC decisions must be recorded.
3. **Preserve original SWOT metadata, units, quality flags, cycle/pass/tile identifiers, and source granule names.**
4. **Do not invent or guess SWOT variable definitions.** Use official product documentation and inspect real files.
5. **Do not hard-code one river, one pass, or one file layout.**
6. **All computational functions must work without an LLM.**
7. **Keep the architecture modular so a future AI agent can call stable scientific tools.**
8. **Do not build another general-purpose SWOT portal.** Focus on the difficult PIXC workflow and complex-river analysis.
9. Prefer reproducible, testable code over one-off notebook logic.
10. Do not introduce a large framework when a small dependency will do.

---

## 3. Initial repository structure

Use this structure unless there is a compelling technical reason to change it:

```text
swot-pixc-lab/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── src/
│   └── swot_pixc_lab/
│       ├── __init__.py
│       ├── discovery.py
│       ├── download.py
│       ├── pixc.py
│       ├── subset.py
│       ├── mosaic.py
│       ├── qc.py
│       ├── export.py
│       └── visualization.py
├── notebooks/
│   └── 01_pixc_aoi_demo.ipynb
├── examples/
├── tests/
└── docs/
```

Do not create `agent.py` yet.

---

## 4. V1 scope

### Phase 1 — PIXC discovery and ingestion

Implement a clean workflow with the following conceptual API:

```python
from swot_pixc_lab import PixcCollection

collection = PixcCollection.search(
    aoi=aoi,
    start_date="2025-01-01",
    end_date="2025-12-31",
)

collection.download("./data")
local = collection.resolve_local()
```

The exact API may change if a better design is justified.

Required capabilities:

- Accept AOI as one of:
  - bounding box
  - GeoJSON Polygon/MultiPolygon
  - GeoDataFrame geometry
- Accept start/end date.
- Discover relevant SWOT L2 HR PIXC granules using an official NASA/Earthdata-compatible method.
- Return a table containing, when available:
  - observation datetime
  - cycle
  - pass
  - tile/granule
  - product version
  - source URL or identifier
  - local path after download
- Avoid duplicate granules.
- Support downloading or opening locally cached files.
- Authentication should use normal Earthdata mechanisms; never store credentials in source code.

### Phase 2 — PIXC parsing, AOI subset, and tile mosaic

Create a standard internal representation for PIXC observations.

Requirements:

- Read actual PIXC NetCDF schema from real data.
- Preserve relevant pixel-level variables and attributes.
- At minimum identify and support the fields needed for:
  - location
  - water-surface elevation/height
  - pixel classification
  - water fraction where available
  - pixel area where available
  - backscatter where available
  - quality information
  - cycle/pass/tile/source granule metadata
- Convert longitude handling consistently.
- Subset pixels spatially to the user AOI.
- Merge/mosaic multiple tiles that intersect the same AOI/observation.
- Handle tile boundaries explicitly.
- Do not silently average duplicate/overlapping pixels.
- Record provenance for every pixel or observation sufficiently to trace it to the source granule.

The preferred output should be compatible with Python geospatial workflows and efficient for large point clouds. Evaluate xarray, pandas/GeoPandas, Arrow/GeoParquet, or a combination. Do not force all raw PIXC data into GeoPandas if it creates unnecessary memory overhead.

### Phase 3 — transparent quality control

QC must be a separate layer from raw ingestion.

Provide:

```python
raw = pixc.raw
clean = pixc.apply_qc(...)
```

Requirements:

- Preserve raw data.
- Expose important SWOT quality flags.
- Make default filtering conservative.
- Every filtering rule must be explicit and documented.
- Return a QC summary such as:
  - pixels before QC
  - pixels removed
  - pixels retained
  - removal reason counts
- Never treat an undocumented numeric flag value as "bad" or "good."
- Add unit tests for all QC logic.

### Phase 4 — PIXC visualization

Create a simple scientific visualization layer before any full web application.

Support pixel visualization colored by available variables such as:

- height/WSE
- classification
- water fraction
- backscatter
- selected quality metrics
- cycle/pass

Requirements:

- Handle hundreds of thousands of points reasonably.
- Avoid plotting millions of points naively with Matplotlib if a more scalable option is appropriate.
- Keep plotting code separate from data-processing code.
- A notebook demo should allow a scientist to inspect one AOI across multiple SWOT observations.

---

## 5. Multiple-channel river functionality

This is the scientific differentiator, but implement it only after the basic PIXC pipeline is stable.

### Cross-section interface

Design an interface similar to:

```python
result = pixc.cross_section(
    transect=line,
    corridor_width=...,
)
```

A transect may intersect multiple wetted branches.

The result should eventually identify distinct wetted segments/branches and report information such as:

```text
Branch A
Branch B
Branch C
```

Potential outputs:

- branch-specific wetted width
- total wetted width
- branch width fraction
- branch-level WSE statistics
- number of valid PIXC pixels
- uncertainty/QC diagnostics

### Important scientific restriction

**Do NOT implement branch width as a naive min/max distance between PIXC points and present it as a validated scientific width.**

PIXC is an irregular pixel-cloud observation and cross-track sampling, classification, geolocation uncertainty, gaps, layover, dark water, and quality flags can affect apparent wetted extent.

Before finalizing a width algorithm:

1. Document the proposed mathematical method.
2. Validate it visually on several known multi-channel reaches.
3. Compare against independent imagery or existing validated measurements where possible.
4. Quantify sensitivity to:
   - transect corridor width
   - PIXC classification selection
   - quality filtering
   - pixel gaps
   - bank-edge definition
5. Add synthetic/unit tests.
6. Clearly label experimental algorithms as experimental.

Do not claim research-grade branch width until validation is complete.

---

## 6. Future capabilities — do not implement in V1

Keep the architecture compatible with these, but do not build them yet:

- automated branch/channel segmentation
- branch width-ratio time series
- branch WSE comparison
- longitudinal water-surface profiles
- anabranch dominance metrics
- bifurcation analysis
- avulsion/change detection
- SWORD/RiverSP integration
- Landsat/Sentinel integration
- DEM/FABDEM integration
- in-situ/gauge validation
- GeoLibre integration
- browser app
- natural-language agent
- manuscript generation

These are later phases.

---

## 7. Future agent architecture

When the scientific functions are stable, an AI layer may call tools such as:

```text
search_pixc()
download_pixc()
open_pixc()
subset_pixc()
mosaic_pixc()
apply_pixc_qc()
plot_pixc()
cross_section()
compare_observations()
```

The LLM should orchestrate and explain analyses. It should not replace the scientific algorithms.

---

## 8. Coding requirements

- Python 3.11+ unless a required dependency prevents this.
- Use `pyproject.toml`.
- Include type hints on public functions.
- Use docstrings for public API.
- Keep functions small and testable.
- Use logging rather than scattered `print()` statements.
- Do not commit secrets, tokens, Earthdata credentials, or local absolute paths.
- Do not hard-code Windows-specific paths.
- Add useful exceptions with actionable error messages.
- Add tests with `pytest`.
- Prefer small test fixtures over committing large PIXC granules.
- If real data are needed for integration testing, make those tests optional and clearly marked.
- Cache downloaded data where reasonable.
- Preserve units and CRS/coordinate metadata.
- Document any coordinate transformation.

---

## 9. Documentation requirements

README must explain:

1. What problem this project solves.
2. Why PIXC is different from standard RiverSP/Reach/Node workflows.
3. Why multiple-channel rivers need pixel-level analysis.
4. Installation.
5. A 5-minute example.
6. Data provenance.
7. QC philosophy.
8. Current limitations.
9. Roadmap.

Do not market unsupported functionality.

---

## 10. First demonstration

Build one reproducible demonstration using a real multiple-channel river AOI supplied by the project owner.

The demo should show:

```text
AOI
 ↓
find SWOT PIXC observations
 ↓
select one or more cycles/passes
 ↓
download/cache
 ↓
open PIXC
 ↓
subset to AOI
 ↓
merge relevant tiles
 ↓
inspect quality flags
 ↓
apply transparent QC
 ↓
visualize water pixels
 ↓
export research-ready subset
```

Export at least:

- a compact tabular/geospatial format suitable for downstream analysis
- a metadata/provenance summary
- one figure or interactive visualization

Do not implement automated scientific interpretation in the first demo.

---

## 11. Acceptance criteria for V1

V1 is successful when a new user can provide an AOI and date range and:

- discover valid PIXC observations
- retrieve the correct files
- load them without manually understanding SWOT filename conventions
- subset/mosaic relevant tiles
- inspect and apply documented QC
- visualize pixel-level observations
- export a research-ready subset
- trace results back to original SWOT granules

The workflow must be reproducible from a clean environment.

---

## 12. How Codex should work on this repository

Before changing code:

1. Read this entire file.
2. Inspect the current repository.
3. Reuse existing working code where appropriate.
4. Identify uncertainties in the official SWOT PIXC schema or APIs.
5. Check authoritative documentation rather than guessing.
6. Propose a short implementation plan.

Then work **one phase at a time**.

Do not attempt the entire roadmap in one run.

After each phase:

- run tests
- summarize files changed
- summarize design decisions
- state remaining uncertainties
- state what should be manually validated by a river/SWOT scientist
- stop before beginning the next major phase unless explicitly requested

---

## 13. First Codex task

Start with **Phase 1 only**.

### Task

Implement SWOT L2 HR PIXC discovery and ingestion scaffolding.

Deliverables:

1. Create or refine the package structure.
2. Add `pyproject.toml`.
3. Implement AOI/date-based PIXC granule discovery using an official NASA/Earthdata-compatible approach.
4. Implement download/cache handling.
5. Create a metadata table for discovered granules.
6. Add tests that do not require user credentials where possible.
7. Create `notebooks/01_pixc_aoi_demo.ipynb` showing the intended workflow.
8. Update README with installation and the Phase-1 example.
9. Do **not** implement multi-channel width algorithms yet.
10. Do **not** build a web UI or AI agent yet.

Before coding, provide a concise plan and identify any API/product-version decisions that need confirmation.

When Phase 1 is complete, run all tests and stop.
