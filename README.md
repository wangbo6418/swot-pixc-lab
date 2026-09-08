# SWOT PIXC Lab

SWOT PIXC Lab is an early-stage scientific Python toolkit for finding and
retrieving NASA Surface Water and Ocean Topography (SWOT) Level 2 High Rate
Pixel Cloud (PIXC) granules. Its long-term purpose is to support reproducible,
pixel-level work on multiple-channel and anabranching rivers. The current
release implements **Phase 1 only**: AOI/date discovery, metadata inspection,
download/cache handling, and a verified local-file manifest.

## Why PIXC?

The [PO.DAAC PIXC Version D collection](https://podaac.jpl.nasa.gov/dataset/SWOT_L2_HR_PIXC_D)
contains the geolocated, irregular pixel cloud of water detections from which
higher-level hydrology products are derived. In contrast, the
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
- product version: `D`
- CMR collection concept ID: `C3233944986-POCLOUD`
- dataset DOI: [`10.5067/SWOT-PIXC-D`](https://doi.org/10.5067/SWOT-PIXC-D)

The concept ID selects Version D rather than asking CMR for an ambiguous
"latest" version. It does not freeze the CMR catalog: PO.DAAC can add granules
or revise metadata, so save the returned metadata table with each analysis.
The default will not silently move to a future product version. A different
verified collection can be requested with `collection_concept_id=...`.

Searches use CMR footprint intersection. They do not clip pixels to the AOI.
Date-only bounds are interpreted as inclusive UTC days. The accepted AOIs are:

- `(west, south, east, north)` in WGS 84 longitude/latitude;
- GeoJSON Polygon or MultiPolygon, including Feature wrappers; or
- a GeoDataFrame-like object with a declared CRS (install the `geo` extra).

GeoDataFrame-like AOIs are explicitly transformed to EPSG:4326. A bounding
box that crosses the antimeridian is split into two CMR searches. Polygon holes
are omitted from the CMR query and recorded as a conservative-search note;
antimeridian-crossing polygons must be supplied as an explicit MultiPolygon.
Exact spatial clipping belongs to Phase 2.

Authoritative product details, schema definitions, and processing notes are in
the [PIXC Product Description Document](https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-docs/web-misc/swot_mission_docs/pdd/D-56411_SWOT_Product_Description_L2_HR_PIXC_20250224a_RevC_clean_sig_final.pdf)
and the [Version D KaRIn release notes](https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-docs/web-misc/swot_mission_docs/SWOT_VersionD_KaRIn_Products_Release_Note_20250423b.pdf).

## Installation

Python 3.12 or newer is required by the current package configuration.

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
python -m pytest
```

## Five-minute Phase 1 example

Replace the example bounds with a small AOI of your own and start with a narrow
date interval. Discovery is normally anonymous; protected PO.DAAC files require
a free [NASA Earthdata Login](https://urs.earthdata.nasa.gov/users/new).

```python
from pathlib import Path

from swot_pixc_lab import PixcCollection

# Example syntax only: west, south, east, north in EPSG:4326.
aoi = (-91.20, 30.00, -91.10, 30.10)

collection = PixcCollection.search(
    aoi=aoi,
    start_date="2025-06-01",
    end_date="2025-06-07",
)

# Inspect before downloading: every match is a complete PIXC tile.
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

# Opt in only after checking the number and total size of the matches.
DOWNLOAD_FULL_TILES = False
cache = Path("data/pixc")
if DOWNLOAD_FULL_TILES and len(collection):
    collection.download(cache, verify="auto", persist_credentials=False)
    local = collection.open(cache, verify="auto")
    print(local.paths)
    local.table.to_csv(cache / "granule_manifest.csv", index=False)
```

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

The fuller, guarded walkthrough is
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

## Cache and local-open semantics

`download(destination)` treats the destination as a filename-based cache.
Existing files are reused only after validation. Missing files are downloaded
to a temporary staging directory and validated as a set. They are then
committed with no-clobber hard links; a commit failure rolls back links already
created, and an existing or concurrently created target is never overwritten.

The default `verify="auto"` verifies the CMR checksum when one is available and
supported, otherwise falling back to the CMR byte size. `verify="size"` avoids
the checksum scan for faster but weaker repeat access, while
`verify="checksum"` requires checksum metadata. `verify="none"` still requires
a regular, non-empty file. An invalid cache entry raises an actionable error
instead of being deleted or replaced automatically.

In Phase 1, `collection.open(cache_dir)` is intentionally local-only. It makes
no network request, does not download anything, and does **not** parse NetCDF
variables. It returns a `LocalPixcCollection`: a verified manifest with
`paths` and a provenance `table`. Scientific PIXC opening, AOI clipping, and
tile mosaicking are Phase 2 work.

## Data provenance

Keep the manifest alongside every downstream analysis. It records the CMR
granule, native, collection, and metadata-revision identifiers; observation,
revision, and production times; cycle/pass/tile identifiers; product, PGE, and
UMM-G specification versions when supplied; source URLs; bounding boxes;
checksums; byte sizes; metadata warnings; and resolved local paths. Also save
`collection.provenance.as_dict()`, which contains the exact normalized temporal
and spatial filters used for discovery. Local paths are machine-specific; the
NASA identifiers and source granule filenames are the durable links back to
the archive.

When publishing work, cite the dataset using the guidance on the
[PO.DAAC collection page](https://podaac.jpl.nasa.gov/dataset/SWOT_L2_HR_PIXC_D),
including the data-access date.

## QC philosophy

No pixel-quality filtering is implemented in Phase 1 because the package does
not yet parse PIXC variables. Future QC will be a separate, explicit layer:
raw values and official flags will be preserved, undocumented numeric values
will never be classified by guesswork, every removal rule will be recorded,
and before/removed/retained counts will be returned. Until that layer exists,
this package makes no claim that downloaded pixels are analysis-ready.

## Current limitations

- **The project owner has not yet supplied the real multiple-channel-river AOI
  required for the Section 10 scientific demonstration.** The notebook
  therefore contains only clearly labeled placeholder AOI syntax and is not a
  validated river-science case study.
- CMR intersection can return tiles that contain no pixels inside the precise
  AOI; full NetCDF tiles are downloaded because pixel subsetting is not yet
  implemented.
- Exact duplicate hits from multiple AOI-component searches are removed by CMR
  granule identity, but distinct reprocessed instances, CRIDs, or product
  counters are retained. Choosing among those scientifically distinct records
  requires an explicit, validated policy and is not automated in Phase 1.
- Phase 1 does not parse PIXC NetCDF groups or variables, subset or mosaic
  tiles, inspect or apply quality flags, visualize pixels, or export a
  research-ready pixel subset.
- No multi-channel segmentation, width, WSE comparison, or morphological
  interpretation is implemented.
- The default collection ID and Version D product assumptions must be reviewed
  when PO.DAAC releases a successor or revises collection guidance.
- Network availability, Earthdata authorization, real granule metadata, file
  sizes, and checksums are external conditions. Unit tests mock these services;
  the optional live CMR test is not a substitute for scientist validation.

## Roadmap

1. **Phase 1 (current):** CMR discovery, metadata, download/cache, and verified
   local manifests.
2. **Phase 2:** inspect real Version D files, define a memory-conscious internal
   representation, parse documented variables, clip to AOIs, and mosaic tiles
   without silently averaging overlaps.
3. **Phase 3:** preserve raw pixels and add documented, auditable QC with
   removal summaries.
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
