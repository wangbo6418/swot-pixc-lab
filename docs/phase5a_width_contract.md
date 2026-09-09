# Phase 5A.1 explicit wet-interval width contract

**Status: EXPERIMENTAL / MANUAL BENCHMARK CONTRACT**

Phase 5A.1 records wet boundaries that a scientist explicitly supplies along
an existing Phase 4 `TransectSample`. It establishes a stable, auditable
measurement contract before any PIXC-to-bank inference is attempted.

> Wet interval boundaries are supplied by the analyst in Phase 5A.1. They are
> not automatically inferred from SWOT PIXC.

This contract is a manual benchmark. It is not a research-grade automatic SWOT
width product.

## Definitions

Let the ordered, disjoint wet set along transect station $s$ be

$$
\mathcal{W}=\bigcup_{i=1}^{n}[a_i,b_i],
\qquad 0\le a_i<b_i\le L,
$$

where $L$ is the finite length of the user transect in the Phase 4 local
azimuthal-equidistant (AEQD) metric CRS. Consecutive intervals must have a
strictly positive gap: $b_i<a_{i+1}$.

The implementation reports four distinct quantities:

1. **Individual wet interval width**

   $$w_i=b_i-a_i.$$

   `interval_id` is only an ordered local identifier. It is not a persistent
   physical branch identifier.

2. **Total wetted width**

   $$W_{\mathrm{wet}}=\sum_{i=1}^{n}w_i.$$

3. **Outer wetted span**

   $$W_{\mathrm{span}}=b_n-a_1.$$

   This includes all internal dry islands, bars, and gaps between the outermost
   supplied wet boundaries.

4. **Internal dry gap**

   $$g_i=a_{i+1}-b_i,
   \qquad G_{\mathrm{internal}}=\sum_{i=1}^{n-1}g_i.$$

For a non-empty valid result,

$$
W_{\mathrm{span}}=W_{\mathrm{wet}}+G_{\mathrm{internal}}.
$$

Blank transect margins before the first wet interval and after the last wet
interval are not internal dry gaps.

## Canonical three-interval example

```python
from swot_pixc_lab import measure_explicit_wet_intervals

result = measure_explicit_wet_intervals(
    sample,
    intervals=[
        (0.0, 120.0),
        (180.0, 260.0),
        (310.0, 350.0),
    ],
)
```

| Ordered record | Station range (m) | Width (m) |
|---|---:|---:|
| Wet interval 1 | 0–120 | 120 |
| Dry gap 1 | 120–180 | 60 |
| Wet interval 2 | 180–260 | 80 |
| Dry gap 2 | 260–310 | 50 |
| Wet interval 3 | 310–350 | 40 |

The result is:

- individual interval widths: 120, 80, and 40 m;
- `total_wetted_width_m = 240.0`;
- `outer_wetted_span_m = 350.0`;
- internal dry gaps: 60 and 50 m; and
- `total_internal_dry_gap_m = 110.0`.

Thus $240+110=350$ m exactly for this example.

## Input and validation contract

`measure_explicit_wet_intervals(sample, intervals)` takes a real Phase 4
`TransectSample` and an ordered iterable of two-value `(start_station_m,
end_station_m)` boundaries. Boundaries need not coincide with PIXC sample
stations. Their order is authoritative.

The function rejects malformed pairs, booleans, NaN, infinity, negative
stations, zero-width or reversed intervals, duplicates, unordered input,
overlap, and endpoints beyond the projected transect length. It never sorts,
merges, removes, fills, or relabels intervals.

Touching intervals such as `[0, 100]` and `[100, 200]` are rejected. Separate
wet components require an explicitly positive dry gap; otherwise the analyst
must supply one interval. Exact comparison is intentional: every positive gap,
however small, remains recorded because Phase 5A.1 has no hidden minimum-gap
threshold.

An empty interval iterable intentionally represents no supplied wet interval:

- `interval_count = 0`;
- `total_wetted_width_m = 0.0`;
- `total_internal_dry_gap_m = 0.0`; and
- `outer_wetted_span_m = None`.

## Results and auditability

The immutable core records are `WetInterval`, `DryGap`,
`WidthSourceSummary`, and `ExplicitIntervalWidthResult`. Overall scalars are
derived from the ordered interval/gap records. `intervals_dataframe()` and
`gaps_dataframe()` return fresh pandas tables, including stable empty-table
schemas.

The result does not retain the `TransectSample` or copy raw pixel arrays. It is
a compact, reference-free snapshot containing:

- geographic transect WKT and its `EPSG:4326` declaration;
- the recorded AEQD CRS WKT, projection center, and projected transect length;
- corridor half-width;
- QC profile name, label, and status when available;
- selected-pixel and classification counts;
- per-source counts, aggregate tile counts, and source-specific granule, tile,
  cycle, pass, CRID, NetCDF/collection version, and CMR context when available;
  and
- a fixed method identifier, experimental status, and analyst-supplied warning.

Per-source records are authoritative when several granules share a tile label
because aggregate tile counts can combine them. Local paths are omitted from
the compact result for portability. The original `TransectSample` remains the
pixel-level source and retains each `source_index`/`source_point_index` pair.

Use `plot_wet_interval_summary(result)` for a one-dimensional review schematic.
It draws only the supplied intervals and the dry gaps reconstructed between
them; it neither reads pixels nor changes measurements.

## What Phase 5A.1 does

- Records explicit analyst measurements on the existing Phase 4 station axis.
- Distinguishes interval widths, total wetted width, outer span, and dry gaps.
- Uses actual projected transect geometry—not selected PIXC point spread—for
  domain validation.
- Preserves compact QC and source provenance without mutating raw PIXC or the
  `TransectSample`.
- Provides deterministic synthetic tests and a scientist-reviewable schematic.

## What Phase 5A.1 does not do

- It does not infer banks from PIXC classifications or point gaps.
- It does not calculate width as `max(station_m) - min(station_m)`.
- It does not create a raster, hull, interpolation, cluster, morphology, or
  connected-component rule.
- It does not generate a centerline, segment branches, assign persistent branch
  identity, or track components across dates.
- It does not calculate area-equivalent width.
- It does not derive, correct, or aggregate WSE.
- It does not use or tune against the real Koshi files or external imagery.

Min/max PIXC point spread is not accepted as validated width because an
isolated false-water classification can expand the span, while missing or
irregular bank sampling can contract it. PIXC station extrema describe the
selected sample distribution, not defensible bank locations.

## Relationship to later phases

Phase 4 supplies the user-defined transect, local AEQD station coordinate,
sampling corridor, selected pixels, QC identity, and provenance. Phase 5A.1
adds only the analyst's measurement records.

Phase 5A.2 now provides a separate experimental fixed station-bin candidate
inference baseline for later comparison against this manual contract; it does
not alter or populate Phase 5A.1 analyst records. See
[`phase5a2_candidate_bank_inference.md`](phase5a2_candidate_bank_inference.md).
Future automatic branch topology remains a separate scientific layer and must
not reinterpret `interval_id` or `candidate_interval_id` as a persistent branch
identifier.
