# Phase 5A.2 fixed station-bin candidate inference

**Status: EXPERIMENTAL / CANDIDATE**

> **Candidate interval edges are not validated physical river banks.**
>
> **An unsampled bin is not treated as confirmed dry land.**

Phase 5A.2 supplies the first transparent baseline for turning the discrete
classification evidence in a Phase 4 `TransectSample` into candidate
wet-support intervals. It is intended for inspectable experiments and
sensitivity analysis. It is not a validated bank detector, channel-width
product, branch extractor, or research-grade river measurement.

The workflow is deliberately narrow:

```text
PIXC observation or QCResult
        |
        v
Phase 4 user-defined TransectSample
        |
        v
explicit classification set and station-bin parameters
        |
        v
three-state bin evidence
        |
        v
candidate wet-support intervals and recorded optional bridges
```

## Public API

All four scientific parameters are mandatory. The values below are only an
illustrative API example; they are not recommended defaults, validated
settings, or values tuned for the Koshi River.

```python
from swot_pixc_lab import (
    CandidateIntervalConfiguration,
    infer_candidate_wet_intervals,
    plot_candidate_wet_interval_inference,
    run_candidate_interval_sensitivity,
)

result = infer_candidate_wet_intervals(
    sample,
    extent_classes=(3, 4, 5),
    station_bin_width_m=25.0,
    min_extent_pixels_per_bin=2,
    max_bridge_gap_m=0.0,
)

print(result.bins_dataframe())
print(result.intervals_dataframe())
print(result.bridges_dataframe())
print(result.audit_summary())
plot_candidate_wet_interval_inference(result)
```

The caller must explicitly select:

- `extent_classes`: a non-empty, duplicate-free sequence of integer PIXC
  `classification` values;
- `station_bin_width_m`: the positive nominal station-bin length;
- `min_extent_pixels_per_bin`: the positive eligible-point count required for
  candidate wet support; and
- `max_bridge_gap_m`: a nonnegative physical gap tolerance, with `0.0`
  disabling all bridging.

The API provides no scientific defaults for these parameters. Class sets such
as `(4,)`, `(3, 4)`, `(3, 4, 5)`, and `(3, 4, 5, 6, 7)` are supported for
explicit sensitivity experiments, but Phase 5A.2 does not endorse any of
them. Interpret the numeric classes using the metadata and official product
documentation for the PIXC version being analyzed.

## Station domain and exact bin definition

The finite user-supplied transect defines the complete station domain

$$
0 \le s \le L,
$$

where `L = sample.transect_length_m` in the local metric CRS recorded by Phase
4 and $\Delta s=$ `station_bin_width_m`. The bins are

$$
B_k=[k\Delta s,\min((k+1)\Delta s,L)),
$$

except that the final bin includes $L$. Thus bins are left-closed and
right-open, the last bin is closed at the transect endpoint, and a point
exactly on an internal edge belongs to the bin on its right. Bin IDs are
one-based in result records.

Every positive final remainder is retained as a partial bin, however small.
All starts and ends are sliced from one stored edge array so adjacent bins share
the same floating-point edge value and form a contiguous partition. The initial
quotient estimate is corrected against those stored edge multiples in both
directions: round-down cannot hide a positive remainder, and round-up cannot
create a zero-length bin at the exact domain endpoint. The domain is never
inferred from the minimum or maximum sampled PIXC station.

Each `StationBinEvidence` records its edges and center, exact `bin_span_m`,
sampled and eligible counts, eligible fraction, classification and source
counts, state, and any bridge identifier. An empty `TransectSample` therefore
still produces bins across the full transect; all of them are `unsampled`.

## Extent eligibility and candidate-wet rule

For bin $B_k$, let $N_k$ be the number of selected sample points in the bin and
let

$$
E_k=\#\{p:s_p\in B_k,\;c_p\text{ is valid},\;c_p\in C\},
$$

where $C$ is the caller's `extent_classes` and $c_p$ is the preserved PIXC
`classification` value. Classification fill values identified by the source
metadata are invalid and cannot contribute to $E_k$.

A bin is candidate wet if and only if

$$
E_k \ge m,
$$

where $m=$ `min_extent_pixels_per_bin`. No additional heuristic changes that
decision. In particular, `height`, `water_frac`, `sig0`, quality values, and
pixel area are not used for bin status.

The method applies no second QC system. To test raw evidence, create the Phase
4 sample from the raw observation. To test a QC policy, create it from the
desired `QCResult`. The result records the sample's QC profile name, label,
and status, when present; it does not reinterpret them.

## Three evidence states

Every bin retains exactly one original evidence state:

| State | Exact condition | Scientific meaning |
|---|---|---|
| `candidate_wet` | $E_k \ge m$ | Meets the caller's classification-support threshold. |
| `sampled_noneligible` | $N_k>0$ and $E_k<m$ | PIXC points were sampled, but the explicit threshold was not met. This is not automatically confirmed dry land. |
| `unsampled` | $N_k=0$ | No selected PIXC point provides evidence in the bin. This is unknown, not confirmed dry land. |

`sampled_noneligible` includes both bins with zero eligible points and bins
with some eligible points below the threshold. The distinction between a
sampled-below-threshold gap and a completely unsampled gap is preserved in bin
records, bridge records, tables, and the diagnostic plot.

## Candidate runs and exact gap-bridging rule

Before bridging, each maximal contiguous sequence of `candidate_wet` bins is
a wet-support run. Let an internal gap between two consecutive runs span the
intervening non-wet bin edges, with physical length $g$. Bridging behaves as
follows:

- if `max_bridge_gap_m == 0.0`, no gap is bridged;
- if the tolerance is positive, an internal gap is bridged if and only if
  $g \le$ `max_bridge_gap_m`;
- no leading gap before the first wet-support run or trailing gap after the
  last run is bridged; and
- adjacent accepted bridges can connect several runs into one final candidate
  interval.

The `<=` comparison absorbs only a small binary floating-point roundoff bound
(eight machine ULPs at the compared length scale), so decimal constructions
such as three 0.1 m bins honor an explicit 0.3 m limit. It is not an additional
physical gap tolerance.

Bridging never changes a bin's original state and never converts unsampled or
sampled-noneligible evidence into observed water. It only connects candidate
runs for the inferred interval span. Each `GapBridgeRecord` retains the left
and right run/bin identifiers, gap bin and station bounds, gap width, number
of unsampled bins, number of sampled-noneligible bins, sampled and eligible
point counts, class/source counts, and the reason `caller-specified gap
bridging`. The gap is therefore reconstructable rather than silently erased.

## Candidate boundaries and boundary resolution

A `CandidateWetInterval` starts at the left edge of its first candidate-wet
bin and ends at the right edge of its last candidate-wet bin, including any
explicitly bridged internal run connections. It does not use the first or last
eligible point as a bank. It does not estimate a sub-bin edge.

`boundary_resolution_m` records `station_bin_width_m`. This is the controlling
discretization scale, not a bank-location accuracy or uncertainty estimate;
the final partial bin can have a smaller span. Reversing or repositioning the
user transect changes its station coordinate and can change the discretized
candidate edges.

`candidate_interval_id` is an ordered result-local identifier. It is not a
`branch_id`, persistent channel identity, or cross-date tracking key.

## Support and span quantities

The result intentionally separates observed classification support from spans
created by inference. For candidate interval $i$, let $\mathcal{C}_i$ be its
candidate-wet bins and $\mathcal{G}_i$ its accepted bridge gaps.

**Observed candidate wet support** is

$$
S_i=\sum_{B_k\in\mathcal{C}_i}|B_k|.
$$

It excludes every bridged gap, including an unsampled one. It appears as
`CandidateWetInterval.observed_candidate_wet_support_m`. The result-level
`individual_candidate_widths_m` contains these $S_i$ values, and
`observed_candidate_wet_support_m` and
`total_candidate_wetted_width_m` both report $\sum_i S_i$. Despite the word
"width" retained in the latter compatibility names, these are experimental
candidate-bin support lengths, not validated river widths.

**Inferred candidate interval span** is

$$
I_i=b_i-a_i=S_i+\sum_{G\in\mathcal{G}_i}|G|,
$$

where $a_i$ and $b_i$ are final bin edges. It appears as
`CandidateWetInterval.inferred_candidate_interval_span_m`; the result-level
`individual_inferred_candidate_interval_spans_m` and
`total_inferred_candidate_interval_span_m` report the individual and summed
spans. The shorter result-level `inferred_candidate_interval_span_m` is an alias
for that total; interval records use the same shorter name only for their own
single span.
`bridged_gap_m` and `total_bridged_gap_m` remain separate and auditable.

`outer_candidate_wetted_span_m` is the distance from the first candidate
interval's left bin edge to the last candidate interval's right bin edge. It
includes all intervening gaps, whether bridged or not, and is `None` when no
candidate interval exists. It must not be substituted for observed support.

## Immutable results and provenance

The separate Phase 5A.2 data model consists of:

- `StationBinEvidence` for complete-domain bin evidence;
- `GapBridgeRecord` for each caller-authorized connection;
- `CandidateWetInterval` for each final connected candidate span;
- `CandidateSourceSummary` for compact source-granule context;
- `CandidateIntervalInferenceResult` for the immutable audit snapshot; and
- `CandidateIntervalConfiguration` and
  `CandidateIntervalSensitivityResult` for explicit sensitivity runs.

The main result records method identifier, method version, experimental
status and warning; all four caller parameters; exact edge semantics and
boundary resolution; transect WKT, CRS, projected length, metric CRS,
projection center, and corridor half-width; upstream QC identity; selected,
classification, eligible, and source counts; aggregate tile counts; source
granule/tile/cycle/pass/CRID/product/CMR context; all bin states; candidate
intervals; all bridge records; and the numerical bin-edge and bridge-comparison
semantics.

Counts stored on a `CandidateWetInterval` cover every sampled point in its full
bridge-inclusive bin-edge span. Consequently, they retain evidence from
sampled-noneligible bridge bins; the separately named eligible count and
candidate-wet support length prevent that evidence from being mistaken for
observed wet support.

It does not copy the raw pixel arrays or retain the source
`TransectSample`. The original sample remains the pixel-level evidence, with
its `source_index` and `source_point_index` provenance. The compact result
omits local file paths for portability. `bins_dataframe()`,
`intervals_dataframe()`, and `bridges_dataframe()` return fresh pandas tables;
no GeoPandas dependency is introduced.

## Diagnostic visualization

`plot_candidate_wet_interval_inference(result, *, ax=None, title=None)` draws a
one-dimensional audit schematic without rerunning inference. It uses distinct
styles for candidate-wet, sampled-noneligible, and unsampled bins; overlays
accepted bridges with a separate hatch; and outlines inferred intervals at
station-bin edges. An unsampled bridged bin retains its unsampled appearance
and therefore is not made to look like observed water.

The title identifies the result as experimental, warns that bin edges are not
validated banks and that unsampled does not mean dry, and lists the class set,
bin width, minimum count, and bridge tolerance. Its annotation distinguishes
observed candidate wet-bin support, summed inferred interval span, outer span,
and bridged gap length. A custom `title` changes only the leading heading, not
the warning or parameter annotation.

## Unranked sensitivity analysis

`run_candidate_interval_sensitivity` evaluates an ordered collection of
explicit `CandidateIntervalConfiguration` objects or mappings:

```python
configurations = [
    CandidateIntervalConfiguration(
        extent_classes=(4,),
        station_bin_width_m=20.0,
        min_extent_pixels_per_bin=1,
        max_bridge_gap_m=0.0,
    ),
    {
        "extent_classes": (3, 4, 5),
        "station_bin_width_m": 30.0,
        "min_extent_pixels_per_bin": 2,
        "max_bridge_gap_m": 30.0,
    },
]

sensitivity = run_candidate_interval_sensitivity(
    sample,
    configurations=configurations,
)
comparison = sensitivity.dataframe()
```

These numbers are also illustrative, not recommended. Mapping inputs must
contain exactly the four configuration fields. The helper preserves caller
order and reports one scalar summary per configuration, including state,
interval, bridge, support, and span quantities. It does not rank, optimize,
select a best setting, or declare a winner. Avoid tuning and evaluating a
configuration against the same observation.

## Relationship to Phase 5A.1

Phase 5A.1 remains the authoritative manual benchmark: the analyst supplies
ordered interval boundaries and `measure_explicit_wet_intervals` records them
without inference. Phase 5A.2 instead generates experimental candidates from
classification support. The two phases use separate result types, method
identifiers, terminology, and audit records.

Do not pass Phase 5A.2 candidate edges to
`measure_explicit_wet_intervals()` in a way that represents them as
analyst-supplied boundaries. Phase 5A.2 does not modify the Phase 5A.1
definitions of explicit interval width, total wetted width, outer wetted span,
or explicit dry gaps. See [`phase5a_width_contract.md`](phase5a_width_contract.md).

## Limitations and excluded methods

This baseline depends on user choices for transect geometry, corridor width,
upstream QC view, class set, bin width, count threshold, and bridge tolerance.
PIXC classification error, irregular or missing sampling, geolocation error,
layover, dark water, corridor mixing, transect orientation, tile boundaries,
and bin alignment can all change the result. Pixel count is a sampling-support
measure; it is not water area or a probability. The method provides no bank
uncertainty model and no claim that `boundary_resolution_m` captures total
error.

As a computational safeguard, a configuration that would require more than
1,000,000 station bins is rejected with an instruction to choose a wider bin.
This ceiling is not a scientific parameter or recommended scale.

Phase 5A.2 does not implement point-extrema banks, `max(station_m) -
min(station_m)` width, hulls or alpha shapes, rasterization, morphology,
interpolation, clustering, optical masks, SWORD, centerline generation,
automatic branch graphs, persistent branch tracking, area-equivalent width,
WSE derivation or aggregation, machine learning, or probabilistic
segmentation. It is not an implementation of braidedSP or RiverObs; it is an
independent fixed-bin baseline for transparent sensitivity tests. No external
method source code is copied.

## Future Koshi validation

No Phase 5A.2 parameters were tuned against the January 14, 2024 Koshi
observation, no manual Koshi bank truth is asserted, and no class set, bin
width, minimum count, or gap tolerance is recommended from Koshi. Synthetic
tests establish deterministic software behavior, not scientific validity.

Future scientific validation should keep the Phase 5A.1 analyst benchmark
separate, use independently defined truth or imagery, review multiple rivers
and observations, and quantify sensitivity to every explicit parameter,
transect placement/orientation, corridor width, upstream QC, classification
choice, gaps, and bin alignment. Until that work is complete, all outputs
remain candidate/experimental.
