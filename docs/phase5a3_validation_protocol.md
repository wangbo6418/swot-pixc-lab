# Phase 5A.3a interval-validation protocol

**Status: validation infrastructure; synthetic verification only**

> **Validation metrics measure agreement with an analyst-supplied reference.
> They do not by themselves prove that the manual reference is error-free.**

Phase 5A.3a compares the Phase 5A.2 experimental candidate intervals with the
Phase 5A.1 manual benchmark on the same finite transect station axis. It does
not infer a new bank, change either upstream method, select a preferred
candidate configuration, or establish validated Koshi performance.

The three roles remain separate:

- **Phase 5A.1 reference:** an analyst supplies ordered wet intervals to
  `measure_explicit_wet_intervals`.
- **Phase 5A.2 prediction:** `infer_candidate_wet_intervals` produces
  experimental station-bin support and optional bridge-inclusive intervals.
- **Phase 5A.3a validation:** continuous one-dimensional interval arithmetic
  describes their agreement and disagreement without changing either input.

## Public comparison API

The primary API keeps the analyst interpretation in the `reference` argument
and the experimental inference in the `candidate` argument:

```python
from swot_pixc_lab import (
    evaluate_candidate_against_explicit,
    plot_interval_validation,
)

validation = evaluate_candidate_against_explicit(reference, candidate)

print(validation.observed_support_metrics)
print(validation.bridge_inclusive_metrics)
print(validation.summary_dataframe())
plot_interval_validation(reference, candidate, validation=validation)
```

The immutable validation records are `IntervalSetMetrics`,
`BoundaryDistanceMetrics`, and `IntervalValidationResult`. A one-row summary
uses distinct `observed_support_` and `bridge_inclusive_` column prefixes; it
never collapses the two predictions into one score.

Before computing metrics, the implementation verifies that both inputs use
the same transect WKT, declared transect CRS, projected transect length, metric
CRS, projection center, and corridor half-width. It does not reproject,
rescale, reverse, or otherwise reconcile incompatible station frames. A
`ValidationError` identifies the incompatible field. `transect_wkt`,
`transect_crs`, and `metric_crs_wkt` require exact string equality. All numeric
values must be finite. For transect length, corridor half-width, and each
projection-center coordinate, recorded values $a$ and $b$ are compatible only
when

$$
|a-b|\leq 8\,\operatorname{ulp}(\max(|a|,|b|)).
$$

This rule is serialized as `FRAME_COMPARISON_SEMANTICS`. It absorbs only
binary floating-point roundoff in values already recorded from the same frame;
it is not a physical matching tolerance.

## The two prediction sets

Let the manual reference be the interval set

$$
M=\bigcup_i[a_i,b_i].
$$

Phase 5A.3a evaluates two separate Phase 5A.2 sets:

1. **Observed candidate support**, $C_{obs}$, is the union of bins whose
   original state is `candidate_wet`. Adjacent candidate-wet bins form an
   observed candidate run. Accepted bridge gaps are excluded.
2. **Bridge-inclusive candidate intervals**, $C_{inf}$, is the union of the
   final `candidate_intervals`. It includes every explicitly accepted bridge
   retained in the Phase 5A.2 result.

Both are bounded by the existing Phase 5A.2 station-bin edges. Validation does
not discretize, snap, or rasterize the manual reference onto the candidate bin
grid. Once the two candidate interval sets have been reconstructed, all set
operations use their continuous station coordinates.

## Continuous overlap metrics

For either candidate set $C$, `IntervalSetMetrics` reports:

$$
\begin{aligned}
L_M &= |M|, & L_C &= |C|,\\
TP &= |M\cap C|, & FP &= |C\setminus M|,\\
FN &= |M\setminus C|, & U &= |M\cup C|.
\end{aligned}
$$

Here vertical bars mean continuous length in metres, not pixel or raster-cell
count. The dimensionless ratios are

$$
\mathrm{precision}=\frac{TP}{TP+FP},\qquad
\mathrm{recall}=\frac{TP}{TP+FN},
$$

$$
F_1=\frac{2TP}{2TP+FP+FN},\qquad
\mathrm{IoU}=\frac{TP}{TP+FP+FN}.
$$

The corresponding fields retain explicit names such as
`true_positive_length_m`, `false_positive_length_m`, and
`false_negative_length_m`. The API does not expose a generic `accuracy` or
`score`, and none of these quantities is a physical-bank validation by itself.

### Empty-set semantics

Empty interval sets are recorded rather than converted to misleading perfect
scores:

| Reference | Prediction | Precision | Recall | F1 | IoU | `exact_empty_match` |
|---|---|---:|---:|---:|---:|---:|
| empty | empty | `None` | `None` | `None` | `None` | `True` |
| non-empty | empty | `None` | `0.0` | `0.0` | `0.0` | `False` |
| empty | non-empty | `0.0` | `None` | `0.0` | `0.0` | `False` |

The core immutable records use `None`, not NaN, for undefined ratios. Pandas
may display those values as missing in a DataFrame. `reference_empty` and
`prediction_empty` accompany `exact_empty_match`.

## Width, span, and component diagnostics

For each prediction, total-width error is

$$
e_W=L_C-L_M,
$$

with `signed_total_width_error_m = e_W`,
`absolute_total_width_error_m = |e_W|`, and
`relative_total_width_error = e_W/L_M` only when $L_M>0$. Relative error is
`None` for an empty reference. These diagnostics are not called accuracy.

Outer span is the distance from the first wet interval start to the last wet
interval end, including intervening non-wet space. When both sets are
non-empty, Phase 5A.3a reports reference and predicted outer spans plus signed
and absolute outer-span errors. If either set is empty, outer-span errors are
`None`. Outer span is not total wetted width.

Component diagnostics report the Phase 5A.1 interval count, the number of
maximal observed candidate runs, and the number of bridge-inclusive candidate
intervals. Differences are prediction count minus reference count. They are
component-count differences, not branch-count errors; no physical or
persistent branch identity is inferred.

## Nearest-boundary diagnostic

Manual boundaries are every start and end in the Phase 5A.1 interval record.
Candidate boundaries are separately every start and end of the observed
candidate runs and every start and end of the bridge-inclusive intervals.

`BoundaryDistanceMetrics` retains the ordered manual-to-candidate and
candidate-to-manual nearest-distance tuples and their mean, median, and maximum
where defined. Symmetric summaries pool the two directed nearest-boundary
collections. If either boundary set is empty, a direction that has no target
has no defined distances or summaries.

This is a **symmetric nearest-boundary diagnostic**. It is not a one-to-one
bank correspondence, assignment, component match, or branch-matching
algorithm. Several boundaries may have the same nearest counterpart. A low
mean can coexist with a large maximum, a misplaced component, or the wrong
number of intervals, so the full tuples and component diagnostics remain
important.

## Explicit bridge-impact accounting

Phase 5A.3a reports the Phase 5A.2 `total_bridged_gap_m` separately and compares
the two prediction sets through, when defined:

- observed-support and bridge-inclusive IoU;
- observed-support and bridge-inclusive F1;
- `delta_iou_due_to_bridging`, computed as bridge-inclusive minus observed;
  and
- `delta_f1_due_to_bridging`, computed in the same direction.

It also intersects the union of recorded bridge gaps with the manual wet set:

$$
L_{G\cap M}=|G\cap M|,\qquad L_{G\setminus M}=|G\setminus M|.
$$

These appear as `bridged_length_over_manual_wet_m` and
`bridged_length_over_manual_nonwet_m`. “Manual nonwet” means the complement of
the manual wet union within the common finite transect domain; it must not be
silently interpreted as verified dry terrain. Positive metric deltas are not
automatically called beneficial: a bridge can restore support over a manually
wet location while also filling an island, bar, or other manually nonwet
interval.

## Synthetic numerical regression

For the documented synthetic case

```text
manual reference:           [0, 100], [150, 250]
candidate observed support: [0, 100], [175, 250]
```

the reference length is 200 m and predicted length is 175 m. Continuous set
arithmetic gives TP = 175 m, FP = 0 m, FN = 25 m, and union = 200 m. Therefore
precision = 1.0, recall = IoU = 0.875,
$F_1=350/375=0.933333\ldots$, and signed total-width error = -25 m. This is a
software regression example, not an observed river or a performance claim.

## Ordered sensitivity validation

The Phase 5A.2 helper produces explicit candidate configurations. Phase 5A.3a
can evaluate them against one manual reference without ranking them:

```python
from swot_pixc_lab import evaluate_sensitivity_against_explicit

sensitivity_validation = evaluate_sensitivity_against_explicit(
    reference,
    sensitivity_results,
)
table = sensitivity_validation.dataframe()
```

`SensitivityValidationResult` retains caller order. Every row includes the
extent classes, station-bin width, minimum eligible-pixel count, bridge-gap
tolerance, and separate observed-support and bridge-inclusive validation
metrics. It does not sort by IoU, assign a rank, choose a best configuration,
or optimize parameters. If configurations are developed on one annotation
set, that same set is not an independent evaluation set.

## Manual benchmark records and JSON format

`ManualBenchmarkMetadata` separates descriptive annotation context from the
scientific Phase 5A.1 measurement. Its fields are:

- required `benchmark_id`, `transect_id`, and unique `annotation_id`;
- optional observation identifier and date, and site/reach label;
- optional analyst ID and display label;
- optional reference source description and acquisition date;
- optional temporal offset from SWOT in hours;
- optional annotation confidence; and
- optional notes.

Create the scientific reference only through Phase 5A.1, then associate the
metadata without recomputing width:

```python
from swot_pixc_lab import (
    ManualBenchmarkMetadata,
    create_manual_benchmark_record,
    measure_explicit_wet_intervals,
)

reference = measure_explicit_wet_intervals(sample, manual_intervals)
metadata = ManualBenchmarkMetadata(
    benchmark_id="example_benchmark",
    transect_id="example_transect_001",
    annotation_id="example_transect_001_analyst_a",
    analyst_id="analyst_a",
    reference_source_description="Describe the independent reference here",
)
record = create_manual_benchmark_record(metadata, reference)
payload = record.as_dict()
```

The serialized record keeps `metadata` separate from
`explicit_interval_reference`. The latter is a compact serialization of the
existing Phase 5A.1 result, including intervals, station frame, method identity,
QC context, and compact source provenance; it is not a second width
implementation.

[`../examples/manual_interval_benchmark_example.json`](../examples/manual_interval_benchmark_example.json)
is an obviously synthetic, human-editable example. A real workflow should:

1. define and freeze the transect geometry and its station direction;
2. select the independent reference image and record its source, acquisition
   time, registration caveats, and temporal mismatch from SWOT;
3. have an analyst record ordered wet interval boundaries in metres on the
   Phase 4 station axis, including uncertainty/confidence and notes;
4. pass those boundaries to `measure_explicit_wet_intervals`;
5. attach one `ManualBenchmarkMetadata` record; and
6. version the serialized `ManualBenchmarkRecord` with the imagery and
   transect definitions needed to reproduce it.

For repeated interpretation, store one independent record per analyst (and,
if appropriate, per repeat annotation), sharing the benchmark and transect IDs
but using distinct annotation IDs. Never average boundaries automatically,
silently select one analyst as truth, or overwrite one record with another.
Inter-analyst disagreement and consensus geometry require a future,
scientist-approved protocol.

## Proposed future Koshi benchmark

The first real-data benchmark should contain approximately 20–40 manually
reviewed Koshi transects. This is a recommended scientific study size, not a
software requirement or an encoded validation threshold. Owner-approved
selection should intentionally span:

- simple single-channel settings;
- multiple wetted branches and island-separated channels;
- exposed bars;
- sparse PIXC sampling;
- dark-water and low-coherence cases;
- bifurcation or confluence vicinity; and
- tile-edge cases where available.

Each reference should use the best available independent, co-registered,
near-date imagery and record the image source, acquisition date/time, temporal
mismatch, analyst, interpretation uncertainty/confidence, and any ambiguity
from cloud, shadow, turbidity, vegetation, geolocation, or changing discharge.
A planned subset should be independently annotated by more than one analyst;
the records remain separate until an inter-analyst analysis is approved.

Transect inclusion, exclusion near junctions, imagery acceptability, temporal
offset limits, and annotation-confidence vocabulary require owner/scientist
approval before collection begins. Phase 5A.3a does not choose Koshi
transects, create Koshi bank truth, download imagery, or recommend a candidate
configuration.

### Development and holdout principle

If Koshi annotations are used to choose extent classes, bin width, minimum
point count, or bridge tolerance, those same annotations must not be presented
as independent final validation. Before inspecting parameter performance, the
owner should approve a documented development/sensitivity subset and a
separate holdout evaluation subset. Phase 5A.3a does not randomly split,
assign, or otherwise manufacture those groups.

## Reporting and limitations

Do not reduce validation to one score. At minimum retain total-width bias,
overlap, false-positive and false-negative extent, directed boundary
distances, component-count differences, outer-span error, and bridge behavior.
A configuration can match total width while placing intervals incorrectly; it
can have good IoU while filling a real island or merging components.

The manual benchmark is an analyst interpretation affected by image date,
resolution, registration, cloud/shadow, stage changes, bank ambiguity, and
station placement. Agreement with it is evidence about reproducibility against
that declared reference, not proof of physical-bank truth or generalization to
another river, date, analyst, sensor, QC profile, or transect geometry.

Phase 5A.3a introduces no bank detector, raster or hull method, morphology,
optical classifier, WSE processing, centerline, branch graph, temporal
tracking, or RiverObs area-equivalent width. It performs no Koshi tuning and
makes no research-grade width claim. Real-data validation and parameter
selection belong to the later Phase 5A.3b work and require owner and SWOT/river
scientist review.
