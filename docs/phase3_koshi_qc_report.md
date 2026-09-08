# Phase 3 Koshi PIXC quality-control reality check

## Scope and status

This report records the Phase 3 scientific quality-control and height-reference
reality check for the exact AOI
`(86.87, 26.49, 87.20, 26.90)`. It does not define a channel-width method and
does not validate any profile for multiple-channel geometry.

The three profiles have deliberately different purposes:

| Profile | Status | Purpose |
|---|---|---|
| `raw` | Baseline; no scientific filtering | Preserve the Phase 2 exact-AOI PIXC observation unchanged. |
| `bo_legacy_strict` | **Legacy / conservative / not NASA-recommended** | Reproduce the project owner's older WSE-oriented screening rules when their input variables are present. |
| `channel_extent_candidate` | **Experimental / unvalidated** | Retain documented edge, dark-water, and low-coherence water classes while rejecting only documented land, invalid quality metadata, in-air pixels, and named bad classification/geolocation states. |

All QC masks are derived from the raw observation. Original `height`,
`classification`, `water_frac`, integer quality fields, and provenance fields
remain unchanged. Independent rejection counts overlap and therefore must not
be summed. Incremental counts are non-overlapping and depend on the documented
rule order.

## Data and provenance

The local regression used these Version D, cycle 9, pass 286, PGD0 granules:

- `SWOT_L2_HR_PIXC_009_286_107L_20240114T081432_20240114T081443_PGD0_01.nc`
- `SWOT_L2_HR_PIXC_009_286_108L_20240114T081442_20240114T081453_PGD0_01.nc`

The files remain local and ignored by Git.

| Source tile | Pixels before exact clipping | Pixels after exact clipping |
|---|---:|---:|
| 107L | 3,969,693 | 0 |
| 108L | 5,889,878 | 2,230,522 |
| **Logical observation** | **9,859,571** | **2,230,522** |

Tile 107L is retained in observation-level provenance even though it has no
pixel inside the exact AOI. Every retained or rejected point remains traceable
through `source_index` and `source_point_index`.

## Authoritative definitions

The implementation and this report use:

- [SWOT L2 HR PIXC Version D collection](https://podaac.jpl.nasa.gov/dataset/SWOT_L2_HR_PIXC_D)
- [L2 HR PIXC Product Description Document, JPL D-56411 Revision C](https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-docs/web-misc/swot_mission_docs/pdd/D-56411_SWOT_Product_Description_L2_HR_PIXC_20250224a_RevC_clean_sig_final.pdf), especially Sections 4.1.2.2-4.1.2.4, the variable tables on pages 45-55, and Appendix B Table 15 on pages 64-67
- [L2 HR PIXC Algorithm Theoretical Basis Document, JPL D-105504](https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-docs/web-misc/swot_mission_docs/atbd/D-105504_SWOT_ATBD_L2_HR_PIXC_20230713a_cite.pdf), Section 3.3.8.3, page 70
- [Version D KaRIn Products Release Note](https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-docs/web-misc/swot_mission_docs/SWOT_VersionD_KaRIn_Products_Release_Note_20250423b.pdf), Table 3 on page 8 and HR known issues on page 18
- [CF Conventions, Section 3.5, Flags](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.13/cf-conventions.html#flags)

The metadata in both real Koshi files agrees with the cited Version D PDD for
the requested variables, including dtypes, dimensions, fill values, valid
ranges, flag names, and flag masks. No PDD-versus-file discrepancy requiring
processing to stop was found.

### Classification

`classification` is `uint8(points)`, has `_FillValue=255`, valid range 1-7,
and the following official enumeration:

| Value | NetCDF meaning | ATBD operational definition |
|---:|---|---|
| 1 | `land` | Land in the retained keep buffer |
| 2 | `land_near_water` | Land on a water edge |
| 3 | `water_near_land` | Detected water on a land edge |
| 4 | `open_water` | Detected interior water |
| 5 | `dark_water` | Dark water |
| 6 | `low_coh_water_near_land` | Low-coherence detected-water edge |
| 7 | `open_low_coh_water` | Low-coherence detected water away from an edge |

Classes 3-7 are therefore the documented water classes used by the
experimental candidate. Classes 1 and 2 are documented land; class 2 may still
be useful as contextual bank information outside the candidate mask.

### Other legacy inputs

| Variable | Stored type and fill | Units / valid range | Product meaning |
|---|---|---|---|
| `water_frac` | float32; `9.96921e36` | `1`; -1000 to 10000 | Noisy estimate of the fraction of a pixel that is water |
| `water_frac_uncert` | float32; `9.96921e36` | `1`; 0 to 999999 | Dispersion/width of the noisy water-fraction estimate |
| `false_detection_rate` | float32; `9.96921e36` | `1`; 0 to 1 | Probability of detecting water when none exists |
| `missed_detection_rate` | float32; `9.96921e36` | `1`; 0 to 1 | Probability of detecting no water when water exists |
| `prior_water_prob` | float32; `9.96921e36` | `1`; 0 to 1 | Prior probability of water |
| `prior_water_change` | float32; `9.96921e36` | `1`; -1 to 1 | Qualitative long-term change in prior water probability |
| `bright_land_flag` | uint8; 255 | values 0, 1, 2 | `not_bright_land`, `bright_land`, `bright_land_or_water` |
| `inc` | float32; `9.96921e36` | degrees; metadata 0 to 999999 | Incidence angle |
| `phase_noise_std` | float32; `9.96921e36` | radians; -999999 to 999999 | Estimated phase-noise standard deviation |

The PDD says `water_frac` is nominally between zero and one but intentionally
does not clip noisy estimates outside that interval. It also cautions against
using the estimate directly for an individual pixel and describes aggregation
as its intended useful context. The broad NetCDF valid range is therefore not
a physical fraction range.

The PDD describes `inc` physically as an angle from 0 to 90 degrees, while the
PDD schema and both real files use the permissive `valid_max=999999`. This is a
schema-versus-narrative nuance, not a PDD-versus-file mismatch. The legacy
0.5-5.0 degree threshold is project-owner input, not a NASA recommendation.

## Version D quality-bit metadata

Each of the following is an unchanged `uint32(points)` variable with
`standard_name="status_flag"`, `_FillValue=4294967295` (`0xffffffff`), and
`valid_min=0`. Names and masks below are the exact NetCDF metadata and match
PDD Table 15.

### `classification_qual`

`valid_max=3893160991` (`0xe80cec1f`).

| Mask | Meaning |
|---:|---|
| 1 | `no_coherent_gain` |
| 2 | `power_close_to_noise_floor` |
| 4 | `detected_water_but_no_prior_water` |
| 8 | `detected_water_but_bright_land` |
| 16 | `water_false_detection_rate_suspect` |
| 1024 | `suspect_karin_telem` |
| 2048 | `coherent_power_suspect` |
| 8192 | `tvp_suspect` |
| 16384 | `sc_event_suspect` |
| 32768 | `small_karin_gap` |
| 262144 | `in_air_pixel_degraded` |
| 524288 | `specular_ringing_degraded` |
| 134217728 | `coherent_power_bad` |
| 536870912 | `tvp_bad` |
| 1073741824 | `sc_event_bad` |
| 2147483648 | `large_karin_gap` |

### `geolocation_qual`

`valid_max=4193842303` (`0xf9f8f47f`).

| Mask | Meaning |
|---:|---|
| 1 | `layover_significant` |
| 2 | `phase_noise_suspect` |
| 4 | `phase_unwrapping_suspect` |
| 8 | `model_dry_tropo_cor_suspect` |
| 16 | `model_wet_tropo_cor_suspect` |
| 32 | `iono_cor_gim_ka_suspect` |
| 64 | `xovercal_suspect` |
| 1024 | `suspect_karin_telem` |
| 4096 | `medium_phase_suspect` |
| 8192 | `tvp_suspect` |
| 16384 | `sc_event_suspect` |
| 32768 | `small_karin_gap` |
| 524288 | `specular_ringing_degraded` |
| 1048576 | `model_dry_tropo_cor_missing` |
| 2097152 | `model_wet_tropo_cor_missing` |
| 4194304 | `iono_cor_gim_ka_missing` |
| 8388608 | `xovercal_missing` |
| 16777216 | `geolocation_is_from_refloc` |
| 134217728 | `no_geolocation_bad` |
| 268435456 | `medium_phase_bad` |
| 536870912 | `tvp_bad` |
| 1073741824 | `sc_event_bad` |
| 2147483648 | `large_karin_gap` |

### `interferogram_qual`

`valid_max=4161600512` (`0xf80cfc00`).

| Mask | Meaning |
|---:|---|
| 1024 | `suspect_karin_telem` |
| 2048 | `rare_power_suspect` |
| 4096 | `rare_phase_suspect` |
| 8192 | `tvp_suspect` |
| 16384 | `sc_event_suspect` |
| 32768 | `small_karin_gap` |
| 262144 | `in_air_pixel_degraded` |
| 524288 | `specular_ringing_degraded` |
| 134217728 | `rare_power_bad` |
| 268435456 | `rare_phase_bad` |
| 536870912 | `tvp_bad` |
| 1073741824 | `sc_event_bad` |
| 2147483648 | `large_karin_gap` |

### `sig0_qual`

`valid_max=3994872847` (`0xee1cec0f`).

| Mask | Meaning |
|---:|---|
| 1 | `sig0_uncert_suspect` |
| 2 | `sig0_cor_atmos_suspect` |
| 4 | `noise_power_suspect` |
| 8 | `xfactor_suspect` |
| 1024 | `suspect_karin_telem` |
| 2048 | `rare_power_suspect` |
| 8192 | `tvp_suspect` |
| 16384 | `sc_event_suspect` |
| 32768 | `small_karin_gap` |
| 262144 | `in_air_pixel_degraded` |
| 524288 | `specular_ringing_degraded` |
| 1048576 | `sig0_cor_atmos_missing` |
| 33554432 | `noise_power_bad` |
| 67108864 | `xfactor_bad` |
| 134217728 | `rare_power_bad` |
| 536870912 | `tvp_bad` |
| 1073741824 | `sc_event_bad` |
| 2147483648 | `large_karin_gap` |

### Individual flags versus summary states

These four variables contain independent bit conditions, not mutually
exclusive summary categories. Under CF semantics, a named bit is true when
`value & flag_mask != 0`. Fill must be detected first: `0xffffffff` has every
bit set and would otherwise be falsely decoded as every condition.

The PDD says zero is nominal and a nonzero value reports suspect or bad
information, but it does not define these four PIXC fields as categorical
`good`, `suspect`, `degraded`, and `bad` summary values. Suffixes in the flag
names and the qualitative color shading in PDD Table 15 do not justify
inventing an undocumented numeric summary. Accordingly, the legacy
`*_qual == 0` rule means that no documented bit may be set; it is substantially
stricter than rejecting named bad conditions.

`pixc_line_qual` is different in shape: in 108L it is
`uint32(num_pixc_lines)` with 3,279 lines. It remains line-level metadata and is
never broadcast or expanded onto the 5,889,878 point records.

## Exact profile rules

### `raw`

No rejection rule is applied. All 2,230,522 Phase 2 exact-AOI points remain.

### `bo_legacy_strict`

Rules are evaluated in this order and only when the source variable exists and
its units/packing support the threshold. For Koshi, every rule was available;
none was skipped.

1. `classification == 4`
2. `water_frac >= 0.90`
3. `water_frac_uncert <= 0.15`
4. `bright_land_flag == 0`
5. `false_detection_rate <= 0.10`
6. `classification_qual == 0`
7. `geolocation_qual == 0`
8. `interferogram_qual == 0`
9. `sig0_qual == 0`
10. `phase_noise_std <= 1.0 radians`
11. `0.5 <= inc <= 5.0 degrees`, inclusively

These rules reproduce an older analysis; they are not claimed to be a
universal WSE standard or NASA-recommended QC.

### `channel_extent_candidate`

Rules are evaluated in this order:

1. Require `classification in {3, 4, 5, 6, 7}`. This excludes the two
   documented land classes but preserves detected edge, interior, dark, and
   low-coherence water.
2. Require `classification_qual` to be non-fill, within its documented valid
   range, and free of undocumented set bits.
3. Require `geolocation_qual` to be non-fill, within its documented valid
   range, and free of undocumented set bits.
4. Exclude these decoded `classification_qual` conditions:
   `in_air_pixel_degraded`, `coherent_power_bad`, `tvp_bad`, `sc_event_bad`,
   and `large_karin_gap`.
5. Exclude these decoded `geolocation_qual` conditions:
   `no_geolocation_bad`, `medium_phase_bad`, `tvp_bad`, `sc_event_bad`, and
   `large_karin_gap`.
6. Exclude `interferogram_qual: in_air_pixel_degraded`, whose independent
   definition also says the range bin does not intersect Earth's surface.

The in-air definition says the range bin does not intersect Earth's surface.
The remaining selected conditions are explicitly named bad or describe the
absence of a meaningful/available measurement. Suspect and other degraded
flags are retained as diagnostics. Apart from the explicit in-air condition,
`interferogram_qual` and all `sig0_qual` conditions are decoded and reported
but do not filter the extent candidate.

No class-4-only, water-fraction, water-fraction-uncertainty, bright-land,
false-detection-rate, phase-noise, incidence-angle, or blanket
`quality == 0` rule is applied by this experimental profile.

## Koshi profile comparison

| Profile | Starting pixels | Final pixels | Retained | Classification after QC | Source tiles after QC |
|---|---:|---:|---:|---|---|
| `raw` | 2,230,522 | 2,230,522 | 100.0000% | 1: 1,824,495; 2: 125,826; 3: 105,528; 4: 131,013; 5: 40,167; 6: 3,326; 7: 167 | 107L: 0; 108L: 2,230,522 |
| `bo_legacy_strict` | 2,230,522 | 0 | 0.0000% | none | 107L: 0; 108L: 0 |
| `channel_extent_candidate` | 2,230,522 | 280,198 | 12.5620% | 3: 105,528; 4: 131,010; 5: 40,167; 6: 3,326; 7: 167 | 107L: 0; 108L: 280,198 |

The candidate retains 280,198 of the 280,201 documented water-class pixels
(99.9989%); its only additional Koshi removal is three class-4 in-air pixels.

### Legacy independent and incremental rejection counts

| Ordered rule | Independent failures | Incremental removals | Pixels remaining |
|---|---:|---:|---:|
| `classification == 4` | 2,099,509 | 2,099,509 | 131,013 |
| `water_frac >= 0.90` | 2,151,407 | 72,452 | 58,561 |
| `water_frac_uncert <= 0.15` | 197,513 | 58,561 | 0 |
| `bright_land_flag == 0` | 111,689 | 0 | 0 |
| `false_detection_rate <= 0.10` | 299 | 0 | 0 |
| `classification_qual == 0` | 572,921 | 0 | 0 |
| `geolocation_qual == 0` | 2,033,625 | 0 | 0 |
| `interferogram_qual == 0` | 63,985 | 0 | 0 |
| `sig0_qual == 0` | 70,674 | 0 | 0 |
| `phase_noise_std <= 1.0` | 60,321 | 0 | 0 |
| `0.5 <= inc <= 5.0` | 221 | 0 | 0 |

No source value used by these rules was fill or outside the documented valid
range. The third ordered rule removes every one of the 58,561 pixels that
survived the first two rules: among that intermediate population,
`water_frac_uncert` ranges from 0.429285 to 3.100230, with median 0.649688.
This result should be checked against the intent and provenance of the older
workflow rather than interpreted as evidence that the data contain no useful
water.

### Candidate independent and incremental rejection counts

| Ordered rule | Independent failures | Incremental removals | Pixels remaining |
|---|---:|---:|---:|
| Classification in classes 3-7 | 1,950,321 | 1,950,321 | 280,201 |
| Valid `classification_qual` | 0 | 0 | 280,201 |
| Valid `geolocation_qual` | 0 | 0 | 280,201 |
| `classification_qual`: `in_air_pixel_degraded` false | 50 | 3 | 280,198 |
| `classification_qual`: `coherent_power_bad` false | 0 | 0 | 280,198 |
| `classification_qual`: `tvp_bad` false | 0 | 0 | 280,198 |
| `classification_qual`: `sc_event_bad` false | 0 | 0 | 280,198 |
| `classification_qual`: `large_karin_gap` false | 0 | 0 | 280,198 |
| `geolocation_qual`: `no_geolocation_bad` false | 0 | 0 | 280,198 |
| `geolocation_qual`: `medium_phase_bad` false | 0 | 0 | 280,198 |
| `geolocation_qual`: `tvp_bad` false | 0 | 0 | 280,198 |
| `geolocation_qual`: `sc_event_bad` false | 0 | 0 | 280,198 |
| `geolocation_qual`: `large_karin_gap` false | 0 | 0 | 280,198 |
| `interferogram_qual`: `in_air_pixel_degraded` false | 50 | 0 | 280,198 |

The classification-quality in-air count includes 47 already-excluded
land-class points, which is why its incremental removal is three. The same 50
points also carry the interferogram in-air bit, so that later rule removes no
additional pixel.

## Class-4 and water-fraction evidence

From the raw exact-AOI population:

- `classification == 4`: 131,013 pixels
- `water_frac >= 0.90`: 79,115 pixels
- both conditions: 58,561 pixels
- documented water pixels outside class 4 that a class-4-only rule loses:
  149,188

| Lost water class | Pixels |
|---:|---:|
| 3, detected water at a land edge | 105,528 |
| 5, dark water | 40,167 |
| 6, low-coherence water edge | 3,326 |
| 7, open low-coherence water | 167 |

### Water-fraction distribution by classification

`n` includes valid water-fraction values in each class. The PDD explicitly
allows estimates outside 0-1.

| Class | n | Minimum | 5th percentile | Median | 95th percentile | Maximum | `water_frac >= 0.90` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,824,495 | -17.055153 | -0.044131 | -0.006375 | 0.071128 | 23.282875 | 59 |
| 2 | 125,826 | -2.223151 | -0.031242 | 0.018642 | 0.124340 | 1.620733 | 14 |
| 3 | 105,528 | -0.443781 | 0.088406 | 0.321942 | 1.340944 | 4.390566 | 20,200 |
| 4 | 131,013 | -1.242896 | 0.180383 | 0.827166 | 1.957156 | 6.290984 | 58,561 |
| 5 | 40,167 | -1.883005 | -0.059078 | -0.003903 | 0.120134 | 7.448197 | 17 |
| 6 | 3,326 | -0.007648 | 0.082276 | 0.215080 | 0.996442 | 1.806238 | 237 |
| 7 | 167 | -0.036007 | 0.076793 | 0.266624 | 1.121301 | 3.041896 | 27 |

These counts show why class 4 and a 0.90 water-fraction threshold cannot be
adopted uncritically for future channel-boundary work. They do not establish
which alternative is scientifically valid.

## Decoded quality frequencies

Counts below are independent frequencies in all 2,230,522 raw exact-AOI
pixels. Named flags with zero occurrences are omitted here but remain exposed
by the decoder. There were no fill or out-of-range quality values.

| Quality variable | Named flag | Pixels |
|---|---|---:|
| `classification_qual` | `no_coherent_gain` | 206,716 |
| | `power_close_to_noise_floor` | 336,885 |
| | `detected_water_but_no_prior_water` | 96,679 |
| | `detected_water_but_bright_land` | 6,700 |
| | `water_false_detection_rate_suspect` | 328 |
| | `coherent_power_suspect` | 31,138 |
| | `in_air_pixel_degraded` | 50 |
| | `specular_ringing_degraded` | 21,986 |
| `geolocation_qual` | `layover_significant` | 56,519 |
| | `phase_noise_suspect` | 40 |
| | `phase_unwrapping_suspect` | 2,031,811 |
| | `medium_phase_suspect` | 7,480 |
| | `specular_ringing_degraded` | 23,891 |
| `interferogram_qual` | `rare_phase_suspect` | 42,250 |
| | `in_air_pixel_degraded` | 50 |
| | `specular_ringing_degraded` | 21,986 |
| `sig0_qual` | `sig0_uncert_suspect` | 48,700 |
| | `in_air_pixel_degraded` | 50 |
| | `specular_ringing_degraded` | 21,986 |

Zero/nonzero raw values are:

| Variable | Exactly zero | Nonzero, non-fill |
|---|---:|---:|
| `classification_qual` | 1,657,601 | 572,921 |
| `geolocation_qual` | 196,897 | 2,033,625 |
| `interferogram_qual` | 2,166,537 | 63,985 |
| `sig0_qual` | 2,159,848 | 70,674 |

Every documented `*_bad` or `*_missing` flag has zero occurrences in this AOI.
`detected_water_but_no_prior_water` is deliberately retained by the candidate:
it can be relevant to new or migrating water. Rejecting every nonzero quality
value would also remove most of the raw AOI because
`phase_unwrapping_suspect` is common.

## Height reference

The real metadata and PDD explicitly establish:

- `height` is float32 ellipsoidal height in metres and remains untouched.
- The root ellipsoid has semi-major axis 6,378,137.0 m and flattening
  0.0033528106647474805, consistent with WGS84.
- `geoid` is float32 in metres, has
  `standard_name="geoid_height_above_reference_ellipsoid"`, and declares
  `source="EGM2008 (Pavlis et al., 2012)"`.
- The supplied geoid includes the mean-tide/permanent-tide adjustment and is
  not already applied to `height`.

Therefore Phase 3 derives, for valid input pairs only:

```text
height_egm2008 = height - geoid
```

The name is justified by the explicit source metadata. It is **not corrected
WSE**. No solid-Earth, load, pole, tropospheric, ionospheric, or other
geophysical correction is applied by this derivation. The Version D release
note reports that Version D fixed an approximately decimetre-scale error in
previously reported geoid values.

| Profile | Height reference | Count | Minimum (m) | Maximum (m) | Mean (m) | Median (m) | Std. dev. (m) |
|---|---|---:|---:|---:|---:|---:|---:|
| `raw` | Ellipsoidal `height` | 2,230,522 | -20.643219 | 1002.676453 | 41.042691 | 27.256669 | 58.390967 |
| `raw` | `height_egm2008` | 2,230,522 | 39.278385 | 1056.908936 | 99.667572 | 86.133530 | 57.513489 |
| `bo_legacy_strict` | Either reference | 0 | - | - | - | - | - |
| `channel_extent_candidate` | Ellipsoidal `height` | 280,198 | -5.212938 | 848.949158 | 29.683965 | 25.754194 | 24.134398 |
| `channel_extent_candidate` | `height_egm2008` | 280,198 | 54.116539 | 903.287476 | 88.450779 | 84.683842 | 23.268894 |

These summaries include all classes admitted by each profile and are not WSE
estimates.

## Scientific uncertainties and owner-review questions

1. Should classes 3-7 all remain eligible for future extent work after visual
   validation against independent imagery? In particular, Version D release
   notes warn that dark-water classification itself can be wrong and that
   land/water boundaries are especially error-prone.
2. Should `geolocation_is_from_refloc` be excluded from exact-boundary work?
   Its definition says a reference location was substituted because computed
   geolocation was not sensible or was unexpectedly displaced. Koshi has zero
   occurrences, so this dataset cannot validate that decision.
3. Should any suspect/degraded flags become optional stricter geometry rules?
   Phase unwrapping, layover, and specular ringing can affect horizontal
   geometry, but their flags do not prove that every affected point is unusable.
   Rejecting `phase_unwrapping_suspect` would also remove all 40,167 class-5
   dark-water points in this Koshi subset.
4. Should `bright_land_flag=1` and `bright_land_flag=2` be treated differently?
   Value 2 explicitly encodes conflicting prior land/water information, which
   can matter in a migrating river corridor.
5. Was the historical `water_frac_uncert <= 0.15` threshold designed for this
   PIXC version and this stored uncertainty definition? It alone removes every
   point surviving the first two legacy rules here.
6. Which validation imagery and reaches should be used before any
   `channel_extent_candidate` mask is promoted or used in a width algorithm?
7. Are the raw and EGM2008-relative height outliers scientifically plausible,
   or should later height-focused work add separately documented QC? That is
   intentionally not decided in Phase 3.

The experimental candidate is evidence-generating scaffolding only. It is not
a validated channel mask, and this report makes no Phase 4 or channel-width
claim.
