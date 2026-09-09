from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from shapely.geometry import LineString

from swot_pixc_lab.bank_inference import infer_candidate_wet_intervals
from swot_pixc_lab.mosaic import PixcObservation
from swot_pixc_lab.qc import (
    QC_PROFILES,
    HeightReferenceResult,
    QCResult,
    decode_quality_flags,
)
from swot_pixc_lab.subset import normalize_exact_aoi
from swot_pixc_lab.transect import sample_transect
from swot_pixc_lab.visualization import (
    CLASSIFICATION_LABELS,
    plot_candidate_wet_interval_inference,
    plot_classification_comparison,
    plot_pixc_map,
    plot_transect_classification,
    plot_transect_corridor,
    plot_transect_height,
    plot_wet_interval_summary,
)
from swot_pixc_lab.width import measure_explicit_wet_intervals

matplotlib.use("Agg", force=True)

FLOAT_FILL = np.float32(9.96921e36)
UINT8_FILL = np.uint8(255)
UINT32_FILL = np.uint32(4294967295)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _pixel_dataset(*, count: int = 7, candidate: bool = False) -> xr.Dataset:
    classification = np.resize(
        np.arange(1, 8, dtype=np.uint8),
        count,
    )
    dataset = xr.Dataset(
        {
            "longitude": xr.DataArray(
                np.linspace(86.9, 87.1, count),
                dims="points",
                attrs={"units": "degrees_east"},
            ),
            "latitude": xr.DataArray(
                np.linspace(26.6, 26.8, count),
                dims="points",
                attrs={"units": "degrees_north"},
            ),
            "classification": xr.DataArray(
                classification,
                dims="points",
                attrs={
                    "_FillValue": UINT8_FILL,
                    "valid_min": np.uint8(1),
                    "valid_max": np.uint8(7),
                },
            ),
            "height": xr.DataArray(
                np.linspace(100.0, 110.0, count, dtype=np.float32),
                dims="points",
                attrs={"_FillValue": FLOAT_FILL, "units": "m"},
            ),
            "height_egm2008": xr.DataArray(
                np.linspace(40.0, 50.0, count, dtype=np.float32),
                dims="points",
                attrs={"units": "m"},
            ),
            "water_frac": xr.DataArray(
                np.linspace(0.0, 1.0, count, dtype=np.float32),
                dims="points",
                attrs={"_FillValue": FLOAT_FILL, "units": "1"},
            ),
            "source_index": xr.DataArray(
                np.arange(count, dtype=np.int32) % 2,
                dims="points",
            ),
            "source_point_index": xr.DataArray(
                np.arange(count, dtype=np.int64),
                dims="points",
            ),
        }
    )
    if candidate:
        dataset.attrs.update(
            {
                "swot_pixc_lab_qc_profile": "channel_extent_candidate",
                "swot_pixc_lab_qc_label": "Channel extent candidate",
                "swot_pixc_lab_qc_status": "experimental / unvalidated",
            }
        )
    return dataset


def _qc_result() -> QCResult:
    raw = _pixel_dataset(count=4)
    quality = np.asarray([0, 1, 1, UINT32_FILL], dtype=np.uint32)
    decoded = decode_quality_flags(
        quality,
        {
            "flag_masks": np.asarray([1], dtype=np.uint32),
            "flag_meanings": "synthetic_suspect",
            "valid_min": np.uint32(0),
            "valid_max": np.uint32(1),
            "_FillValue": UINT32_FILL,
        },
        variable_name="classification_qual",
    )
    raw["classification_qual"] = xr.DataArray(quality, dims="points")
    observation = PixcObservation(
        pixels=raw,
        sources=(),
        aoi=normalize_exact_aoi((86.8, 26.5, 87.2, 26.9)),
    )
    mask = np.asarray([True, True, False, True], dtype=np.bool_)
    filtered = raw.isel(points=np.flatnonzero(mask)).copy(deep=True)
    filtered.attrs.update(
        {
            "swot_pixc_lab_qc_profile": "channel_extent_candidate",
            "swot_pixc_lab_qc_label": "Channel extent candidate",
            "swot_pixc_lab_qc_status": "experimental / unvalidated",
        }
    )
    height_reference = HeightReferenceResult(
        status="skipped",
        variable_name=None,
        formula="height - geoid",
        egm2008_confirmed=False,
        valid_count=0,
        invalid_count=4,
        definition_source="synthetic",
        skipped_reason="synthetic fixture",
    )
    return QCResult(
        profile=QC_PROFILES["channel_extent_candidate"],
        observation=observation,
        mask=mask,
        filtered=filtered,
        reason_masks={},
        rule_outcomes=(),
        decoded_flags={"classification_qual": decoded},
        height_reference=height_reference,
    )


def _sample(*, empty: bool = False) -> SimpleNamespace:
    pixels = _pixel_dataset(count=0 if empty else 5, candidate=True)
    pixels["station_m"] = xr.DataArray(
        np.asarray([] if empty else [0.0, 25.0, 50.0, 75.0, 100.0]),
        dims="points",
        attrs={"units": "m"},
    )
    pixels["distance_to_transect_m"] = xr.DataArray(
        np.asarray([] if empty else [0.0, 5.0, 10.0, 5.0, 0.0]),
        dims="points",
        attrs={"units": "m"},
    )
    line = LineString([(86.9, 26.6), (87.1, 26.8)])
    return SimpleNamespace(
        pixels=pixels,
        transect=line,
        corridor=line.buffer(0.001),
        corridor_half_width_m=50.0,
        profile_name="channel_extent_candidate",
        profile_label="Channel extent candidate",
        profile_status="experimental / unvalidated",
    )


def _width_result(intervals):
    sample = sample_transect(
        _pixel_dataset(candidate=True),
        LineString([(86.9, 26.6), (87.1, 26.8)]),
        corridor_half_width_m=100.0,
    )
    return sample, measure_explicit_wet_intervals(sample, intervals)


def _candidate_inference_result(*, empty: bool = False):
    sample = sample_transect(
        _pixel_dataset(count=0 if empty else 7, candidate=True),
        LineString([(86.9, 26.6), (87.1, 26.8)]),
        corridor_half_width_m=100.0,
    )
    bin_width = sample.transect_length_m / 5.0
    if not empty:
        sample.pixels["station_m"] = xr.DataArray(
            bin_width * np.asarray([0.25, 0.5, 1.25, 3.25, 3.5, 4.25, 4.5]),
            dims="points",
            attrs={"units": "m"},
        )
        sample.pixels["classification"] = xr.DataArray(
            np.asarray([4, 4, 1, 4, 4, 1, 2], dtype=np.uint8),
            dims="points",
            attrs={"_FillValue": UINT8_FILL},
        )
    result = infer_candidate_wet_intervals(
        sample,
        extent_classes=(4,),
        station_bin_width_m=bin_width,
        min_extent_pixels_per_bin=2,
        max_bridge_gap_m=0.0 if empty else 2.1 * bin_width,
    )
    return sample, result


def test_classification_map_is_one_rasterized_collection_and_does_not_mutate() -> None:
    dataset = _pixel_dataset()
    before = dataset.copy(deep=True)

    axes = plot_pixc_map(dataset, color_by="classification")

    assert isinstance(axes, Axes)
    assert len(axes.collections) == 1
    assert axes.collections[0].get_rasterized() is True
    assert axes.collections[0].get_offsets().shape == (7, 2)
    assert axes.get_xlabel() == "Longitude (degrees east)"
    assert axes.get_ylabel() == "Latitude (degrees north)"
    legend_labels = [text.get_text() for text in axes.get_legend().get_texts()]
    assert legend_labels == [
        f"{value}: {label}" for value, label in CLASSIFICATION_LABELS.items()
    ]
    xr.testing.assert_identical(dataset, before)


@pytest.mark.parametrize("color_by", ["height", "height_egm2008", "water_frac"])
def test_continuous_map_variables_use_colorbar_and_filter_fill(color_by: str) -> None:
    dataset = _pixel_dataset()
    dataset[color_by].values[-1] = FLOAT_FILL
    dataset[color_by].attrs["_FillValue"] = FLOAT_FILL

    axes = plot_pixc_map(dataset, color_by=color_by)

    assert len(axes.collections) == 1
    assert axes.collections[0].get_offsets().shape == (6, 2)
    assert len(axes.figure.axes) == 2
    if color_by in {"height", "height_egm2008"}:
        assert "not corrected WSE" in axes.figure.axes[1].get_ylabel()


def test_source_map_and_experimental_qc_result_title() -> None:
    result = _qc_result()

    axes = plot_pixc_map(result, color_by="source_index", show_colorbar=False)

    assert len(axes.collections) == 1
    assert "experimental / unvalidated" in axes.get_title()
    assert [text.get_text() for text in axes.get_legend().get_texts()] == [
        "0: source",
        "1: source",
    ]


def test_decoded_flag_selector_aligns_qc_mask_and_excludes_fill() -> None:
    result = _qc_result()

    axes = plot_pixc_map(
        result,
        color_by="classification_qual:synthetic_suspect",
    )

    collection = axes.collections[0]
    assert collection.get_offsets().shape == (2, 2)
    np.testing.assert_array_equal(collection.get_array(), [0, 1])
    assert "experimental / unvalidated" in axes.get_title()


def test_decoded_flag_selector_aligns_a_reordered_filtered_result_by_provenance() -> (
    None
):
    result = _qc_result()
    reordered = replace(
        result,
        filtered=result.filtered.isel(points=[1, 0, 2]).copy(deep=True),
    )

    axes = plot_pixc_map(
        reordered,
        color_by="classification_qual:synthetic_suspect",
    )

    collection = axes.collections[0]
    np.testing.assert_array_equal(collection.get_array(), [1, 0])
    np.testing.assert_allclose(
        collection.get_offsets(),
        np.column_stack(
            (
                reordered.filtered["longitude"].values[:2],
                reordered.filtered["latitude"].values[:2],
            )
        ),
    )


def test_large_map_submits_every_point_without_artist_per_pixel() -> None:
    dataset = _pixel_dataset(count=300_001)

    axes = plot_pixc_map(
        dataset,
        color_by="classification",
        show_legend=False,
    )

    assert len(axes.collections) == 1
    assert axes.collections[0].get_offsets().shape == (300_001, 2)


def test_classification_comparison_has_shared_extent_counts_and_legend() -> None:
    raw = _pixel_dataset()
    candidate = raw.isel(points=[2, 3, 4, 5, 6]).copy(deep=True)
    candidate.attrs.update(
        {
            "swot_pixc_lab_qc_profile": "channel_extent_candidate",
            "swot_pixc_lab_qc_label": "Channel extent candidate",
            "swot_pixc_lab_qc_status": "experimental / unvalidated",
        }
    )

    figure, axes = plot_classification_comparison(
        raw,
        candidate,
        title="Synthetic classification review",
        extent=(86.87, 26.49, 87.20, 26.90),
    )

    assert isinstance(figure, Figure)
    assert axes.shape == (3,)
    assert [len(axis.collections) for axis in axes] == [1, 1, 1]
    assert [axis.collections[0].get_offsets().shape[0] for axis in axes] == [
        7,
        5,
        5,
    ]
    assert all(axis.get_xlim() == pytest.approx((86.87, 87.20)) for axis in axes)
    assert "not a validated boundary" in axes[1].get_title()
    assert "experimental / unvalidated" in axes[2].get_title()
    assert len(figure.legends[0].get_texts()) == 7
    assert "Synthetic classification review" in figure._suptitle.get_text()
    assert "no boundary or width inference" in figure._suptitle.get_text()


def test_classification_comparison_rejects_candidate_from_another_observation() -> None:
    raw_result = _qc_result()
    unrelated_candidate = _qc_result()

    with pytest.raises(ValueError, match="different observation"):
        plot_classification_comparison(
            raw_result.observation,
            candidate=unrelated_candidate,
        )


def test_classification_comparison_validates_standalone_candidate_provenance() -> None:
    raw = _pixel_dataset()
    candidate = raw.isel(points=[2, 3]).copy(deep=True)
    candidate.attrs.update(
        {
            "swot_pixc_lab_qc_profile": "channel_extent_candidate",
            "swot_pixc_lab_qc_label": "Channel extent candidate",
            "swot_pixc_lab_qc_status": "experimental / unvalidated",
        }
    )
    candidate["source_point_index"].values[1] = 999

    with pytest.raises(ValueError, match="provenance pair"):
        plot_classification_comparison(raw, candidate=candidate)


def test_transect_diagnostics_show_corridor_samples_and_no_interpolation() -> None:
    sample = _sample()

    corridor_axes = plot_transect_corridor(sample)
    class_axes = plot_transect_classification(sample)
    height_axes = plot_transect_height(sample)

    assert len(corridor_axes.collections) == 1
    assert len(corridor_axes.lines) >= 4
    assert "±50 m" in corridor_axes.get_title()
    assert corridor_axes.collections[0].get_rasterized() is True
    assert len(class_axes.collections) == 1
    assert class_axes.collections[0].get_rasterized() is True
    assert "no width inference" in class_axes.get_title()
    assert len(height_axes.collections) == 2
    assert all(item.get_rasterized() for item in height_axes.collections)
    assert "no interpolation" in height_axes.get_title()
    height_labels = [text.get_text() for text in height_axes.get_legend().get_texts()]
    assert all("not corrected WSE" in label for label in height_labels)
    forbidden = {"channel_width", "branch_width", "bank_position", "branch_id"}
    assert forbidden.isdisjoint(vars(sample))


def test_custom_transect_titles_cannot_hide_corridor_or_scientific_cautions() -> None:
    sample = _sample()

    corridor_axes = plot_transect_corridor(sample, title="Owner corridor")
    class_axes = plot_transect_classification(sample, title="Owner classes")
    height_axes = plot_transect_height(sample, title="Owner heights")

    for axes in (corridor_axes, class_axes, height_axes):
        assert "Owner" in axes.get_title()
        assert "±50 m corridor" in axes.get_title()
        assert "width inference" in axes.get_title()
    assert "sampling corridor only" in class_axes.get_title()
    assert "no interpolation" in height_axes.get_title()


def test_empty_transect_plots_remain_auditable() -> None:
    sample = _sample(empty=True)

    corridor_axes = plot_transect_corridor(sample)
    class_axes = plot_transect_classification(sample)
    height_axes = plot_transect_height(sample)

    assert corridor_axes.collections[0].get_offsets().shape == (0, 2)
    assert class_axes.collections[0].get_offsets().shape == (0, 2)
    assert [item.get_offsets().shape for item in height_axes.collections] == [
        (0, 2),
        (0, 2),
    ]


def test_explicit_width_plot_draws_exact_supplied_intervals_and_gaps() -> None:
    sample, result = _width_result([(0.0, 120.0), (180.0, 260.0), (310.0, 350.0)])
    before = result.audit_summary()

    axes = plot_wet_interval_summary(result)

    assert isinstance(axes, Axes)
    assert [patch.get_x() for patch in axes.patches] == [
        0.0,
        180.0,
        310.0,
        120.0,
        260.0,
    ]
    assert [patch.get_width() for patch in axes.patches] == [
        120.0,
        80.0,
        40.0,
        60.0,
        50.0,
    ]
    assert [patch.get_gid() for patch in axes.patches] == [
        "swot-pixc-lab:explicit-wet-interval-1",
        "swot-pixc-lab:explicit-wet-interval-2",
        "swot-pixc-lab:explicit-wet-interval-3",
        "swot-pixc-lab:explicit-dry-gap-1",
        "swot-pixc-lab:explicit-dry-gap-2",
    ]
    labels = [text.get_text() for text in axes.texts]
    assert "I1\n120 m" in labels
    assert "G1: 60 m" in labels
    assert any("Total wetted width: 240 m" in label for label in labels)
    assert any("outer wetted span: 350 m" in label for label in labels)
    assert any("internal dry gaps: 110 m" in label for label in labels)
    assert axes.get_xlim() == pytest.approx((0.0, sample.transect_length_m))
    assert "analyst-supplied measurements only" in axes.get_title()
    assert "no bank inference" in axes.get_title()
    assert result.audit_summary() == before


def test_explicit_width_plot_uses_supplied_axes_and_keeps_caution_visible() -> None:
    _, result = _width_result([(10.0, 20.0)])
    import matplotlib.pyplot as plt

    _, supplied_axes = plt.subplots()
    returned = plot_wet_interval_summary(result, ax=supplied_axes, title="Owner review")

    assert returned is supplied_axes
    assert "Owner review" in returned.get_title()
    assert "EXPERIMENTAL / MANUAL BENCHMARK CONTRACT" in returned.get_title()
    assert "analyst-supplied measurements only" in returned.get_title()
    assert "no bank inference" in returned.get_title()


def test_empty_explicit_width_plot_has_no_misleading_interval_legend() -> None:
    _, result = _width_result([])

    axes = plot_wet_interval_summary(result)

    assert len(axes.patches) == 0
    assert axes.get_legend() is None
    labels = [text.get_text() for text in axes.texts]
    assert "No wet intervals supplied" in labels
    assert any("outer wetted span: not defined" in label for label in labels)
    assert any("Total wetted width: 0 m" in label for label in labels)


def test_candidate_inference_plot_preserves_states_bridges_and_bin_boundaries() -> None:
    _, result = _candidate_inference_result()
    before = result.audit_summary()

    axes = plot_candidate_wet_interval_inference(result)

    assert isinstance(axes, Axes)
    assert [item.state for item in result.bins] == [
        "candidate_wet",
        "sampled_noneligible",
        "unsampled",
        "candidate_wet",
        "sampled_noneligible",
    ]
    patches_by_gid = {item.get_gid(): item for item in axes.patches}
    bin_patches = [
        patches_by_gid[f"swot-pixc-lab:candidate-bin-{bin_id}"]
        for bin_id in range(1, 6)
    ]
    assert (
        len({(item.get_facecolor(), item.get_hatch()) for item in bin_patches[:3]}) == 3
    )
    assert bin_patches[1].get_hatch() == ".."
    assert bin_patches[2].get_hatch() == "xx"

    bridge = result.bridge_records[0]
    bridge_patch = patches_by_gid[f"swot-pixc-lab:candidate-bridge-{bridge.bridge_id}"]
    assert bridge_patch.get_x() == pytest.approx(bridge.gap_start_station_m)
    assert bridge_patch.get_width() == pytest.approx(bridge.gap_width_m)
    assert bridge_patch.get_facecolor()[3] == 0.0
    assert bridge_patch.get_hatch() == "////"
    assert bin_patches[2].get_facecolor() != bin_patches[0].get_facecolor()

    interval = result.candidate_intervals[0]
    interval_patch = patches_by_gid[
        f"swot-pixc-lab:candidate-interval-{interval.candidate_interval_id}"
    ]
    assert interval_patch.get_x() == pytest.approx(interval.start_station_m)
    assert interval_patch.get_x() + interval_patch.get_width() == pytest.approx(
        interval.end_station_m
    )
    assert interval_patch.get_facecolor()[3] == 0.0
    assert interval_patch.get_linestyle() == "--"
    annotation = "\n".join(item.get_text() for item in axes.texts)
    assert "Observed candidate wet-bin support:" in annotation
    assert "summed inferred interval span:" in annotation
    assert "bridged gaps:" in annotation
    assert result.audit_summary() == before


def test_candidate_inference_plot_uses_supplied_axes_and_keeps_full_caution() -> None:
    _, result = _candidate_inference_result()
    import matplotlib.pyplot as plt

    _, supplied_axes = plt.subplots()
    returned = plot_candidate_wet_interval_inference(
        result,
        ax=supplied_axes,
        title="Owner candidate review",
    )

    assert returned is supplied_axes
    plot_title = returned.get_title()
    assert "Owner candidate review" in plot_title
    assert result.method_status in plot_title
    assert "not validated physical banks" in plot_title
    assert "unsampled ≠ dry" in plot_title
    assert "classes=(4)" in plot_title
    assert f"bin={result.station_bin_width_m:g} m" in plot_title
    assert f"minimum={result.min_extent_pixels_per_bin}" in plot_title
    assert f"bridge tolerance={result.max_bridge_gap_m:g} m" in plot_title
    legend_labels = [text.get_text() for text in returned.get_legend().get_texts()]
    assert "candidate wet-support bin (classification evidence)" in legend_labels
    assert "sampled below threshold (not confirmed dry)" in legend_labels
    assert "unsampled (unknown; not confirmed dry)" in legend_labels
    assert "caller-authorized bridged gap" in legend_labels
    assert "candidate interval boundary (station-bin edge)" in legend_labels


def test_empty_candidate_inference_plot_retains_unsampled_domain_and_cautions() -> None:
    sample, result = _candidate_inference_result(empty=True)
    before = result.audit_summary()

    axes = plot_candidate_wet_interval_inference(result)

    assert result.candidate_intervals == ()
    assert result.bridge_records == ()
    assert result.unsampled_bin_count == result.bin_count == 5
    assert len(axes.patches) == result.bin_count
    assert all(item.get_hatch() == "xx" for item in axes.patches)
    assert all(
        item.get_gid() == f"swot-pixc-lab:candidate-bin-{index}"
        for index, item in enumerate(axes.patches, start=1)
    )
    labels = [text.get_text() for text in axes.texts]
    assert any("Observed candidate wet-bin support: 0 m" in label for label in labels)
    assert any("summed inferred interval span: 0 m" in label for label in labels)
    assert any("outer candidate span: not defined" in label for label in labels)
    assert axes.get_xlim() == pytest.approx((0.0, sample.transect_length_m))
    assert "not validated physical banks" in axes.get_title()
    assert "unsampled ≠ dry" in axes.get_title()
    legend_labels = [text.get_text() for text in axes.get_legend().get_texts()]
    assert legend_labels == [
        "candidate wet-support bin (classification evidence)",
        "sampled below threshold (not confirmed dry)",
        "unsampled (unknown; not confirmed dry)",
    ]
    assert result.audit_summary() == before


def test_invalid_color_requests_are_actionable() -> None:
    dataset = _pixel_dataset()

    with pytest.raises(ValueError, match="Unsupported color variable"):
        plot_pixc_map(dataset, color_by="branch_id")
    with pytest.raises(ValueError, match="require a QCResult"):
        plot_pixc_map(dataset, color_by="classification_qual:first")
    with pytest.raises(ValueError, match="west < east"):
        plot_pixc_map(dataset, extent=(87.2, 26.5, 86.9, 26.9))


def test_nonnumeric_coordinates_are_rejected_before_plotting() -> None:
    dataset = _pixel_dataset()
    dataset["longitude"] = xr.DataArray(
        np.asarray(["east"] * dataset.sizes["points"]),
        dims="points",
    )

    with pytest.raises(ValueError, match="real numeric coordinates"):
        plot_pixc_map(dataset)


@pytest.mark.parametrize(
    ("name", "values", "message"),
    [
        ("classification", ["water"] * 7, "integer point variable"),
        ("source_index", np.linspace(0.0, 1.0, 7), "integer point variable"),
        ("height", ["high"] * 7, "real numeric point variable"),
    ],
)
def test_plot_color_variables_require_scientifically_valid_dtypes(
    name: str,
    values: object,
    message: str,
) -> None:
    dataset = _pixel_dataset()
    dataset[name] = xr.DataArray(values, dims="points")

    with pytest.raises(ValueError, match=message):
        plot_pixc_map(dataset, color_by=name)
