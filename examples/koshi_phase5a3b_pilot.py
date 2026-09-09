"""Build the guarded Phase 5A.3b Koshi pilot benchmark.

This cache-only script opens the configured observation once, applies QC once,
and reuses the result for transect proposals, annotation packets, and eventual
validation. It never searches for or downloads PIXC or reference imagery.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from swot_pixc_lab import (
    PHASE3_POINT_VARIABLES,
    apply_qc,
    generate_annotation_packets,
    load_benchmark_config,
    load_manual_annotations,
    load_transect_manifest,
    open_pixc,
    propose_candidate_transects,
    run_benchmark,
    write_transect_manifest,
)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("koshi_phase5a3b_pilot_config.json")
ANNOTATION_REQUIRED_MESSAGE = (
    "Annotation packets created. Manual reference annotations are required "
    "before quantitative validation can run."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"benchmark JSON (default: {DEFAULT_CONFIG_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    """Prepare packets and run validation only after both human gates pass."""

    args = _parse_args()
    config = load_benchmark_config(args.config)
    missing = [path for path in config.observation.local_files if not path.is_file()]
    if missing:
        missing_lines = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(
            "Required local Koshi PIXC file(s) are missing:\n"
            f"{missing_lines}\n"
            "This script never searches for or downloads data. Restore only the "
            "listed file(s), then rerun."
        )

    configurations = config.sensitivity_configurations
    print("PHASE 5A.3b KOSHI PILOT: cache-only, exploratory, and unranked.")
    print(f"Configured sensitivity grid: {len(configurations)} configurations")
    print("Opening the configured observation once...")
    observation = open_pixc(
        config.observation.local_files,
        aoi=config.observation.aoi,
        variables=PHASE3_POINT_VARIABLES,
    )
    print(f"Applying QC profile {config.qc_profile!r} once...")
    qc_result = apply_qc(observation, profile=config.qc_profile)

    if config.transect_manifest_path.is_file():
        manifest = load_transect_manifest(config.transect_manifest_path)
        print(f"Loaded transect manifest: {config.transect_manifest_path}")
    else:
        if not config.transect_proposals.enabled:
            raise SystemExit(
                "No transect manifest exists and deterministic proposals are "
                "disabled. Supply an owner-reviewed EPSG:4326 manifest."
            )
        manifest = propose_candidate_transects(
            qc_result,
            benchmark_id=config.benchmark_id,
            settings=config.transect_proposals,
            corridor_half_width_m=config.corridor_half_width_m,
            diagnostic_configuration=config.packet_configuration,
        )
        write_transect_manifest(manifest, config.transect_manifest_path)
        print(
            f"Wrote {len(manifest.transects)} deterministic proposals to "
            f"{config.transect_manifest_path}"
        )
        print("Every proposal is PROPOSED / REQUIRES SCIENTIST REVIEW.")

    preparation = generate_annotation_packets(qc_result, config, manifest)
    print(f"Generated {len(preparation.packets)} annotation packet(s):")
    print(config.output_directory / "annotation_packets")
    print(f"Preparation manifest: {preparation.preparation_manifest_path}")

    unapproved = [
        transect.transect_id for transect in manifest.transects if not transect.approved
    ]
    if unapproved:
        print(
            "Scientist approval is still required before these transects may be "
            "annotated or validated: " + ", ".join(unapproved)
        )
        print(
            "Review the manifest and independent imagery; edit or replace each "
            "geometry as needed, then explicitly set review_status to APPROVED."
        )

    annotations = load_manual_annotations(
        config.annotation_directory,
        benchmark_id=config.benchmark_id,
    )
    if not annotations:
        print(ANNOTATION_REQUIRED_MESSAGE)
        print(f"Save completed annotations under: {config.annotation_directory}")
        return

    result = run_benchmark(config, manifest, preparation.samples, annotations)
    print(
        f"Completed {len(result.individual_results)} annotation × configuration "
        "evaluations without ranking configurations."
    )
    print(f"Individual results: {result.results_csv}")
    print(f"Aggregate descriptions: {result.aggregate_csv}")
    print(f"Report: {result.report_path}")
    print(f"Run manifest: {result.manifest_path}")


if __name__ == "__main__":
    main()
