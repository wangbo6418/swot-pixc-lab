from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_script_module(
    filename: str = "annotate_from_reference_imagery.py",
) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "examples" / filename
    module_name = f"{path.stem}_test_module"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_real_packet_shape_preserves_signed_temporal_offset(tmp_path: Path) -> None:
    module = _load_script_module()
    image = tmp_path / "rgb.tif"
    image.write_bytes(b"test raster placeholder")
    packet = tmp_path / "reference_imagery_metadata.json"
    packet.write_text(
        json.dumps(
            {
                "transect_id": "T002F",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[87.0, 26.6], [87.1, 26.5]],
                },
                "primary_candidate": {
                    "provider": "Element84 Earth Search",
                    "collection": "sentinel-2-l2a",
                    "acquisition_datetime": "2024-01-12T05:01:13.589Z",
                    "signed_offset_hours": -51.2247,
                    "item_ids": ["S2A_45RVK_20240112_0_L2A"],
                    "rgb_geotiff_path": str(image),
                    "chip_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    "local_quality": {"cloud_fraction": 0.001},
                },
                "candidates": [],
            }
        ),
        encoding="utf-8",
    )

    imagery, payload = module._selected_imagery(
        packet,
        transect_id="T002F",
        candidate_index=None,
    )

    assert payload["transect_id"] == "T002F"
    assert imagery.temporal_offset_hours == -51.2247
    assert imagery.local_image_path == image.resolve()
    assert imagery.source_identifier == ("sentinel-2-l2a:S2A_45RVK_20240112_0_L2A")


def test_preparation_loader_labels_t001_and_leaves_f_geojson_byte_unchanged(
    tmp_path: Path,
) -> None:
    module = _load_script_module("koshi_phase5a3c_reference_imagery.py")
    features = []
    for number in range(1, 8):
        transect_id = f"T{number:03d}F"
        features.append(
            {
                "type": "Feature",
                "id": transect_id,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [86.90 + number * 0.001, 26.50 + number * 0.01],
                        [86.96 + number * 0.001, 26.46 + number * 0.01],
                    ],
                },
                "properties": {
                    "transect_id": transect_id,
                    "review_status": "PROPOSED / REQUIRES SCIENTIST REVIEW",
                    "proposal_diagnostics": {"azimuth_deg": 130.0},
                },
            }
        )
    path = tmp_path / "refined_transects_proposed.geojson"
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    _, transects = module._load_proposed_f_transects(
        path,
        context_margin_m=1500.0,
    )

    assert transects[0].role == "AOI-EDGE / FAILURE-CONTROL CASE"
    assert transects[1].role == "PRIMARY BENCHMARK CANDIDATE"
    assert transects[-1].role == "CHALLENGE MORPHOLOGY CASE"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_owner_review_rerun_never_blanks_existing_human_decisions(
    tmp_path: Path,
) -> None:
    module = _load_script_module("koshi_phase5a3c_reference_imagery.py")
    path = tmp_path / "geometry_imagery_owner_review.csv"
    module._write_owner_review(path)
    contents = path.read_text(encoding="utf-8")
    path.write_text(
        contents.replace("T002F,,,", "T002F,APPROVE,APPROVE,reviewed"),
        encoding="utf-8",
    )
    before = path.read_bytes()

    module._write_owner_review(path)

    assert path.read_bytes() == before
