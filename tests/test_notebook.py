from __future__ import annotations

import ast
from pathlib import Path

import nbformat


def test_phase1_demo_notebook_is_valid_and_guarded() -> None:
    notebook_path = (
        Path(__file__).resolve().parents[1] / "notebooks" / "01_pixc_aoi_demo.ipynb"
    )
    notebook = nbformat.read(notebook_path, as_version=4)

    nbformat.validate(notebook)
    code = "\n\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    ast.parse(code)

    assert "OWNER_AOI = None" in code
    assert "DOWNLOAD_FULL_TILES = False" in code
    assert "WRITE_PROVENANCE_FILES = False" in code
    assert "collection.resolve_local(" in code
    assert "collection.open(" not in code


def test_phase4_demo_notebook_is_valid_guarded_and_uses_manual_transect() -> None:
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "02_pixc_visualization_and_transect_demo.ipynb"
    )
    notebook = nbformat.read(notebook_path, as_version=4)

    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    code = "\n\n".join(cell.source for cell in code_cells)
    ast.parse(code)

    assert "LOCAL_OBSERVATION_GROUPS" in code
    assert "available_observation_groups" in code
    assert "for label, paths in LOCAL_OBSERVATION_GROUPS.items()" in code
    assert "download(" not in code
    assert "SYNTHETIC_ENDPOINTS" in code
    assert "corridor_half_width_m=50.0" in code
    assert "sample_transect(" in code
    assert "channel_width" not in code
    assert all(cell.execution_count is None for cell in code_cells)
    assert all(not cell.outputs for cell in code_cells)
