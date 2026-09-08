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
