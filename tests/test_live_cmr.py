from __future__ import annotations

import os

import pytest

from swot_pixc_lab import PixcCollection


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("SWOT_PIXC_LIVE_TEST") != "1",
    reason="set SWOT_PIXC_LIVE_TEST=1 to contact NASA CMR",
)
def test_live_version_d_cmr_search() -> None:
    collection = PixcCollection.search(
        (27.6, 33.8, 28.6, 34.4),
        "2023-03-28",
        "2023-03-28",
    )

    assert len(collection) >= 1
    table = collection.table
    assert set(table["product_short_name"]) == {"SWOT_L2_HR_PIXC_D"}
    row = table.iloc[0]
    assert row["filename"].endswith(".nc")
    assert row["cycle"] is not None
    assert row["pass"] is not None
    assert row["tile"]
    assert row["size_bytes"] > 0
    assert row["checksum"]
    assert row["source_url"].startswith("https://")
    assert row["s3_url"].startswith("s3://")
    assert row["granule_bbox"] is not None
    assert row["pge_version"]
    assert row["revision_id"] > 0
    assert row["revision_date"] is not None
