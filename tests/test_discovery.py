from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from swot_pixc_lab import DEFAULT_PIXC_COLLECTION_CONCEPT_ID, PixcCollection
from swot_pixc_lab.constants import METADATA_COLUMNS
from swot_pixc_lab.discovery import parse_granule, safe_filename
from swot_pixc_lab.exceptions import DiscoveryError


def _multipolygon():
    return {
        "type": "MultiPolygon",
        "coordinates": [
            [[[-100, 40], [-99, 40], [-99, 41], [-100, 41], [-100, 40]]],
            [[[-90, 30], [-89, 30], [-89, 31], [-90, 31], [-90, 30]]],
        ],
    }


def test_collection_reserves_open_for_phase2() -> None:
    assert callable(PixcCollection.resolve_local)
    assert not hasattr(PixcCollection, "open")


def test_search_queries_components_deduplicates_and_sorts(
    granule_factory, fake_backend_class
) -> None:
    later = granule_factory(concept_id="G2", start="2025-01-03T03:04:05Z")
    earlier = granule_factory(concept_id="G1", start="2025-01-01T03:04:05Z")
    backend = fake_backend_class([[later], [later, earlier]])

    collection = PixcCollection.search(
        _multipolygon(),
        "2025-01-01",
        "2025-01-31",
        backend=backend,
    )

    assert len(collection) == 2
    assert collection.table["concept_id"].tolist() == ["G1", "G2"]
    assert len(backend.search_calls) == 2
    assert all(call["count"] == -1 for call in backend.search_calls)
    assert all(
        call["concept_id"] == DEFAULT_PIXC_COLLECTION_CONCEPT_ID
        for call in backend.search_calls
    )
    assert backend.search_calls[0]["temporal"] == (
        "2025-01-01T00:00:00Z",
        "2025-01-31T23:59:59.999999Z",
    )


def test_collection_concept_id_can_be_pinned_explicitly(fake_backend_class) -> None:
    backend = fake_backend_class([[]])

    collection = PixcCollection.search(
        (-100, 40, -99, 41),
        "2025-01-01",
        "2025-01-02",
        collection_concept_id="C123-TEST",
        backend=backend,
    )

    assert collection.collection_concept_id == "C123-TEST"
    assert backend.search_calls[0]["concept_id"] == "C123-TEST"


def test_failed_component_does_not_return_partial_results(
    granule_factory, fake_backend_class
) -> None:
    backend = fake_backend_class([[granule_factory()]], fail_search_call=2)

    with pytest.raises(DiscoveryError, match="no partial collection"):
        PixcCollection.search(
            _multipolygon(),
            "2025-01-01",
            "2025-01-31",
            backend=backend,
        )


def test_metadata_parser_preserves_cmr_fields(granule_factory) -> None:
    raw = granule_factory()

    record = parse_granule(raw)

    assert record.observation_start == datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert record.cycle == 12
    assert record.pass_number == 34
    assert record.tile == "056L"
    assert record.product_short_name == "SWOT_L2_HR_PIXC_D"
    assert record.product_version == "D"
    assert record.native_id == record.granule_id
    assert record.revision_id == 2
    assert record.revision_date == datetime(2025, 5, 2, 12, tzinfo=UTC)
    assert record.metadata_specification_version == "1.6.7"
    assert record.source_url.endswith(".nc")
    assert record.s3_url.startswith("s3://")
    assert record.size_bytes == 4
    assert record.checksum_algorithm == "MD5"
    assert record.granule_bbox == (-100.0, 40.0, -99.0, 41.0)
    assert record.metadata_warnings == ()


def test_track_fields_use_conservative_official_filename_fallback(
    granule_factory,
) -> None:
    record = parse_granule(granule_factory(include_track=False))

    assert (record.cycle, record.pass_number, record.tile) == (12, 34, "056L")
    assert len(record.metadata_warnings) == 3
    assert all("official PIXC filename" in item for item in record.metadata_warnings)


def test_missing_optional_metadata_stays_null() -> None:
    raw = {
        "meta": {"concept-id": "G-MINIMAL"},
        "umm": {
            "GranuleUR": "unknown-granule",
            "CollectionReference": {"ShortName": "SWOT_L2_HR_PIXC_D"},
        },
    }

    record = parse_granule(raw)

    assert record.filename is None
    assert record.cycle is None
    assert record.source_url is None
    assert record.size_bytes is None


def test_data_link_selection_is_netcdf_only_and_deterministic(granule_factory) -> None:
    filename = "SWOT_L2_HR_PIXC_012_034_056L_20250102T030405_20250102T030412_PID0_01.nc"
    links = [
        {"URL": "https://example.test/b.png", "Type": "GET RELATED VISUALIZATION"},
        {"URL": f"https://z.test/{filename}", "Type": "GET DATA"},
        {"URL": f"https://a.test/{filename}?token=ignored", "Type": "GET DATA"},
        {"URL": f"s3://bucket/{filename}", "Type": "GET DATA VIA DIRECT ACCESS"},
    ]

    record = parse_granule(granule_factory(related_urls=links))

    assert record.source_url.startswith("https://a.test/")
    assert record.filename == filename


@pytest.mark.parametrize("filename", ["../bad.nc", "folder/bad.nc", "folder\\bad.nc"])
def test_unsafe_filenames_are_rejected(filename) -> None:
    with pytest.raises(DiscoveryError, match="Unsafe"):
        safe_filename(filename)


def test_empty_collection_has_stable_table_schema(fake_backend_class) -> None:
    collection = PixcCollection.search(
        (-100, 40, -99, 41),
        "2025-01-01",
        "2025-01-02",
        backend=fake_backend_class([[]]),
    )

    table = collection.table
    assert tuple(table.columns) == METADATA_COLUMNS
    assert table.empty
    assert isinstance(table["observation_datetime"].dtype, pd.DatetimeTZDtype)
    assert str(table["cycle"].dtype) == "Int64"
    assert str(table["concept_id"].dtype) == "string"


def test_search_provenance_can_reconstruct_normalized_query(fake_backend_class) -> None:
    collection = PixcCollection.search(
        (170, -10, -175, 10),
        "2025-01-01",
        "2025-01-02",
        backend=fake_backend_class([[], []]),
    )

    assert collection.provenance is not None
    assert collection.provenance.as_dict() == {
        "collection_concept_id": DEFAULT_PIXC_COLLECTION_CONCEPT_ID,
        "temporal": [
            "2025-01-01T00:00:00Z",
            "2025-01-02T23:59:59.999999Z",
        ],
        "spatial_queries": [
            {"bounding_box": (170.0, -10.0, 180.0, 10.0)},
            {"bounding_box": (-180.0, -10.0, -175.0, 10.0)},
        ],
        "notes": [
            "Antimeridian-crossing bounding box was split into two CMR searches."
        ],
        "count": -1,
    }


def test_collection_id_format_is_validated_before_network(fake_backend_class) -> None:
    backend = fake_backend_class()

    with pytest.raises(ValueError, match="CMR collection ID"):
        PixcCollection.search(
            (-100, 40, -99, 41),
            "2025-01-01",
            "2025-01-02",
            collection_concept_id="C-not-an-id",
            backend=backend,
        )

    assert backend.search_calls == []


def test_non_pixc_collection_result_is_rejected(
    granule_factory, fake_backend_class
) -> None:
    raw = granule_factory()
    raw["umm"]["CollectionReference"]["ShortName"] = "ATL06"

    with pytest.raises(DiscoveryError, match="non-PIXC product"):
        PixcCollection.search(
            (-100, 40, -99, 41),
            "2025-01-01",
            "2025-01-02",
            collection_concept_id="C123-TEST",
            backend=fake_backend_class([[raw]]),
        )


def test_table_is_a_defensive_copy(granule_factory, fake_backend_class) -> None:
    collection = PixcCollection.search(
        (-100, 40, -99, 41),
        "2025-01-01",
        "2025-01-02",
        backend=fake_backend_class([[granule_factory()]]),
    )

    first = collection.table
    first.loc[0, "tile"] = "changed"

    assert collection.table.loc[0, "tile"] == "056L"
