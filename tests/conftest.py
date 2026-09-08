from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def granule_factory():
    def make_granule(
        *,
        concept_id: str = "G100-POCLOUD",
        cycle: int = 12,
        pass_number: int = 34,
        tile: str = "056L",
        start: str = "2025-01-02T03:04:05Z",
        payload: bytes = b"data",
        include_track: bool = True,
        related_urls: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(UTC)
        end_dt = start_dt.replace(second=min(start_dt.second + 7, 59))
        timestamp = start_dt.strftime("%Y%m%dT%H%M%S")
        end_timestamp = end_dt.strftime("%Y%m%dT%H%M%S")
        filename = (
            f"SWOT_L2_HR_PIXC_{cycle:03d}_{pass_number:03d}_{tile}_"
            f"{timestamp}_{end_timestamp}_PID0_01.nc"
        )
        granule_ur = filename.removesuffix(".nc")
        if related_urls is None:
            related_urls = [
                {
                    "URL": f"https://example.test/{filename}.iso.xml",
                    "Type": "EXTENDED METADATA",
                },
                {
                    "URL": f"s3://bucket/{filename}",
                    "Type": "GET DATA VIA DIRECT ACCESS",
                },
                {
                    "URL": f"https://example.test/{filename}",
                    "Type": "GET DATA",
                },
            ]
        horizontal: dict[str, Any] = {
            "Geometry": {
                "BoundingRectangles": [
                    {
                        "WestBoundingCoordinate": -100.0,
                        "SouthBoundingCoordinate": 40.0,
                        "EastBoundingCoordinate": -99.0,
                        "NorthBoundingCoordinate": 41.0,
                    }
                ]
            }
        }
        if include_track:
            horizontal["Track"] = {
                "Cycle": cycle,
                "Passes": [{"Pass": pass_number, "Tiles": [tile]}],
            }
        return {
            "meta": {
                "concept-id": concept_id,
                "native-id": granule_ur,
                "collection-concept-id": "C3233944986-POCLOUD",
                "provider-id": "POCLOUD",
                "revision-id": 2,
                "revision-date": "2025-05-02T12:00:00Z",
            },
            "umm": {
                "GranuleUR": granule_ur,
                "TemporalExtent": {
                    "RangeDateTime": {
                        "BeginningDateTime": start,
                        "EndingDateTime": end_dt.isoformat().replace("+00:00", "Z"),
                    }
                },
                "SpatialExtent": {"HorizontalSpatialDomain": horizontal},
                "CollectionReference": {
                    "ShortName": "SWOT_L2_HR_PIXC_D",
                    "Version": "D",
                },
                "PGEVersionClass": {
                    "PGEName": "PGE_L2_HR_PIXC",
                    "PGEVersion": "5.4.2",
                },
                "RelatedUrls": related_urls,
                "DataGranule": {
                    "ProductionDateTime": "2025-05-01T12:00:00Z",
                    "Identifiers": [
                        {
                            "Identifier": filename,
                            "IdentifierType": "ProducerGranuleId",
                        }
                    ],
                    "ArchiveAndDistributionInformation": [
                        {
                            "Name": f"{granule_ur}.log",
                            "SizeInBytes": 11,
                        },
                        {
                            "Name": filename,
                            "SizeInBytes": len(payload),
                            "Checksum": {
                                "Value": hashlib.md5(payload).hexdigest(),
                                "Algorithm": "MD5",
                            },
                        },
                        {
                            "Name": f"{filename}.iso.xml",
                            "SizeInBytes": 12,
                        },
                    ],
                },
                "MetadataSpecification": {
                    "Name": "UMM-G",
                    "Version": "1.6.7",
                },
            },
            "_test_payload": payload,
        }

    return make_granule


class FakeBackend:
    def __init__(
        self,
        search_results: list[list[dict[str, Any]]] | None = None,
        *,
        fail_search_call: int | None = None,
        fail_download: bool = False,
        omit_downloads: bool = False,
    ) -> None:
        self.search_results = search_results or []
        self.fail_search_call = fail_search_call
        self.fail_download = fail_download
        self.omit_downloads = omit_downloads
        self.search_calls: list[dict[str, Any]] = []
        self.download_calls: list[dict[str, Any]] = []

    def search_data(self, *, count: int = -1, **kwargs: Any):
        self.search_calls.append({"count": count, **kwargs})
        call_number = len(self.search_calls)
        if call_number == self.fail_search_call:
            raise RuntimeError("synthetic CMR failure")
        if call_number <= len(self.search_results):
            return self.search_results[call_number - 1]
        return []

    def download(
        self,
        granules,
        destination: Path,
        *,
        auth_strategy: str | None,
        persist_credentials: bool,
        threads: int,
        show_progress: bool | None,
    ):
        self.download_calls.append(
            {
                "granules": list(granules),
                "destination": destination,
                "auth_strategy": auth_strategy,
                "persist_credentials": persist_credentials,
                "threads": threads,
                "show_progress": show_progress,
            }
        )
        if self.fail_download:
            raise RuntimeError("synthetic auth failure")
        if self.omit_downloads:
            return []
        paths: list[Path] = []
        for raw in granules:
            identifier = raw["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]
            path = destination / identifier
            path.write_bytes(raw["_test_payload"])
            paths.append(path)
        return paths


@pytest.fixture
def fake_backend_class():
    return FakeBackend
