from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from swot_pixc_lab.aoi import normalize_aoi, normalize_temporal_bounds
from swot_pixc_lab.exceptions import AoiError, DateRangeError


def test_bbox_normalization() -> None:
    normalized = normalize_aoi((-100, 40, -99, 41))

    assert normalized.notes == ()
    assert [query.as_kwargs() for query in normalized.queries] == [
        {"bounding_box": (-100.0, 40.0, -99.0, 41.0)}
    ]


def test_antimeridian_bbox_is_split_without_global_broadening() -> None:
    normalized = normalize_aoi((170, -10, -175, 10))

    assert [query.as_kwargs() for query in normalized.queries] == [
        {"bounding_box": (170.0, -10.0, 180.0, 10.0)},
        {"bounding_box": (-180.0, -10.0, -175.0, 10.0)},
    ]
    assert "Antimeridian" in normalized.notes[0]


@pytest.mark.parametrize(
    "aoi",
    [
        (-100, 40, -100, 41),
        (-100, 41, -99, 40),
        (-181, 40, -99, 41),
        (-100, -91, -99, 41),
        (-100, 40, float("nan"), 41),
        (-100, True, -99, 41),
    ],
)
def test_invalid_bboxes_are_rejected(aoi) -> None:
    with pytest.raises(AoiError):
        normalize_aoi(aoi)


def test_polygon_is_closed_and_counter_clockwise() -> None:
    clockwise = {
        "type": "Polygon",
        "coordinates": [[[-100, 40], [-100, 41], [-99, 41], [-99, 40], [-100, 40]]],
    }

    coordinates = normalize_aoi(clockwise).queries[0].coordinates
    assert coordinates[0] == coordinates[-1]
    signed_area = sum(
        left[0] * right[1] - right[0] * left[1]
        for left, right in zip(coordinates, coordinates[1:], strict=False)
    )
    assert signed_area > 0


def test_multipolygon_becomes_separate_queries() -> None:
    aoi = {
        "type": "MultiPolygon",
        "coordinates": [
            [[[-100, 40], [-99, 40], [-99, 41], [-100, 41], [-100, 40]]],
            [[[-90, 30], [-89, 30], [-89, 31], [-90, 31], [-90, 30]]],
        ],
    }

    assert len(normalize_aoi(aoi).queries) == 2


def test_polygon_holes_are_disclosed() -> None:
    aoi = {
        "type": "Polygon",
        "coordinates": [
            [[-100, 40], [-98, 40], [-98, 42], [-100, 42], [-100, 40]],
            [[-99.5, 40.5], [-99.5, 41], [-99, 41], [-99, 40.5], [-99.5, 40.5]],
        ],
    }

    with pytest.warns(UserWarning, match="cannot encode 1 AOI hole"):
        normalized = normalize_aoi(aoi)
    assert "searched conservatively" in normalized.notes[0]


def test_unsplit_antimeridian_polygon_is_rejected() -> None:
    aoi = {
        "type": "Polygon",
        "coordinates": [[[170, 0], [-170, 0], [-170, 5], [170, 5], [170, 0]]],
    }

    with pytest.raises(AoiError, match="antimeridian"):
        normalize_aoi(aoi)


def test_feature_collection_rejects_non_polygon_geometry() -> None:
    aoi = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Point", "coordinates": [-100, 40]},
            }
        ],
    }

    with pytest.raises(AoiError, match="Polygon"):
        normalize_aoi(aoi)


class _FakeGeoDataFrame:
    geometry = object()

    def __init__(self, crs="EPSG:3857") -> None:
        self.crs = crs
        self.transformed_to = None

    def to_crs(self, *, epsg: int):
        self.transformed_to = epsg
        return self

    @property
    def __geo_interface__(self):
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[-100, 40], [-99, 40], [-99, 41], [-100, 41], [-100, 40]]
                        ],
                    },
                }
            ],
        }


def test_geodataframe_like_aoi_is_transformed() -> None:
    aoi = _FakeGeoDataFrame()

    normalized = normalize_aoi(aoi)

    assert aoi.transformed_to == 4326
    assert len(normalized.queries) == 1


def test_geodataframe_without_crs_is_rejected() -> None:
    with pytest.raises(AoiError, match="declare a CRS"):
        normalize_aoi(_FakeGeoDataFrame(crs=None))


def test_date_only_bounds_include_complete_end_day() -> None:
    assert normalize_temporal_bounds("2024-02-29", date(2024, 3, 1)) == (
        "2024-02-29T00:00:00Z",
        "2024-03-01T23:59:59.999999Z",
    )


def test_aware_datetimes_are_converted_to_utc() -> None:
    start = datetime.fromisoformat("2025-01-01T01:00:00+01:00")
    end = datetime(2025, 1, 1, 2, tzinfo=UTC)

    assert normalize_temporal_bounds(start, end) == (
        "2025-01-01T00:00:00Z",
        "2025-01-01T02:00:00Z",
    )


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2025-02-30", "2025-03-01"),
        ("", "2025-03-01"),
        ("2025-03-02", "2025-03-01"),
    ],
)
def test_invalid_temporal_bounds_are_rejected(start, end) -> None:
    with pytest.raises(DateRangeError):
        normalize_temporal_bounds(start, end)
