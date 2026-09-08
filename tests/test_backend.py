from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from swot_pixc_lab.discovery import EarthaccessBackend


class _FakeEarthaccess:
    def __init__(self) -> None:
        self.search_calls = []
        self.login_calls = []
        self.download_calls = []

    def search_data(self, **kwargs):
        self.search_calls.append(kwargs)
        return ["result"]

    def login(self, **kwargs):
        self.login_calls.append(kwargs)
        return SimpleNamespace(authenticated=True)

    def download(self, granules, destination, **kwargs):
        self.download_calls.append((granules, destination, kwargs))
        return [Path(destination) / "granule.nc"]


def test_earthaccess_backend_forwards_current_public_api(monkeypatch) -> None:
    module = _FakeEarthaccess()
    monkeypatch.setattr(
        "swot_pixc_lab.discovery._import_earthaccess",
        lambda: module,
    )
    backend = EarthaccessBackend()

    assert backend.search_data(count=2, concept_id="C123-TEST") == ["result"]
    assert module.search_calls == [{"count": 2, "concept_id": "C123-TEST"}]

    paths = backend.download(
        ["granule"],
        Path("staging"),
        auth_strategy="all",
        persist_credentials=False,
        threads=3,
        show_progress=False,
    )

    assert paths == [Path("staging/granule.nc")]
    assert module.login_calls == [{"strategy": "all", "persist": False}]
    assert module.download_calls == [
        (
            ["granule"],
            Path("staging"),
            {"threads": 3, "show_progress": False},
        )
    ]


def test_earthaccess_backend_can_reuse_an_existing_login(monkeypatch) -> None:
    module = _FakeEarthaccess()
    monkeypatch.setattr(
        "swot_pixc_lab.discovery._import_earthaccess",
        lambda: module,
    )

    EarthaccessBackend().download(
        ["granule"],
        Path("staging"),
        auth_strategy=None,
        persist_credentials=False,
        threads=1,
        show_progress=None,
    )

    assert module.login_calls == []
