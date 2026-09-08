from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from swot_pixc_lab import PixcCollection
from swot_pixc_lab.exceptions import (
    CacheIntegrityError,
    DownloadError,
    LocalFilesUnavailableError,
)


def _collection(backend):
    return PixcCollection.search(
        (-100, 40, -99, 41),
        "2025-01-01",
        "2025-01-31",
        backend=backend,
    )


def test_download_resolve_local_flow_is_transactional_and_preserves_provenance(
    tmp_path, granule_factory, fake_backend_class
) -> None:
    raw = granule_factory()
    backend = fake_backend_class([[raw]])
    collection = _collection(backend)

    returned = collection.download(tmp_path, show_progress=False)
    local = collection.resolve_local()

    assert returned is collection
    assert len(backend.download_calls) == 1
    assert backend.download_calls[0]["auth_strategy"] == "all"
    assert len(local.paths) == 1
    assert isinstance(local.paths[0], Path)
    assert local.paths[0].is_absolute()
    assert local.paths[0].read_bytes() == b"data"
    assert collection.table.loc[0, "local_path"] == str(local.paths[0])


def test_existing_verified_cache_skips_login_and_download(
    tmp_path, granule_factory, fake_backend_class
) -> None:
    raw = granule_factory()
    filename = raw["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]
    (tmp_path / filename).write_bytes(b"data")
    backend = fake_backend_class([[raw]])
    collection = _collection(backend)

    collection.download(tmp_path)

    assert backend.download_calls == []
    assert collection.resolve_local().paths == ((tmp_path / filename).resolve(),)


def test_second_download_is_idempotent(
    tmp_path, granule_factory, fake_backend_class
) -> None:
    raw = granule_factory()
    backend = fake_backend_class([[raw]])
    collection = _collection(backend)

    collection.download(tmp_path)
    collection.download(tmp_path)

    assert len(backend.download_calls) == 1


def test_invalid_existing_cache_is_never_silently_reused(
    tmp_path, granule_factory, fake_backend_class
) -> None:
    raw = granule_factory()
    filename = raw["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]
    (tmp_path / filename).write_bytes(b"wrong-size")
    collection = _collection(fake_backend_class([[raw]]))

    with pytest.raises(CacheIntegrityError, match="size mismatch"):
        collection.download(tmp_path)


def test_default_auto_verification_detects_same_size_corruption(
    tmp_path, granule_factory, fake_backend_class
) -> None:
    raw = granule_factory()
    filename = raw["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]
    (tmp_path / filename).write_bytes(b"fail")
    collection = _collection(fake_backend_class([[raw]]))

    with pytest.raises(CacheIntegrityError, match="checksum mismatch"):
        collection.download(tmp_path)


def test_auto_verification_falls_back_to_size_when_checksum_is_absent(
    tmp_path, granule_factory, fake_backend_class
) -> None:
    raw = granule_factory()
    archive = raw["umm"]["DataGranule"]["ArchiveAndDistributionInformation"]
    netcdf_entry = next(item for item in archive if item["Name"].endswith(".nc"))
    netcdf_entry.pop("Checksum")
    backend = fake_backend_class([[raw]])
    collection = _collection(backend)

    collection.download(tmp_path)

    assert len(collection.resolve_local().paths) == 1


def test_explicit_checksum_verification_requires_cmr_checksum(
    tmp_path, granule_factory, fake_backend_class
) -> None:
    raw = granule_factory()
    archive = raw["umm"]["DataGranule"]["ArchiveAndDistributionInformation"]
    netcdf_entry = next(item for item in archive if item["Name"].endswith(".nc"))
    netcdf_entry.pop("Checksum")
    filename = raw["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]
    (tmp_path / filename).write_bytes(b"data")
    collection = _collection(fake_backend_class([[raw]]))

    with pytest.raises(CacheIntegrityError, match="does not provide a checksum"):
        collection.resolve_local(tmp_path, verify="checksum")


def test_resolve_local_is_local_only_and_actionable(
    tmp_path, granule_factory, fake_backend_class
) -> None:
    raw = granule_factory()
    backend = fake_backend_class([[raw]])
    collection = _collection(backend)

    with pytest.raises(LocalFilesUnavailableError, match=r"Call download\(\)"):
        collection.resolve_local(tmp_path)
    assert backend.download_calls == []


def test_resolve_local_can_resolve_an_existing_cache_without_download(
    tmp_path, granule_factory, fake_backend_class
) -> None:
    raw = granule_factory()
    filename = raw["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]
    (tmp_path / filename).write_bytes(b"data")
    backend = fake_backend_class([[raw]])
    collection = _collection(backend)

    local = collection.resolve_local(tmp_path)

    assert local.paths == ((tmp_path / filename).resolve(),)
    assert backend.download_calls == []


def test_downloader_failure_is_wrapped_without_credentials(
    tmp_path, granule_factory, fake_backend_class
) -> None:
    raw = granule_factory()
    collection = _collection(fake_backend_class([[raw]], fail_download=True))

    with pytest.raises(DownloadError, match="Earthdata Login") as error:
        collection.download(tmp_path)

    assert "password" not in str(error.value).lower()


def test_missing_data_url_fails_before_authentication(
    tmp_path, granule_factory, fake_backend_class
) -> None:
    raw = granule_factory(related_urls=[])
    backend = fake_backend_class([[raw]])
    collection = _collection(backend)

    with pytest.raises(DownloadError, match="no downloadable NetCDF URL"):
        collection.download(tmp_path)

    assert backend.download_calls == []


def test_partial_downloader_result_is_not_committed(
    tmp_path, granule_factory, fake_backend_class
) -> None:
    raw = granule_factory()
    filename = raw["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]
    collection = _collection(fake_backend_class([[raw]], omit_downloads=True))

    with pytest.raises(DownloadError, match="was not created"):
        collection.download(tmp_path)

    assert not (tmp_path / filename).exists()


def test_empty_collection_download_and_resolve_local_are_noops(
    tmp_path, fake_backend_class
) -> None:
    backend = fake_backend_class([[]])
    collection = _collection(backend)

    local = collection.download(tmp_path).resolve_local()

    assert len(local) == 0
    assert local.paths == ()
    assert backend.download_calls == []


def test_filename_collision_is_rejected(
    tmp_path, granule_factory, fake_backend_class
) -> None:
    first = granule_factory(concept_id="G1")
    second = granule_factory(concept_id="G2")
    backend = fake_backend_class([[first, second]])
    collection = _collection(backend)

    with pytest.raises(DownloadError, match="same cache filename"):
        collection.download(tmp_path)


def test_cache_commit_rolls_back_if_a_later_link_fails(
    tmp_path, granule_factory, fake_backend_class, monkeypatch
) -> None:
    first = granule_factory(concept_id="G1", cycle=12)
    second = granule_factory(concept_id="G2", cycle=13)
    backend = fake_backend_class([[first, second]])
    collection = _collection(backend)
    filenames = [
        raw["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]
        for raw in (first, second)
    ]
    real_link = os.link
    calls = 0

    def fail_second_link(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(errno.EIO, "synthetic commit failure")
        real_link(source, target)

    monkeypatch.setattr("swot_pixc_lab.download.os.link", fail_second_link)

    with pytest.raises(DownloadError, match="rolled back"):
        collection.download(tmp_path)

    assert all(not (tmp_path / filename).exists() for filename in filenames)


def test_cache_commit_copies_exclusively_when_hard_links_are_unsupported(
    tmp_path, granule_factory, fake_backend_class, monkeypatch
) -> None:
    raw = granule_factory()
    filename = raw["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]
    collection = _collection(fake_backend_class([[raw]]))

    def unsupported_link(source, target):
        raise OSError(errno.ENOTSUP, "hard links are not supported")

    monkeypatch.setattr("swot_pixc_lab.download.os.link", unsupported_link)

    collection.download(tmp_path)

    target = (tmp_path / filename).resolve()
    assert target.read_bytes() == b"data"
    assert collection.resolve_local().paths == (target,)


def test_fallback_copy_is_verified_and_removed_if_corrupted(
    tmp_path, granule_factory, fake_backend_class, monkeypatch
) -> None:
    raw = granule_factory()
    filename = raw["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]
    collection = _collection(fake_backend_class([[raw]]))

    def unsupported_link(source, target):
        raise OSError(errno.ENOTSUP, "hard links are not supported")

    def copy_corrupt_data(source, target, *, length):
        target.write(b"fail")

    monkeypatch.setattr("swot_pixc_lab.download.os.link", unsupported_link)
    monkeypatch.setattr(
        "swot_pixc_lab.download.shutil.copyfileobj",
        copy_corrupt_data,
    )

    with pytest.raises(CacheIntegrityError, match="checksum mismatch"):
        collection.download(tmp_path)

    assert not (tmp_path / filename).exists()


def test_fallback_copy_is_rolled_back_if_a_later_commit_fails(
    tmp_path, granule_factory, fake_backend_class, monkeypatch
) -> None:
    first = granule_factory(concept_id="G1", cycle=12)
    second = granule_factory(concept_id="G2", cycle=13)
    collection = _collection(fake_backend_class([[first, second]]))
    filenames = [
        raw["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]
        for raw in (first, second)
    ]
    calls = 0

    def fail_after_fallback(source, target):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(errno.ENOTSUP, "hard links are not supported")
        raise OSError(errno.EIO, "synthetic commit failure")

    monkeypatch.setattr("swot_pixc_lab.download.os.link", fail_after_fallback)

    with pytest.raises(DownloadError, match="rolled back"):
        collection.download(tmp_path)

    assert all(not (tmp_path / filename).exists() for filename in filenames)


def test_fallback_rollback_preserves_a_concurrently_replaced_target(
    tmp_path, granule_factory, fake_backend_class, monkeypatch
) -> None:
    first = granule_factory(concept_id="G1", cycle=12)
    second = granule_factory(concept_id="G2", cycle=13)
    collection = _collection(fake_backend_class([[first, second]]))
    first_filename = first["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]
    first_target = tmp_path / first_filename
    calls = 0

    def replace_before_failure(source, target):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(errno.ENOTSUP, "hard links are not supported")
        first_target.unlink()
        first_target.write_bytes(b"external")
        raise OSError(errno.EIO, "synthetic commit failure")

    monkeypatch.setattr("swot_pixc_lab.download.os.link", replace_before_failure)

    with pytest.raises(DownloadError, match="rolled back"):
        collection.download(tmp_path)

    assert first_target.read_bytes() == b"external"


def test_cache_commit_never_clobbers_a_concurrent_target(
    tmp_path, granule_factory, fake_backend_class, monkeypatch
) -> None:
    raw = granule_factory()
    filename = raw["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]
    backend = fake_backend_class([[raw]])
    collection = _collection(backend)
    real_link = os.link

    def create_racing_target(source, target):
        Path(target).write_bytes(b"external")
        real_link(source, target)

    monkeypatch.setattr(
        "swot_pixc_lab.download.os.link",
        create_racing_target,
    )

    with pytest.raises(DownloadError, match="appeared during download"):
        collection.download(tmp_path)

    assert (tmp_path / filename).read_bytes() == b"external"


def test_fallback_copy_never_clobbers_a_concurrent_target(
    tmp_path, granule_factory, fake_backend_class, monkeypatch
) -> None:
    raw = granule_factory()
    filename = raw["umm"]["DataGranule"]["Identifiers"][0]["Identifier"]
    target = tmp_path / filename
    collection = _collection(fake_backend_class([[raw]]))
    real_open = Path.open

    def unsupported_link(source, target):
        raise OSError(errno.ENOTSUP, "hard links are not supported")

    def create_racing_target(path, mode="r", *args, **kwargs):
        if path == target and mode == "xb":
            path.write_bytes(b"external")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("swot_pixc_lab.download.os.link", unsupported_link)
    monkeypatch.setattr(Path, "open", create_racing_target)

    with pytest.raises(DownloadError, match="appeared during download"):
        collection.download(tmp_path)

    assert target.read_bytes() == b"external"
