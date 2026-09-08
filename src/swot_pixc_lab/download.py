"""Transactional download and local-cache verification for PIXC granules."""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from .discovery import EarthdataBackend, GranuleRecord, safe_filename
from .exceptions import CacheIntegrityError, DiscoveryError, DownloadError

type VerificationMode = Literal["auto", "none", "size", "checksum"]
type _FileIdentity = tuple[int, int, int, int, int, int]

LOGGER = logging.getLogger(__name__)

_HARD_LINK_FALLBACK_ERRNOS = frozenset(
    value
    for name in (
        "EACCES",
        "EMLINK",
        "ENOSYS",
        "ENOTSUP",
        "EOPNOTSUPP",
        "EPERM",
        "EXDEV",
    )
    if (value := getattr(errno, name, None)) is not None
)
_HARD_LINK_FALLBACK_WINERRORS = frozenset({1, 17, 50})


@dataclass(frozen=True, slots=True)
class _CacheCommit:
    source: Path
    target: Path
    method: Literal["link", "copy"]
    copied_identity: _FileIdentity | None = None


def download_records(
    records: Sequence[GranuleRecord],
    handles: Mapping[str, Any],
    backend: EarthdataBackend,
    destination: str | os.PathLike[str],
    *,
    auth_strategy: str | None,
    persist_credentials: bool,
    threads: int,
    verify: VerificationMode,
    show_progress: bool | None,
) -> tuple[GranuleRecord, ...]:
    """Resolve cache hits and transactionally download all missing records."""

    _validate_verification_mode(verify)
    target_dir = Path(destination).expanduser().resolve()
    if target_dir.exists() and not target_dir.is_dir():
        raise DownloadError(f"Download destination is not a directory: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    _reject_filename_collisions(records)

    resolved: dict[str, GranuleRecord] = {}
    missing: list[GranuleRecord] = []
    for record in records:
        filename = _record_filename(record)
        target = target_dir / filename
        if target.exists():
            verify_cached_file(target, record, verify=verify)
            resolved[record.identity] = record.with_local_path(target)
        else:
            missing.append(record)

    if missing:
        missing_handles: list[Any] = []
        for record in missing:
            if not record.source_url and not record.s3_url:
                raise DownloadError(
                    f"Granule {record.granule_id or record.identity} has no "
                    "downloadable NetCDF URL in its CMR metadata."
                )
            try:
                missing_handles.append(handles[record.identity])
            except KeyError as exc:
                raise DownloadError(
                    f"No Earthdata handle is available for granule {record.identity}; "
                    "run a new search before downloading."
                ) from exc

        with TemporaryDirectory(
            prefix=".swot_pixc_lab_", dir=target_dir
        ) as staging_text:
            staging = Path(staging_text).resolve()
            try:
                returned = backend.download(
                    missing_handles,
                    staging,
                    auth_strategy=auth_strategy,
                    persist_credentials=persist_credentials,
                    threads=threads,
                    show_progress=show_progress,
                )
            except Exception as exc:
                raise DownloadError(
                    "Earthdata download failed. Check the network connection and "
                    "Earthdata Login configuration (.netrc, environment, or the "
                    "selected earthaccess login strategy)."
                ) from exc

            if returned is None:
                raise DownloadError("Earthdata downloader returned no path list.")
            returned_by_name = _index_returned_paths(returned, staging)
            staged: dict[str, Path] = {}
            for record in missing:
                filename = _record_filename(record)
                candidate = returned_by_name.get(filename, staging / filename)
                _require_inside_staging(candidate, staging)
                if not candidate.is_file():
                    raise DownloadError(
                        f"Earthdata reported success but granule was not created: "
                        f"{filename}"
                    )
                verify_cached_file(candidate, record, verify=verify)
                staged[record.identity] = candidate

            committed: list[_CacheCommit] = []
            try:
                for record in missing:
                    filename = _record_filename(record)
                    source = staged[record.identity]
                    target = target_dir / filename
                    commit = _commit_staged_file(source, target)
                    committed.append(commit)
                    if commit.method == "copy":
                        verify_cached_file(target, record, verify=verify)
            except Exception as exc:
                rollback_failures = _rollback_commits(committed)
                if rollback_failures:
                    raise DownloadError(
                        "Cache commit failed and rollback could not remove: "
                        + ", ".join(str(path) for path in rollback_failures)
                    ) from exc
                if isinstance(exc, DownloadError):
                    raise
                raise DownloadError(
                    "Cache commit failed; newly committed files were rolled back."
                ) from exc

            for record in missing:
                target = target_dir / _record_filename(record)
                resolved[record.identity] = record.with_local_path(target)

    return tuple(resolved[record.identity] for record in records)


def resolve_local_records(
    records: Sequence[GranuleRecord],
    *,
    cache_dir: str | os.PathLike[str] | None,
    verify: VerificationMode,
) -> tuple[GranuleRecord, ...]:
    """Resolve and verify records locally without making any network calls."""

    _validate_verification_mode(verify)
    base = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
    resolved: list[GranuleRecord] = []
    missing: list[str] = []
    for record in records:
        if base is not None:
            try:
                path = base / _record_filename(record)
            except DownloadError:
                missing.append(record.granule_id or record.identity)
                continue
        else:
            path = record.local_path
        if path is None or not path.is_file():
            missing.append(record.filename or record.granule_id or record.identity)
            continue
        verify_cached_file(path, record, verify=verify)
        resolved.append(record.with_local_path(path))

    if missing:
        from .exceptions import LocalFilesUnavailableError

        preview = ", ".join(missing[:3])
        suffix = " ..." if len(missing) > 3 else ""
        raise LocalFilesUnavailableError(
            f"{len(missing)} PIXC granule(s) are not available locally: "
            f"{preview}{suffix}. Call download() or pass the correct cache_dir."
        )
    return tuple(resolved)


def verify_cached_file(
    path: Path, record: GranuleRecord, *, verify: VerificationMode
) -> None:
    """Verify a regular, non-empty cache file against available CMR metadata."""

    if not path.is_file():
        raise CacheIntegrityError(f"Cached granule is not a regular file: {path}")
    actual_size = path.stat().st_size
    if actual_size <= 0:
        raise CacheIntegrityError(f"Cached granule is empty: {path}")

    checksum_available = bool(record.checksum and record.checksum_algorithm)
    effective_verify = verify
    if verify == "auto":
        effective_verify = "checksum" if checksum_available else "size"
        if not checksum_available and record.size_bytes is None:
            LOGGER.warning(
                "CMR provides neither checksum nor byte size for %s; only "
                "regular-file and non-empty checks were possible.",
                path.name,
            )

    if effective_verify in {"size", "checksum"} and record.size_bytes is not None:
        if actual_size != record.size_bytes:
            raise CacheIntegrityError(
                f"Cached granule size mismatch for {path.name}: expected "
                f"{record.size_bytes} bytes from CMR, found {actual_size}. "
                "Remove the invalid cache file and retry download()."
            )

    if effective_verify == "checksum":
        if not record.checksum or not record.checksum_algorithm:
            raise CacheIntegrityError(
                f"CMR does not provide a checksum for {path.name}; "
                "checksum verification cannot be completed."
            )
        algorithm = _hashlib_name(record.checksum_algorithm)
        try:
            digest = hashlib.new(algorithm)
        except ValueError as exc:
            if verify == "auto":
                LOGGER.warning(
                    "CMR checksum algorithm %s is unsupported for %s; "
                    "cache verification fell back to available size/non-empty checks.",
                    record.checksum_algorithm,
                    path.name,
                )
                return
            raise CacheIntegrityError(
                f"Unsupported CMR checksum algorithm for {path.name}: "
                f"{record.checksum_algorithm}"
            ) from exc
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest().lower() != record.checksum.lower():
            raise CacheIntegrityError(
                f"Cached granule checksum mismatch for {path.name}. Remove the "
                "invalid cache file and retry download()."
            )


def _record_filename(record: GranuleRecord) -> str:
    try:
        return safe_filename(record.filename)
    except DiscoveryError as exc:
        raise DownloadError(
            f"Granule {record.granule_id or record.identity} has no safe filename."
        ) from exc


def _reject_filename_collisions(records: Sequence[GranuleRecord]) -> None:
    names = [_record_filename(record) for record in records]
    collisions = [name for name, count in Counter(names).items() if count > 1]
    if collisions:
        raise DownloadError(
            "Distinct granules resolve to the same cache filename; refusing to "
            f"overwrite: {', '.join(sorted(collisions))}"
        )


def _index_returned_paths(returned: Sequence[Path], staging: Path) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for value in returned:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = staging / path
        path = path.resolve()
        _require_inside_staging(path, staging)
        if path.name in indexed and indexed[path.name] != path:
            raise DownloadError(
                f"Downloader returned duplicate files named {path.name!r}."
            )
        indexed[path.name] = path
    return indexed


def _require_inside_staging(path: Path, staging: Path) -> None:
    try:
        path.resolve().relative_to(staging.resolve())
    except ValueError as exc:
        raise DownloadError(
            f"Downloader returned a path outside its staging directory: {path}"
        ) from exc


def _validate_verification_mode(verify: str) -> None:
    if verify not in {"auto", "none", "size", "checksum"}:
        raise ValueError(
            "verify must be one of: 'auto', 'none', 'size', or 'checksum'."
        )


def _hashlib_name(cmr_name: str) -> str:
    return cmr_name.lower().replace("-", "").replace("_", "")


def _commit_staged_file(source: Path, target: Path) -> _CacheCommit:
    try:
        os.link(source, target)
    except FileExistsError as exc:
        raise _target_appeared_error(target) from exc
    except OSError as exc:
        if not _hard_link_fallback_allowed(exc):
            raise
        LOGGER.debug(
            "Hard-link cache commit is unavailable for %s; using an exclusive copy.",
            target.name,
        )
        return _copy_staged_file_exclusively(source, target)
    return _CacheCommit(source=source, target=target, method="link")


def _copy_staged_file_exclusively(source: Path, target: Path) -> _CacheCommit:
    copied_identity: _FileIdentity | None = None
    try:
        with source.open("rb") as source_stream:
            try:
                target_stream = target.open("xb", buffering=0)
            except FileExistsError as exc:
                raise _target_appeared_error(target) from exc

            try:
                with target_stream:
                    copied_identity = _file_identity(os.fstat(target_stream.fileno()))
                    try:
                        shutil.copyfileobj(
                            source_stream,
                            target_stream,
                            length=1024 * 1024,
                        )
                    finally:
                        copied_identity = _file_identity(
                            os.fstat(target_stream.fileno())
                        )
            except Exception as exc:
                commit = _CacheCommit(
                    source=source,
                    target=target,
                    method="copy",
                    copied_identity=copied_identity,
                )
                rollback_failures = _rollback_commits((commit,))
                if rollback_failures:
                    raise DownloadError(
                        "Exclusive cache copy failed and rollback could not remove: "
                        f"{target}"
                    ) from exc
                raise
    except DownloadError:
        raise

    return _CacheCommit(
        source=source,
        target=target,
        method="copy",
        copied_identity=copied_identity,
    )


def _hard_link_fallback_allowed(error: OSError) -> bool:
    return (
        error.errno in _HARD_LINK_FALLBACK_ERRNOS
        or getattr(error, "winerror", None) in _HARD_LINK_FALLBACK_WINERRORS
    )


def _target_appeared_error(target: Path) -> DownloadError:
    return DownloadError(
        f"Cache target appeared during download and was not replaced: {target}"
    )


def _file_identity(file_stat: os.stat_result) -> _FileIdentity:
    birth_or_change_time = getattr(
        file_stat,
        "st_birthtime_ns",
        file_stat.st_ctime_ns,
    )
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        birth_or_change_time,
    )


def _rollback_commits(committed: Sequence[_CacheCommit]) -> list[Path]:
    failures: list[Path] = []
    for commit in reversed(committed):
        try:
            if commit.method == "link":
                should_remove = (
                    commit.target.exists()
                    and commit.source.exists()
                    and os.path.samefile(commit.source, commit.target)
                )
            elif commit.copied_identity is None:
                failures.append(commit.target)
                continue
            else:
                try:
                    current_identity = _file_identity(
                        commit.target.stat(follow_symlinks=False)
                    )
                except FileNotFoundError:
                    continue
                should_remove = _same_file_identity(
                    current_identity,
                    commit.copied_identity,
                )

            if should_remove:
                commit.target.unlink()
        except OSError:
            failures.append(commit.target)
    return failures


def _same_file_identity(
    current: _FileIdentity,
    expected: _FileIdentity,
) -> bool:
    """Return whether every recorded identity field still matches.

    Device and inode alone are insufficient because a filesystem may quickly
    reuse an inode after the original cache target is unlinked.
    """

    return current == expected
