"""Transactional download and local-cache verification for PIXC granules."""

from __future__ import annotations

import hashlib
import logging
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from .discovery import EarthdataBackend, GranuleRecord, safe_filename
from .exceptions import CacheIntegrityError, DiscoveryError, DownloadError

type VerificationMode = Literal["auto", "none", "size", "checksum"]

LOGGER = logging.getLogger(__name__)


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

            committed: list[tuple[Path, Path]] = []
            try:
                for record in missing:
                    filename = _record_filename(record)
                    source = staged[record.identity]
                    target = target_dir / filename
                    try:
                        os.link(source, target)
                    except FileExistsError as exc:
                        raise DownloadError(
                            "Cache target appeared during download and was not "
                            f"replaced: {target}"
                        ) from exc
                    committed.append((source, target))
            except Exception as exc:
                rollback_failures = _rollback_links(committed)
                if rollback_failures:
                    raise DownloadError(
                        "Cache commit failed and rollback could not remove: "
                        + ", ".join(str(path) for path in rollback_failures)
                    ) from exc
                if isinstance(exc, DownloadError):
                    raise
                raise DownloadError(
                    "Cache commit failed; newly linked files were rolled back."
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


def _rollback_links(committed: Sequence[tuple[Path, Path]]) -> list[Path]:
    failures: list[Path] = []
    for source, target in reversed(committed):
        try:
            if target.exists() and source.exists() and os.path.samefile(source, target):
                target.unlink()
        except OSError:
            failures.append(target)
    return failures
