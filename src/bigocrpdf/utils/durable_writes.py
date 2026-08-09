"""Crash-safe publication of durable files."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

TextWriter = Callable[[TextIO], object]
TargetCandidates = Callable[[int], Sequence[str | Path]]
ExpectedSourceContent = tuple[str, int]

_JOURNAL_PATTERN = re.compile(r"^\.bigocr-publish-([0-9a-f]{32})\.journal$")
_NEXT_JOURNAL_PATTERN = re.compile(r"^\.bigocr-publish-([0-9a-f]{32})\.journal\.next$")
_INTERNAL_PREFIX = ".bigocr-publish-"
_JOURNAL_VERSION = 2
_LEGACY_JOURNAL_VERSION = 1
_MAX_JOURNAL_BYTES = 64 * 1024 * 1024
_FICLONE = 0x40049409
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


def read_regular_file_bytes(path: str | Path) -> bytes:
    """Read a regular file without following a final symbolic link."""
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f"Not a regular file: {path}")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            return stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


class PublicationRecoveryError(OSError):
    """A publication could not be committed or rolled back with certainty."""


class _TargetCollisionError(FileExistsError):
    """A non-overwriting destination was claimed by another process."""


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int


@dataclass(frozen=True)
class _FileState:
    identity: _FileIdentity
    size: int
    mode: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class RetirementTarget:
    """A dynamically selected retirement path bound to its observed file state."""

    path: Path
    _expected_state: _FileState

    @classmethod
    def capture(cls, path: str | Path) -> RetirementTarget:
        canonical_path = _canonical_path(Path(path))
        return cls(canonical_path, _file_state(canonical_path))


RetireCandidates = Callable[[Sequence[Path]], Sequence[RetirementTarget]]


@dataclass(frozen=True)
class _JournalEntry:
    target_name: str
    new_identity: _FileIdentity | None
    new_mode: int | None
    original_identity: _FileIdentity | None
    backup_identity: _FileIdentity | None
    backup_mode: int | None
    retire: bool = False


@dataclass(frozen=True)
class _Journal:
    path: Path
    transaction_id: str
    phase: str
    overwrite: bool
    entries: tuple[_JournalEntry, ...]
    identity: _FileIdentity


@dataclass(frozen=True)
class _PreparedPublication:
    source: Path | None
    target: Path
    source_identity: _FileIdentity | None
    original_state: _FileState | None
    journal_entry: _JournalEntry


@dataclass(frozen=True)
class _RetirementRequest:
    path: Path
    expected_state: _FileState | None


def write_text_file_atomically(
    path: str | Path,
    writer: TextWriter,
    *,
    overwrite: bool = True,
) -> Path:
    """Write text without exposing a partial target or following target symlinks."""
    target = Path(path)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temp_path = Path(temp_name)

    try:
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = -1
        with stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        return publish_file_atomically(temp_path, target, overwrite=overwrite)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def write_text_atomically(
    path: str | Path,
    text: str,
    *,
    overwrite: bool = True,
) -> Path:
    """Atomically publish an already-rendered UTF-8 text payload."""
    return write_text_file_atomically(
        path,
        lambda stream: stream.write(text),
        overwrite=overwrite,
    )


def copy_file_atomically(
    source: str | Path,
    requested_target: str | Path,
    *,
    overwrite: bool,
) -> Path:
    """Copy a regular file beside its destination, then publish the complete copy."""
    source_path = _canonical_path(Path(source))
    source_stat = source_path.lstat()
    if not stat.S_ISREG(source_stat.st_mode):
        raise ValueError(f"Copy source is not a regular file: {source_path}")

    target = _canonical_path(Path(requested_target))
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    source_descriptor = os.open(source_path, source_flags)
    staged_descriptor = -1
    try:
        opened_stat = os.fstat(source_descriptor)
        opened_identity = _FileIdentity(opened_stat.st_dev, opened_stat.st_ino)
        expected_identity = _FileIdentity(source_stat.st_dev, source_stat.st_ino)
        if not stat.S_ISREG(opened_stat.st_mode) or opened_identity != expected_identity:
            raise PublicationRecoveryError(f"Copy source changed: {source_path}")
        with tempfile.TemporaryDirectory(
            prefix=f".{target.name}.",
            dir=target.parent,
        ) as staging_name:
            staging_dir = Path(staging_name)
            staged_path = staging_dir / "payload"
            staged_flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            staged_descriptor = os.open(staged_path, staged_flags, 0o600)
            with os.fdopen(source_descriptor, "rb") as source_stream:
                source_descriptor = -1
                with os.fdopen(staged_descriptor, "wb") as staged_stream:
                    staged_descriptor = -1
                    shutil.copyfileobj(source_stream, staged_stream)
                    staged_stream.flush()
                    os.fchmod(
                        staged_stream.fileno(),
                        stat.S_IMODE(opened_stat.st_mode) & 0o777,
                    )
                    os.fsync(staged_stream.fileno())
                    final_source_stat = os.fstat(source_stream.fileno())
                    if (
                        final_source_stat.st_size,
                        final_source_stat.st_mtime_ns,
                        final_source_stat.st_ctime_ns,
                    ) != (
                        opened_stat.st_size,
                        opened_stat.st_mtime_ns,
                        opened_stat.st_ctime_ns,
                    ):
                        raise PublicationRecoveryError(
                            f"Copy source changed while it was copied: {source_path}"
                        )
            return publish_file_atomically(staged_path, target, overwrite=overwrite)
    finally:
        if source_descriptor >= 0:
            os.close(source_descriptor)
        if staged_descriptor >= 0:
            os.close(staged_descriptor)


def publish_file_atomically(
    source: str | Path,
    requested_target: str | Path,
    *,
    overwrite: bool,
) -> Path:
    """Publish one staged regular file and return the destination actually used."""
    return publish_files_transactionally(
        [(source, requested_target)],
        overwrite=overwrite,
    )[0]


def publish_files_transactionally(
    publications: Iterable[tuple[str | Path, str | Path]],
    *,
    overwrite: bool,
    target_candidates: TargetCandidates | None = None,
    retire_targets: Iterable[str | Path] = (),
    retire_candidates: RetireCandidates | None = None,
    expected_source_content: Mapping[
        str | Path,
        ExpectedSourceContent,
    ]
    | None = None,
) -> list[Path]:
    """Publish a recoverable file set whose destinations share one directory.

    ``target_candidates`` is evaluated under the destination-directory lock
    when a non-overwriting batch needs one shared collision counter.
    Explicit or dynamically selected retirement targets participate in the
    same journaled commit and are restored if publication rolls back.
    ``expected_source_content`` binds selected staged paths to a SHA-256 and
    byte count verified against the private snapshot that will be installed.
    """
    raw_pairs = [(Path(source), Path(target)) for source, target in publications]
    raw_retire_targets = [Path(target) for target in retire_targets]
    if not raw_pairs and not raw_retire_targets and retire_candidates is None:
        return []
    if retire_candidates is not None and not raw_pairs:
        raise ValueError("Dynamic retirement requires at least one publication")
    if retire_candidates is not None and raw_retire_targets:
        raise ValueError("Use either retire_targets or retire_candidates, not both")
    if (raw_retire_targets or retire_candidates is not None) and not overwrite:
        raise ValueError("Retiring prior outputs requires overwrite=True")

    sources = [_canonical_path(source) for source, _target in raw_pairs]
    requested_targets = [_canonical_path(target) for _source, target in raw_pairs]
    retire_requests = [
        _RetirementRequest(_canonical_path(target), None) for target in raw_retire_targets
    ]
    source_content = _canonical_expected_source_content(
        expected_source_content,
        sources,
    )
    if len(set(requested_targets)) != len(requested_targets):
        raise ValueError("A publication batch contains duplicate target paths")

    target_parents = {
        target.parent
        for target in (
            *requested_targets,
            *(request.path for request in retire_requests),
        )
    }
    if len(target_parents) != 1:
        raise ValueError("Every publication target must use the same directory")
    directory = next(iter(target_parents))

    for target in (
        *requested_targets,
        *(request.path for request in retire_requests),
    ):
        if target.name.startswith(_INTERNAL_PREFIX):
            raise ValueError("Publication targets cannot use the reserved internal prefix")

    with _locked_directory(directory):
        _recover_pending_publications_locked(directory)
        while True:
            targets = (
                requested_targets
                if overwrite
                else _available_targets(
                    requested_targets,
                    directory,
                    target_candidates,
                )
            )
            source_identities = _validate_sources(sources, targets, directory)
            resolved_retire_requests = retire_requests
            if retire_candidates is not None:
                resolved_retire_requests = [
                    _retirement_request(target) for target in retire_candidates(tuple(targets))
                ]
            _validate_retire_targets(
                resolved_retire_requests,
                targets,
                directory,
            )
            try:
                return _publish_prepared_set(
                    list(zip(sources, targets, source_identities, strict=True)),
                    overwrite=overwrite,
                    directory=directory,
                    retire_targets=resolved_retire_requests,
                    expected_source_content=source_content,
                )
            except _TargetCollisionError:
                if overwrite:
                    raise PublicationRecoveryError(
                        "A destination appeared during overwriting publication"
                    ) from None
                continue


def recover_pending_publications(directory: str | Path) -> None:
    """Resolve every valid interrupted publication transaction in a directory."""
    canonical_directory = Path(directory).resolve(strict=True)
    if not canonical_directory.is_dir():
        raise NotADirectoryError(canonical_directory)
    with _locked_directory(canonical_directory):
        _recover_pending_publications_locked(canonical_directory)


def _publish_prepared_set(
    publications: list[tuple[Path, Path, _FileIdentity]],
    *,
    overwrite: bool,
    directory: Path,
    retire_targets: list[_RetirementRequest],
    expected_source_content: Mapping[Path, ExpectedSourceContent],
) -> list[Path]:
    transaction_id = uuid.uuid4().hex
    journal_path = _journal_path(directory, transaction_id)
    published_targets = [target for _source, target, _identity in publications]
    preparing_publications = tuple(
        _JournalEntry(
            target_name=target.name,
            new_identity=None,
            new_mode=None,
            original_identity=None,
            backup_identity=None,
            backup_mode=None,
        )
        for _source, target, _identity in publications
    )
    preparing_retirements = tuple(
        _JournalEntry(
            target_name=target.name,
            new_identity=None,
            new_mode=None,
            original_identity=None,
            backup_identity=None,
            backup_mode=None,
            retire=True,
        )
        for request in retire_targets
        for target in (request.path,)
    )
    preparing_entries = preparing_publications + preparing_retirements
    _write_new_journal(
        journal_path,
        transaction_id,
        "PREPARING",
        overwrite,
        preparing_entries,
    )

    prepared: list[_PreparedPublication] = []
    installation_started = False
    try:
        for index, (source, target, source_identity) in enumerate(publications):
            new_path = _artifact_path(directory, transaction_id, index, "new")
            backup_path = _artifact_path(directory, transaction_id, index, "backup")
            expected_content = expected_source_content.get(source)
            if expected_content is None:
                new_identity, new_mode = _snapshot_regular_file(
                    source,
                    new_path,
                    source_identity,
                )
            else:
                new_identity, new_mode = _snapshot_regular_file(
                    source,
                    new_path,
                    source_identity,
                    expected_content=expected_content,
                )
            (
                original_identity,
                backup_identity,
                backup_mode,
                original_state,
            ) = (
                _snapshot_existing_target(target, backup_path)
                if overwrite
                else (None, None, None, None)
            )

            prepared.append(
                _PreparedPublication(
                    source=source,
                    target=target,
                    source_identity=source_identity,
                    original_state=original_state,
                    journal_entry=_JournalEntry(
                        target_name=target.name,
                        new_identity=new_identity,
                        new_mode=new_mode,
                        original_identity=original_identity,
                        backup_identity=backup_identity,
                        backup_mode=backup_mode,
                    ),
                )
            )

        for index, request in enumerate(
            retire_targets,
            start=len(publications),
        ):
            target = request.path
            backup_path = _artifact_path(directory, transaction_id, index, "backup")
            (
                original_identity,
                backup_identity,
                backup_mode,
                original_state,
            ) = _snapshot_existing_target(
                target,
                backup_path,
                expected_state=request.expected_state,
            )
            prepared.append(
                _PreparedPublication(
                    source=None,
                    target=target,
                    source_identity=None,
                    original_state=original_state,
                    journal_entry=_JournalEntry(
                        target_name=target.name,
                        new_identity=None,
                        new_mode=None,
                        original_identity=original_identity,
                        backup_identity=backup_identity,
                        backup_mode=backup_mode,
                        retire=True,
                    ),
                )
            )

        _fsync_directory(directory)
        prepared_entries = tuple(item.journal_entry for item in prepared)
        _encode_journal_payload(
            transaction_id,
            "PREPARED",
            overwrite,
            prepared_entries,
        )
        _encode_journal_payload(
            transaction_id,
            "COMMITTED",
            overwrite,
            prepared_entries,
        )
        _replace_journal(
            journal_path,
            transaction_id,
            "PREPARED",
            overwrite,
            prepared_entries,
        )

        installation_started = True
        _install_prepared_files(
            prepared,
            transaction_id=transaction_id,
            overwrite=overwrite,
            directory=directory,
        )
        _fsync_directory(directory)
        _replace_journal(
            journal_path,
            transaction_id,
            "COMMITTED",
            overwrite,
            prepared_entries,
        )
    except Exception as publication_error:
        visible_journal = _try_read_owned_journal(journal_path, transaction_id)
        if visible_journal is not None and visible_journal.phase == "COMMITTED":
            try:
                _fsync_directory(directory)
            except OSError as sync_error:
                raise PublicationRecoveryError(
                    "Publication committed, but its durable state is uncertain; "
                    f"recovery journal preserved at {journal_path}"
                ) from sync_error
            _verify_committed_targets(visible_journal, directory)
            _remove_staged_sources(prepared)
            _cleanup_committed_best_effort(visible_journal, directory)
            return published_targets

        try:
            if installation_started:
                _rollback_prepared_entries(
                    tuple(item.journal_entry for item in prepared),
                    journal_path,
                    directory,
                    overwrite=overwrite,
                )
            else:
                _abandon_preparing_transaction(
                    transaction_id,
                    len(preparing_entries),
                    journal_path,
                    directory,
                )
        except Exception as rollback_error:
            raise PublicationRecoveryError(
                "Publication could not be rolled back; inspect the preserved "
                f"transaction state in {directory}"
            ) from rollback_error
        raise publication_error

    committed_journal = _read_journal(journal_path)
    _verify_committed_targets(committed_journal, directory)
    _remove_staged_sources(prepared)
    _cleanup_committed_best_effort(committed_journal, directory)
    return published_targets


def _snapshot_existing_target(
    target: Path,
    backup_path: Path,
    *,
    expected_state: _FileState | None = None,
) -> tuple[
    _FileIdentity | None,
    _FileIdentity | None,
    int | None,
    _FileState | None,
]:
    if not os.path.lexists(target):
        if expected_state is not None:
            raise PublicationRecoveryError(
                f"Retirement target changed before it was prepared: {target}"
            )
        return None, None, None, None
    target_stat = target.lstat()
    original_state = _state_from_stat(target_stat)
    if expected_state is not None and original_state != expected_state:
        raise PublicationRecoveryError(
            f"Retirement target changed before it was prepared: {target}"
        )
    original_identity = _FileIdentity(
        target_stat.st_dev,
        target_stat.st_ino,
    )
    backup_mode = None
    if stat.S_ISREG(target_stat.st_mode):
        backup_identity, _snapshot_mode = _snapshot_regular_file(
            target,
            backup_path,
            original_identity,
        )
        backup_mode = stat.S_IMODE(target_stat.st_mode)
    elif stat.S_ISLNK(target_stat.st_mode):
        os.symlink(os.readlink(target), backup_path)
        backup_identity = _identity(backup_path)
    else:
        raise ValueError(f"Existing output is not a regular file: {target}")
    if _optional_file_state(target) != original_state:
        raise PublicationRecoveryError(f"Destination changed while creating its backup: {target}")
    return original_identity, backup_identity, backup_mode, original_state


def _install_prepared_files(
    prepared: list[_PreparedPublication],
    *,
    transaction_id: str,
    overwrite: bool,
    directory: Path,
) -> None:
    _validate_prepared_artifacts(
        tuple(item.journal_entry for item in prepared),
        transaction_id,
        directory,
    )
    for item in prepared:
        expected_identity = item.journal_entry.original_identity
        target_state = _optional_file_state(item.target)
        target_identity = target_state.identity if target_state is not None else None
        if target_identity != expected_identity or (
            item.original_state is not None and target_state != item.original_state
        ):
            if not overwrite and expected_identity is None:
                raise _TargetCollisionError(item.target)
            raise PublicationRecoveryError(f"Destination changed during publication: {item.target}")

    for index, item in enumerate(prepared):
        entry = item.journal_entry
        retired_path = _artifact_path(
            directory,
            transaction_id,
            index,
            "retired",
        )
        if entry.original_identity is not None:
            _move_owned_without_replacement(
                item.target,
                retired_path,
                entry.original_identity,
                expected_state=item.original_state,
            )
        if item.journal_entry.retire:
            continue
        new_path = _artifact_path(directory, transaction_id, index, "new")
        new_identity = entry.new_identity
        new_mode = entry.new_mode
        if new_identity is None or new_mode is None:
            raise ValueError("Invalid prepared publication journal")
        _rename_without_replacement(new_path, item.target)
        if _optional_identity(item.target) != new_identity:
            raise PublicationRecoveryError(
                f"Published destination has an unexpected identity: {item.target}"
            )
        _set_regular_file_mode(
            item.target,
            new_identity,
            new_mode,
        )


def _recover_pending_publications_locked(directory: Path) -> None:
    orphan_next_paths = _next_journal_paths(directory)
    for next_path in orphan_next_paths:
        _unlink_internal_file(next_path)
    if orphan_next_paths:
        _fsync_directory(directory)

    for journal_path in _journal_paths(directory):
        journal = _read_journal(journal_path)
        if journal.phase == "PREPARING":
            _abandon_preparing_transaction(
                journal.transaction_id,
                len(journal.entries),
                journal.path,
                directory,
                journal.identity,
            )
        elif journal.phase == "PREPARED":
            _rollback_prepared_entries(
                journal.entries,
                journal.path,
                directory,
                overwrite=journal.overwrite,
                journal_identity=journal.identity,
            )
        elif journal.phase == "ROLLED_BACK":
            _cleanup_rolled_back_journal(journal, directory)
        else:
            _verify_committed_targets(journal, directory)
            _cleanup_committed_journal(journal, directory)


def _rollback_prepared_entries(
    entries: tuple[_JournalEntry, ...],
    journal_path: Path,
    directory: Path,
    *,
    overwrite: bool,
    journal_identity: _FileIdentity | None = None,
) -> None:
    expected_journal_identity = journal_identity or _identity(journal_path)
    if _identity(journal_path) != expected_journal_identity:
        raise PublicationRecoveryError(
            f"Publication journal changed before rollback: {journal_path}"
        )
    transaction_id = _journal_transaction_id(journal_path)
    _validate_prepared_artifacts(entries, transaction_id, directory)
    actions = tuple(
        _rollback_action(entry, transaction_id, index, directory)
        for index, entry in enumerate(entries)
    )

    for index, (entry, action) in enumerate(zip(entries, actions, strict=True)):
        if action == "restore":
            _rollback_restore_entry(
                entry,
                transaction_id,
                index,
                directory,
            )
        elif action == "remove":
            _rollback_remove_new_entry(
                entry,
                transaction_id,
                index,
                directory,
            )

    for entry in entries:
        _restore_target_mode(entry, directory / entry.target_name)

    _fsync_directory(directory)
    _replace_journal(
        journal_path,
        transaction_id,
        "ROLLED_BACK",
        overwrite,
        entries,
    )
    rolled_back_journal = _read_journal(journal_path)
    _cleanup_rolled_back_journal(rolled_back_journal, directory)


def _rollback_action(
    entry: _JournalEntry,
    transaction_id: str,
    index: int,
    directory: Path,
) -> str:
    if entry.retire:
        return _retirement_rollback_action(
            entry,
            transaction_id,
            index,
            directory,
        )
    if entry.new_identity is None:
        raise ValueError("Invalid prepared publication journal")
    new_path = _artifact_path(directory, transaction_id, index, "new")
    new_artifact_identity = _optional_identity(new_path)
    target = directory / entry.target_name
    target_identity = _optional_identity(target)
    if new_artifact_identity is not None:
        if new_artifact_identity != entry.new_identity:
            raise PublicationRecoveryError(f"Publication artifact changed: {new_path}")
        retired_path = _artifact_path(
            directory,
            transaction_id,
            index,
            "retired",
        )
        retired_identity = _optional_identity(retired_path)
        if retired_identity is not None:
            if retired_identity != entry.original_identity:
                raise PublicationRecoveryError(
                    f"Retired publication artifact changed: {retired_path}"
                )
            if target_identity in {
                entry.original_identity,
                entry.backup_identity,
            }:
                return "preserve"
            if target_identity is None:
                return "restore"
            raise PublicationRecoveryError(
                f"Cannot restore externally changed destination: {target}"
            )
        if target_identity in {
            entry.original_identity,
            entry.backup_identity,
        }:
            return "preserve"
        if target_identity is None:
            return "restore" if entry.original_identity is not None else "preserve"
        return "preserve"

    if entry.backup_identity is not None:
        if target_identity == entry.new_identity:
            if not _original_artifact_available(
                entry,
                transaction_id,
                index,
                directory,
            ):
                raise PublicationRecoveryError(
                    f"Cannot restore destination without its backup: {target}"
                )
            return "restore"
        if target_identity in {
            entry.original_identity,
            entry.backup_identity,
        }:
            return "preserve"
        if target_identity is None and _original_artifact_available(
            entry,
            transaction_id,
            index,
            directory,
        ):
            return "restore"
        raise PublicationRecoveryError(f"Cannot restore externally changed destination: {target}")

    if target_identity == entry.new_identity:
        return "remove"
    if target_identity is None:
        return "preserve"
    raise PublicationRecoveryError(f"Cannot remove externally changed destination: {target}")


def _retirement_rollback_action(
    entry: _JournalEntry,
    transaction_id: str,
    index: int,
    directory: Path,
) -> str:
    target = directory / entry.target_name
    target_identity = _optional_identity(target)
    if entry.original_identity is None:
        if target_identity is None:
            return "preserve"
        raise PublicationRecoveryError(
            f"Cannot restore externally created retired destination: {target}"
        )

    if target_identity in {
        entry.original_identity,
        entry.backup_identity,
    }:
        return "preserve"
    if target_identity is None and _original_artifact_available(
        entry,
        transaction_id,
        index,
        directory,
    ):
        return "restore"
    raise PublicationRecoveryError(f"Cannot restore retired destination: {target}")


def _original_artifact_available(
    entry: _JournalEntry,
    transaction_id: str,
    index: int,
    directory: Path,
) -> bool:
    retired_path = _artifact_path(
        directory,
        transaction_id,
        index,
        "retired",
    )
    retired_identity = _optional_identity(retired_path)
    if retired_identity is not None:
        if retired_identity != entry.original_identity:
            raise PublicationRecoveryError(f"Retired publication artifact changed: {retired_path}")
        return True
    backup_path = _artifact_path(
        directory,
        transaction_id,
        index,
        "backup",
    )
    return _optional_identity(backup_path) == entry.backup_identity


def _rollback_restore_entry(
    entry: _JournalEntry,
    transaction_id: str,
    index: int,
    directory: Path,
) -> None:
    target = directory / entry.target_name
    new_path = _artifact_path(directory, transaction_id, index, "new")
    target_identity = _optional_identity(target)
    if not entry.retire and target_identity == entry.new_identity:
        if entry.new_identity is None:
            raise ValueError("Invalid prepared publication journal")
        _move_owned_without_replacement(
            target,
            new_path,
            entry.new_identity,
        )
        target_identity = None
    if target_identity in {
        entry.original_identity,
        entry.backup_identity,
    }:
        return
    if target_identity is not None:
        raise PublicationRecoveryError(f"Cannot restore externally changed destination: {target}")

    retired_path = _artifact_path(
        directory,
        transaction_id,
        index,
        "retired",
    )
    if _optional_identity(retired_path) == entry.original_identity:
        if entry.original_identity is None:
            raise ValueError("Invalid original publication identity")
        _move_owned_without_replacement(
            retired_path,
            target,
            entry.original_identity,
        )
        return

    backup_path = _artifact_path(
        directory,
        transaction_id,
        index,
        "backup",
    )
    if _optional_identity(backup_path) != entry.backup_identity:
        raise PublicationRecoveryError(f"Cannot restore destination without its backup: {target}")
    if entry.backup_identity is None:
        raise ValueError("Invalid publication backup identity")
    _move_owned_without_replacement(
        backup_path,
        target,
        entry.backup_identity,
    )


def _rollback_remove_new_entry(
    entry: _JournalEntry,
    transaction_id: str,
    index: int,
    directory: Path,
) -> None:
    if entry.new_identity is None:
        raise ValueError("Invalid prepared publication journal")
    target = directory / entry.target_name
    new_path = _artifact_path(directory, transaction_id, index, "new")
    target_identity = _optional_identity(target)
    if target_identity == entry.new_identity:
        _move_owned_without_replacement(
            target,
            new_path,
            entry.new_identity,
        )
        return
    if target_identity is None and _optional_identity(new_path) == entry.new_identity:
        return
    raise PublicationRecoveryError(f"Cannot remove externally changed destination: {target}")


def _restore_target_mode(entry: _JournalEntry, target: Path) -> None:
    if entry.backup_mode is None:
        return
    target_identity = _optional_identity(target)
    if target_identity is None or target_identity not in {
        entry.original_identity,
        entry.backup_identity,
    }:
        return
    _set_regular_file_mode(target, target_identity, entry.backup_mode)


def _set_regular_file_mode(
    target: Path,
    expected_identity: _FileIdentity,
    access_mode: int,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(target, flags)
    try:
        target_stat = os.fstat(descriptor)
        target_identity = _FileIdentity(target_stat.st_dev, target_stat.st_ino)
        if target_identity != expected_identity:
            raise PublicationRecoveryError(
                f"Cannot restore permissions on changed destination: {target}"
            )
        if not stat.S_ISREG(target_stat.st_mode):
            raise PublicationRecoveryError(
                f"Cannot restore permissions on invalid destination: {target}"
            )
        os.fchmod(descriptor, access_mode)
        os.fsync(descriptor)
        final_stat = os.fstat(descriptor)
        if _FileIdentity(final_stat.st_dev, final_stat.st_ino) != expected_identity:
            raise PublicationRecoveryError(
                f"Destination changed while restoring permissions: {target}"
            )
    finally:
        os.close(descriptor)


def _verify_committed_targets(journal: _Journal, directory: Path) -> None:
    if journal.phase != "COMMITTED":
        raise PublicationRecoveryError(f"Publication journal is not committed: {journal.path}")
    for entry in journal.entries:
        target = directory / entry.target_name
        expected_identity = None if entry.retire else entry.new_identity
        if _optional_identity(target) != expected_identity:
            raise PublicationRecoveryError(
                f"Committed destination changed before recovery: {target}"
            )


def _cleanup_committed_best_effort(journal: _Journal, directory: Path) -> None:
    try:
        _cleanup_committed_journal(journal, directory)
    except PublicationRecoveryError:
        raise
    except OSError:
        # The COMMITTED decision and its targets were verified. A later
        # publication or explicit recovery can finish artifact cleanup.
        pass


def _cleanup_committed_journal(journal: _Journal, directory: Path) -> None:
    _cleanup_entry_artifacts(
        journal.entries,
        journal.transaction_id,
        directory,
    )
    _remove_journal(journal.path, journal.identity, directory)


def _cleanup_rolled_back_journal(journal: _Journal, directory: Path) -> None:
    if journal.phase != "ROLLED_BACK":
        raise PublicationRecoveryError(f"Publication journal is not rolled back: {journal.path}")
    _cleanup_entry_artifacts(
        journal.entries,
        journal.transaction_id,
        directory,
    )
    _remove_journal(journal.path, journal.identity, directory)


def _validate_prepared_artifacts(
    entries: tuple[_JournalEntry, ...],
    transaction_id: str,
    directory: Path,
) -> None:
    for index, entry in enumerate(entries):
        new_path = _artifact_path(directory, transaction_id, index, "new")
        new_identity = _optional_identity(new_path)
        retired_path = _artifact_path(
            directory,
            transaction_id,
            index,
            "retired",
        )
        retired_identity = _optional_identity(retired_path)
        if retired_identity is not None and retired_identity != entry.original_identity:
            raise PublicationRecoveryError(f"Retired publication artifact changed: {retired_path}")
        if entry.retire:
            if entry.new_identity is not None or entry.new_mode is not None:
                raise ValueError("Invalid retired publication journal")
            if new_identity is not None:
                raise PublicationRecoveryError(
                    f"Unexpected retirement publication artifact: {new_path}"
                )
        elif entry.new_identity is None:
            raise ValueError("Invalid prepared publication journal")
        elif new_identity is not None and new_identity != entry.new_identity:
            raise PublicationRecoveryError(f"Publication artifact changed: {new_path}")
        backup_path = _artifact_path(directory, transaction_id, index, "backup")
        backup_identity = _optional_identity(backup_path)
        if backup_identity is not None and backup_identity != entry.backup_identity:
            raise PublicationRecoveryError(f"Publication backup changed: {backup_path}")


def _cleanup_entry_artifacts(
    entries: tuple[_JournalEntry, ...],
    transaction_id: str,
    directory: Path,
) -> None:
    for index, entry in enumerate(entries):
        if not entry.retire:
            if entry.new_identity is None:
                raise ValueError("Invalid publication journal")
            _unlink_owned(
                _artifact_path(directory, transaction_id, index, "new"),
                entry.new_identity,
            )
        if entry.backup_identity is not None:
            _unlink_owned(
                _artifact_path(directory, transaction_id, index, "backup"),
                entry.backup_identity,
            )
        if entry.original_identity is not None:
            _unlink_owned(
                _artifact_path(
                    directory,
                    transaction_id,
                    index,
                    "retired",
                ),
                entry.original_identity,
            )
    _unlink_internal_file(_next_journal_path(directory, transaction_id))
    _fsync_directory(directory)


def _abandon_preparing_transaction(
    transaction_id: str,
    entry_count: int,
    journal_path: Path,
    directory: Path,
    journal_identity: _FileIdentity | None = None,
) -> None:
    for index in range(entry_count):
        for kind in ("new", "backup", "retired"):
            _unlink_internal_file(_artifact_path(directory, transaction_id, index, kind))
    _unlink_internal_file(_next_journal_path(directory, transaction_id))
    if os.path.lexists(journal_path):
        _remove_journal(
            journal_path,
            journal_identity or _identity(journal_path),
            directory,
        )
    else:
        _fsync_directory(directory)


def _remove_staged_sources(prepared: list[_PreparedPublication]) -> None:
    changed_directories: set[Path] = set()
    for item in prepared:
        if item.source is None or item.source_identity is None:
            continue
        try:
            _unlink_owned(item.source, item.source_identity)
        except OSError:
            continue
        changed_directories.add(item.source.parent)
    for directory in changed_directories:
        try:
            _fsync_directory(directory)
        except OSError:
            pass


def _validate_sources(
    sources: list[Path],
    targets: list[Path],
    directory: Path,
) -> list[_FileIdentity]:
    if set(sources) & set(targets):
        raise ValueError("A staged source cannot also be a publication target")
    directory_device = directory.stat().st_dev
    identities: list[_FileIdentity] = []
    for source in sources:
        source_stat = source.lstat()
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError(f"Staged output is not a regular file: {source}")
        if source_stat.st_dev != directory_device:
            raise OSError("Staged output and destination must be on the same filesystem")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(source, flags)
        try:
            opened_stat = os.fstat(descriptor)
            identity = _FileIdentity(opened_stat.st_dev, opened_stat.st_ino)
            if identity != _FileIdentity(source_stat.st_dev, source_stat.st_ino):
                raise PublicationRecoveryError(f"Staged output changed: {source}")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        identities.append(identity)
    return identities


def _snapshot_regular_file(
    source: Path,
    destination: Path,
    expected_source_identity: _FileIdentity,
    *,
    expected_content: ExpectedSourceContent | None = None,
) -> tuple[_FileIdentity, int]:
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    destination_flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    source_descriptor = os.open(source, source_flags)
    destination_descriptor = -1
    snapshot_identity: _FileIdentity | None = None
    try:
        source_stat = os.fstat(source_descriptor)
        source_identity = _FileIdentity(source_stat.st_dev, source_stat.st_ino)
        if not stat.S_ISREG(source_stat.st_mode) or source_identity != expected_source_identity:
            raise PublicationRecoveryError(f"Snapshot source changed: {source}")
        os.fsync(source_descriptor)
        destination_descriptor = os.open(destination, destination_flags, 0o600)
        try:
            fcntl.ioctl(destination_descriptor, _FICLONE, source_descriptor)
        except OSError as clone_error:
            if clone_error.errno not in {
                errno.EBADF,
                errno.EINVAL,
                errno.ENOTTY,
                errno.EOPNOTSUPP,
                errno.EXDEV,
                errno.ENOSYS,
                errno.EPERM,
            }:
                raise
            os.ftruncate(destination_descriptor, 0)
            os.lseek(destination_descriptor, 0, os.SEEK_SET)
            os.lseek(source_descriptor, 0, os.SEEK_SET)
            _copy_file_descriptors(source_descriptor, destination_descriptor)
        os.fchmod(destination_descriptor, 0o600)
        os.fsync(destination_descriptor)
        if (
            expected_content is not None
            and _descriptor_content_fingerprint(destination_descriptor) != expected_content
        ):
            raise PublicationRecoveryError(
                f"Staged source content changed before publication snapshot: {source}"
            )
        final_source_stat = os.fstat(source_descriptor)
        if (
            final_source_stat.st_size,
            final_source_stat.st_mtime_ns,
            final_source_stat.st_ctime_ns,
        ) != (
            source_stat.st_size,
            source_stat.st_mtime_ns,
            source_stat.st_ctime_ns,
        ):
            raise PublicationRecoveryError(f"Snapshot source changed while it was copied: {source}")
        destination_stat = os.fstat(destination_descriptor)
        snapshot_identity = _FileIdentity(
            destination_stat.st_dev,
            destination_stat.st_ino,
        )
    finally:
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        os.close(source_descriptor)
    if snapshot_identity is None or _identity(destination) != snapshot_identity:
        raise PublicationRecoveryError(f"Publication snapshot changed: {destination}")
    return snapshot_identity, stat.S_IMODE(source_stat.st_mode) & 0o777


def _copy_file_descriptors(source_descriptor: int, destination_descriptor: int) -> None:
    while data := os.read(source_descriptor, 1024 * 1024):
        remaining = memoryview(data)
        while remaining:
            written = os.write(destination_descriptor, remaining)
            if written == 0:
                raise OSError("Could not write publication snapshot")
            remaining = remaining[written:]


def _descriptor_content_fingerprint(
    descriptor: int,
) -> ExpectedSourceContent:
    digest = hashlib.sha256()
    size = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while data := os.read(descriptor, 1024 * 1024):
        digest.update(data)
        size += len(data)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest(), size


def _canonical_expected_source_content(
    expected_content: Mapping[
        str | Path,
        ExpectedSourceContent,
    ]
    | None,
    sources: Sequence[Path],
) -> dict[Path, ExpectedSourceContent]:
    if expected_content is None:
        return {}
    canonical: dict[Path, ExpectedSourceContent] = {}
    for raw_path, fingerprint in expected_content.items():
        path = _canonical_path(Path(raw_path))
        if path in canonical:
            raise ValueError("Duplicate expected source content path")
        if (
            not isinstance(fingerprint, tuple)
            or len(fingerprint) != 2
            or not isinstance(fingerprint[0], str)
            or len(fingerprint[0]) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint[0])
            or type(fingerprint[1]) is not int
            or fingerprint[1] < 0
        ):
            raise ValueError("Invalid expected source content fingerprint")
        canonical[path] = fingerprint
    if not set(canonical) <= set(sources):
        raise ValueError("Expected source content paths must belong to the publication set")
    return canonical


def _available_targets(
    requested_targets: list[Path],
    directory: Path,
    target_candidates: TargetCandidates | None,
) -> list[Path]:
    counter = 0
    while True:
        if target_candidates is None:
            candidates = [
                (
                    requested
                    if counter == 0
                    else requested.with_name(f"{requested.stem}-{counter}{requested.suffix}")
                )
                for requested in requested_targets
            ]
        else:
            candidates = [
                _canonical_path(Path(candidate)) for candidate in target_candidates(counter)
            ]
            if counter == 0 and candidates != requested_targets:
                raise ValueError(
                    "Collision target candidates must preserve requested paths at counter zero"
                )
        _validate_target_candidates(
            candidates,
            requested_targets,
            directory,
        )
        if not any(os.path.lexists(candidate) for candidate in candidates):
            return candidates
        counter += 1


def _validate_target_candidates(
    candidates: list[Path],
    requested_targets: list[Path],
    directory: Path,
) -> None:
    if len(candidates) != len(requested_targets):
        raise ValueError("Collision target candidate count must match the publication batch")
    if len(set(candidates)) != len(candidates):
        raise ValueError("Collision target candidates contain duplicate paths")
    for candidate in candidates:
        if candidate.parent != directory:
            raise ValueError("Every collision target candidate must use the publication directory")
        if candidate.name.startswith(_INTERNAL_PREFIX):
            raise ValueError("Collision target candidates cannot use the reserved internal prefix")


def _validate_retire_targets(
    retire_targets: list[_RetirementRequest],
    publication_targets: list[Path],
    directory: Path,
) -> None:
    paths = [request.path for request in retire_targets]
    if len(set(paths)) != len(paths):
        raise ValueError("A publication batch contains duplicate retirement paths")
    if set(publication_targets) & set(paths):
        raise ValueError("A publication target cannot also be retired")
    for request in retire_targets:
        target = request.path
        if target.parent != directory:
            raise ValueError("Every retirement target must use the publication directory")
        if target.name.startswith(_INTERNAL_PREFIX):
            raise ValueError("Retirement targets cannot use the reserved internal prefix")
        if (
            request.expected_state is not None
            and _optional_file_state(target) != request.expected_state
        ):
            raise PublicationRecoveryError(f"Retirement target changed after selection: {target}")


def _retirement_request(target: RetirementTarget) -> _RetirementRequest:
    if not isinstance(target, RetirementTarget):
        raise TypeError(
            "Dynamic retirement callbacks must return RetirementTarget.capture() values"
        )
    path = _canonical_path(target.path)
    return _RetirementRequest(path, target._expected_state)


def _canonical_path(path: Path) -> Path:
    parent = path.parent.resolve(strict=True)
    if not path.name or path.name in {".", ".."}:
        raise ValueError(f"Invalid publication path: {path}")
    return parent / path.name


@contextmanager
def _locked_directory(directory: Path) -> Iterator[None]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(directory, flags)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _rename_without_replacement(source: Path, target: Path) -> None:
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise OSError(
            errno.ENOSYS,
            "Atomic no-replace rename is unavailable on this Linux system",
            target,
        ) from error
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(target),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise _TargetCollisionError(target)
    raise OSError(error_number, os.strerror(error_number), target)


def _write_new_journal(
    path: Path,
    transaction_id: str,
    phase: str,
    overwrite: bool,
    entries: tuple[_JournalEntry, ...],
) -> None:
    payload = _encode_journal_payload(
        transaction_id,
        phase,
        overwrite,
        entries,
    )
    next_path = _next_journal_path(path.parent, transaction_id)
    try:
        _write_bytes_exclusively(next_path, payload)
        os.replace(next_path, path)
        _fsync_directory(path.parent)
    finally:
        try:
            next_path.unlink(missing_ok=True)
        except OSError:
            pass


def _replace_journal(
    path: Path,
    transaction_id: str,
    phase: str,
    overwrite: bool,
    entries: tuple[_JournalEntry, ...],
) -> None:
    next_path = _next_journal_path(path.parent, transaction_id)
    payload = _encode_journal_payload(
        transaction_id,
        phase,
        overwrite,
        entries,
    )
    try:
        _write_bytes_exclusively(next_path, payload)
        os.replace(next_path, path)
        _fsync_directory(path.parent)
    finally:
        try:
            next_path.unlink(missing_ok=True)
        except OSError:
            pass


def _write_bytes_exclusively(path: Path, payload: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    opened_stat = os.fstat(descriptor)
    expected_identity = _FileIdentity(opened_stat.st_dev, opened_stat.st_ino)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            _unlink_owned(path, expected_identity)
            _fsync_directory(path.parent)
        except OSError:
            pass
        raise


def _encode_journal_payload(
    transaction_id: str,
    phase: str,
    overwrite: bool,
    entries: tuple[_JournalEntry, ...],
) -> bytes:
    payload = {
        "version": _JOURNAL_VERSION,
        "transaction": transaction_id,
        "phase": phase,
        "overwrite": overwrite,
        "entries": [
            {
                "target": entry.target_name,
                "new_identity": _identity_payload(entry.new_identity),
                "new_mode": entry.new_mode,
                "original_identity": _identity_payload(entry.original_identity),
                "backup_identity": _identity_payload(entry.backup_identity),
                "backup_mode": entry.backup_mode,
                "retire": entry.retire,
            }
            for entry in entries
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(encoded) > _MAX_JOURNAL_BYTES:
        raise ValueError(
            f"Publication journal is too large for safe recovery ({len(encoded)} bytes)"
        )
    return encoded


def _identity_payload(identity: _FileIdentity | None) -> list[int] | None:
    if identity is None:
        return None
    return [identity.device, identity.inode]


def _read_journal(path: Path) -> _Journal:
    match = _JOURNAL_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Invalid publication journal name: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"Invalid publication journal: {path}") from error
    try:
        file_stat = os.fstat(descriptor)
        private_mode = stat.S_IMODE(file_stat.st_mode)
        if (
            not stat.S_ISREG(file_stat.st_mode)
            or file_stat.st_uid != os.geteuid()
            or private_mode & 0o077
            or file_stat.st_nlink != 1
            or file_stat.st_size > _MAX_JOURNAL_BYTES
        ):
            raise ValueError(f"Invalid publication journal: {path}")
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = -1
            payload = json.load(stream, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid publication journal: {path}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    transaction_id = match.group(1)
    overwrite, entries = _parse_journal_payload(
        payload,
        transaction_id,
        path.parent,
    )
    return _Journal(
        path=path,
        transaction_id=transaction_id,
        phase=payload["phase"],
        overwrite=overwrite,
        entries=entries,
        identity=_FileIdentity(file_stat.st_dev, file_stat.st_ino),
    )


def _parse_journal_payload(
    payload: object,
    transaction_id: str,
    directory: Path,
) -> tuple[bool, tuple[_JournalEntry, ...]]:
    if not isinstance(payload, dict) or set(payload) != {
        "version",
        "transaction",
        "phase",
        "overwrite",
        "entries",
    }:
        raise ValueError("Invalid publication journal schema")
    phase = payload["phase"]
    overwrite = payload["overwrite"]
    raw_entries = payload["entries"]
    version = payload["version"]
    if (
        type(version) is not int
        or version not in {_LEGACY_JOURNAL_VERSION, _JOURNAL_VERSION}
        or payload["transaction"] != transaction_id
        or phase not in {"PREPARING", "PREPARED", "ROLLED_BACK", "COMMITTED"}
        or type(overwrite) is not bool
        or not isinstance(raw_entries, list)
        or not raw_entries
    ):
        raise ValueError("Invalid publication journal schema")

    directory_device = directory.stat().st_dev
    entries: list[_JournalEntry] = []
    target_names: set[str] = set()
    for raw_entry in raw_entries:
        entry_fields = {
            "target",
            "new_identity",
            "new_mode",
            "original_identity",
            "backup_identity",
            "backup_mode",
        }
        if version == _JOURNAL_VERSION:
            entry_fields.add("retire")
        if not isinstance(raw_entry, dict) or set(raw_entry) != entry_fields:
            raise ValueError("Invalid publication journal entry")
        retire = raw_entry.get("retire", False)
        if type(retire) is not bool:
            raise ValueError("Invalid publication retirement state")
        if retire and not overwrite:
            raise ValueError("Invalid non-overwriting retirement")
        target_name = raw_entry["target"]
        if (
            not isinstance(target_name, str)
            or not target_name
            or target_name in {".", ".."}
            or "/" in target_name
            or target_name.startswith(_INTERNAL_PREFIX)
            or target_name in target_names
        ):
            raise ValueError("Invalid publication journal target")
        new_identity = _parse_identity(
            raw_entry["new_identity"],
            directory_device,
        )
        new_mode = raw_entry["new_mode"]
        if new_mode is not None and (type(new_mode) is not int or not 0 <= new_mode <= 0o777):
            raise ValueError("Invalid publication output mode")
        original_identity = _parse_identity(
            raw_entry["original_identity"],
            directory_device,
        )
        backup_identity = _parse_identity(
            raw_entry["backup_identity"],
            directory_device,
        )
        backup_mode = raw_entry["backup_mode"]
        if backup_mode is not None and (
            type(backup_mode) is not int or not 0 <= backup_mode <= 0o7777
        ):
            raise ValueError("Invalid publication journal backup mode")
        if phase == "PREPARING":
            if any(
                value is not None
                for value in (
                    new_identity,
                    new_mode,
                    original_identity,
                    backup_identity,
                    backup_mode,
                )
            ):
                raise ValueError("Invalid preparing publication journal")
        elif retire:
            if not overwrite or new_identity is not None or new_mode is not None:
                raise ValueError("Invalid retired publication journal")
            if (original_identity is None) != (backup_identity is None):
                raise ValueError("Invalid retirement backup identity")
            if backup_mode is not None and backup_identity is None:
                raise ValueError("Invalid retirement backup mode")
        else:
            if new_identity is None or new_mode is None:
                raise ValueError("Invalid prepared publication journal")
            if (original_identity is None) != (backup_identity is None):
                raise ValueError("Invalid publication backup identity")
            if not overwrite and backup_identity is not None:
                raise ValueError("Invalid non-overwriting publication backup")
            if backup_mode is not None and backup_identity is None:
                raise ValueError("Invalid publication backup mode")
        target_names.add(target_name)
        entries.append(
            _JournalEntry(
                target_name=target_name,
                new_identity=new_identity,
                new_mode=new_mode,
                original_identity=original_identity,
                backup_identity=backup_identity,
                backup_mode=backup_mode,
                retire=retire,
            )
        )
    return overwrite, tuple(entries)


def _parse_identity(value: object, directory_device: int) -> _FileIdentity | None:
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(part) is not int or part < 0 for part in value)
        or value[0] != directory_device
    ):
        raise ValueError("Invalid publication journal identity")
    return _FileIdentity(value[0], value[1])


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Invalid publication journal: duplicate key")
        result[key] = value
    return result


def _try_read_owned_journal(path: Path, transaction_id: str) -> _Journal | None:
    if not os.path.lexists(path):
        return None
    try:
        journal = _read_journal(path)
    except ValueError:
        return None
    return journal if journal.transaction_id == transaction_id else None


def _journal_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.name.startswith(_INTERNAL_PREFIX) and path.name.endswith(".journal")
    )


def _next_journal_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if _NEXT_JOURNAL_PATTERN.fullmatch(path.name) is not None
    )


def _journal_path(directory: Path, transaction_id: str) -> Path:
    return directory / f"{_INTERNAL_PREFIX}{transaction_id}.journal"


def _next_journal_path(directory: Path, transaction_id: str) -> Path:
    return directory / f"{_INTERNAL_PREFIX}{transaction_id}.journal.next"


def _artifact_path(
    directory: Path,
    transaction_id: str,
    index: int,
    kind: str,
) -> Path:
    return directory / f"{_INTERNAL_PREFIX}{transaction_id}.{index}.{kind}"


def _journal_transaction_id(path: Path) -> str:
    match = _JOURNAL_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Invalid publication journal: {path}")
    return match.group(1)


def _identity(path: Path) -> _FileIdentity:
    file_stat = path.lstat()
    return _FileIdentity(file_stat.st_dev, file_stat.st_ino)


def _state_from_stat(file_stat: os.stat_result) -> _FileState:
    return _FileState(
        identity=_FileIdentity(file_stat.st_dev, file_stat.st_ino),
        size=file_stat.st_size,
        mode=file_stat.st_mode,
        mtime_ns=file_stat.st_mtime_ns,
        ctime_ns=file_stat.st_ctime_ns,
    )


def _file_state(path: Path) -> _FileState:
    return _state_from_stat(path.lstat())


def _optional_file_state(path: Path) -> _FileState | None:
    try:
        return _file_state(path)
    except FileNotFoundError:
        return None


def _optional_identity(path: Path) -> _FileIdentity | None:
    file_state = _optional_file_state(path)
    return file_state.identity if file_state is not None else None


def _move_owned_without_replacement(
    source: Path,
    destination: Path,
    expected_identity: _FileIdentity,
    *,
    expected_state: _FileState | None = None,
) -> None:
    source_state = _optional_file_state(source)
    if source_state is None:
        raise PublicationRecoveryError(f"Owned publication path disappeared: {source}")
    if source_state.identity != expected_identity or (
        expected_state is not None and source_state != expected_state
    ):
        raise PublicationRecoveryError(f"Owned publication path changed: {source}")

    _rename_without_replacement(source, destination)
    moved_state = _optional_file_state(destination)
    if (
        moved_state is not None
        and moved_state.identity == expected_identity
        and (
            expected_state is None
            or (
                moved_state.size == expected_state.size
                and moved_state.mode == expected_state.mode
                and moved_state.mtime_ns == expected_state.mtime_ns
            )
        )
    ):
        return

    try:
        _rename_without_replacement(destination, source)
    except Exception as restore_error:
        raise PublicationRecoveryError(
            "A concurrently changed file was preserved at "
            f"{destination}; publication recovery requires inspection"
        ) from restore_error
    raise PublicationRecoveryError(f"Owned publication path changed while it was moved: {source}")


def _unlink_owned(path: Path, expected_identity: _FileIdentity) -> None:
    identity = _optional_identity(path)
    if identity is None:
        return
    if identity != expected_identity:
        raise PublicationRecoveryError(f"Refusing to remove changed artifact: {path}")
    tombstone = path.parent / f"{_INTERNAL_PREFIX}{uuid.uuid4().hex}.unlink"
    _move_owned_without_replacement(
        path,
        tombstone,
        expected_identity,
    )
    if _optional_identity(tombstone) != expected_identity:
        raise PublicationRecoveryError(f"Refusing to remove changed artifact: {tombstone}")
    tombstone.unlink()


def _unlink_internal_file(path: Path) -> None:
    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        return
    mode = file_stat.st_mode
    if not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
        raise PublicationRecoveryError(f"Invalid publication artifact: {path}")
    _unlink_owned(
        path,
        _FileIdentity(file_stat.st_dev, file_stat.st_ino),
    )


def _remove_journal(
    journal_path: Path,
    expected_identity: _FileIdentity,
    directory: Path,
) -> None:
    _unlink_owned(journal_path, expected_identity)
    _fsync_directory(directory)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
