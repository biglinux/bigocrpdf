"""Atomic publication of user documents and program state.

A reader sees the old file or the complete new one, never a partial file. User
documents are synced before and after the rename, so a power cut keeps one of
the two as well; program state (settings, queue, history, flags) is not,
because losing its last seconds costs nothing and its loaders fall back to
defaults.
"""

from __future__ import annotations

import ctypes
import errno
import os
import shutil
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

TextWriter = Callable[[TextIO], object]

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
# Filesystems without rename flags (FUSE, NFS) or without hard links (vfat).
_UNSUPPORTED = {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EPERM, errno.EOPNOTSUPP}
_NOFOLLOW_READ = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)


def read_regular_file_bytes(path: str | Path) -> bytes:
    """Read a regular file without following a final symbolic link."""
    descriptor = os.open(path, _NOFOLLOW_READ | getattr(os, "O_NONBLOCK", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f"Not a regular file: {path}")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            return stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def write_text_file_atomically(
    path: str | Path,
    writer: TextWriter,
    *,
    overwrite: bool = True,
    durable: bool = True,
) -> Path:
    """Write UTF-8 text through *writer* and publish it; return the path used.

    ``durable=False`` is for program state: atomic, but never synced.
    """
    target = Path(path)
    temp_path = _temp_beside(target)
    try:
        with open(temp_path, "w", encoding="utf-8") as stream:
            writer(stream)
        return publish_file_atomically(temp_path, target, overwrite=overwrite, durable=durable)
    finally:
        temp_path.unlink(missing_ok=True)


def write_text_atomically(
    path: str | Path,
    text: str,
    *,
    overwrite: bool = True,
    durable: bool = True,
) -> Path:
    """Publish an already-rendered UTF-8 text payload."""
    return write_text_file_atomically(
        path, lambda stream: stream.write(text), overwrite=overwrite, durable=durable
    )


def copy_file_atomically(
    source: str | Path,
    requested_target: str | Path,
    *,
    overwrite: bool,
) -> Path:
    """Copy a regular file beside its destination, then publish the complete copy."""
    target = Path(requested_target)
    temp_path = _temp_beside(target)
    try:
        _copy_regular_file(Path(source), temp_path)
        return publish_file_atomically(temp_path, target, overwrite=overwrite)
    finally:
        temp_path.unlink(missing_ok=True)


def publish_file_atomically(
    source: str | Path,
    requested_target: str | Path,
    *,
    overwrite: bool,
    durable: bool = True,
) -> Path:
    """Move a complete staged file onto its destination and return the path used.

    Without ``overwrite`` an existing name is never replaced: the file gets the
    first free ``name-N`` instead. A staged file on another filesystem is
    copied beside the destination first.
    """
    staged = Path(source)
    target = Path(requested_target)
    try:
        return _publish(staged, target, overwrite=overwrite, durable=durable)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
    local = _temp_beside(target)
    try:
        _copy_regular_file(staged, local)
        published = _publish(local, target, overwrite=overwrite, durable=durable)
    finally:
        local.unlink(missing_ok=True)
    staged.unlink()
    return published


def _publish(staged: Path, target: Path, *, overwrite: bool, durable: bool) -> Path:
    # A published document never carries set-user-ID, set-group-ID or sticky bits.
    mode = os.lstat(staged).st_mode
    if mode & 0o7000:
        os.chmod(staged, stat.S_IMODE(mode) & 0o777)
    if durable:
        _fsync_file(staged)
    if overwrite:
        os.replace(staged, target)
    else:
        counter = 0
        while True:
            candidate = numbered_path(target, counter)
            try:
                publish_file_exclusively(staged, candidate, durable=False)
                target = candidate
                break
            except FileExistsError:
                counter += 1
    if durable:
        fsync_directory(target.parent)
    return target


def publish_file_exclusively(source: Path, target: Path, *, durable: bool = True) -> None:
    """Rename a staged file to *target*; raise FileExistsError if it exists."""
    if durable:
        _fsync_file(source)
    try:
        _rename_without_replacement(source, target)
    except OSError as error:
        if error.errno not in _UNSUPPORTED:
            raise
        _link_without_replacement(source, target)
    if durable:
        fsync_directory(target.parent)


def numbered_path(path: Path, counter: int) -> Path:
    """``path`` itself for 0, else ``stem-counter.suffix``."""
    return path if counter == 0 else path.with_name(f"{path.stem}-{counter}{path.suffix}")


def _temp_beside(target: Path) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    return Path(name)


def _copy_regular_file(source: Path, destination: Path) -> None:
    descriptor = os.open(source, _NOFOLLOW_READ)
    with os.fdopen(descriptor, "rb") as source_stream:
        source_stat = os.fstat(source_stream.fileno())
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError(f"Copy source is not a regular file: {source}")
        with open(destination, "wb") as destination_stream:
            shutil.copyfileobj(source_stream, destination_stream)
            os.fchmod(destination_stream.fileno(), stat.S_IMODE(source_stat.st_mode) & 0o777)


def _rename_without_replacement(source: Path, target: Path) -> None:
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable", target) from error
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if (
        renameat2(_AT_FDCWD, os.fsencode(source), _AT_FDCWD, os.fsencode(target), _RENAME_NOREPLACE)
        == 0
    ):
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), target)
    raise OSError(error_number, os.strerror(error_number), target)


def _link_without_replacement(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno not in _UNSUPPORTED:
            raise
        # No hard links: claim the name, then replace the empty placeholder.
        os.close(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
        os.replace(source, target)
        return
    source.unlink()


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, _NOFOLLOW_READ)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
