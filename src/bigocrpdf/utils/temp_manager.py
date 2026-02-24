"""
Centralized temporary file management.

Provides smart temp directory selection based on available space, automatic
cleanup on exit (including SIGTERM), and tracking of all temp files/dirs
to prevent orphaned files.
"""

import atexit
import os
import shutil
import signal
import tempfile
from pathlib import Path

from bigocrpdf.utils.logger import logger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_APP_CACHE_DIR = Path.home() / ".cache" / "bigocrpdf"
_TMPFS_HEADROOM_RATIO = 0.50  # Use /tmp only if needed < 50% of free RAM
_MIN_FREE_BYTES = 100 * 1024 * 1024  # 100 MB absolute minimum

# ---------------------------------------------------------------------------
# Global registry (de-duplication safe; cleaned at process exit)
# ---------------------------------------------------------------------------
_tracked_files: set[str] = set()
_tracked_dirs: set[str] = set()
_cleanup_registered = False


def _register_cleanup() -> None:
    """Register atexit and SIGTERM handlers exactly once."""
    global _cleanup_registered
    if _cleanup_registered:
        return
    _cleanup_registered = True

    atexit.register(cleanup_all)

    # Also handle SIGTERM (e.g. system shutdown, kill <pid>)
    prev_handler = signal.getsignal(signal.SIGTERM)

    def _on_sigterm(signum, frame):
        cleanup_all()
        # Chain to previous handler
        if callable(prev_handler):
            prev_handler(signum, frame)
        else:
            raise SystemExit(1)

    signal.signal(signal.SIGTERM, _on_sigterm)


# ---------------------------------------------------------------------------
# Space queries
# ---------------------------------------------------------------------------


def get_free_space(path: str | Path) -> int:
    """Return free bytes on the filesystem containing *path*."""
    try:
        st = os.statvfs(str(path))
        return st.f_bavail * st.f_frsize
    except OSError:
        return 0


def check_disk_space(path: str | Path, needed_bytes: int) -> tuple[bool, str]:
    """Check whether *path* has at least *needed_bytes* free.

    Returns:
        (ok, message) — *ok* is False when space is insufficient.
    """
    free = get_free_space(path)
    if free == 0:
        return False, f"Cannot determine free space on {path}"
    if free < needed_bytes:
        from bigocrpdf.utils.i18n import _

        needed_mb = needed_bytes / (1024 * 1024)
        free_mb = free / (1024 * 1024)
        return False, _(
            "Not enough disk space. Need {needed:.0f} MB but only {free:.0f} MB available."
        ).format(needed=needed_mb, free=free_mb)
    return True, ""


def check_writable(path: str | Path) -> tuple[bool, str]:
    """Check whether *path* (or its nearest existing parent) is writable."""
    p = Path(path)
    while not p.exists():
        p = p.parent
    if not os.access(str(p), os.W_OK):
        from bigocrpdf.utils.i18n import _

        return False, _("No write permission for folder: {path}").format(path=str(p))
    return True, ""


# ---------------------------------------------------------------------------
# Smart temp directory selection
# ---------------------------------------------------------------------------


def choose_temp_base(needed_bytes: int = 0) -> Path:
    """Choose the best base directory for temporary files.

    Strategy:
      1. If *needed_bytes* < 50 % of free space on /tmp → use /tmp (fast, tmpfs).
      2. Otherwise try ~/.cache/bigocrpdf if it has enough room.
      3. Fall back to /tmp regardless (let the OS handle it).
    """
    tmp_free = get_free_space("/tmp")

    # Prefer /tmp when plenty of headroom
    if needed_bytes == 0 or (tmp_free > 0 and needed_bytes < tmp_free * _TMPFS_HEADROOM_RATIO):
        return Path(tempfile.gettempdir())

    # Try persistent cache dir
    cache_free = get_free_space(str(_APP_CACHE_DIR.parent))
    if cache_free > needed_bytes + _MIN_FREE_BYTES:
        _APP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return _APP_CACHE_DIR

    # Fallback — /tmp is still best-effort
    return Path(tempfile.gettempdir())


# ---------------------------------------------------------------------------
# Tracked temp creation helpers
# ---------------------------------------------------------------------------


def mkstemp(
    suffix: str = "",
    prefix: str = "bigocrpdf_",
    needed_bytes: int = 0,
) -> tuple[int, str]:
    """Create a temp file in the best directory, tracked for cleanup."""
    _register_cleanup()
    base = choose_temp_base(needed_bytes)
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=str(base))
    _tracked_files.add(path)
    return fd, path


def mkdtemp(
    suffix: str = "",
    prefix: str = "bigocrpdf_",
    needed_bytes: int = 0,
) -> str:
    """Create a temp directory in the best location, tracked for cleanup."""
    _register_cleanup()
    base = choose_temp_base(needed_bytes)
    path = tempfile.mkdtemp(suffix=suffix, prefix=prefix, dir=str(base))
    _tracked_dirs.add(path)
    return path


# ---------------------------------------------------------------------------
# Explicit tracking / un-tracking
# ---------------------------------------------------------------------------


def track_file(path: str) -> None:
    """Add an existing file to the cleanup registry."""
    _register_cleanup()
    _tracked_files.add(path)


def track_dir(path: str) -> None:
    """Add an existing directory to the cleanup registry."""
    _register_cleanup()
    _tracked_dirs.add(path)


def untrack_file(path: str) -> None:
    """Remove a file from tracking (caller takes ownership)."""
    _tracked_files.discard(path)


def untrack_dir(path: str) -> None:
    """Remove a directory from tracking."""
    _tracked_dirs.discard(path)


def remove_file(path: str) -> None:
    """Remove a tracked temp file immediately."""
    _tracked_files.discard(path)
    try:
        os.unlink(path)
    except OSError:
        pass


def remove_dir(path: str) -> None:
    """Remove a tracked temp directory immediately."""
    _tracked_dirs.discard(path)
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Global cleanup
# ---------------------------------------------------------------------------


def cleanup_all() -> None:
    """Remove all tracked temp files and directories.

    Safe to call multiple times (idempotent).
    """
    for f in list(_tracked_files):
        try:
            if os.path.exists(f):
                os.unlink(f)
                logger.debug(f"Cleaned temp file: {f}")
        except OSError:
            pass
    _tracked_files.clear()

    for d in list(_tracked_dirs):
        try:
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
                logger.debug(f"Cleaned temp dir: {d}")
        except OSError:
            pass
    _tracked_dirs.clear()

    # Also sweep stale bigocrpdf files in ~/.cache/bigocrpdf
    try:
        if _APP_CACHE_DIR.is_dir():
            for item in _APP_CACHE_DIR.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                except OSError:
                    pass
    except OSError:
        pass
