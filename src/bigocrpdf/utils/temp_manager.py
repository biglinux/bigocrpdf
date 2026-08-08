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
import threading
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
_sigterm_registered = False
_run_cache_dir: Path | None = None


def _register_cleanup() -> None:
    """Register atexit and SIGTERM handlers exactly once."""
    global _cleanup_registered, _sigterm_registered
    if not _cleanup_registered:
        _cleanup_registered = True
        atexit.register(cleanup_all)

    # SIGTERM handler can only be set from the main thread
    if not _sigterm_registered and threading.current_thread() is threading.main_thread():
        prev_handler = signal.getsignal(signal.SIGTERM)

        def _on_sigterm(signum, frame):
            cleanup_all()
            # Chain to previous handler
            if callable(prev_handler):
                prev_handler(signum, frame)
            else:
                raise SystemExit(1)

        signal.signal(signal.SIGTERM, _on_sigterm)
        _sigterm_registered = True


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
      1. If *needed_bytes* < 50 % of free space on the system temp dir → use it.
      2. Otherwise try ~/.cache/bigocrpdf if it has enough room.
      3. Fall back to the system temp dir regardless (let the OS handle it).
    """
    system_temp_dir = tempfile.gettempdir()
    tmp_free = get_free_space(system_temp_dir)

    # Prefer /tmp when plenty of headroom
    if needed_bytes == 0 or (tmp_free > 0 and needed_bytes < tmp_free * _TMPFS_HEADROOM_RATIO):
        return Path(tempfile.gettempdir())

    # Try persistent cache dir
    cache_free = get_free_space(str(_APP_CACHE_DIR.parent))
    if cache_free > needed_bytes + _MIN_FREE_BYTES:
        return _get_run_cache_dir()

    # Fallback — the system temp dir is still best-effort
    return Path(system_temp_dir)


def _get_run_cache_dir() -> Path:
    """Return this process's private cache namespace."""
    global _run_cache_dir
    if _run_cache_dir is not None and _run_cache_dir.is_dir():
        return _run_cache_dir

    _APP_CACHE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    run_dir = Path(
        tempfile.mkdtemp(
            prefix=f"run-{os.getpid()}-",
            dir=_APP_CACHE_DIR,
        )
    )
    _tracked_dirs.add(str(run_dir))
    _run_cache_dir = run_dir
    return run_dir


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


def remove_tracked_file(path: str) -> bool:
    """Remove an exact file only when this process registered its ownership."""
    if path not in _tracked_files:
        return False
    try:
        os.unlink(path)
    except FileNotFoundError:
        _tracked_files.discard(path)
        return True
    except OSError:
        return False
    _tracked_files.discard(path)
    return True


def remove_file(path: str) -> None:
    """Remove a tracked temp file immediately."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        _tracked_files.discard(path)
    except OSError:
        return
    else:
        _tracked_files.discard(path)


def remove_dir(path: str) -> None:
    """Remove a tracked temp directory immediately."""
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        _tracked_dirs.discard(path)
    except OSError:
        return
    else:
        _tracked_dirs.discard(path)


# ---------------------------------------------------------------------------
# Global cleanup
# ---------------------------------------------------------------------------


def _cleanup_tracked_files() -> None:
    """Remove tracked temporary files."""
    for path in list(_tracked_files):
        try:
            os.unlink(path)
        except FileNotFoundError:
            _tracked_files.discard(path)
        except OSError:
            continue
        else:
            _tracked_files.discard(path)
            logger.debug(f"Cleaned temp file: {path}")


def _cleanup_tracked_dirs() -> None:
    """Remove tracked temporary directories."""
    for path in list(_tracked_dirs):
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            _tracked_dirs.discard(path)
        except OSError:
            continue
        else:
            _tracked_dirs.discard(path)
            logger.debug(f"Cleaned temp dir: {path}")


def cleanup_all() -> None:
    """Remove all tracked temp files and directories.

    Safe to call multiple times (idempotent).
    """
    global _run_cache_dir
    _cleanup_tracked_files()
    _cleanup_tracked_dirs()
    _run_cache_dir = None
