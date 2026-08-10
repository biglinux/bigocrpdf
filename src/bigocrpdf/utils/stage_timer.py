"""Cheap process measurements: per-stage wall time, peak RSS, scratch size.

Named ``stage_timer`` rather than ``timer`` because ``utils.timer`` is a GLib
source-removal helper and has nothing to do with measurement.

Everything here reads ``/proc`` or walks a directory -- no sampling thread, no
psutil, no dependency.  Continuous sampling belongs in the benchmark harness,
which already does it from outside the process; in-process the interesting
number is the high-water mark the kernel already tracks.
"""

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_KIB_PER_MIB = 1024.0


class StageTimer:
    """Accumulates wall time per named stage across a whole run."""

    __slots__ = ("_totals",)

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._totals[name] = self._totals.get(name, 0.0) + elapsed_ms

    def add(self, name: str, milliseconds: float) -> None:
        """Fold in a duration measured elsewhere, such as a worker's trace."""
        self._totals[name] = self._totals.get(name, 0.0) + float(milliseconds)

    def totals(self) -> dict[str, float]:
        return {name: round(value, 3) for name, value in sorted(self._totals.items())}


def _proc_status_field(path: str, field: str) -> float | None:
    """Read a ``/proc/.../status`` value in kB, or None when unavailable.

    Absent outside Linux and for processes that have already exited, both of
    which are ordinary rather than exceptional.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(field):
                    return float(line.split()[1]) / _KIB_PER_MIB
    except (OSError, ValueError, IndexError):
        return None
    return None


def peak_rss_mb() -> float | None:
    """This process's high-water resident set size, in MiB.

    VmHWM is maintained by the kernel, so reading it costs one file read and
    cannot miss a spike the way periodic sampling can.
    """
    return _proc_status_field("/proc/self/status", "VmHWM:")


def process_rss_mb(pid: int) -> float | None:
    """Current resident set size of another process, in MiB.

    Used for the OCR subprocess, whose cost the pipeline budgets for but has
    never measured.
    """
    return _proc_status_field(f"/proc/{pid}/status", "VmRSS:")


def dir_bytes(path: Path | str) -> int:
    """Total size of the regular files under ``path``.

    Scratch directories hold rendered pages and can dwarf the PDF itself, and
    nothing has ever reported how large they get.  Symlinks are not followed
    and unreadable entries are skipped: this is a measurement, never a reason
    to fail a run.
    """
    total = 0
    stack = [Path(path)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total
