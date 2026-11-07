"""Ownership tests for temporary resource cleanup."""

import fcntl
import os
import time
from pathlib import Path

from bigocrpdf.utils import temp_manager


def test_cleanup_preserves_other_process_cache_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_dir = tmp_path / "cache" / "bigocrpdf"
    cache_dir.mkdir(parents=True)
    other_run = cache_dir / "run-other"
    other_run.mkdir()
    other_file = other_run / "active.pdf"
    other_file.write_bytes(b"active")
    unrelated_file = cache_dir / "user-data.txt"
    unrelated_file.write_text("keep", encoding="utf-8")

    monkeypatch.setattr(temp_manager, "_APP_CACHE_DIR", cache_dir)
    monkeypatch.setattr(temp_manager, "_run_cache_dir", None, raising=False)
    monkeypatch.setattr(temp_manager, "_cleanup_registered", True)
    temp_manager._tracked_files.clear()
    temp_manager._tracked_dirs.clear()

    system_temp = Path(temp_manager.tempfile.gettempdir())

    def free_space(path: str | Path) -> int:
        return 100 if Path(path) == system_temp else 1_000_000_000

    monkeypatch.setattr(temp_manager, "get_free_space", free_space)
    descriptor, owned_path_string = temp_manager.mkstemp(needed_bytes=1_000)
    os.close(descriptor)
    owned_path = Path(owned_path_string)
    owned_run = owned_path.parent

    temp_manager.cleanup_all()

    assert not owned_path.exists()
    assert not owned_run.exists()
    assert other_file.read_bytes() == b"active"
    assert unrelated_file.read_text(encoding="utf-8") == "keep"


def test_remove_file_retries_after_transient_failure(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "temporary.pdf"
    path.write_bytes(b"data")
    temp_manager._tracked_files.clear()
    temp_manager.track_file(str(path))
    real_unlink = os.unlink
    attempts = 0

    def unlink(file_path: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("busy")
        real_unlink(file_path)

    monkeypatch.setattr(temp_manager.os, "unlink", unlink)

    temp_manager.remove_file(str(path))

    assert path.exists()
    assert str(path) in temp_manager._tracked_files

    temp_manager.cleanup_all()

    assert not path.exists()
    assert str(path) not in temp_manager._tracked_files


def test_remove_dir_retries_after_transient_failure(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "temporary"
    path.mkdir()
    (path / "file").write_bytes(b"data")
    temp_manager._tracked_dirs.clear()
    temp_manager.track_dir(str(path))
    real_rmtree = temp_manager.shutil.rmtree
    attempts = 0

    def rmtree(directory: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("busy")
        real_rmtree(directory)

    monkeypatch.setattr(temp_manager.shutil, "rmtree", rmtree)

    temp_manager.remove_dir(str(path))

    assert path.exists()
    assert str(path) in temp_manager._tracked_dirs

    temp_manager.cleanup_all()

    assert not path.exists()
    assert str(path) not in temp_manager._tracked_dirs


def test_staging_sweep_removes_only_dead_runs_of_this_user(tmp_path: Path) -> None:
    import socket
    import subprocess
    import sys

    dead = subprocess.Popen([sys.executable, "-c", ""])
    dead.wait(timeout=30)
    host = socket.gethostname()
    crashed = tmp_path / f".bigocr_ocr_bundle_{dead.pid}@{host}@abc"
    (crashed / "nested").mkdir(parents=True)
    (crashed / "nested" / "0-payload.pdf").write_bytes(b"%PDF")
    running = tmp_path / f".bigocr_stage_{os.getpid()}@{host}@def"
    running.mkdir()
    other_host = tmp_path / f".bigocr_stage_{dead.pid}@elsewhere.{host}@ghi"
    other_host.mkdir()
    target = tmp_path / "linked"
    target.mkdir()
    linked = tmp_path / f".bigocr_ocr_{dead.pid}@{host}@jkl"
    linked.symlink_to(target)
    document = tmp_path / "scan.pdf"
    document.write_bytes(b"%PDF")
    crashed_scratch = tmp_path / f"bigocrpdf_imgs_{dead.pid}@{host}@mno"
    (crashed_scratch / "chunk_imgs").mkdir(parents=True)
    running_scratch = tmp_path / f"bigocrpdf_imgs_{os.getpid()}@{host}@pqr"
    running_scratch.mkdir()
    # A PID from another namespace that looks dead here, its lock still held.
    foreign = tmp_path / f"bigocrpdf_imgs_{dead.pid}@{host}@stu"
    foreign.mkdir()
    held = os.open(foreign, os.O_RDONLY | os.O_DIRECTORY)
    fcntl.flock(held, fcntl.LOCK_EX)
    overlong = tmp_path / f"bigocrpdf_imgs_{'9' * 25}@{host}@vwx"
    overlong.mkdir()
    an_hour_ago = time.time() - 3600
    for path in tmp_path.iterdir():
        os.utime(path, (an_hour_ago, an_hour_ago), follow_symlinks=False)
    # Just made by a run in another PID namespace, not locked yet.
    unlocked = tmp_path / f"bigocrpdf_imgs_{dead.pid}@{host}@yza"
    unlocked.mkdir()

    with temp_manager.staging_directory(tmp_path, ".bigocr_stage_") as staging:
        assert Path(staging).name.startswith(f".bigocr_stage_{os.getpid()}@{host}@")
        survivors = {path.name for path in tmp_path.iterdir()}

    os.close(held)
    assert survivors == {
        running.name,
        running_scratch.name,
        foreign.name,
        unlocked.name,
        overlong.name,
        other_host.name,
        target.name,
        linked.name,
        document.name,
        Path(staging).name,
    }


_LATE_SIGTERM = """
import sys, threading, time
from bigocrpdf import run_quitting_on_termination
from bigocrpdf.utils.temp_manager import track_file

open(sys.argv[1], "w").close()
track_file(sys.argv[1])


class App:
    def run(self, argv):
        return 0


# A job's thread still finishing after the window closed.
threading.Thread(target=time.sleep, args=(30,)).start()
run_quitting_on_termination(App())
print("closed", flush=True)
"""


def test_sigterm_after_close_removes_tracked_files(tmp_path: Path) -> None:
    import signal
    import subprocess
    import sys

    tracked = tmp_path / "clipboard.tiff"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    gui = subprocess.Popen(
        [sys.executable, "-c", _LATE_SIGTERM, str(tracked)],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert gui.stdout is not None
        assert gui.stdout.readline() == "closed\n"
        gui.send_signal(signal.SIGTERM)
        assert gui.wait(timeout=10) == -signal.SIGTERM
    finally:
        gui.kill()
        gui.wait()
    assert not tracked.exists()
