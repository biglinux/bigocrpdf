"""Ownership tests for temporary resource cleanup."""

import os
import signal
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


def test_main_thread_can_install_sigterm_after_worker_registered_atexit(monkeypatch) -> None:
    registered_atexit = []
    registered_signals = []
    current_thread = object()
    main_thread = object()
    monkeypatch.setattr(temp_manager, "_cleanup_registered", False)
    monkeypatch.setattr(temp_manager, "_sigterm_registered", False)
    monkeypatch.setattr(temp_manager.atexit, "register", registered_atexit.append)
    monkeypatch.setattr(temp_manager.threading, "current_thread", lambda: current_thread)
    monkeypatch.setattr(temp_manager.threading, "main_thread", lambda: main_thread)
    monkeypatch.setattr(
        temp_manager.signal,
        "signal",
        lambda signal_number, handler: registered_signals.append((signal_number, handler)),
    )

    temp_manager._register_cleanup()

    assert len(registered_atexit) == 1
    assert registered_signals == []

    current_thread = main_thread
    monkeypatch.setattr(temp_manager.signal, "getsignal", lambda _signal_number: signal.SIG_DFL)
    temp_manager._register_cleanup()

    assert len(registered_atexit) == 1
    assert len(registered_signals) == 1
    assert registered_signals[0][0] == signal.SIGTERM


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
