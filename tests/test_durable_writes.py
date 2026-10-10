"""Tests for atomic publication of documents and program state."""

import errno
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from bigocrpdf.utils import durable_writes
from bigocrpdf.utils.durable_writes import (
    copy_file_atomically,
    publish_file_exclusively,
    write_text_atomically,
    write_text_file_atomically,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_REPO_ROOT / "src")
    return environment


def test_writer_failure_preserves_target_and_removes_temporary_file(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    target.write_text("original", encoding="utf-8")

    def fail_after_partial_write(stream) -> None:
        stream.write("partial")
        raise OSError("simulated disk failure")

    with pytest.raises(OSError, match="simulated disk failure"):
        write_text_file_atomically(target, fail_after_partial_write)

    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.glob(".settings.json.*.tmp")) == []


@pytest.mark.parametrize("overwrite", [False, True])
def test_atomic_write_accepts_overlay_directory_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overwrite: bool
) -> None:
    target = tmp_path / "settings.json"
    if overwrite:
        target.write_text("original", encoding="utf-8")
    real_stat = Path.stat

    def overlay_stat(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if path == tmp_path:
            fields = list(result)
            fields[2] += 1  # OverlayFS can report distinct directory and file devices.
            return os.stat_result(fields)
        return result

    monkeypatch.setattr(Path, "stat", overlay_stat)

    published = write_text_atomically(target, '{"language": "pt"}', overwrite=overwrite)

    assert published == target
    assert json.loads(target.read_text(encoding="utf-8")) == {"language": "pt"}
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_copy_publishes_complete_file_and_preserves_source(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"complete pdf payload")
    source.chmod(0o644)
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    target = output_directory / "copy.pdf"

    published = copy_file_atomically(source, target, overwrite=True)

    assert published == target
    assert source.read_bytes() == b"complete pdf payload"
    assert target.read_bytes() == b"complete pdf payload"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert list(output_directory.iterdir()) == [target]


def test_atomic_copy_failure_preserves_existing_target(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"new payload")
    target = tmp_path / "target.pdf"
    target.write_bytes(b"original payload")

    with (
        patch(
            "bigocrpdf.utils.durable_writes.shutil.copyfileobj",
            side_effect=OSError("simulated copy failure"),
        ),
        pytest.raises(OSError, match="simulated copy failure"),
    ):
        copy_file_atomically(source, target, overwrite=True)

    assert source.read_bytes() == b"new payload"
    assert target.read_bytes() == b"original payload"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "source.pdf",
        "target.pdf",
    ]


@pytest.mark.parametrize(
    ("overwrite", "source_mode", "existing_mode"),
    (
        (False, 0o644, None),
        (True, 0o664, 0o640),
    ),
)
def test_publication_preserves_regular_source_access_mode(
    tmp_path: Path,
    overwrite: bool,
    source_mode: int,
    existing_mode: int | None,
) -> None:
    source = tmp_path / ".source"
    source.write_text("new payload", encoding="utf-8")
    source.chmod(source_mode | stat.S_ISUID)
    target = tmp_path / "output.pdf"
    if existing_mode is not None:
        target.write_text("old payload", encoding="utf-8")
        target.chmod(existing_mode)

    durable_writes.publish_file_atomically(source, target, overwrite=overwrite)

    assert target.read_text(encoding="utf-8") == "new payload"
    assert stat.S_IMODE(target.stat().st_mode) == source_mode


def test_valid_posix_filename_with_backslash_completes_transaction(
    tmp_path: Path,
) -> None:
    source = tmp_path / ".source"
    source.write_text("payload", encoding="utf-8")
    target = tmp_path / "page\\scan.pdf"

    published = durable_writes.publish_file_atomically(
        source,
        target,
        overwrite=True,
    )

    assert published == target
    assert target.read_text(encoding="utf-8") == "payload"


def test_concurrent_non_overwriting_publishers_choose_unique_names(
    tmp_path: Path,
) -> None:
    child_code = """
import sys
from pathlib import Path
from bigocrpdf.utils.durable_writes import publish_file_atomically

root = Path(sys.argv[1])
index = sys.argv[2]
source = root / f".source-{index}"
source.write_text(f"payload {index}", encoding="utf-8")
published = publish_file_atomically(source, root / "output.pdf", overwrite=False)
print(published.name)
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", child_code, str(tmp_path), str(index)],
            cwd=_REPO_ROOT,
            env=_subprocess_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in range(4)
    ]
    results = [process.communicate(timeout=10) for process in processes]

    assert [process.returncode for process in processes] == [0, 0, 0, 0]
    published_names = [stdout.strip() for stdout, _stderr in results]
    assert len(set(published_names)) == 4
    assert {path.read_text(encoding="utf-8") for path in tmp_path.glob("output*.pdf")} == {
        "payload 0",
        "payload 1",
        "payload 2",
        "payload 3",
    }


def test_program_state_is_replaced_without_syncing(tmp_path: Path) -> None:
    target = tmp_path / "config.json"
    target.write_text("{}", encoding="utf-8")

    with patch.object(durable_writes.os, "fsync") as fsync:
        write_text_atomically(target, '{"language": "pt"}', durable=False)

    assert fsync.call_count == 0
    assert json.loads(target.read_text(encoding="utf-8")) == {"language": "pt"}


def test_a_document_is_synced_before_and_after_its_rename(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    synced: list[str] = []
    real_fsync = os.fsync

    def record(descriptor: int) -> None:
        synced.append("directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file")
        real_fsync(descriptor)

    with patch.object(durable_writes.os, "fsync", side_effect=record):
        write_text_atomically(target, "text")

    assert synced == ["file", "directory"]


@pytest.mark.parametrize("kill_after", [1, 2, 3, 4])
def test_a_kill_at_any_rename_leaves_the_old_or_the_new_target(
    tmp_path: Path, kill_after: int
) -> None:
    """A settings file that went missing after a crash reset every setting."""
    target = tmp_path / "config.json"
    target.write_text("original", encoding="utf-8")
    child_code = """
import os, signal, sys
from bigocrpdf.utils import durable_writes

kill_after = int(sys.argv[2])
calls = 0

def counted(real):
    def run(*args, **kwargs):
        global calls
        result = real(*args, **kwargs)
        calls += 1
        if calls == kill_after:
            os.kill(os.getpid(), signal.SIGKILL)
        return result
    return run

durable_writes.os.replace = counted(os.replace)
durable_writes._rename_without_replacement = counted(durable_writes._rename_without_replacement)
durable_writes.write_text_atomically(sys.argv[1], "new")
"""
    subprocess.run(
        [sys.executable, "-c", child_code, str(target), str(kill_after)],
        cwd=_REPO_ROOT,
        env=_subprocess_environment(),
        check=False,
    )

    assert target.read_text(encoding="utf-8") in {"original", "new"}


@pytest.mark.parametrize("unsupported", [errno.EINVAL, errno.ENOSYS])
def test_no_replace_install_falls_back_where_rename_flags_are_unsupported(
    tmp_path: Path, unsupported: int
) -> None:
    source = tmp_path / ".staged"
    source.write_text("new", encoding="utf-8")
    taken = tmp_path / "taken.pdf"
    taken.write_text("existing", encoding="utf-8")

    with patch.object(
        durable_writes,
        "_rename_without_replacement",
        side_effect=OSError(unsupported, "unsupported"),
    ):
        with pytest.raises(FileExistsError):
            publish_file_exclusively(source, taken)
        publish_file_exclusively(source, tmp_path / "free.pdf")

    assert taken.read_text(encoding="utf-8") == "existing"
    assert (tmp_path / "free.pdf").read_text(encoding="utf-8") == "new"
    assert not source.exists()


def test_no_replace_install_works_without_hard_links(tmp_path: Path) -> None:
    source = tmp_path / ".staged"
    source.write_text("new", encoding="utf-8")
    target = tmp_path / "out.pdf"

    with (
        patch.object(
            durable_writes,
            "_rename_without_replacement",
            side_effect=OSError(errno.EINVAL, "unsupported"),
        ),
        patch.object(durable_writes.os, "link", side_effect=OSError(errno.EPERM, "no links")),
    ):
        publish_file_exclusively(source, target)

    assert target.read_text(encoding="utf-8") == "new"
    assert not source.exists()


def test_a_staged_file_on_another_filesystem_is_copied_beside_the_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / ".staged"
    source.write_text("new", encoding="utf-8")
    target = tmp_path / "out.pdf"
    real_replace = os.replace

    def cross_device_once(src, dst):
        if Path(src) == source:
            raise OSError(errno.EXDEV, "cross-device link")
        real_replace(src, dst)

    with patch.object(durable_writes.os, "replace", side_effect=cross_device_once):
        published = durable_writes.publish_file_atomically(source, target, overwrite=True)

    assert published == target
    assert target.read_text(encoding="utf-8") == "new"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["out.pdf"]
