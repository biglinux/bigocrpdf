"""Tests for crash-safe durable file publication."""

import errno
import hashlib
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
    PublicationRecoveryError,
    copy_file_atomically,
    publish_files_transactionally,
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


def test_expected_source_content_rejects_a_changed_staged_payload(
    tmp_path: Path,
) -> None:
    source = tmp_path / ".source"
    source.write_bytes(b"changed payload")
    target = tmp_path / "target.pdf"
    target.write_bytes(b"original payload")
    expected_payload = b"expected payload"

    with pytest.raises(PublicationRecoveryError, match="content changed"):
        publish_files_transactionally(
            [(source, target)],
            overwrite=True,
            expected_source_content={
                source: (
                    hashlib.sha256(expected_payload).hexdigest(),
                    len(expected_payload),
                )
            },
        )

    assert source.read_bytes() == b"changed payload"
    assert target.read_bytes() == b"original payload"
    assert list(tmp_path.glob(".bigocr-publish-*")) == []


def test_atomic_copy_rejects_source_changed_during_copy(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"initial payload")
    target = tmp_path / "target.pdf"
    target.write_bytes(b"original target")

    def copy_then_mutate(source_stream, staged_stream) -> None:
        staged_stream.write(source_stream.read())
        source.write_bytes(b"changed while copying")

    with (
        patch(
            "bigocrpdf.utils.durable_writes.shutil.copyfileobj",
            side_effect=copy_then_mutate,
        ),
        pytest.raises(PublicationRecoveryError, match="Copy source changed"),
    ):
        copy_file_atomically(source, target, overwrite=True)

    assert target.read_bytes() == b"original target"
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


def test_prepared_snapshots_remain_private_until_publication(
    tmp_path: Path,
) -> None:
    source = tmp_path / ".source"
    source.write_text("new payload", encoding="utf-8")
    source.chmod(0o644)
    target = tmp_path / "output.pdf"
    target.write_text("old payload", encoding="utf-8")
    target.chmod(0o640)
    child_code = """
import os
import sys
from pathlib import Path
from bigocrpdf.utils import durable_writes

root = Path(sys.argv[1])
real_replace_journal = durable_writes._replace_journal

def replace_journal_and_die(path, transaction_id, phase, overwrite, entries):
    real_replace_journal(path, transaction_id, phase, overwrite, entries)
    if phase == "PREPARED":
        os._exit(94)

durable_writes._replace_journal = replace_journal_and_die
durable_writes.publish_file_atomically(
    root / ".source",
    root / "output.pdf",
    overwrite=True,
)
"""
    child = subprocess.run(
        [sys.executable, "-c", child_code, str(tmp_path)],
        capture_output=True,
        cwd=_REPO_ROOT,
        env=_subprocess_environment(),
        text=True,
        timeout=10,
        check=False,
    )

    assert child.returncode == 94, child.stderr
    artifacts = sorted(
        (
            *tmp_path.glob(".bigocr-publish-*.new"),
            *tmp_path.glob(".bigocr-publish-*.backup"),
        )
    )
    assert len(artifacts) == 2
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in artifacts)

    durable_writes.recover_pending_publications(tmp_path)
    assert target.read_text(encoding="utf-8") == "old payload"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert list(tmp_path.glob(".bigocr-publish-*")) == []


def test_post_replace_sync_failure_restores_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    target.write_text("original", encoding="utf-8")
    original_sync = durable_writes._fsync_directory
    failure_injected = False

    def fail_once_after_publication(directory: Path) -> None:
        nonlocal failure_injected
        if not failure_injected and target.read_text(encoding="utf-8") == "replacement":
            failure_injected = True
            raise OSError("simulated directory sync failure")
        original_sync(directory)

    with (
        patch.object(
            durable_writes,
            "_fsync_directory",
            side_effect=fail_once_after_publication,
        ),
        pytest.raises(OSError, match="simulated directory sync failure"),
    ):
        write_text_atomically(target, "replacement")

    assert failure_injected
    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.glob(".settings.json.*")) == []


def test_post_replace_sync_failure_removes_new_target(tmp_path: Path) -> None:
    target = tmp_path / "new.json"
    original_sync = durable_writes._fsync_directory
    failure_injected = False

    def fail_once_after_publication(directory: Path) -> None:
        nonlocal failure_injected
        if not failure_injected and target.exists():
            failure_injected = True
            raise OSError("simulated directory sync failure")
        original_sync(directory)

    with (
        patch.object(
            durable_writes,
            "_fsync_directory",
            side_effect=fail_once_after_publication,
        ),
        pytest.raises(OSError, match="simulated directory sync failure"),
    ):
        write_text_atomically(target, "new")

    assert failure_injected
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_file_set_failure_removes_every_new_publication(tmp_path: Path) -> None:
    first_staged = tmp_path / ".first.staged"
    first_staged.write_text("first", encoding="utf-8")
    missing_staged = tmp_path / ".missing.staged"
    first_target = tmp_path / "first.pdf"
    second_target = tmp_path / "second.pdf"

    with pytest.raises(FileNotFoundError):
        publish_files_transactionally(
            [(first_staged, first_target), (missing_staged, second_target)],
            overwrite=False,
        )

    assert not first_target.exists()
    assert not second_target.exists()


def test_non_overwrite_uses_one_collision_counter_for_the_entire_batch(
    tmp_path: Path,
) -> None:
    first_staged = tmp_path / ".first.staged"
    second_staged = tmp_path / ".second.staged"
    first_staged.write_text("new first", encoding="utf-8")
    second_staged.write_text("new second", encoding="utf-8")
    first_target = tmp_path / "first.pdf"
    second_target = tmp_path / "second.pdf"
    second_target.write_text("existing second", encoding="utf-8")

    published = publish_files_transactionally(
        [(first_staged, first_target), (second_staged, second_target)],
        overwrite=False,
    )

    assert published == [
        tmp_path / "first-1.pdf",
        tmp_path / "second-1.pdf",
    ]
    assert not first_target.exists()
    assert second_target.read_text(encoding="utf-8") == "existing second"
    assert published[0].read_text(encoding="utf-8") == "new first"
    assert published[1].read_text(encoding="utf-8") == "new second"


@pytest.mark.parametrize(
    ("case", "error_match"),
    (
        ("counter_zero", "preserve requested paths"),
        ("count", "candidate count"),
        ("duplicate", "duplicate paths"),
        ("directory", "publication directory"),
        ("reserved", "reserved internal prefix"),
    ),
)
def test_collision_target_candidates_reject_invalid_families(
    tmp_path: Path,
    case: str,
    error_match: str,
) -> None:
    first_staged = tmp_path / ".first.staged"
    second_staged = tmp_path / ".second.staged"
    first_staged.write_text("new first", encoding="utf-8")
    second_staged.write_text("new second", encoding="utf-8")
    first_target = tmp_path / "first.pdf"
    second_target = tmp_path / "second.pdf"
    first_target.write_text("existing first", encoding="utf-8")
    other_dir = tmp_path / "other"
    other_dir.mkdir()

    def candidates(counter: int) -> list[Path]:
        if case == "counter_zero":
            return [
                tmp_path / f"first-{counter + 1}.pdf",
                tmp_path / f"second-{counter + 1}.pdf",
            ]
        if counter == 0:
            return [first_target, second_target]
        if case == "count":
            return [tmp_path / "first-1.pdf"]
        if case == "duplicate":
            duplicate = tmp_path / "duplicate.pdf"
            return [duplicate, duplicate]
        if case == "directory":
            return [tmp_path / "first-1.pdf", other_dir / "second-1.pdf"]
        return [
            tmp_path / "first-1.pdf",
            tmp_path / ".bigocr-publish-forbidden",
        ]

    with pytest.raises(ValueError, match=error_match):
        publish_files_transactionally(
            [
                (first_staged, first_target),
                (second_staged, second_target),
            ],
            overwrite=False,
            target_candidates=candidates,
        )

    assert first_target.read_text(encoding="utf-8") == "existing first"
    assert not second_target.exists()


def test_file_set_failure_restores_every_overwritten_target(tmp_path: Path) -> None:
    first_target = tmp_path / "first.pdf"
    second_target = tmp_path / "second.pdf"
    first_target.write_text("original first", encoding="utf-8")
    second_target.write_text("original second", encoding="utf-8")
    first_staged = tmp_path / ".first.staged"
    first_staged.write_text("replacement", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        publish_files_transactionally(
            [(first_staged, first_target), (tmp_path / ".missing.staged", second_target)],
            overwrite=True,
        )

    assert first_target.read_text(encoding="utf-8") == "original first"
    assert second_target.read_text(encoding="utf-8") == "original second"
    assert list(tmp_path.glob(".*.backup")) == []


def test_publication_retires_superseded_outputs_in_the_same_transaction(
    tmp_path: Path,
) -> None:
    target = tmp_path / "output.pdf"
    target.write_text("old output", encoding="utf-8")
    first_stale = tmp_path / "output-01.pdf"
    second_stale = tmp_path / "output-02.pdf"
    first_stale.write_text("old part one", encoding="utf-8")
    second_stale.write_text("old part two", encoding="utf-8")
    staged = tmp_path / ".output.staged"
    staged.write_text("new output", encoding="utf-8")

    published = publish_files_transactionally(
        [(staged, target)],
        overwrite=True,
        retire_targets=[first_stale, second_stale],
    )

    assert published == [target]
    assert target.read_text(encoding="utf-8") == "new output"
    assert not first_stale.exists()
    assert not second_stale.exists()
    assert list(tmp_path.glob(".bigocr-publish-*")) == []


def test_retirement_failure_restores_publications_and_retired_outputs(
    tmp_path: Path,
) -> None:
    target = tmp_path / "output.pdf"
    target.write_text("old output", encoding="utf-8")
    first_stale = tmp_path / "output-01.pdf"
    second_stale = tmp_path / "output-02.pdf"
    first_stale.write_text("old part one", encoding="utf-8")
    second_stale.write_text("old part two", encoding="utf-8")
    staged = tmp_path / ".output.staged"
    staged.write_text("new output", encoding="utf-8")
    real_move_owned = durable_writes._move_owned_without_replacement
    retirements_seen = 0

    def fail_second_retirement(
        source: Path,
        destination: Path,
        expected_identity,
        *,
        expected_state=None,
    ) -> None:
        nonlocal retirements_seen
        if source in {first_stale, second_stale}:
            retirements_seen += 1
            if retirements_seen == 2:
                raise OSError("simulated retirement failure")
        real_move_owned(
            source,
            destination,
            expected_identity,
            expected_state=expected_state,
        )

    with (
        patch.object(
            durable_writes,
            "_move_owned_without_replacement",
            side_effect=fail_second_retirement,
        ),
        pytest.raises(OSError, match="simulated retirement failure"),
    ):
        publish_files_transactionally(
            [(staged, target)],
            overwrite=True,
            retire_targets=[first_stale, second_stale],
        )

    assert target.read_text(encoding="utf-8") == "old output"
    assert first_stale.read_text(encoding="utf-8") == "old part one"
    assert second_stale.read_text(encoding="utf-8") == "old part two"
    assert list(tmp_path.glob(".bigocr-publish-*")) == []


def test_retirement_never_deletes_a_concurrent_replacement(tmp_path: Path) -> None:
    target = tmp_path / "output.pdf"
    target.write_text("old output", encoding="utf-8")
    stale = tmp_path / "output-01.pdf"
    stale.write_text("old part", encoding="utf-8")
    staged = tmp_path / ".output.staged"
    staged.write_text("new output", encoding="utf-8")
    real_optional_file_state = durable_writes._optional_file_state
    stale_state_checks = 0

    def replace_after_state_check(path: Path):
        nonlocal stale_state_checks
        file_state = real_optional_file_state(path)
        if path == stale:
            stale_state_checks += 1
            if stale_state_checks == 3:
                stale.unlink()
                stale.write_text("concurrent replacement", encoding="utf-8")
        return file_state

    with (
        patch.object(
            durable_writes,
            "_optional_file_state",
            side_effect=replace_after_state_check,
        ),
        pytest.raises(PublicationRecoveryError),
    ):
        publish_files_transactionally(
            [(staged, target)],
            overwrite=True,
            retire_targets=[stale],
        )

    assert stale.read_text(encoding="utf-8") == "concurrent replacement"


def test_rollback_never_deletes_a_concurrent_replacement(tmp_path: Path) -> None:
    target = tmp_path / "output.pdf"
    staged = tmp_path / ".output.staged"
    staged.write_text("new output", encoding="utf-8")
    real_sync = durable_writes._fsync_directory
    real_rollback_action = durable_writes._rollback_action
    publication_seen = False
    replacement_injected = False

    def fail_sync_after_publication(directory: Path) -> None:
        nonlocal publication_seen
        if target.exists() and target.read_text(encoding="utf-8") == "new output":
            publication_seen = True
        if publication_seen:
            raise OSError("force rollback")
        real_sync(directory)

    def replace_after_rollback_decision(entry, transaction_id, index, directory):
        nonlocal replacement_injected
        action = real_rollback_action(entry, transaction_id, index, directory)
        if not replacement_injected and action == "remove":
            target.unlink()
            target.write_text("concurrent replacement", encoding="utf-8")
            replacement_injected = True
        return action

    with (
        patch.object(
            durable_writes,
            "_fsync_directory",
            side_effect=fail_sync_after_publication,
        ),
        patch.object(
            durable_writes,
            "_rollback_action",
            side_effect=replace_after_rollback_decision,
        ),
        pytest.raises(PublicationRecoveryError),
    ):
        publish_files_transactionally(
            [(staged, target)],
            overwrite=True,
        )

    assert replacement_injected
    assert target.read_text(encoding="utf-8") == "concurrent replacement"


def test_recovery_restores_retired_output_after_process_death(
    tmp_path: Path,
) -> None:
    target = tmp_path / "output.pdf"
    target.write_text("old output", encoding="utf-8")
    stale = tmp_path / "output-01.pdf"
    stale.write_text("old part", encoding="utf-8")
    staged = tmp_path / ".output.staged"
    staged.write_text("new output", encoding="utf-8")
    child_code = """
import os
import sys
from pathlib import Path
from bigocrpdf.utils import durable_writes

root = Path(sys.argv[1])
stale = root / "output-01.pdf"
real_move_owned = durable_writes._move_owned_without_replacement

def move_retirement_and_die(source, destination, expected_identity, *, expected_state=None):
    real_move_owned(
        source,
        destination,
        expected_identity,
        expected_state=expected_state,
    )
    if Path(source) == stale:
        os._exit(79)

durable_writes._move_owned_without_replacement = move_retirement_and_die
durable_writes.publish_files_transactionally(
    [(root / ".output.staged", root / "output.pdf")],
    overwrite=True,
    retire_targets=[stale],
)
"""
    child = subprocess.run(
        [sys.executable, "-c", child_code, str(tmp_path)],
        capture_output=True,
        cwd=_REPO_ROOT,
        env=_subprocess_environment(),
        text=True,
        timeout=10,
        check=False,
    )

    assert child.returncode == 79, child.stderr
    durable_writes.recover_pending_publications(tmp_path)

    assert target.read_text(encoding="utf-8") == "old output"
    assert stale.read_text(encoding="utf-8") == "old part"
    assert list(tmp_path.glob(".bigocr-publish-*")) == []


def test_file_set_rollback_restores_symlink_without_touching_victim(tmp_path: Path) -> None:
    victim = tmp_path / "victim.pdf"
    victim.write_text("protected", encoding="utf-8")
    linked_target = tmp_path / "first.pdf"
    linked_target.symlink_to(victim)
    second_target = tmp_path / "second.pdf"
    second_target.write_text("original second", encoding="utf-8")
    first_staged = tmp_path / ".first.staged"
    first_staged.write_text("replacement", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        publish_files_transactionally(
            [(first_staged, linked_target), (tmp_path / ".missing.staged", second_target)],
            overwrite=True,
        )

    assert linked_target.is_symlink()
    assert linked_target.resolve() == victim
    assert victim.read_text(encoding="utf-8") == "protected"


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("kill_after_publications", [1, 2])
def test_recovery_rolls_back_process_death_before_commit(
    tmp_path: Path,
    overwrite: bool,
    kill_after_publications: int,
) -> None:
    first_target = tmp_path / "first.pdf"
    second_target = tmp_path / "second.pdf"
    if overwrite:
        first_target.write_text("old first", encoding="utf-8")
        second_target.write_text("old second", encoding="utf-8")
    first_staged = tmp_path / ".first.staged"
    second_staged = tmp_path / ".second.staged"
    first_staged.write_text("new first", encoding="utf-8")
    second_staged.write_text("new second", encoding="utf-8")
    child_code = """
import os
import sys
from pathlib import Path
from bigocrpdf.utils import durable_writes

root = Path(sys.argv[1])
overwrite = sys.argv[2] == "1"
kill_after = int(sys.argv[3])
targets = {root / "first.pdf", root / "second.pdf"}
published = 0
real_replace = durable_writes.os.replace
real_rename_without_replacement = durable_writes._rename_without_replacement

def maybe_die(destination):
    global published
    if Path(destination) not in targets:
        return
    published += 1
    if published == kill_after:
        os._exit(77)

def replace_and_maybe_die(source, destination):
    real_replace(source, destination)
    maybe_die(destination)

def rename_and_maybe_die(source, destination):
    real_rename_without_replacement(source, destination)
    maybe_die(destination)

durable_writes.os.replace = replace_and_maybe_die
durable_writes._rename_without_replacement = rename_and_maybe_die
durable_writes.publish_files_transactionally(
    [
        (root / ".first.staged", root / "first.pdf"),
        (root / ".second.staged", root / "second.pdf"),
    ],
    overwrite=overwrite,
)
"""

    child = subprocess.run(
        [
            sys.executable,
            "-c",
            child_code,
            str(tmp_path),
            "1" if overwrite else "0",
            str(kill_after_publications),
        ],
        capture_output=True,
        cwd=_REPO_ROOT,
        env=_subprocess_environment(),
        text=True,
        timeout=10,
        check=False,
    )

    assert child.returncode == 77, child.stderr
    durable_writes.recover_pending_publications(tmp_path)

    if overwrite:
        assert first_target.read_text(encoding="utf-8") == "old first"
        assert second_target.read_text(encoding="utf-8") == "old second"
    else:
        assert not first_target.exists()
        assert not second_target.exists()
    assert list(tmp_path.glob(".bigocr-publish-*")) == []


def test_recovery_rejects_untrusted_journal_without_touching_target(tmp_path: Path) -> None:
    target = tmp_path / "important.pdf"
    target.write_text("keep", encoding="utf-8")
    journal = tmp_path / ".bigocr-publish-forged.journal"
    journal.write_text('{"version": 1, "entries": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="publication journal"):
        durable_writes.recover_pending_publications(tmp_path)

    assert target.read_text(encoding="utf-8") == "keep"


def test_recovery_discards_partial_initial_journal_write(tmp_path: Path) -> None:
    source = tmp_path / ".source"
    source.write_text("new payload", encoding="utf-8")
    target = tmp_path / "output.pdf"
    child_code = """
import os
import sys
from pathlib import Path
from bigocrpdf.utils import durable_writes

root = Path(sys.argv[1])

def write_partial_journal_and_die(path, payload):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    os.write(descriptor, payload[:8])
    os.fsync(descriptor)
    os._exit(92)

durable_writes._write_bytes_exclusively = write_partial_journal_and_die
durable_writes.publish_file_atomically(
    root / ".source",
    root / "output.pdf",
    overwrite=True,
)
"""
    child = subprocess.run(
        [sys.executable, "-c", child_code, str(tmp_path)],
        capture_output=True,
        cwd=_REPO_ROOT,
        env=_subprocess_environment(),
        text=True,
        timeout=10,
        check=False,
    )

    assert child.returncode == 92, child.stderr
    durable_writes.recover_pending_publications(tmp_path)
    assert list(tmp_path.glob(".bigocr-publish-*")) == []

    published = durable_writes.publish_file_atomically(source, target, overwrite=True)
    assert published == target
    assert target.read_text(encoding="utf-8") == "new payload"


def test_recovery_restores_mode_after_death_during_backup_restore(
    tmp_path: Path,
) -> None:
    source = tmp_path / ".source"
    source.write_text("new payload", encoding="utf-8")
    target = tmp_path / "output.pdf"
    target.write_text("old payload", encoding="utf-8")
    target.chmod(0o640)
    child_code = """
import os
import sys
from pathlib import Path
from bigocrpdf.utils import durable_writes

root = Path(sys.argv[1])
target = root / "output.pdf"
real_move_owned = durable_writes._move_owned_without_replacement
real_sync = durable_writes._fsync_directory
publication_seen = False

def fail_sync_after_publication(directory):
    global publication_seen
    if target.exists() and target.read_text(encoding="utf-8") == "new payload":
        publication_seen = True
    if publication_seen:
        raise OSError("force rollback")
    real_sync(directory)

def move_and_die_after_original_restore(
    source,
    destination,
    expected_identity,
    *,
    expected_state=None,
):
    real_move_owned(
        source,
        destination,
        expected_identity,
        expected_state=expected_state,
    )
    if Path(source).name.endswith(".0.retired") and Path(destination) == target:
        os._exit(93)

durable_writes._fsync_directory = fail_sync_after_publication
durable_writes._move_owned_without_replacement = move_and_die_after_original_restore
durable_writes.publish_file_atomically(
    root / ".source",
    target,
    overwrite=True,
)
"""
    child = subprocess.run(
        [sys.executable, "-c", child_code, str(tmp_path)],
        capture_output=True,
        cwd=_REPO_ROOT,
        env=_subprocess_environment(),
        text=True,
        timeout=10,
        check=False,
    )

    assert child.returncode == 93, child.stderr
    assert target.read_text(encoding="utf-8") == "old payload"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640

    restored_identity = (target.stat().st_dev, target.stat().st_ino)
    target_fsync_seen = False
    real_fsync = durable_writes.os.fsync

    def record_target_fsync(descriptor: int) -> None:
        nonlocal target_fsync_seen
        descriptor_stat = os.fstat(descriptor)
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) == restored_identity:
            target_fsync_seen = True
        real_fsync(descriptor)

    with patch.object(durable_writes.os, "fsync", side_effect=record_target_fsync):
        durable_writes.recover_pending_publications(tmp_path)

    assert target_fsync_seen
    assert target.read_text(encoding="utf-8") == "old payload"
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert list(tmp_path.glob(".bigocr-publish-*")) == []


def test_file_set_rejects_targets_in_different_directories(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_staged = first_dir / ".first.staged"
    second_staged = second_dir / ".second.staged"
    first_staged.write_text("first", encoding="utf-8")
    second_staged.write_text("second", encoding="utf-8")

    with pytest.raises(ValueError, match="same directory"):
        publish_files_transactionally(
            [
                (first_staged, first_dir / "first.pdf"),
                (second_staged, second_dir / "second.pdf"),
            ],
            overwrite=True,
        )

    assert first_staged.exists()
    assert second_staged.exists()
    assert not (first_dir / "first.pdf").exists()
    assert not (second_dir / "second.pdf").exists()


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize(
    ("kill_point", "expect_new"),
    [
        ("PREPARING", False),
        ("PREPARED", False),
        ("TARGETS_SYNCED", False),
        ("COMMITTED", True),
        ("CLEANUP", True),
    ],
)
def test_recovery_honors_the_durable_transaction_phase(
    tmp_path: Path,
    overwrite: bool,
    kill_point: str,
    expect_new: bool,
) -> None:
    first_target = tmp_path / "first.pdf"
    second_target = tmp_path / "second.pdf"
    if overwrite:
        first_target.write_text("old first", encoding="utf-8")
        second_target.write_text("old second", encoding="utf-8")
    (tmp_path / ".first.staged").write_text("new first", encoding="utf-8")
    (tmp_path / ".second.staged").write_text("new second", encoding="utf-8")
    child_code = """
import os
import sys
from pathlib import Path
from bigocrpdf.utils import durable_writes

root = Path(sys.argv[1])
overwrite = sys.argv[2] == "1"
kill_point = sys.argv[3]
targets = [root / "first.pdf", root / "second.pdf"]

real_write_new_journal = durable_writes._write_new_journal
real_replace_journal = durable_writes._replace_journal
real_sync_directory = durable_writes._fsync_directory
real_snapshot_regular_file = durable_writes._snapshot_regular_file
real_remove_journal = durable_writes._remove_journal

def write_new_journal_and_maybe_die(*args, **kwargs):
    real_write_new_journal(*args, **kwargs)

def replace_journal_and_maybe_die(path, transaction_id, phase, overwrite, entries):
    real_replace_journal(path, transaction_id, phase, overwrite, entries)
    if kill_point == phase:
        os._exit(79)

def sync_directory_and_maybe_die(directory):
    real_sync_directory(directory)
    if (
        kill_point == "TARGETS_SYNCED"
        and all(path.exists() and path.read_text(encoding="utf-8").startswith("new") for path in targets)
    ):
        journals = list(root.glob(".bigocr-publish-*.journal"))
        if journals and '"phase":"PREPARED"' in journals[0].read_text(encoding="utf-8"):
            os._exit(79)

def snapshot_and_maybe_die(source, destination, expected_identity):
    snapshot_identity = real_snapshot_regular_file(source, destination, expected_identity)
    if kill_point == "PREPARING" and Path(destination).name.endswith(".0.new"):
        os._exit(79)
    return snapshot_identity

def remove_journal_and_maybe_die(path, expected_identity, directory):
    if kill_point == "CLEANUP":
        os._exit(79)
    real_remove_journal(path, expected_identity, directory)

durable_writes._write_new_journal = write_new_journal_and_maybe_die
durable_writes._replace_journal = replace_journal_and_maybe_die
durable_writes._fsync_directory = sync_directory_and_maybe_die
durable_writes._snapshot_regular_file = snapshot_and_maybe_die
durable_writes._remove_journal = remove_journal_and_maybe_die
durable_writes.publish_files_transactionally(
    [
        (root / ".first.staged", root / "first.pdf"),
        (root / ".second.staged", root / "second.pdf"),
    ],
    overwrite=overwrite,
)
"""

    child = subprocess.run(
        [
            sys.executable,
            "-c",
            child_code,
            str(tmp_path),
            "1" if overwrite else "0",
            kill_point,
        ],
        capture_output=True,
        cwd=_REPO_ROOT,
        env=_subprocess_environment(),
        text=True,
        timeout=10,
        check=False,
    )

    assert child.returncode == 79, child.stderr
    durable_writes.recover_pending_publications(tmp_path)

    if expect_new:
        assert first_target.read_text(encoding="utf-8") == "new first"
        assert second_target.read_text(encoding="utf-8") == "new second"
    elif overwrite:
        assert first_target.read_text(encoding="utf-8") == "old first"
        assert second_target.read_text(encoding="utf-8") == "old second"
    else:
        assert not first_target.exists()
        assert not second_target.exists()
    assert list(tmp_path.glob(".bigocr-publish-*")) == []


def test_non_overwrite_retries_after_noncooperating_name_collision(
    tmp_path: Path,
) -> None:
    source = tmp_path / ".source"
    source.write_text("our payload", encoding="utf-8")
    target = tmp_path / "output.pdf"
    real_rename = durable_writes._rename_without_replacement
    collision_injected = False

    def collide_once(source_path, destination):
        nonlocal collision_injected
        if not collision_injected and Path(destination) == target:
            collision_injected = True
            target.write_text("external payload", encoding="utf-8")
        return real_rename(source_path, destination)

    with patch.object(
        durable_writes,
        "_rename_without_replacement",
        side_effect=collide_once,
    ):
        published = durable_writes.publish_file_atomically(
            source,
            target,
            overwrite=False,
        )

    assert collision_injected
    assert target.read_text(encoding="utf-8") == "external payload"
    assert published == tmp_path / "output-1.pdf"
    assert published.read_text(encoding="utf-8") == "our payload"
    assert list(tmp_path.glob(".bigocr-publish-*")) == []


def test_non_overwrite_preserves_same_inode_collision_before_retry(
    tmp_path: Path,
) -> None:
    source = tmp_path / ".source"
    source.write_text("our payload", encoding="utf-8")
    target = tmp_path / "output.pdf"
    real_rename = durable_writes._rename_without_replacement
    collision_inode = None

    def collide_once(snapshot_path, destination):
        nonlocal collision_inode
        if collision_inode is None and Path(destination) == target:
            os.link(snapshot_path, target)
            collision_inode = target.stat().st_ino
        return real_rename(snapshot_path, destination)

    with patch.object(
        durable_writes,
        "_rename_without_replacement",
        side_effect=collide_once,
    ):
        published = durable_writes.publish_file_atomically(
            source,
            target,
            overwrite=False,
        )

    assert collision_inode is not None
    assert target.stat().st_ino == collision_inode
    assert target.read_text(encoding="utf-8") == "our payload"
    assert published == tmp_path / "output-1.pdf"
    assert published.read_text(encoding="utf-8") == "our payload"
    assert list(tmp_path.glob(".bigocr-publish-*")) == []


def test_collision_recovery_survives_a_second_sync_failure(tmp_path: Path) -> None:
    source = tmp_path / ".source"
    source.write_text("our payload", encoding="utf-8")
    target = tmp_path / "output.pdf"
    real_rename = durable_writes._rename_without_replacement
    real_sync = durable_writes._fsync_directory
    collision_injected = False

    def collide_once(snapshot_path, destination):
        nonlocal collision_injected
        if not collision_injected and Path(destination) == target:
            collision_injected = True
            target.write_text("external payload", encoding="utf-8")
        return real_rename(snapshot_path, destination)

    def fail_rollback_sync(directory: Path) -> None:
        if collision_injected:
            raise OSError("simulated rollback sync failure")
        real_sync(directory)

    with (
        patch.object(
            durable_writes,
            "_rename_without_replacement",
            side_effect=collide_once,
        ),
        patch.object(
            durable_writes,
            "_fsync_directory",
            side_effect=fail_rollback_sync,
        ),
        pytest.raises(PublicationRecoveryError, match="could not be rolled back"),
    ):
        durable_writes.publish_file_atomically(source, target, overwrite=False)

    assert target.read_text(encoding="utf-8") == "external payload"
    assert list(tmp_path.glob(".bigocr-publish-*.journal"))

    durable_writes.recover_pending_publications(tmp_path)
    published = durable_writes.publish_file_atomically(
        source,
        target,
        overwrite=False,
    )
    assert target.read_text(encoding="utf-8") == "external payload"
    assert published == tmp_path / "output-1.pdf"
    assert published.read_text(encoding="utf-8") == "our payload"
    assert list(tmp_path.glob(".bigocr-publish-*")) == []


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
    assert list(tmp_path.glob(".bigocr-publish-*")) == []


def test_published_snapshot_is_independent_when_staged_unlink_fails(
    tmp_path: Path,
) -> None:
    source = tmp_path / ".source"
    source.write_text("original snapshot", encoding="utf-8")
    target = tmp_path / "output.pdf"
    real_unlink_owned = durable_writes._unlink_owned

    def refuse_staged_unlink(path: Path, expected_identity):
        if path == source:
            raise PermissionError("simulated staged unlink failure")
        return real_unlink_owned(path, expected_identity)

    with patch.object(
        durable_writes,
        "_unlink_owned",
        side_effect=refuse_staged_unlink,
    ):
        published = durable_writes.publish_file_atomically(
            source,
            target,
            overwrite=True,
        )

    assert published == target
    assert source.exists()
    assert source.stat().st_ino != target.stat().st_ino
    source.write_text("mutated staged file", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "original snapshot"
    assert list(tmp_path.glob(".bigocr-publish-*")) == []


def test_staged_cleanup_preserves_a_concurrent_replacement(tmp_path: Path) -> None:
    source = tmp_path / ".source"
    source.write_text("original staged payload", encoding="utf-8")
    replacement = tmp_path / ".replacement"
    replacement.write_text("concurrent replacement", encoding="utf-8")
    target = tmp_path / "output.pdf"
    real_optional_identity = durable_writes._optional_identity
    replacement_injected = False

    def replace_after_identity_check(path: Path):
        nonlocal replacement_injected
        identity = real_optional_identity(path)
        if path == source and identity is not None and not replacement_injected:
            replacement.replace(source)
            replacement_injected = True
        return identity

    with patch.object(
        durable_writes,
        "_optional_identity",
        side_effect=replace_after_identity_check,
    ):
        durable_writes.publish_file_atomically(source, target, overwrite=True)

    assert replacement_injected
    assert source.read_text(encoding="utf-8") == "concurrent replacement"
    assert target.read_text(encoding="utf-8") == "original staged payload"
    assert list(tmp_path.glob(".bigocr-publish-*")) == []


@pytest.mark.parametrize("clone_errno", [errno.EOPNOTSUPP, errno.EBADF])
def test_snapshot_copy_fallback_does_not_require_reflinks(
    tmp_path: Path,
    clone_errno: int,
) -> None:
    source = tmp_path / ".source"
    source.write_text("replacement", encoding="utf-8")
    target = tmp_path / "output.pdf"
    target.write_text("original", encoding="utf-8")

    with patch.object(
        durable_writes.fcntl,
        "ioctl",
        side_effect=OSError(clone_errno, "reflink unavailable"),
    ):
        durable_writes.publish_file_atomically(source, target, overwrite=True)

    assert target.read_text(encoding="utf-8") == "replacement"
    assert list(tmp_path.glob(".bigocr-publish-*")) == []


def test_recovery_preserves_uninstalled_external_target(tmp_path: Path) -> None:
    first_target = tmp_path / "first.pdf"
    second_target = tmp_path / "second.pdf"
    first_target.write_text("old first", encoding="utf-8")
    second_target.write_text("old second", encoding="utf-8")
    (tmp_path / ".first.staged").write_text("new first", encoding="utf-8")
    (tmp_path / ".second.staged").write_text("new second", encoding="utf-8")
    child_code = """
import os
import sys
from pathlib import Path
from bigocrpdf.utils import durable_writes

root = Path(sys.argv[1])
target = root / "first.pdf"
real_rename = durable_writes._rename_without_replacement

def rename_and_die(source, destination):
    real_rename(source, destination)
    if Path(source).name.endswith(".0.new") and Path(destination) == target:
        os._exit(80)

durable_writes._rename_without_replacement = rename_and_die
durable_writes.publish_files_transactionally(
    [
        (root / ".first.staged", root / "first.pdf"),
        (root / ".second.staged", root / "second.pdf"),
    ],
    overwrite=True,
)
"""
    child = subprocess.run(
        [sys.executable, "-c", child_code, str(tmp_path)],
        capture_output=True,
        cwd=_REPO_ROOT,
        env=_subprocess_environment(),
        text=True,
        timeout=10,
        check=False,
    )
    assert child.returncode == 80, child.stderr

    external = tmp_path / ".external"
    external.write_text("external second", encoding="utf-8")
    os.replace(external, second_target)

    durable_writes.recover_pending_publications(tmp_path)

    assert first_target.read_text(encoding="utf-8") == "old first"
    assert second_target.read_text(encoding="utf-8") == "external second"
    assert list(tmp_path.glob(".bigocr-publish-*")) == []


def test_recovery_refuses_replaced_transaction_artifact(tmp_path: Path) -> None:
    source = tmp_path / ".source"
    source.write_text("new payload", encoding="utf-8")
    target = tmp_path / "output.pdf"
    target.write_text("old payload", encoding="utf-8")
    child_code = """
import os
import sys
from pathlib import Path
from bigocrpdf.utils import durable_writes

root = Path(sys.argv[1])
real_replace_journal = durable_writes._replace_journal

def replace_journal_and_die(path, transaction_id, phase, overwrite, entries):
    real_replace_journal(path, transaction_id, phase, overwrite, entries)
    if phase == "PREPARED":
        os._exit(81)

durable_writes._replace_journal = replace_journal_and_die
durable_writes.publish_file_atomically(
    root / ".source",
    root / "output.pdf",
    overwrite=True,
)
"""
    child = subprocess.run(
        [sys.executable, "-c", child_code, str(tmp_path)],
        capture_output=True,
        cwd=_REPO_ROOT,
        env=_subprocess_environment(),
        text=True,
        timeout=10,
        check=False,
    )
    assert child.returncode == 81, child.stderr

    artifact = next(tmp_path.glob(".bigocr-publish-*.0.new"))
    artifact.unlink()
    artifact.write_text("foreign artifact", encoding="utf-8")

    with pytest.raises(PublicationRecoveryError, match="artifact changed"):
        durable_writes.recover_pending_publications(tmp_path)

    assert target.read_text(encoding="utf-8") == "old payload"
    assert artifact.read_text(encoding="utf-8") == "foreign artifact"
    assert list(tmp_path.glob(".bigocr-publish-*.journal"))


def test_recovery_resumes_after_partial_multi_file_rollback(
    tmp_path: Path,
) -> None:
    first_target = tmp_path / "first.pdf"
    second_target = tmp_path / "second.pdf"
    first_target.write_text("old first", encoding="utf-8")
    second_target.write_text("old second", encoding="utf-8")
    (tmp_path / ".first.staged").write_text("new first", encoding="utf-8")
    (tmp_path / ".second.staged").write_text("new second", encoding="utf-8")
    child_code = """
import os
import sys
from pathlib import Path
from bigocrpdf.utils import durable_writes

root = Path(sys.argv[1])
targets = [root / "first.pdf", root / "second.pdf"]
real_sync = durable_writes._fsync_directory

def sync_and_die_after_targets(directory):
    real_sync(directory)
    if all(path.read_text(encoding="utf-8").startswith("new") for path in targets):
        journals = list(root.glob(".bigocr-publish-*.journal"))
        if journals and '"phase":"PREPARED"' in journals[0].read_text(encoding="utf-8"):
            os._exit(82)

durable_writes._fsync_directory = sync_and_die_after_targets
durable_writes.publish_files_transactionally(
    [
        (root / ".first.staged", root / "first.pdf"),
        (root / ".second.staged", root / "second.pdf"),
    ],
    overwrite=True,
)
"""
    child = subprocess.run(
        [sys.executable, "-c", child_code, str(tmp_path)],
        capture_output=True,
        cwd=_REPO_ROOT,
        env=_subprocess_environment(),
        text=True,
        timeout=10,
        check=False,
    )
    assert child.returncode == 82, child.stderr

    real_move_owned = durable_writes._move_owned_without_replacement

    def fail_second_restore(
        source_path,
        destination,
        expected_identity,
        *,
        expected_state=None,
    ):
        if Path(source_path).name.endswith(".1.retired") and Path(destination) == second_target:
            raise OSError("simulated second restore failure")
        return real_move_owned(
            source_path,
            destination,
            expected_identity,
            expected_state=expected_state,
        )

    for _attempt in range(2):
        with (
            patch.object(
                durable_writes,
                "_move_owned_without_replacement",
                side_effect=fail_second_restore,
            ),
            pytest.raises(OSError, match="second restore failure"),
        ):
            durable_writes.recover_pending_publications(tmp_path)
        assert first_target.read_text(encoding="utf-8") == "old first"
        assert not second_target.exists()

    durable_writes.recover_pending_publications(tmp_path)
    assert first_target.read_text(encoding="utf-8") == "old first"
    assert second_target.read_text(encoding="utf-8") == "old second"
    assert list(tmp_path.glob(".bigocr-publish-*")) == []


def test_committed_target_mismatch_is_not_suppressed_as_cleanup(
    tmp_path: Path,
) -> None:
    source = tmp_path / ".source"
    source.write_text("published payload", encoding="utf-8")
    target = tmp_path / "output.pdf"
    target.write_text("old payload", encoding="utf-8")
    real_replace_journal = durable_writes._replace_journal

    def replace_target_after_commit(
        path,
        transaction_id,
        phase,
        overwrite,
        entries,
    ):
        real_replace_journal(
            path,
            transaction_id,
            phase,
            overwrite,
            entries,
        )
        if phase == "COMMITTED":
            external = tmp_path / ".external"
            external.write_text("external payload", encoding="utf-8")
            os.replace(external, target)

    with (
        patch.object(
            durable_writes,
            "_replace_journal",
            side_effect=replace_target_after_commit,
        ),
        pytest.raises(
            PublicationRecoveryError,
            match="Committed destination changed",
        ),
    ):
        durable_writes.publish_file_atomically(source, target, overwrite=True)

    assert target.read_text(encoding="utf-8") == "external payload"
    assert list(tmp_path.glob(".bigocr-publish-*.journal"))


def test_persistent_rollback_sync_failure_preserves_recovery_journal(
    tmp_path: Path,
) -> None:
    target = tmp_path / "settings.json"
    target.write_text("original", encoding="utf-8")
    staged = tmp_path / ".settings.staged"
    staged.write_text("replacement", encoding="utf-8")
    original_sync = durable_writes._fsync_directory
    publication_was_visible = False

    def fail_after_publication(directory: Path) -> None:
        nonlocal publication_was_visible
        if target.exists() and target.read_text(encoding="utf-8") == "replacement":
            publication_was_visible = True
        if publication_was_visible:
            raise OSError("persistent directory sync failure")
        original_sync(directory)

    with (
        patch.object(
            durable_writes,
            "_fsync_directory",
            side_effect=fail_after_publication,
        ),
        pytest.raises(PublicationRecoveryError, match="could not be rolled back"),
    ):
        durable_writes.publish_file_atomically(staged, target, overwrite=True)

    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.glob(".bigocr-publish-*.journal"))

    durable_writes.recover_pending_publications(tmp_path)
    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.glob(".bigocr-publish-*")) == []


def test_oversized_journal_is_rejected_before_publication(tmp_path: Path) -> None:
    source = tmp_path / ".source"
    source.write_text("new payload", encoding="utf-8")
    target = tmp_path / "output.pdf"
    target.write_text("old payload", encoding="utf-8")

    with (
        patch.object(durable_writes, "_MAX_JOURNAL_BYTES", 128),
        pytest.raises(ValueError, match="journal is too large"),
    ):
        durable_writes.publish_file_atomically(source, target, overwrite=True)

    assert source.read_text(encoding="utf-8") == "new payload"
    assert target.read_text(encoding="utf-8") == "old payload"
    assert list(tmp_path.glob(".bigocr-publish-*")) == []


def test_recovery_rejects_path_traversal_in_validly_named_journal(
    tmp_path: Path,
) -> None:
    target = tmp_path / "important.pdf"
    target.write_text("keep", encoding="utf-8")
    transaction_id = "0" * 32
    journal = tmp_path / f".bigocr-publish-{transaction_id}.journal"
    journal.write_text(
        json.dumps(
            {
                "version": 1,
                "transaction": transaction_id,
                "phase": "PREPARING",
                "overwrite": True,
                "entries": [
                    {
                        "target": "../important.pdf",
                        "new_identity": None,
                        "new_mode": None,
                        "original_identity": None,
                        "backup_identity": None,
                        "backup_mode": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    journal.chmod(0o600)

    with pytest.raises(ValueError, match="publication journal target"):
        durable_writes.recover_pending_publications(tmp_path)

    assert target.read_text(encoding="utf-8") == "keep"


def test_recovery_rejects_boolean_journal_version(tmp_path: Path) -> None:
    transaction_id = "3" * 32
    journal = tmp_path / f".bigocr-publish-{transaction_id}.journal"
    journal.write_text(
        json.dumps(
            {
                "version": True,
                "transaction": transaction_id,
                "phase": "PREPARING",
                "overwrite": True,
                "entries": [
                    {
                        "target": "important.pdf",
                        "new_identity": None,
                        "new_mode": None,
                        "original_identity": None,
                        "backup_identity": None,
                        "backup_mode": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    journal.chmod(0o600)

    with pytest.raises(ValueError, match="publication journal schema"):
        durable_writes.recover_pending_publications(tmp_path)


def test_recovery_rejects_non_private_journal(tmp_path: Path) -> None:
    target = tmp_path / "important.pdf"
    target.write_text("keep", encoding="utf-8")
    transaction_id = "2" * 32
    journal = tmp_path / f".bigocr-publish-{transaction_id}.journal"
    journal.write_text(
        json.dumps(
            {
                "version": 1,
                "transaction": transaction_id,
                "phase": "PREPARING",
                "overwrite": True,
                "entries": [
                    {
                        "target": "important.pdf",
                        "new_identity": None,
                        "new_mode": None,
                        "original_identity": None,
                        "backup_identity": None,
                        "backup_mode": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    journal.chmod(0o644)

    with pytest.raises(ValueError, match="publication journal"):
        durable_writes.recover_pending_publications(tmp_path)

    assert target.read_text(encoding="utf-8") == "keep"


def test_recovery_rejects_symlink_journal(tmp_path: Path) -> None:
    target = tmp_path / "important.pdf"
    target.write_text("keep", encoding="utf-8")
    transaction_id = "1" * 32
    journal_payload = tmp_path / "payload.json"
    journal_payload.write_text(
        json.dumps(
            {
                "version": 1,
                "transaction": transaction_id,
                "phase": "PREPARING",
                "overwrite": True,
                "entries": [
                    {
                        "target": "important.pdf",
                        "new_identity": None,
                        "new_mode": None,
                        "original_identity": None,
                        "backup_identity": None,
                        "backup_mode": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    journal = tmp_path / f".bigocr-publish-{transaction_id}.journal"
    journal.symlink_to(journal_payload)

    with pytest.raises(ValueError, match="publication journal"):
        durable_writes.recover_pending_publications(tmp_path)

    assert target.read_text(encoding="utf-8") == "keep"


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
    assert list(tmp_path.glob(".bigocr-publish-*")) == []
