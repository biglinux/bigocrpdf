"""Queue UI behavior and persistence contracts."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import gi

gi.require_version("Gdk", "4.0")
gi.require_version("Gio", "2.0")
from gi.repository import Gdk, Gio

from bigocrpdf.ui.settings_queue_mixin import SettingsQueueMixin


def _manager(settings) -> Any:
    manager: Any = SettingsQueueMixin.__new__(SettingsQueueMixin)
    manager.window = SimpleNamespace(
        settings=settings,
        ui=SimpleNamespace(custom_header_bar=MagicMock(), show_toast=MagicMock()),
        announce_status=MagicMock(),
    )
    manager.refresh_queue_status = MagicMock()
    return manager


def test_reorder_refreshes_only_after_persisted_move() -> None:
    settings = SimpleNamespace(_move_file=MagicMock(return_value=False))
    manager = _manager(settings)
    manager._populate_file_list = MagicMock()

    assert manager._on_reorder_drop(0, 1) is False
    manager._populate_file_list.assert_not_called()

    settings._move_file.return_value = True

    assert manager._on_reorder_drop(0, 1) is True
    manager._populate_file_list.assert_called_once_with()


def test_failed_remove_does_not_announce_success() -> None:
    settings = SimpleNamespace(
        selected_files=["/tmp/input.pdf"],
        _remove_file=MagicMock(return_value=False),
    )
    manager = _manager(settings)

    manager._remove_single_file(0)

    manager.refresh_queue_status.assert_not_called()
    manager.window.announce_status.assert_not_called()


def test_failed_clear_does_not_announce_success() -> None:
    settings = SimpleNamespace(
        selected_files=["/tmp/input.pdf"],
        _clear_files=MagicMock(return_value=False),
    )
    manager = _manager(settings)

    manager._remove_all_files()

    manager.refresh_queue_status.assert_not_called()
    manager.window.announce_status.assert_not_called()


def test_drop_uses_gdk_file_list_and_ignores_remote_files(tmp_path: Path) -> None:
    local_pdf = tmp_path / "local.pdf"
    local_pdf.write_bytes(b"%PDF-1.4\n")
    files = Gdk.FileList.new_from_array(
        [
            Gio.File.new_for_uri("https://example.com/remote.pdf"),
            Gio.File.new_for_path(str(local_pdf)),
        ]
    )
    settings = SimpleNamespace(
        selected_files=[],
        original_file_paths={},
        add_files=MagicMock(return_value=1),
    )
    manager = _manager(settings)

    assert manager._on_drop(MagicMock(), files, 0, 0) is True

    settings.add_files.assert_called_once_with([str(local_pdf)])
    manager.refresh_queue_status.assert_called_once_with()


def test_drop_rejects_values_outside_declared_target_type() -> None:
    settings = SimpleNamespace(selected_files=[], original_file_paths={})
    manager = _manager(settings)

    assert manager._on_drop(MagicMock(), Gio.File.new_for_path("/tmp/file.pdf"), 0, 0) is False
    manager.refresh_queue_status.assert_not_called()


def test_filter_rejects_duplicates_aliases_and_non_regular_files(tmp_path: Path) -> None:
    queued = tmp_path / "queued.pdf"
    queued.write_bytes(b"%PDF-1.4\n")
    queued_alias = tmp_path / "queued-alias.pdf"
    queued_alias.symlink_to(queued)
    original_image = tmp_path / "source.png"
    original_image.write_bytes(b"png")
    new_pdf = tmp_path / "new.pdf"
    new_pdf.write_bytes(b"%PDF-1.4\n")
    directory = tmp_path / "folder.pdf"
    directory.mkdir()

    settings = SimpleNamespace(
        selected_files=[str(queued)],
        original_file_paths={"/tmp/generated.pdf": str(original_image)},
    )
    manager = _manager(settings)

    assert manager._filter_supported_files(
        [
            str(queued_alias),
            str(original_image),
            str(directory),
            str(new_pdf),
            str(new_pdf),
        ]
    ) == [str(new_pdf)]


def test_list_refresh_does_not_render_hidden_grid() -> None:
    settings = SimpleNamespace(selected_files=[])
    manager = _manager(settings)
    manager.file_list_box = MagicMock()
    manager.file_list_box.get_first_child.return_value = None
    manager.placeholder = MagicMock()
    manager._item_popover = None
    manager._queue_view_stack = MagicMock()
    manager._queue_view_stack.get_visible_child_name.return_value = "list"
    manager._populate_grid = MagicMock()

    manager._populate_file_list()

    manager._populate_grid.assert_not_called()


def test_cleanup_cancels_queue_workers_and_callbacks() -> None:
    manager = _manager(SimpleNamespace())
    manager._queue_metadata_closed = False
    manager._queue_metadata_waiters = {"/tmp/input.pdf": [MagicMock()]}
    pool = MagicMock()
    manager._queue_metadata_pool = pool
    manager._item_popover = None

    manager.cleanup()

    assert manager._queue_metadata_closed is True
    assert manager._queue_metadata_waiters == {}
    assert manager._queue_metadata_pool is None
    pool.shutdown.assert_called_once_with(
        wait=False,
        cancel_futures=True,
    )
