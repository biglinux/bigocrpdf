import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

from bigocrpdf.ui.pdf_editor import thumbnail_renderer
from bigocrpdf.ui.pdf_editor.page_grid import PageGrid
from bigocrpdf.ui.pdf_editor.page_thumbnail import PageThumbnail


class RecordingPool:
    def __init__(self) -> None:
        self.submissions: list[tuple[Callable[..., Any], tuple[Any, ...]]] = []
        self.futures: list[RecordingFuture] = []
        self.shutdown_calls: list[tuple[bool, bool]] = []

    def submit(self, callback: Callable[..., Any], *args: Any) -> "RecordingFuture":
        self.submissions.append((callback, args))
        future = RecordingFuture()
        self.futures.append(future)
        return future

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        self.shutdown_calls.append((wait, cancel_futures))


class RecordingFuture:
    def __init__(self, *, cancel_result: bool = True) -> None:
        self.cancelled = False
        self.cancel_result = cancel_result

    def cancel(self) -> bool:
        self.cancelled = True
        return self.cancel_result


def _renderer_with_recording_pool(
    monkeypatch,
) -> tuple[thumbnail_renderer.ThumbnailRenderer, RecordingPool]:
    pool = RecordingPool()
    monkeypatch.setattr(thumbnail_renderer, "ThreadPoolExecutor", lambda **_kwargs: pool)
    renderer = thumbnail_renderer.ThumbnailRenderer()
    return renderer, pool


def test_duplicate_thumbnail_requests_share_render_and_notify_every_waiter(
    monkeypatch, tmp_path: Path
) -> None:
    renderer, pool = _renderer_with_recording_pool(monkeypatch)
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")
    rendered = object()
    received: list[tuple[str, object]] = []

    monkeypatch.setattr(
        renderer,
        "_render_pdf_page_pdftoppm",
        lambda *_args: rendered,
    )
    monkeypatch.setattr(
        thumbnail_renderer.GLib,
        "idle_add",
        lambda callback, *args: callback(*args),
    )

    renderer.render_page_thumbnail_async(
        str(pdf_path),
        0,
        lambda pixbuf: received.append(("first", pixbuf)),
    )
    renderer.render_page_thumbnail_async(
        str(pdf_path),
        0,
        lambda pixbuf: received.append(("second", pixbuf)),
    )

    assert len(pool.submissions) == 1
    worker, args = pool.submissions[0]
    worker(*args)
    assert received == [("first", rendered), ("second", rendered)]


def test_thumbnail_cache_key_changes_when_document_is_replaced(monkeypatch, tmp_path: Path) -> None:
    renderer, _pool = _renderer_with_recording_pool(monkeypatch)
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"first")
    first_key = renderer._get_cache_key(str(pdf_path), 0, 200, 0)

    replacement = tmp_path / "replacement.pdf"
    replacement.write_bytes(b"second version")
    replacement.replace(pdf_path)
    second_key = renderer._get_cache_key(str(pdf_path), 0, 200, 0)

    assert second_key != first_key


def test_cache_clear_invalidates_inflight_thumbnail_callback(monkeypatch, tmp_path: Path) -> None:
    renderer, pool = _renderer_with_recording_pool(monkeypatch)
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")
    rendered = object()
    received: list[str] = []

    monkeypatch.setattr(
        renderer,
        "_render_pdf_page_pdftoppm",
        lambda *_args: rendered,
    )
    monkeypatch.setattr(
        thumbnail_renderer.GLib,
        "idle_add",
        lambda callback, *args: callback(*args),
    )

    renderer.render_page_thumbnail_async(
        str(pdf_path),
        0,
        lambda _pixbuf: received.append("stale"),
    )
    renderer.clear_document_cache(str(pdf_path))
    renderer.render_page_thumbnail_async(
        str(pdf_path),
        0,
        lambda _pixbuf: received.append("current"),
    )

    assert len(pool.submissions) == 2
    first_worker, first_args = pool.submissions[0]
    second_worker, second_args = pool.submissions[1]
    first_worker(*first_args)
    second_worker(*second_args)
    assert received == ["current"]


def test_page_reload_invalidates_only_its_page(monkeypatch, tmp_path: Path) -> None:
    renderer = SimpleNamespace(
        clear_page_cache=MagicMock(),
        clear_document_cache=MagicMock(),
    )
    thumbnail = PageThumbnail.__new__(PageThumbnail)
    thumbnail._pdf_path = str(tmp_path / "document.pdf")
    thumbnail._page_state = SimpleNamespace(page_number=3)
    thumbnail._discard_thumbnail = MagicMock()
    thumbnail.load_thumbnail = MagicMock()
    monkeypatch.setattr(
        "bigocrpdf.ui.pdf_editor.page_thumbnail.get_thumbnail_renderer",
        lambda: renderer,
    )

    thumbnail.reload_thumbnail()

    renderer.clear_page_cache.assert_called_once_with(thumbnail._pdf_path, 2)
    renderer.clear_document_cache.assert_not_called()
    thumbnail.load_thumbnail.assert_called_once_with()


def test_unallocated_grid_prefetch_is_bounded_to_viewport() -> None:
    assert (
        PageGrid._initial_thumbnail_load_limit(
            item_count=500,
            columns=4,
            viewport_height=700,
            item_height=260,
        )
        == 20
    )


def test_stale_widget_thumbnail_callback_is_ignored() -> None:
    thumbnail = PageThumbnail.__new__(PageThumbnail)
    thumbnail._thumbnail_generation = 2
    thumbnail._thumbnail_loaded = False

    thumbnail._on_thumbnail_loaded(cast(Any, object()), generation=1)

    assert thumbnail._thumbnail_loaded is False


def test_cancelled_queued_thumbnail_drops_callback_and_future(monkeypatch, tmp_path: Path) -> None:
    renderer, pool = _renderer_with_recording_pool(monkeypatch)
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")
    received: list[object] = []

    request = renderer.render_page_thumbnail_async(
        str(pdf_path),
        0,
        received.append,
    )
    request.cancel()

    assert pool.futures[0].cancelled is True
    assert received == []


def test_cancelled_running_thumbnail_terminates_pdftoppm(monkeypatch, tmp_path: Path) -> None:
    renderer, pool = _renderer_with_recording_pool(monkeypatch)
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")
    request = renderer.render_page_thumbnail_async(str(pdf_path), 0, lambda _pixbuf: None)
    cache_key = renderer._get_cache_key(str(pdf_path), 0, 200, 0)
    job = renderer._jobs[cache_key]
    process = MagicMock()
    process.poll.return_value = None
    pool.futures[0].cancel_result = False
    job.process = process

    request.cancel()

    assert job.cancel_event.is_set()
    process.terminate.assert_called_once_with()


def test_transient_render_error_is_not_cached(monkeypatch, tmp_path: Path) -> None:
    renderer, pool = _renderer_with_recording_pool(monkeypatch)
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")
    error_pixbuf = object()
    received: list[object] = []
    monkeypatch.setattr(renderer, "_render_pdf_page_pdftoppm", lambda *_args: None)
    monkeypatch.setattr(renderer, "_create_error_pixbuf", lambda _size: error_pixbuf)
    monkeypatch.setattr(
        thumbnail_renderer.GLib,
        "idle_add",
        lambda callback, *args: callback(*args),
    )

    renderer.render_page_thumbnail_async(str(pdf_path), 0, received.append)
    first_worker, first_args = pool.submissions[0]
    first_worker(*first_args)
    renderer.render_page_thumbnail_async(str(pdf_path), 0, received.append)

    assert received == [error_pixbuf]
    assert len(pool.submissions) == 2


def test_invalidated_render_result_remains_retryable() -> None:
    thumbnail = PageThumbnail.__new__(PageThumbnail)
    thumbnail._thumbnail_generation = 1
    thumbnail._thumbnail_request = object()
    thumbnail._thumbnail_loading = True
    thumbnail._thumbnail_loaded = False
    thumbnail._current_rotation = None
    thumbnail._spinner = MagicMock()
    thumbnail._image = MagicMock()
    thumbnail._page_state = SimpleNamespace(
        rotation=0,
        flip_horizontal=False,
        flip_vertical=False,
        thumbnail_pixbuf=object(),
    )

    thumbnail._on_thumbnail_loaded(None, generation=1, requested_rotation=0)

    assert thumbnail._thumbnail_loading is False
    assert thumbnail._thumbnail_loaded is False
    assert thumbnail._page_state.thumbnail_pixbuf is None
    thumbnail._image.set_visible.assert_called_once_with(False)


def test_loaded_thumbnail_reapplies_persisted_flips(monkeypatch) -> None:
    thumbnail = PageThumbnail.__new__(PageThumbnail)
    thumbnail._thumbnail_generation = 1
    thumbnail._thumbnail_request = object()
    thumbnail._thumbnail_loading = True
    thumbnail._thumbnail_loaded = False
    thumbnail._current_rotation = None
    thumbnail._spinner = MagicMock()
    thumbnail._image = MagicMock()
    horizontal = MagicMock()
    vertical = MagicMock()
    source = MagicMock()
    source.flip.return_value = horizontal
    horizontal.flip.return_value = vertical
    thumbnail._page_state = SimpleNamespace(
        rotation=0,
        flip_horizontal=True,
        flip_vertical=True,
        thumbnail_pixbuf=None,
    )
    monkeypatch.setattr(
        "bigocrpdf.ui.pdf_editor.page_thumbnail.Gdk.Texture.new_for_pixbuf",
        lambda _pixbuf: object(),
    )

    thumbnail._on_thumbnail_loaded(source, generation=1, requested_rotation=0)

    source.flip.assert_called_once_with(True)
    horizontal.flip.assert_called_once_with(False)
    assert thumbnail._page_state.thumbnail_pixbuf is vertical


def test_thumbnail_accessible_name_reports_rotation_and_flips(monkeypatch) -> None:
    thumbnail = PageThumbnail.__new__(PageThumbnail)
    thumbnail._page_state = SimpleNamespace(
        page_number=3,
        deleted=False,
        rotation=90,
        flip_horizontal=True,
        flip_vertical=True,
    )
    thumbnail._page_label = MagicMock()
    thumbnail.update_property = MagicMock()
    monkeypatch.setattr(
        "bigocrpdf.ui.pdf_editor.page_thumbnail._",
        lambda message: message,
    )

    thumbnail._update_page_label()

    thumbnail._page_label.set_text.assert_called_once_with("3")
    assert thumbnail.update_property.call_args.args[1] == [
        "Page 3 — Rotated 90° — Horizontal Flip — Vertical Flip"
    ]


def test_inline_thumbnail_rotation_refreshes_accessible_state() -> None:
    page_state = SimpleNamespace(rotate=MagicMock())
    thumbnail = SimpleNamespace(
        rotate_thumbnail_in_place=MagicMock(),
        update_from_state=MagicMock(),
    )
    document = SimpleNamespace(mark_modified=MagicMock())
    grid = PageGrid.__new__(PageGrid)
    grid.on_before_mutate = MagicMock()
    grid._document = document

    grid._on_thumbnail_rotate_right(cast(Any, thumbnail), cast(Any, page_state))

    page_state.rotate.assert_called_once_with(90)
    thumbnail.rotate_thumbnail_in_place.assert_called_once_with(90)
    thumbnail.update_from_state.assert_called_once_with()
    document.mark_modified.assert_called_once_with()


def test_unload_never_loaded_thumbnail_stops_spinner() -> None:
    thumbnail = PageThumbnail.__new__(PageThumbnail)
    thumbnail._thumbnail_generation = 0
    thumbnail._thumbnail_request = None
    thumbnail._thumbnail_loading = False
    thumbnail._thumbnail_loaded = False
    thumbnail._current_rotation = None
    thumbnail._page_state = SimpleNamespace(thumbnail_pixbuf=None)
    thumbnail._spinner = MagicMock()
    thumbnail._image = MagicMock()

    thumbnail.unload_thumbnail()

    thumbnail._spinner.stop.assert_called_once_with()
    thumbnail._spinner.set_visible.assert_called_once_with(False)


def test_grid_close_cancels_every_owned_thumbnail_request() -> None:
    thumbnails = [MagicMock(), MagicMock()]
    grid = PageGrid.__new__(PageGrid)
    grid._thumbnails = thumbnails
    grid._stop_auto_scroll = MagicMock()

    grid.cancel_thumbnail_requests()

    for thumbnail in thumbnails:
        thumbnail.unload_thumbnail.assert_called_once_with()
    grid._stop_auto_scroll.assert_called_once_with()


def test_grid_deferred_load_is_coalesced_and_removed_on_teardown(monkeypatch) -> None:
    callbacks: dict[int, Callable[[], bool]] = {}
    next_source_id = iter((41, 42))
    removed: list[int] = []

    def timeout_add(_delay: int, callback: Callable[[], bool]) -> int:
        source_id = next(next_source_id)
        callbacks[source_id] = callback
        return source_id

    monkeypatch.setattr(
        "bigocrpdf.ui.pdf_editor.page_grid.GLib.timeout_add",
        timeout_add,
    )
    monkeypatch.setattr(
        "bigocrpdf.ui.pdf_editor.page_grid.GLib.source_remove",
        removed.append,
    )
    grid = PageGrid.__new__(PageGrid)
    grid._thumbnail_load_source_id = None
    grid._tearing_down = False
    grid._thumbnails = []
    grid._stop_auto_scroll = MagicMock()
    grid._load_visible_thumbnails = MagicMock(return_value=False)

    grid._schedule_visible_thumbnail_load()
    grid._schedule_visible_thumbnail_load()
    grid.cancel_thumbnail_requests()
    grid.cancel_thumbnail_requests()

    assert removed == [41, 42]
    assert callbacks[42]() is False
    grid._load_visible_thumbnails.assert_not_called()


def test_grid_scroll_after_teardown_does_not_load_thumbnails() -> None:
    grid = PageGrid.__new__(PageGrid)
    grid._tearing_down = True
    grid._load_visible_thumbnails = MagicMock()

    grid._on_scroll_changed()

    grid._load_visible_thumbnails.assert_not_called()


def test_pdftoppm_renders_the_cropbox(monkeypatch) -> None:
    renderer, _pool = _renderer_with_recording_pool(monkeypatch)
    process = MagicMock()
    process.communicate.return_value = b"", b""
    process.returncode = 1
    process.args = ["pdftoppm"]
    monkeypatch.setattr(thumbnail_renderer.subprocess, "Popen", MagicMock(return_value=process))
    monkeypatch.setattr(renderer, "_render_pdf_to_pixbuf", MagicMock(return_value=None))

    renderer._render_pdf_page_pdftoppm(
        "document.pdf",
        0,
        200,
        threading.Event(),
        None,
    )

    command = thumbnail_renderer.subprocess.Popen.call_args.args[0]
    assert command.count("-cropbox") == 1


def test_renderer_shutdown_cancels_jobs_and_does_not_create_a_replacement_pool(
    monkeypatch,
    tmp_path: Path,
) -> None:
    renderer, pool = _renderer_with_recording_pool(monkeypatch)
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")
    renderer.render_page_thumbnail_async(str(pdf_path), 0, lambda _pixbuf: None)
    job = next(iter(renderer._jobs.values()))
    process = MagicMock()
    process.poll.return_value = None
    job.process = process

    renderer.shutdown(wait=True)

    assert pool.shutdown_calls == [(True, True)]
    assert pool.futures[0].cancelled is True
    process.terminate.assert_called_once_with()
    assert renderer._jobs == {}


def test_pdftoppm_cancellation_stops_before_poppler_fallback(monkeypatch) -> None:
    renderer, _pool = _renderer_with_recording_pool(monkeypatch)
    cancel_event = threading.Event()
    process = MagicMock()
    process.communicate.side_effect = subprocess.TimeoutExpired("pdftoppm", 0.05)
    process.poll.return_value = None

    communicate_calls = 0

    def cancel_during_communicate(*_args, **_kwargs):
        nonlocal communicate_calls
        communicate_calls += 1
        if communicate_calls == 1:
            cancel_event.set()
            raise subprocess.TimeoutExpired("pdftoppm", 0.05)
        return b"", b""

    process.communicate.side_effect = cancel_during_communicate
    monkeypatch.setattr(thumbnail_renderer.subprocess, "Popen", lambda *_args, **_kwargs: process)
    fallback = MagicMock()
    monkeypatch.setattr(renderer, "_render_pdf_to_pixbuf", fallback)

    result = renderer._render_pdf_page_pdftoppm(
        "document.pdf",
        0,
        200,
        cancel_event,
        None,
    )

    assert result is None
    process.terminate.assert_called_once_with()
    fallback.assert_not_called()
