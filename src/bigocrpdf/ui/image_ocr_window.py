"""
BigOcrPdf - Image OCR Window

Standalone window for Image OCR using the configured RapidOCR model.
Supports opening image files and capturing screen regions.
"""

import os
import threading
from dataclasses import dataclass

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

gi.require_version("Gdk", "4.0")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk

from bigocrpdf import OcrDependencyState
from bigocrpdf.config import IMAGE_WINDOW_STATE_KEY
from bigocrpdf.services.screen_capture import (
    DIRECT_IMAGE_HARD_MAX_MEGAPIXELS,
    ImageOcrOutcome,
    ImageOcrRequest,
    ImageOcrStatus,
    ScreenCaptureService,
    get_supported_image_extensions,
)
from bigocrpdf.services.settings import OcrSettings
from bigocrpdf.ui.widgets import (
    get_default_clipboard,
    parse_clipboard_file_paths,
    present_ocr_unavailable_dialog,
)
from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.config_manager import get_config_manager
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.temp_manager import mkstemp, remove_file
from bigocrpdf.utils.timer import safe_remove_source


@dataclass(frozen=True)
class _ClipboardEncodeJob:
    """One immutable texture waiting in the bounded clipboard encoder."""

    texture: Gdk.Texture
    generation: int | None
    cancellable: Gio.Cancellable | None


class ImageOcrWindow(Adw.ApplicationWindow):
    """Standalone window for Image OCR operations using RapidOCR."""

    _MAX_CLIPBOARD_URI_BYTES = 64 * 1024

    def __init__(
        self,
        application: Gtk.Application,
        image_path: str | None = None,
        *,
        ocr_dependency: OcrDependencyState,
    ) -> None:
        """Initialize the Image OCR Window.

        Args:
            application: The Gtk Application instance
            image_path: Optional path to an image file to process immediately
            ocr_dependency: Resolved OCR dependency state from startup
        """
        width, height = self._load_window_size()

        super().__init__(
            application=application,
            default_width=width,
            default_height=height,
        )

        self.set_title("Big Image OCR")
        self.set_icon_name("bigocrimage")

        self.ocr_dependency = ocr_dependency
        self._ocr_unavailable_dialog: Adw.AlertDialog | None = None

        # Initialize services
        self._settings = OcrSettings()
        self._screen_capture_service = ScreenCaptureService()

        # UI state
        self._alive = True
        self._active_request: ImageOcrRequest | None = None
        self._is_hidden_for_capture = False
        self._hidden_capture_generation: int | None = None
        self._processing_generation = 0
        self._capture_delay_source_id = 0
        self._focus_idle_source_id = 0
        self._input_cancellable: Gio.Cancellable | None = None
        self._input_generation = 0
        self._stable_page_name = "welcome"
        self._clipboard_encode_lock = threading.Lock()
        self._clipboard_encode_pending: _ClipboardEncodeJob | None = None
        self._clipboard_encode_active: _ClipboardEncodeJob | None = None
        self._clipboard_encode_thread: threading.Thread | None = None
        self._clipboard_encode_stopping = False

        self._setup_ui()
        self._setup_window_actions()
        self.connect("close-request", self._on_close_request)

        # Process provided image or show welcome state
        if image_path and self.ocr_dependency.is_available:
            self._start_processing(image_path)
        else:
            self._stack.set_visible_child_name("welcome")

    # ── Window State ────────────────────────────────────────────────────

    def _load_window_size(self) -> tuple[int, int]:
        """Load window size from configuration."""
        config = get_config_manager()
        width = config.get(f"{IMAGE_WINDOW_STATE_KEY}.width", 800)
        height = config.get(f"{IMAGE_WINDOW_STATE_KEY}.height", 500)
        return max(width, 400), max(height, 300)

    def _save_window_size(self) -> None:
        """Save current window size to configuration."""
        if self.is_maximized() or self.is_fullscreen():
            return

        config = get_config_manager()
        width = self.get_width()
        height = self.get_height()
        if width > 0 and height > 0:
            config.set(f"{IMAGE_WINDOW_STATE_KEY}.width", width, save_immediately=False)
            config.set(f"{IMAGE_WINDOW_STATE_KEY}.height", height, save_immediately=True)

    def _on_close_request(self, _window: Gtk.Window) -> bool:
        """Handle window close request — save state."""
        self.prepare_close()
        return False

    def prepare_close(self) -> None:
        """Idempotently invalidate callbacks and cancel window-owned work."""
        if not getattr(self, "_alive", False):
            return

        self._save_window_size()
        self._alive = False
        self._processing_generation += 1
        self._input_generation += 1
        self._remove_source("_capture_delay_source_id")
        self._remove_source("_focus_idle_source_id")

        request = self._active_request
        self._active_request = None
        if request is not None:
            request.cancel()

        cancellable = self._input_cancellable
        self._input_cancellable = None
        if cancellable is not None:
            cancellable.cancel()

        ImageOcrWindow._shutdown_clipboard_encode_lane(self)
        self._screen_capture_service.shutdown(wait=False)
        self._hidden_capture_generation = None
        self._is_hidden_for_capture = False
        self._set_cancel_enabled(False)

    def _setup_window_actions(self) -> None:
        """Expose window-owned cancellation and paste actions."""
        cancel_action = Gio.SimpleAction.new("cancel-processing", None)
        cancel_action.connect("activate", self._on_cancel_processing)
        cancel_action.set_enabled(False)
        self.add_action(cancel_action)

        paste_action = Gio.SimpleAction.new("paste-clipboard", None)
        paste_action.connect("activate", lambda *_args: self.paste_from_clipboard())
        paste_action.set_enabled(self.ocr_dependency.is_available)
        self.add_action(paste_action)

    def _on_cancel_processing(self, *_args: object) -> None:
        """Cancel the current request and restore the last stable page."""
        if not getattr(self, "_alive", False):
            return

        generation = self._processing_generation
        request = self._active_request
        self._active_request = None
        if request is not None:
            request.cancel()

        self._remove_source("_capture_delay_source_id")
        self._remove_source("_focus_idle_source_id")
        self._restore_after_capture(generation)
        self._processing_generation += 1
        self._stack.set_visible_child_name(self._stable_page_name)
        self._sync_copy_button_state()
        self._set_cancel_enabled(False)

    def _begin_operation(self) -> int:
        """Cancel older work and reserve ownership for a new operation."""
        if not getattr(self, "_alive", True):
            return -1

        previous_generation = getattr(self, "_processing_generation", 0)
        previous_request = getattr(self, "_active_request", None)
        self._active_request = None
        if previous_request is not None:
            previous_request.cancel()

        self._remove_source("_capture_delay_source_id")
        self._remove_source("_focus_idle_source_id")
        self._restore_after_capture(previous_generation)

        input_cancellable = getattr(self, "_input_cancellable", None)
        self._input_cancellable = None
        if input_cancellable is not None:
            input_cancellable.cancel()

        generation = previous_generation + 1
        self._processing_generation = generation
        return generation

    def _is_current_operation(
        self,
        generation: int,
        request: ImageOcrRequest | None = None,
    ) -> bool:
        """Whether an async callback still owns this live window."""
        if not getattr(self, "_alive", True):
            return False
        if generation != getattr(self, "_processing_generation", 0):
            return False
        return request is None or request is getattr(self, "_active_request", None)

    def _remove_source(self, attribute: str) -> None:
        """Remove one tracked GLib source before its ID can be reused."""
        source_id = int(getattr(self, attribute, 0) or 0)
        setattr(self, attribute, 0)
        if source_id:
            safe_remove_source(source_id)

    def _set_cancel_enabled(self, enabled: bool) -> None:
        """Keep the visible control and window action synchronized."""
        cancel_button = getattr(self, "_cancel_button", None)
        if cancel_button is not None:
            cancel_button.set_sensitive(enabled)
        lookup_action = getattr(self, "lookup_action", None)
        if lookup_action is not None:
            action = lookup_action("cancel-processing")
            if action is not None:
                action.set_enabled(enabled)

    # ── UI Setup ────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        """Set up the window UI following Adwaita HIG patterns.

        Layout:
        - Header bar (raised)
        - Content stack: welcome / loading / results pages
        - Results page includes a bottom action bar with labeled buttons
        """
        toolbar_view = Adw.ToolbarView()
        toolbar_view.set_top_bar_style(Adw.ToolbarStyle.RAISED)
        self.set_content(toolbar_view)

        # Header Bar
        header = Adw.HeaderBar()
        toolbar_view.add_top_bar(header)

        # Toast overlay wraps all content for non-intrusive feedback
        self._toast_overlay = Adw.ToastOverlay()
        toolbar_view.set_content(self._toast_overlay)

        # Content Stack (welcome / loading / results)
        self._stack = Gtk.Stack()
        self._stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        self._toast_overlay.set_child(self._stack)

        self._build_welcome_page()
        self._build_loading_page()
        self._build_results_page()
        self._build_empty_page()

        # Enable drag-and-drop for image files
        self._setup_drop_target()

    _SUPPORTED_IMAGE_EXTENSIONS = get_supported_image_extensions()

    def _setup_drop_target(self) -> None:
        """Set up drag-and-drop target for image files."""
        drop = Gtk.DropTarget.new(Gio.File, Gdk.DragAction.COPY)
        drop.connect("drop", self._on_drop)
        self.add_controller(drop)

    def _on_drop(self, _target: Gtk.DropTarget, value: Gio.File, _x: float, _y: float) -> bool:
        """Handle dropped file."""
        if not self._require_ocr_available():
            return False

        path = value.get_path()
        if path:
            import os

            ext = os.path.splitext(path)[1].lower()
            if ext in self._SUPPORTED_IMAGE_EXTENSIONS:
                self._start_processing(path)
                return True
            else:
                self._toast_overlay.add_toast(Adw.Toast(title=_("Unsupported file format")))
        return False

    # ── Clipboard Paste ─────────────────────────────────────────────────

    def paste_from_clipboard(self) -> None:
        """Paste image from clipboard (Ctrl+V)."""
        if not self._require_ocr_available():
            return

        generation, cancellable = self._begin_input_operation()
        clipboard = get_default_clipboard()
        if clipboard is None:
            self._finish_input_operation(generation, cancellable)
            self._toast_overlay.add_toast(Adw.Toast(title=_("No image found in clipboard")))
            return
        formats = clipboard.get_formats()
        if formats.contain_gtype(Gdk.Texture):
            clipboard.read_texture_async(
                cancellable,
                lambda source, result: self._on_clipboard_texture_ready(
                    source,
                    result,
                    generation,
                    cancellable,
                ),
            )
        elif formats.contain_mime_type("text/uri-list"):
            clipboard.read_async(
                ["text/uri-list"],
                GLib.PRIORITY_DEFAULT,
                cancellable,
                lambda source, result: self._on_clipboard_uri_ready(
                    source,
                    result,
                    generation,
                    cancellable,
                ),
            )
        else:
            self._finish_input_operation(generation, cancellable)
            self._toast_overlay.add_toast(Adw.Toast(title=_("No image found in clipboard")))

    def _begin_input_operation(self) -> tuple[int, Gio.Cancellable]:
        """Reserve ownership for one GTK input operation."""
        previous = getattr(self, "_input_cancellable", None)
        if previous is not None:
            previous.cancel()
        self._input_generation = getattr(self, "_input_generation", 0) + 1
        cancellable = Gio.Cancellable()
        self._input_cancellable = cancellable
        return self._input_generation, cancellable

    def _is_current_input_operation(
        self,
        generation: int | None,
        cancellable: Gio.Cancellable | None,
    ) -> bool:
        """Whether an input callback still owns this live window."""
        if generation is None or cancellable is None:
            return getattr(self, "_alive", True)
        return (
            getattr(self, "_alive", True)
            and generation == getattr(self, "_input_generation", 0)
            and cancellable is getattr(self, "_input_cancellable", None)
            and not cancellable.is_cancelled()
        )

    def _finish_input_operation(
        self,
        generation: int | None,
        cancellable: Gio.Cancellable | None,
    ) -> None:
        """Release an input token if it is still current."""
        if generation is None or cancellable is None:
            return
        if generation == getattr(self, "_input_generation", 0) and cancellable is getattr(
            self, "_input_cancellable", None
        ):
            self._input_cancellable = None

    def _on_clipboard_texture_ready(
        self,
        clipboard: Gdk.Clipboard,
        result: Gio.AsyncResult,
        generation: int | None = None,
        cancellable: Gio.Cancellable | None = None,
    ) -> None:
        """Handle clipboard texture read completion."""
        if not self._is_current_input_operation(generation, cancellable):
            return
        worker_started = False
        try:
            texture = clipboard.read_texture_finish(result)
            if not texture:
                self._toast_overlay.add_toast(Adw.Toast(title=_("No image found in clipboard")))
                return
            width = int(texture.get_width())
            height = int(texture.get_height())
            hard_pixel_limit = int(DIRECT_IMAGE_HARD_MAX_MEGAPIXELS * 1_000_000)
            if width <= 0 or height <= 0 or width * height > hard_pixel_limit:
                self._toast_overlay.add_toast(
                    Adw.Toast(title=_("Image dimensions exceed the configured safety limit."))
                )
                return
            worker_started = ImageOcrWindow._queue_clipboard_texture(
                self,
                texture,
                generation,
                cancellable,
            )
        except Exception as e:
            if self._is_current_input_operation(generation, cancellable):
                logger.error(f"Clipboard paste error: {e}")
                self._toast_overlay.add_toast(
                    Adw.Toast(title=_("Could not read the clipboard image"))
                )
        finally:
            if not worker_started:
                self._finish_input_operation(generation, cancellable)

    def _queue_clipboard_texture(
        self,
        texture: Gdk.Texture,
        generation: int | None,
        cancellable: Gio.Cancellable | None,
    ) -> bool:
        """Retain only the newest texture behind one encoder thread."""
        if not hasattr(self, "_clipboard_encode_lock"):
            self._clipboard_encode_lock = threading.Lock()
            self._clipboard_encode_pending = None
            self._clipboard_encode_active = None
            self._clipboard_encode_thread = None
            self._clipboard_encode_stopping = False

        job = _ClipboardEncodeJob(texture, generation, cancellable)
        thread_to_start: threading.Thread | None = None
        with self._clipboard_encode_lock:
            if self._clipboard_encode_stopping:
                return False
            superseded = self._clipboard_encode_pending
            active = self._clipboard_encode_active
            self._clipboard_encode_pending = job
            if self._clipboard_encode_thread is None:
                thread_to_start = threading.Thread(
                    target=ImageOcrWindow._run_clipboard_encode_lane.__get__(self),
                    args=(),
                    daemon=True,
                )
                self._clipboard_encode_thread = thread_to_start

        for old_job in (superseded, active):
            if (
                old_job is not None
                and old_job.cancellable is not None
                and old_job.cancellable is not cancellable
            ):
                old_job.cancellable.cancel()
        if thread_to_start is not None:
            try:
                thread_to_start.start()
            except BaseException:
                with self._clipboard_encode_lock:
                    if self._clipboard_encode_thread is thread_to_start:
                        self._clipboard_encode_thread = None
                    if self._clipboard_encode_pending is job:
                        self._clipboard_encode_pending = None
                raise
        return True

    def _run_clipboard_encode_lane(self) -> None:
        """Encode at most one texture while coalescing pending requests."""
        while True:
            with self._clipboard_encode_lock:
                job = self._clipboard_encode_pending
                self._clipboard_encode_pending = None
                if job is None:
                    self._clipboard_encode_thread = None
                    return
                self._clipboard_encode_active = job

            self._save_clipboard_texture_worker(job)

            with self._clipboard_encode_lock:
                if self._clipboard_encode_active is job:
                    self._clipboard_encode_active = None

    def _save_clipboard_texture_worker(self, job: _ClipboardEncodeJob) -> None:
        """Encode one bounded immutable texture outside the GTK main loop."""
        error: Exception | None = None
        saved = False
        tmp_path: str | None = None
        try:
            if job.cancellable is None or not job.cancellable.is_cancelled():
                fd, tmp_path = mkstemp(suffix=".png", prefix="bigocrimage_paste_")
                os.close(fd)
                saved = bool(job.texture.save_to_png(tmp_path))
                if not saved:
                    error = OSError("GTK could not encode the clipboard texture")
        except Exception as caught_error:
            error = caught_error
        GLib.idle_add(
            self._on_clipboard_texture_saved,
            tmp_path,
            saved,
            error,
            job.generation,
            job.cancellable,
        )

    def _on_clipboard_texture_saved(
        self,
        tmp_path: str | None,
        saved: bool,
        error: Exception | None,
        generation: int | None,
        cancellable: Gio.Cancellable | None,
    ) -> bool:
        """Publish a worker-encoded clipboard image only to its live owner."""
        is_current = self._is_current_input_operation(generation, cancellable)
        if not saved or not is_current:
            if tmp_path is not None:
                remove_file(tmp_path)
            if error is not None and is_current:
                logger.error(f"Clipboard paste error: {error}")
                self._toast_overlay.add_toast(
                    Adw.Toast(title=_("Could not read the clipboard image"))
                )
            self._finish_input_operation(generation, cancellable)
            return GLib.SOURCE_REMOVE

        assert tmp_path is not None
        self._finish_input_operation(generation, cancellable)
        self._start_processing(tmp_path, cleanup_path=tmp_path)
        return GLib.SOURCE_REMOVE

    def _shutdown_clipboard_encode_lane(self) -> None:
        """Cancel and discard pending clipboard work during window teardown."""
        if not hasattr(self, "_clipboard_encode_lock"):
            return
        with self._clipboard_encode_lock:
            self._clipboard_encode_stopping = True
            pending = self._clipboard_encode_pending
            active = self._clipboard_encode_active
            self._clipboard_encode_pending = None
        for job in (pending, active):
            if job is not None and job.cancellable is not None:
                job.cancellable.cancel()

    def _on_clipboard_uri_ready(
        self,
        clipboard: Gdk.Clipboard,
        result: Gio.AsyncResult,
        generation: int | None = None,
        cancellable: Gio.Cancellable | None = None,
    ) -> None:
        """Handle clipboard URI read completion."""
        stream = None
        try:
            stream = clipboard.read_finish(result)[0]
            if not stream:
                self._finish_input_operation(generation, cancellable)
                return
            if not self._is_current_input_operation(generation, cancellable):
                self._close_stream_async(stream)
                return
            self._read_clipboard_uri_chunk(
                stream,
                bytearray(),
                generation,
                cancellable,
            )
        except Exception as e:
            should_log = self._is_current_input_operation(generation, cancellable)
            if stream is not None:
                self._close_stream_async(stream)
            if should_log:
                logger.error(f"Clipboard URI paste error: {e}")
            self._finish_input_operation(generation, cancellable)

    def _read_clipboard_uri_chunk(
        self,
        stream: Gio.InputStream,
        payload: bytearray,
        generation: int | None,
        cancellable: Gio.Cancellable | None,
    ) -> None:
        """Read URI text without blocking GTK and with a strict byte budget."""
        if not self._is_current_input_operation(generation, cancellable):
            self._close_stream_async(stream)
            return
        remaining = self._MAX_CLIPBOARD_URI_BYTES + 1 - len(payload)
        stream.read_bytes_async(
            min(8192, remaining),
            GLib.PRIORITY_DEFAULT,
            cancellable,
            self._on_clipboard_uri_chunk_ready,
            payload,
            generation,
            cancellable,
        )

    def _on_clipboard_uri_chunk_ready(
        self,
        stream: Gio.InputStream,
        result: Gio.AsyncResult,
        payload: bytearray,
        generation: int | None,
        cancellable: Gio.Cancellable | None,
    ) -> None:
        """Accumulate one async URI chunk and consume it at EOF."""
        keep_open = False
        try:
            if not self._is_current_input_operation(generation, cancellable):
                return
            raw_data = stream.read_bytes_finish(result).get_data() or b""
            if raw_data:
                payload.extend(raw_data)
                if len(payload) > self._MAX_CLIPBOARD_URI_BYTES:
                    self._toast_overlay.add_toast(
                        Adw.Toast(title=_("Clipboard file list is too large"))
                    )
                    return
                self._read_clipboard_uri_chunk(
                    stream,
                    payload,
                    generation,
                    cancellable,
                )
                keep_open = True
                return

            data = bytes(payload).decode("utf-8", errors="replace")
            paths = parse_clipboard_file_paths(data)
            image_path = next(
                (
                    path
                    for path in paths
                    if os.path.splitext(path)[1].lower() in self._SUPPORTED_IMAGE_EXTENSIONS
                ),
                None,
            )
            self._finish_input_operation(generation, cancellable)
            if image_path is not None:
                self._start_processing(image_path)
            else:
                self._toast_overlay.add_toast(Adw.Toast(title=_("No image found in clipboard")))
        except Exception as e:
            if getattr(self, "_alive", True) and not (
                cancellable is not None and cancellable.is_cancelled()
            ):
                logger.error(f"Clipboard URI paste error: {e}")
        finally:
            if not keep_open:
                self._close_stream_async(stream)
                self._finish_input_operation(generation, cancellable)

    @staticmethod
    def _close_stream_async(stream: Gio.InputStream) -> None:
        """Close an input stream without blocking the GTK main loop."""

        def on_closed(source: Gio.InputStream, result: Gio.AsyncResult) -> None:
            try:
                source.close_finish(result)
            except GLib.Error:
                pass

        try:
            stream.close_async(
                GLib.PRIORITY_DEFAULT,
                None,
                on_closed,
            )
        except (AttributeError, TypeError):
            stream.close(None)

    def _build_source_buttons(self) -> Gtk.Box:
        """Build a fresh pair of source actions for one status page.

        Every page that has no text to show offers the same two ways to get some,
        and a widget belongs to a single parent, so each page needs its own pair.
        """
        btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        btn_box.set_halign(Gtk.Align.CENTER)

        open_content = Adw.ButtonContent()
        open_content.set_icon_name("document-open-symbolic")
        open_content.set_label(_("Open Image"))
        open_btn = Gtk.Button()
        open_btn.set_child(open_content)
        open_btn.add_css_class("pill")
        set_a11y_label(open_btn, _("Open Image"))
        open_btn.connect("clicked", self._on_open_image_clicked)
        self._apply_ocr_availability_to_button(open_btn)
        btn_box.append(open_btn)

        capture_content = Adw.ButtonContent()
        capture_content.set_icon_name("camera-photo-symbolic")
        capture_content.set_label(_("Screen Capture"))
        capture_btn = Gtk.Button()
        capture_btn.set_child(capture_content)
        capture_btn.add_css_class("pill")
        capture_btn.add_css_class("suggested-action")
        set_a11y_label(capture_btn, _("Screen Capture"))
        capture_btn.connect("clicked", self._on_new_capture_clicked)
        self._apply_ocr_availability_to_button(capture_btn)
        btn_box.append(capture_btn)

        return btn_box

    def _build_welcome_page(self) -> None:
        """Build the welcome page with Adw.StatusPage and action buttons."""
        status = Adw.StatusPage()
        status.set_icon_name("camera-photo-symbolic")
        status.set_title(_("Image OCR"))
        status.set_description(_("Extract text from images or screen captures using OCR."))
        status.set_child(self._build_source_buttons())
        self._stack.add_named(status, "welcome")

    def _build_empty_page(self) -> None:
        """Build the page shown when OCR ran and found no readable text."""
        status = Adw.StatusPage()
        status.set_icon_name("x-office-document-symbolic")
        status.set_title(_("No text found"))
        status.set_description(
            _(
                "OCR finished but found no readable text. Capture a tighter region, "
                "or try a sharper image."
            )
        )
        status.set_child(self._build_source_buttons())
        self._stack.add_named(status, "empty")

    def _build_loading_page(self) -> None:
        """Build the loading page with Adw.StatusPage and spinner."""
        status = Adw.StatusPage()
        status.set_icon_name("content-loading-symbolic")
        status.set_title(_("Extracting text…"))

        spinner = Gtk.Spinner()
        spinner.set_size_request(32, 32)
        spinner.start()
        spinner.set_halign(Gtk.Align.CENTER)

        self._cancel_button = Gtk.Button(label=_("Cancel"))
        self._cancel_button.set_halign(Gtk.Align.CENTER)
        self._cancel_button.add_css_class("pill")
        self._cancel_button.set_tooltip_text(_("Cancel text extraction"))
        set_a11y_label(self._cancel_button, _("Cancel text extraction"))
        self._cancel_button.connect("clicked", self._on_cancel_processing)

        loading_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        loading_box.set_halign(Gtk.Align.CENTER)
        loading_box.append(spinner)
        loading_box.append(self._cancel_button)
        status.set_child(loading_box)

        self._stack.add_named(status, "loading")

    def _build_results_page(self) -> None:
        """Build the results page with text view and bottom action bar."""
        results_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        # Scrollable text view
        text_scroll = Gtk.ScrolledWindow()
        text_scroll.set_vexpand(True)
        text_scroll.set_hexpand(True)

        self._text_view = Gtk.TextView()
        self._text_view.set_editable(True)
        self._text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        # The formatter aligns columns and indents with spaces, which only line up
        # under a fixed-width font.
        self._text_view.add_css_class("monospace")
        self._text_view.set_left_margin(18)
        self._text_view.set_right_margin(18)
        self._text_view.set_top_margin(12)
        self._text_view.set_bottom_margin(12)
        set_a11y_label(self._text_view, _("Extracted text"))

        self._text_buffer = self._text_view.get_buffer()
        text_scroll.set_child(self._text_view)
        results_box.append(text_scroll)

        # Bottom action bar with labeled buttons
        action_bar = Gtk.ActionBar()

        # Left: Copy button
        self._copy_button = Gtk.Button(
            icon_name="edit-copy-symbolic",
            label=_("Copy"),
        )
        self._copy_button.set_tooltip_text(_("Copy text to clipboard"))
        self._copy_button.update_property(
            [Gtk.AccessibleProperty.LABEL], [_("Copy text to clipboard")]
        )
        self._copy_button.connect("clicked", self._on_copy_clicked)
        action_bar.pack_start(self._copy_button)

        # Right: New Capture (primary action) and Open Image
        capture_btn = Gtk.Button(
            icon_name="camera-photo-symbolic",
            label=_("New Capture"),
        )
        capture_btn.add_css_class("suggested-action")
        capture_btn.set_tooltip_text(_("Capture a screen region"))
        set_a11y_label(capture_btn, _("Capture a screen region"))
        capture_btn.connect("clicked", self._on_new_capture_clicked)
        self._apply_ocr_availability_to_button(capture_btn)
        action_bar.pack_end(capture_btn)

        open_btn = Gtk.Button(
            icon_name="document-open-symbolic",
            label=_("Open Image"),
        )
        open_btn.set_tooltip_text(_("Open an image file"))
        set_a11y_label(open_btn, _("Open an image file"))
        open_btn.connect("clicked", self._on_open_image_clicked)
        self._apply_ocr_availability_to_button(open_btn)
        action_bar.pack_end(open_btn)

        results_box.append(action_bar)
        self._stack.add_named(results_box, "results")

    # ── OCR Processing ──────────────────────────────────────────────────

    def _start_processing(self, image_path: str, *, cleanup_path: str | None = None) -> bool:
        """Start OCR processing for an image file.

        Args:
            image_path: Path to the image file
        """
        if not self._require_ocr_available():
            if cleanup_path is not None:
                remove_file(cleanup_path)
            return False

        generation = self._begin_operation()
        if generation < 0:
            if cleanup_path is not None:
                remove_file(cleanup_path)
            return False

        self._stack.set_visible_child_name("loading")
        self._copy_button.set_sensitive(False)
        self._set_cancel_enabled(True)

        request: ImageOcrRequest | None = None
        completed = False

        def callback(outcome: ImageOcrOutcome) -> None:
            nonlocal completed
            completed = True
            if cleanup_path is not None:
                remove_file(cleanup_path)
            if self._is_current_operation(generation, request):
                self._active_request = None
                self._set_cancel_enabled(False)
                self._on_processing_complete(outcome)

        try:
            self._settings.load_settings()
            config = self._settings._snapshot_ocr_config()
            request = self._screen_capture_service.process_image_file(
                image_path,
                callback=callback,
                config=config,
            )
            if completed or not self._is_current_operation(generation):
                request.cancel()
            else:
                self._active_request = request
        except Exception:
            if cleanup_path is not None:
                remove_file(cleanup_path)
            if self._is_current_operation(generation):
                self._stack.set_visible_child_name(self._stable_page_name)
                self._set_cancel_enabled(False)
            raise
        return True

    def _on_processing_complete(self, outcome: ImageOcrOutcome) -> None:
        """Handle processing completion.

        Args:
            outcome: Explicit OCR completion state
        """
        if outcome.status == ImageOcrStatus.CANCELLED:
            self._stack.set_visible_child_name(self._stable_page_name)
            self._sync_copy_button_state()
            return

        if outcome.status == ImageOcrStatus.ERROR:
            self._show_error(outcome.message or _("OCR processing failed."))
            self._stack.set_visible_child_name(self._stable_page_name)
            self._sync_copy_button_state()
            return

        if outcome.status == ImageOcrStatus.SUCCESS and outcome.text:
            self._text_buffer.set_text(outcome.text)
            self._copy_button.set_sensitive(True)
            self._stable_page_name = "results"
        else:
            # An empty result is its own state. Putting the explanation in the
            # editable buffer would offer prose the OCR never read as if it were
            # the extracted text.
            self._text_buffer.set_text("")
            self._copy_button.set_sensitive(False)
            self._stable_page_name = "empty"

        self._stack.set_visible_child_name(self._stable_page_name)

    def _sync_copy_button_state(self) -> None:
        """Enable Copy exactly when the current result buffer has text."""
        start_iter, end_iter = self._text_buffer.get_bounds()
        text = self._text_buffer.get_text(start_iter, end_iter, True)
        self._copy_button.set_sensitive(bool(text))

    # ── Capture & Open ──────────────────────────────────────────────────

    def _on_new_capture_clicked(self, *_args: object) -> None:
        """Start a new screen capture.

        Hides the window entirely (unmap) instead of minimizing, so that
        re-showing it after capture triggers a fresh map on the compositor,
        which reliably grants focus on KDE Plasma / Wayland.
        """
        if not self._require_ocr_available():
            return

        generation = self._begin_operation()
        if generation < 0:
            return

        self._is_hidden_for_capture = True
        self._hidden_capture_generation = generation
        self.set_visible(False)
        self._capture_delay_source_id = GLib.timeout_add(
            200,
            self._on_capture_delay,
            generation,
        )

    def _on_capture_delay(self, generation: int) -> bool:
        """Start only the capture that still owns the delayed source."""
        self._capture_delay_source_id = 0
        if (
            not self._is_current_operation(generation)
            or self._hidden_capture_generation != generation
        ):
            return GLib.SOURCE_REMOVE
        return self._trigger_capture(generation)

    def _trigger_capture(self, generation: int | None = None) -> bool:
        """Trigger screen capture after minimize delay."""
        if generation is None:
            generation = self._begin_operation()
        if not self._require_ocr_available():
            self._restore_after_capture(generation)
            return False
        if generation < 0 or not self._is_current_operation(generation):
            return False

        request: ImageOcrRequest | None = None
        completed = False

        def on_processing() -> None:
            if self._is_current_operation(generation, request):
                self._on_capture_taken(generation)

        def on_complete(outcome: ImageOcrOutcome) -> None:
            nonlocal completed
            completed = True
            if self._is_current_operation(generation, request):
                self._active_request = None
                self._set_cancel_enabled(False)
                self._restore_after_capture(generation)
                self._on_processing_complete(outcome)

        try:
            self._settings.load_settings()
            config = self._settings._snapshot_ocr_config()
            request = self._screen_capture_service.capture_screen_region(
                callback=on_complete,
                on_processing=on_processing,
                config=config,
            )
        except Exception as error:
            logger.error(f"Could not start screen capture: {error}")
            if not completed and self._is_current_operation(generation):
                self._active_request = None
                self._set_cancel_enabled(False)
                self._restore_after_capture(generation)
                self._stack.set_visible_child_name(self._stable_page_name)
                self._show_error(
                    _("Screen capture failed. Please try again or open an image file.")
                )
            return False
        if completed or not self._is_current_operation(generation):
            request.cancel()
        else:
            self._active_request = request
            self._set_cancel_enabled(True)
        return False

    def _on_capture_taken(self, generation: int) -> None:
        """Handle the moment right after the screenshot is captured (before OCR).

        Re-maps the hidden window (set_visible + present), which on Wayland
        compositors treats it as a fresh surface and grants focus reliably.
        Falls back to the modal window hack if focus is still not obtained.
        """
        if not self._is_current_operation(generation):
            return
        self._stack.set_visible_child_name("loading")
        self._restore_after_capture(generation)
        self._set_cancel_enabled(True)

        # Fallback: modal hack in case present() alone didn't work
        def _check_and_apply_hack() -> bool:
            self._focus_idle_source_id = 0
            if self._is_current_operation(generation) and not self.is_active():
                logger.info("Window not active after re-map, applying modal hack.")
                hack_window = Gtk.Window(transient_for=self, modal=True)
                hack_window.set_default_size(1, 1)
                hack_window.set_decorated(False)
                hack_window.present()
                GLib.idle_add(hack_window.destroy)
            return GLib.SOURCE_REMOVE

        self._focus_idle_source_id = GLib.idle_add(
            _check_and_apply_hack,
            priority=GLib.PRIORITY_LOW,
        )

    def _restore_after_capture(self, generation: int | None = None) -> None:
        """Re-map the window after a capture succeeds, fails, or is cancelled."""
        hidden_generation = getattr(self, "_hidden_capture_generation", None)
        if generation is None:
            generation = hidden_generation
        if hidden_generation is not None and generation != hidden_generation:
            return
        if not getattr(self, "_is_hidden_for_capture", False):
            return

        self._is_hidden_for_capture = False
        self._hidden_capture_generation = None
        if not getattr(self, "_alive", True):
            return
        self.set_visible(True)
        self.present()

    def _on_open_image_clicked(self, *_args: object) -> None:
        """Open file chooser to select an image."""
        if not self._require_ocr_available():
            return

        generation, cancellable = self._begin_input_operation()
        dialog = Gtk.FileDialog()
        dialog.set_title(_("Open Image to OCR"))

        filter_images = Gtk.FileFilter()
        filter_images.set_name(_("Images"))
        filter_images.add_mime_type("image/*")

        store = Gio.ListStore.new(Gtk.FileFilter)
        store.append(filter_images)
        dialog.set_filters(store)

        dialog.open(
            self,
            cancellable,
            lambda source, result: self._on_file_opened(
                source,
                result,
                generation,
                cancellable,
            ),
        )

    def _on_file_opened(
        self,
        dialog: Gtk.FileDialog,
        result: Gio.AsyncResult,
        generation: int | None = None,
        cancellable: Gio.Cancellable | None = None,
    ) -> None:
        """Handle file selection result."""
        try:
            if not self._is_current_input_operation(generation, cancellable):
                return
            file = dialog.open_finish(result)
            if file:
                file_path = file.get_path()
                if file_path:
                    self._finish_input_operation(generation, cancellable)
                    self._start_processing(file_path)
        except GLib.Error as e:
            dismissed = e.matches(
                Gtk.DialogError.quark(),
                Gtk.DialogError.CANCELLED,
            ) or e.matches(
                Gtk.DialogError.quark(),
                Gtk.DialogError.DISMISSED,
            )
            cancelled = e.matches(
                Gio.io_error_quark(),
                Gio.IOErrorEnum.CANCELLED,
            )
            if not dismissed and not cancelled:
                logger.error(f"Error opening file: {e}")
                self._toast_overlay.add_toast(
                    Adw.Toast(title=_("Could not open the selected image"))
                )
        finally:
            self._finish_input_operation(generation, cancellable)

    # ── Copy & Clipboard ────────────────────────────────────────────────

    def _on_copy_clicked(self, _btn: Gtk.Button) -> None:
        """Copy extracted text through the standard GTK4 content provider API."""
        start_iter, end_iter = self._text_buffer.get_bounds()
        text = self._text_buffer.get_text(start_iter, end_iter, True)

        if not text:
            return

        logger.info(f"Copying to clipboard: {len(text)} chars")
        clipboard = get_default_clipboard()
        if clipboard is None:
            logger.warning("Clipboard is unavailable because no display is active")
            return
        data = GLib.Bytes.new(text.encode("utf-8"))
        provider = Gdk.ContentProvider.new_for_bytes("text/plain;charset=utf-8", data)
        if not clipboard.set_content(provider):
            logger.warning("Clipboard rejected the extracted text content")
            self._toast_overlay.add_toast(
                Adw.Toast(title=_("Could not copy text to the clipboard"))
            )
            return

        self._toast_overlay.add_toast(Adw.Toast(title=_("Copied to clipboard")))

    # ── Helpers ─────────────────────────────────────────────────────────

    def _apply_ocr_availability_to_button(self, button: Gtk.Button) -> None:
        button.set_sensitive(self.ocr_dependency.is_available)
        if not self.ocr_dependency.is_available:
            button.set_tooltip_text(
                _("OCR is unavailable. Install the required engine and restart the application.")
            )

    def _require_ocr_available(self) -> bool:
        if self.ocr_dependency.is_available:
            return True
        self.show_ocr_unavailable_dialog()
        return False

    def show_ocr_unavailable_dialog(self) -> bool:
        """Explain why OCR is blocked and the action required to recover."""
        if self.ocr_dependency.is_available or self._ocr_unavailable_dialog is not None:
            return False

        self._ocr_unavailable_dialog = present_ocr_unavailable_dialog(
            self,
            self.ocr_dependency.error,
            self._on_ocr_unavailable_response,
        )
        return False

    def _on_ocr_unavailable_response(
        self,
        _dialog: Adw.AlertDialog,
        _response: str,
    ) -> None:
        self._ocr_unavailable_dialog = None

    def open_image(self, file_path: str) -> None:
        """Open and process an image file.

        Public method called when a file is passed via command line.

        Args:
            file_path: Path to the image file to process
        """
        if file_path:
            self._start_processing(file_path)

    def _show_error(self, message: str) -> None:
        """Show error dialog.

        Args:
            message: Error message to display
        """
        alert = Adw.AlertDialog()
        alert.set_heading(_("Error"))
        alert.set_body(message)
        alert.add_response("ok", _("OK"))
        alert.present(self)
