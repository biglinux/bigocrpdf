"""
BigOcrPdf - Image OCR Window

Standalone window for Image OCR using RapidOCR PP-OCRv5.
Supports opening image files and capturing screen regions.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk

import tempfile

from bigocrpdf.config import IMAGE_WINDOW_STATE_KEY
from bigocrpdf.services.processor import OcrProcessor
from bigocrpdf.services.screen_capture import ScreenCaptureService
from bigocrpdf.services.settings import OcrSettings
from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.config_manager import get_config_manager
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class ImageOcrWindow(Adw.ApplicationWindow):
    """Standalone window for Image OCR operations using RapidOCR."""

    def __init__(
        self,
        application: Gtk.Application,
        image_path: str | None = None,
    ) -> None:
        """Initialize the Image OCR Window.

        Args:
            application: The Gtk Application instance
            image_path: Optional path to an image file to process immediately
        """
        width, height = self._load_window_size()

        super().__init__(
            application=application,
            default_width=width,
            default_height=height,
        )

        self.set_title("Big Image OCR")
        self.set_icon_name("bigocrimage")

        self._image_path = image_path

        # Initialize services
        self._settings = OcrSettings()
        self._screen_capture_service = ScreenCaptureService(self)
        self._ocr_processor = OcrProcessor(self._settings)

        # UI state
        self._current_image_path: str | None = None
        self._lang_codes: list[str] = []

        # Load language from config
        config = get_config_manager()
        config_lang = config.get("ocr.language")
        if config_lang:
            self._settings.lang = config_lang

        self._setup_ui()
        self.connect("close-request", self._on_close_request)

        # Process provided image or show welcome state
        if self._image_path:
            self._start_processing(self._image_path)
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
        config = get_config_manager()
        width = self.get_width()
        height = self.get_height()
        if width > 0 and height > 0:
            config.set(f"{IMAGE_WINDOW_STATE_KEY}.width", width, save_immediately=False)
            config.set(f"{IMAGE_WINDOW_STATE_KEY}.height", height, save_immediately=True)

    def _on_close_request(self, _window: Gtk.Window) -> bool:
        """Handle window close request — save state."""
        self._save_window_size()
        return False

    # ── UI Setup ────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        """Set up the window UI following Adwaita HIG patterns.

        Layout:
        - Header bar (raised): language dropdown on the left
        - Content stack: welcome / loading / results pages
        - Results page includes a bottom action bar with labeled buttons
        """
        toolbar_view = Adw.ToolbarView()
        toolbar_view.set_top_bar_style(Adw.ToolbarStyle.RAISED)
        self.set_content(toolbar_view)

        # Header Bar — clean, with language selector only
        header = Adw.HeaderBar()
        self._lang_dropdown = self._create_language_dropdown()
        header.pack_start(self._lang_dropdown)
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

        # Enable drag-and-drop for image files
        self._setup_drop_target()

    _SUPPORTED_IMAGE_EXTENSIONS = frozenset((
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".gif",
        ".avif",
        ".heif",
        ".heic",
        ".jxl",
        ".ico",
        ".psd",
        ".eps",
        ".tga",
        ".pbm",
        ".pgm",
        ".ppm",
        ".xbm",
        ".xpm",
    ))

    def _setup_drop_target(self) -> None:
        """Set up drag-and-drop target for image files."""
        drop = Gtk.DropTarget.new(Gio.File, Gdk.DragAction.COPY)
        drop.connect("drop", self._on_drop)
        self.add_controller(drop)

    def _on_drop(self, _target: Gtk.DropTarget, value: Gio.File, _x: float, _y: float) -> bool:
        """Handle dropped file."""
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
        clipboard = Gdk.Display.get_default().get_clipboard()
        formats = clipboard.get_formats()
        if formats.contain_gtype(Gdk.Texture):
            clipboard.read_texture_async(None, self._on_clipboard_texture_ready)
        elif formats.contain_mime_type("text/uri-list"):
            clipboard.read_async(
                ["text/uri-list"],
                GLib.PRIORITY_DEFAULT,
                None,
                self._on_clipboard_uri_ready,
            )
        else:
            self._toast_overlay.add_toast(Adw.Toast(title=_("No image found in clipboard")))

    def _on_clipboard_texture_ready(
        self, clipboard: Gdk.Clipboard, result: Gio.AsyncResult
    ) -> None:
        """Handle clipboard texture read completion."""
        try:
            texture = clipboard.read_texture_finish(result)
            if not texture:
                return
            fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="bigocrimage_paste_")
            import os

            os.close(fd)
            texture.save_to_png(tmp_path)
            self._start_processing(tmp_path)
        except Exception as e:
            logger.error(f"Clipboard paste error: {e}")

    def _on_clipboard_uri_ready(self, clipboard: Gdk.Clipboard, result: Gio.AsyncResult) -> None:
        """Handle clipboard URI read completion."""
        try:
            stream = clipboard.read_finish(result)[0]
            if not stream:
                return
            data = stream.read_bytes(65536, None).get_data().decode("utf-8", errors="replace")
            stream.close(None)
            import os
            from urllib.parse import unquote, urlparse

            for line in data.splitlines():
                line = line.strip()
                if line.startswith("file://"):
                    path = unquote(urlparse(line).path)
                    ext = os.path.splitext(path)[1].lower()
                    if ext in self._SUPPORTED_IMAGE_EXTENSIONS and os.path.isfile(path):
                        self._start_processing(path)
                        return
            self._toast_overlay.add_toast(Adw.Toast(title=_("No image found in clipboard")))
        except Exception as e:
            logger.error(f"Clipboard URI paste error: {e}")

    def _build_welcome_page(self) -> None:
        """Build the welcome page with Adw.StatusPage and action buttons."""
        status = Adw.StatusPage()
        status.set_icon_name("camera-photo-symbolic")
        status.set_title(_("Image OCR"))
        status.set_description(_("Extract text from images or screen captures using OCR."))

        # Action buttons embedded in the status page
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
        btn_box.append(capture_btn)

        status.set_child(btn_box)
        self._stack.add_named(status, "welcome")

    def _build_loading_page(self) -> None:
        """Build the loading page with Adw.StatusPage and spinner."""
        status = Adw.StatusPage()
        status.set_icon_name("content-loading-symbolic")
        status.set_title(_("Extracting text…"))

        spinner = Gtk.Spinner()
        spinner.set_size_request(32, 32)
        spinner.start()
        spinner.set_halign(Gtk.Align.CENTER)
        status.set_child(spinner)

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
        action_bar.pack_end(capture_btn)

        open_btn = Gtk.Button(
            icon_name="document-open-symbolic",
            label=_("Open Image"),
        )
        open_btn.set_tooltip_text(_("Open an image file"))
        set_a11y_label(open_btn, _("Open an image file"))
        open_btn.connect("clicked", self._on_open_image_clicked)
        action_bar.pack_end(open_btn)

        results_box.append(action_bar)
        self._stack.add_named(results_box, "results")

    # ── Language Dropdown ───────────────────────────────────────────────

    def _create_language_dropdown(self) -> Gtk.DropDown:
        """Create the language dropdown with available RapidOCR languages.

        Returns:
            Configured Gtk.DropDown widget
        """
        dropdown = Gtk.DropDown()
        dropdown.set_can_focus(False)
        string_list = Gtk.StringList()

        # Get available languages from RapidOCR models
        languages = self._ocr_processor.get_available_ocr_languages()
        self._lang_codes = []
        default_index = 0

        for i, (lang_code, lang_name) in enumerate(languages):
            string_list.append(lang_name)
            self._lang_codes.append(lang_code)
            if lang_code == self._settings.lang:
                default_index = i

        dropdown.set_model(string_list)
        dropdown.set_tooltip_text(_("Choose the language of the text in the image"))
        set_a11y_label(dropdown, _("OCR language"))

        # Defer selection and signal connection until widget is first mapped
        def on_map(_widget: Gtk.Widget) -> None:
            dropdown.set_can_focus(True)
            dropdown.set_selected(default_index)
            dropdown.connect("notify::selected", self._on_lang_changed)
            dropdown.disconnect(handler_id)

        handler_id = dropdown.connect("map", on_map)
        return dropdown

    def _on_lang_changed(self, dropdown: Gtk.DropDown, _pspec: object) -> None:
        """Handle language selection change."""
        selected = dropdown.get_selected()
        if 0 <= selected < len(self._lang_codes):
            new_lang = self._lang_codes[selected]
            self._settings.lang = new_lang
            config = get_config_manager()
            config.set("ocr.language", new_lang, save_immediately=True)
            logger.info(f"OCR language changed to: {new_lang}")

    # ── OCR Processing ──────────────────────────────────────────────────

    def _start_processing(self, image_path: str) -> None:
        """Start OCR processing for an image file.

        Args:
            image_path: Path to the image file
        """
        self._stack.set_visible_child_name("loading")
        self._copy_button.set_sensitive(False)

        # Store current image path
        self._current_image_path = image_path

        self._screen_capture_service.process_image_file(
            image_path,
            callback=self._on_processing_complete,
            language=self._settings.lang,
        )

    def _on_processing_complete(self, text: str | None, error: str | None) -> None:
        """Handle processing completion.

        Args:
            text: Extracted text or None
            error: Error message or None
        """
        if error:
            self._show_error(error)
            self._stack.set_visible_child_name("results")
            return

        if text:
            self._text_buffer.set_text(text)
            self._copy_button.set_sensitive(True)
        else:
            self._text_buffer.set_text(_("No text extracted."))
            self._copy_button.set_sensitive(False)

        self._stack.set_visible_child_name("results")

    # ── Capture & Open ──────────────────────────────────────────────────

    def _on_new_capture_clicked(self, *_args: object) -> None:
        """Start a new screen capture.

        Hides the window entirely (unmap) instead of minimizing, so that
        re-showing it after capture triggers a fresh map on the compositor,
        which reliably grants focus on KDE Plasma / Wayland.
        """
        self.set_visible(False)
        GLib.timeout_add(200, self._trigger_capture)

    def _trigger_capture(self) -> bool:
        """Trigger screen capture after minimize delay."""
        self._settings.load_settings()
        self._screen_capture_service.capture_screen_region(
            callback=self._on_processing_complete,
            on_processing=self._on_capture_taken,
            language=self._settings.lang,
        )
        return False

    def _on_capture_taken(self) -> None:
        """Handle the moment right after the screenshot is captured (before OCR).

        Re-maps the hidden window (set_visible + present), which on Wayland
        compositors treats it as a fresh surface and grants focus reliably.
        Falls back to the modal window hack if focus is still not obtained.
        """
        self._stack.set_visible_child_name("loading")
        self.set_visible(True)
        self.present()

        # Fallback: modal hack in case present() alone didn't work
        def _check_and_apply_hack() -> bool:
            if not self.is_active():
                logger.info("Window not active after re-map, applying modal hack.")
                hack_window = Gtk.Window(transient_for=self, modal=True)
                hack_window.set_default_size(1, 1)
                hack_window.set_decorated(False)
                hack_window.present()
                GLib.idle_add(hack_window.destroy)
            return GLib.SOURCE_REMOVE

        GLib.idle_add(_check_and_apply_hack, priority=GLib.PRIORITY_LOW)

    def _on_open_image_clicked(self, *_args: object) -> None:
        """Open file chooser to select an image."""
        dialog = Gtk.FileDialog()
        dialog.set_title(_("Open Image to OCR"))

        filter_images = Gtk.FileFilter()
        filter_images.set_name(_("Images"))
        filter_images.add_mime_type("image/*")

        store = Gio.ListStore.new(Gtk.FileFilter)
        store.append(filter_images)
        dialog.set_filters(store)

        dialog.open(self, None, self._on_file_opened)

    def _on_file_opened(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        """Handle file selection result."""
        try:
            file = dialog.open_finish(result)
            if file:
                self._start_processing(file.get_path())
        except GLib.Error as e:
            if "dismissed" not in str(e):
                logger.error(f"Error opening file: {e}")

    # ── Copy & Clipboard ────────────────────────────────────────────────

    def _on_copy_clicked(self, _btn: Gtk.Button) -> None:
        """Copy extracted text to clipboard.

        Uses a triple-fallback chain due to inconsistent GDK4 clipboard API
        across GTK versions and Wayland compositors:
        1. set_text() — available in some GDK4 builds
        2. ContentProvider.new_for_bytes() — standard GTK4 approach
        3. ContentProvider.new_for_value() — last resort fallback
        This workaround can be simplified once GTK >= 4.14 is the minimum.
        """
        start_iter, end_iter = self._text_buffer.get_bounds()
        text = self._text_buffer.get_text(start_iter, end_iter, True)

        if not text:
            return

        logger.info(f"Copying to clipboard: {len(text)} chars")
        clipboard = Gdk.Display.get_default().get_clipboard()

        try:
            clipboard.set_text(text)
        except AttributeError:
            # Fallback for GDK4 environments where set_text is missing
            try:
                data = GLib.Bytes.new(text.encode("utf-8"))
                provider = Gdk.ContentProvider.new_for_bytes("text/plain", data)
                clipboard.set_content(provider)
                logger.info("Used fallback clipboard method")
            except Exception as e:
                logger.error(f"Clipboard fallback failed: {e}")
                clipboard.set(Gdk.ContentProvider.new_for_value(text))

        self._toast_overlay.add_toast(Adw.Toast(title=_("Copied to clipboard")))

    # ── Helpers ─────────────────────────────────────────────────────────

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
