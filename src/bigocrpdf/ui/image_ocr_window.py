"""
BigOcrPdf - Image OCR Window

This module provides a standalone window for Image OCR functionality.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, GLib, Gtk

from bigocrpdf.services.screen_capture import ScreenCaptureService
from bigocrpdf.services.settings import OcrSettings
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger
from bigocrpdf.config import (
    IMAGE_WINDOW_STATE_KEY,
    DEFAULT_WINDOW_WIDTH,
    DEFAULT_WINDOW_HEIGHT,
)
from bigocrpdf.utils.config_manager import get_config_manager


class ImageOcrWindow(Adw.Window):
    """Standalone window for Image OCR operations."""

    def __init__(self, application: Gtk.Application, image_path: str | None = None) -> None:
        """Initialize the Image OCR Window.

        Args:
            application: The Gtk Application instance
            image_path: Optional path to an image file to process immediately
        """
        # Load saved window size
        width, height = self._load_window_size()

        super().__init__(application=application, default_width=width, default_height=height)

        self.set_title(_("Big OCR Image"))
        self.set_icon_name("bigocrimage")

        # Important: Destroying this window should potentially quit the app
        # if it's the only one, which Gtk.Application handles if the window is properly added.

        self._image_path = image_path
        # self._text is used for initial load, but buffer is truth for editing
        self._text: str | None = None

        # Initialize services
        self._settings = OcrSettings()
        self._screen_capture_service = ScreenCaptureService(self)

        self._setup_ui()

        self.connect("close-request", self._on_close_request)

        # Start processing if image provided
        if self._image_path:
            self._start_processing(self._image_path)
        else:
            # If no image, maybe start correctly in "New Capture" mode or empty state?
            # For now, let's trigger a new capture automatically if opened without args?
            # Or just show empty state. Let's show empty state.
            self._stack.set_visible_child_name("results")
            self._update_ui_state(empty=True)

    def _load_window_size(self) -> tuple[int, int]:
        """Load window size from configuration."""
        config = get_config_manager()
        width = config.get(f"{IMAGE_WINDOW_STATE_KEY}.width", 600)
        height = config.get(f"{IMAGE_WINDOW_STATE_KEY}.height", 500)

        # Ensure minimum reasonable size
        width = max(width, 400)
        height = max(height, 300)

        return width, height

    def _save_window_size(self) -> None:
        """Save current window size to configuration."""
        config = get_config_manager()
        width = self.get_width()
        height = self.get_height()

        if width > 0 and height > 0:
            config.set(f"{IMAGE_WINDOW_STATE_KEY}.width", width, save_immediately=False)
            config.set(f"{IMAGE_WINDOW_STATE_KEY}.height", height, save_immediately=True)

    def _on_close_request(self, window: Gtk.Window) -> bool:
        """Handle window close request - save state."""
        self._save_window_size()
        return False

    def _setup_ui(self) -> None:
        """Set up the window UI."""
        # Main layout
        toolbar_view = Adw.ToolbarView()
        self.set_content(toolbar_view)

        # Header Bar
        header = Adw.HeaderBar()
        toolbar_view.add_top_bar(header)

        # Menu Button (Optional, maybe for settings?)
        # For now, keep it simple.

        # Content Stack
        self._stack = Gtk.Stack()
        self._stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        toolbar_view.set_content(self._stack)

        # --- Loading Page ---
        loading_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        loading_box.set_halign(Gtk.Align.CENTER)
        loading_box.set_valign(Gtk.Align.CENTER)

        spinner = Gtk.Spinner()
        spinner.set_size_request(48, 48)
        spinner.start()
        loading_box.append(spinner)

        loading_label = Gtk.Label(label=_("Processing extracted text..."))
        loading_label.add_css_class("title-4")
        loading_box.append(loading_label)

        self._stack.add_named(loading_box, "loading")

        # --- Results Page ---
        results_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        # Toolbar/Action Bar at top of content? Or just use header buttons?
        # Let's put buttons at bottom like the dialog.

        # Text View
        text_scroll = Gtk.ScrolledWindow()
        text_scroll.set_vexpand(True)
        text_scroll.set_hexpand(True)

        self._text_view = Gtk.TextView()
        self._text_view.set_editable(True)  # Allow editing
        self._text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self._text_view.set_left_margin(16)
        self._text_view.set_right_margin(16)
        self._text_view.set_top_margin(16)
        self._text_view.set_bottom_margin(16)
        self._text_view.set_monospace(True)

        self._text_buffer = self._text_view.get_buffer()
        text_scroll.set_child(self._text_view)
        results_box.append(text_scroll)

        # Bottom Bar
        action_bar = Gtk.ActionBar()
        results_box.append(action_bar)

        # Copy Button
        self._copy_button = Gtk.Button(icon_name="edit-copy-symbolic", label=_("Copy"))
        self._copy_button.connect("clicked", self._on_copy_clicked)
        action_bar.pack_start(self._copy_button)

        # New Capture Button
        capture_btn = Gtk.Button(icon_name="camera-photo-symbolic", label=_("New Capture"))
        capture_btn.add_css_class("suggested-action")
        capture_btn.connect("clicked", self._on_new_capture_clicked)
        action_bar.pack_end(capture_btn)

        # Open Image Button
        open_btn = Gtk.Button(icon_name="document-open-symbolic", label=_("Open Image"))
        open_btn.connect("clicked", self._on_open_image_clicked)
        action_bar.pack_end(open_btn)

        self._stack.add_named(results_box, "results")

    def _start_processing(self, image_path: str) -> None:
        """Start OCR processing for a file."""
        self._stack.set_visible_child_name("loading")
        self._copy_button.set_sensitive(False)

        # Use existing service
        # Ensure we have current settings
        self._settings.load_settings()
        lang = self._settings.lang

        self._screen_capture_service.process_image_file(
            image_path, callback=self._on_processing_complete, lang=lang
        )

    def _on_processing_complete(self, text: str | None, error: str | None) -> None:
        """Handle processing completion."""
        if error:
            self._show_error(error)
            self._stack.set_visible_child_name("results")
            return

        if text:
            self._text = text
            self._text_buffer.set_text(text)
            self._copy_button.set_sensitive(True)
        else:
            self._text_buffer.set_text(_("No text extracted."))
            self._copy_button.set_sensitive(False)

        self._stack.set_visible_child_name("results")

    def _on_copy_clicked(self, btn: Gtk.Button) -> None:
        """Copy extracted text to clipboard."""
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
            # We use explicit bytes to ensure text/plain is handled correctly
            try:
                data = GLib.Bytes.new(text.encode("utf-8"))
                provider = Gdk.ContentProvider.new_for_bytes("text/plain", data)
                clipboard.set_content(provider)
                logger.info("Used fallback clipboard set_content with bytes")
            except Exception as e:
                logger.error(f"Clipboard fallback failed: {e}")
                # Last resort try the simple value provider
                clipboard.set(Gdk.ContentProvider.new_for_value(text))

        # Feedback
        orig_label = btn.get_label()
        btn.set_label(_("Copied!"))
        btn.set_sensitive(False)

        GLib.timeout_add(
            1500, lambda: (btn.set_label(orig_label), btn.set_sensitive(True), False)[2]
        )

    def _on_new_capture_clicked(self, *args) -> None:
        """Start a new screen capture."""
        # Minimize window to allow capturing correct area
        self.minimize()

        # Wait for minimize animation?
        # Using a small timeout to be safe
        GLib.timeout_add(200, self._trigger_capture)

    def _trigger_capture(self) -> bool:
        """Actually trigger the capture after minimize delay."""
        self._settings.load_settings()
        self._screen_capture_service.capture_screen_region(
            callback=self._on_capture_complete, lang=self._settings.lang
        )
        return False

    def _on_capture_complete(self, text: str | None, error: str | None) -> None:
        """Handle screen capture completion."""
        # Restore window
        self.present()
        self._on_processing_complete(text, error)

    def _on_open_image_clicked(self, *args) -> None:
        """Open file chooser to select an image."""
        dialog = Gtk.FileDialog()
        dialog.set_title(_("Open Image to OCR"))

        filters = Gdk.ContentFormats.new(
            ["image/png", "image/jpeg", "image/gif", "image/bmp", "image/webp", "image/tiff"]
        )
        # FileDialog filters are a bit different in GTK4, easier to stick to basic open for now
        # or construct Gtk.FileFilter

        filter_images = Gtk.FileFilter()
        filter_images.set_name(_("Images"))
        filter_images.add_mime_type("image/*")

        store = Gio.ListStore.new(Gtk.FileFilter)
        store.append(filter_images)
        dialog.set_filters(store)

        dialog.open(self, None, self._on_file_opened)

    def _on_file_opened(self, dialog, result) -> None:
        """Handle file selection result."""
        try:
            file = dialog.open_finish(result)
            if file:
                self._start_processing(file.get_path())
        except GLib.Error as e:
            # Cancelled or error
            if "dismissed" not in str(e):  # Check cancellation
                logger.error(f"Error opening file: {e}")

    def _show_error(self, message: str) -> None:
        """Show error dialog."""
        alert = Adw.AlertDialog()
        alert.set_heading(_("Error"))
        alert.set_body(message)
        alert.add_response("ok", _("OK"))
        alert.present(self)

    def _update_ui_state(self, empty: bool = False) -> None:
        """Update UI based on state."""
        if empty:
            self._text_buffer.set_text(_("Open an image or take a screenshot to extract text."))
            self._copy_button.set_sensitive(False)


# Import Gio strictly for type hinting if needed, but we used it inside method
from gi.repository import Gio
