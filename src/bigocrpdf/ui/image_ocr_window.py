"""
BigOcrPdf - Image OCR Window

This module provides a standalone window for Image OCR functionality.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, GLib, Gtk

from bigocrpdf.services.processor import OcrProcessor
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
        self._ocr_processor = OcrProcessor(self._settings)

        # UI references
        self._lang_dropdown: Gtk.DropDown | None = None
        self._psm_dropdown: Gtk.DropDown | None = None
        self._oem_dropdown: Gtk.DropDown | None = None
        self._redo_button: Gtk.Button | None = None
        self._current_image_path: str | None = None  # Track current image for redo

        # Load OCR settings from config (with defaults and system language detection)
        config = get_config_manager()

        # Language: Use config, fall back to settings (which auto-detects), then to 'eng'
        config_lang = config.get("ocr.language")
        if config_lang:
            self._settings.lang = config_lang
        # else: settings.lang is already set by OcrSettings.__init__ with detection

        self._current_psm: int = config.get("ocr.psm", 3)  # Default: Auto
        self._current_oem: int = config.get("ocr.oem", 3)  # Default: Auto

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

        # Language selector (left side)
        lang_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        lang_icon = Gtk.Image.new_from_icon_name("preferences-desktop-locale-symbolic")
        lang_box.append(lang_icon)
        self._lang_dropdown = self._create_language_dropdown()
        lang_box.append(self._lang_dropdown)
        action_bar.pack_start(lang_box)

        # PSM dropdown
        psm_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        psm_label = Gtk.Label(label="PSM:")
        psm_label.add_css_class("dim-label")
        psm_box.append(psm_label)
        self._psm_dropdown = self._create_psm_dropdown()
        psm_box.append(self._psm_dropdown)
        action_bar.pack_start(psm_box)

        # OEM dropdown
        oem_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        oem_label = Gtk.Label(label="OEM:")
        oem_label.add_css_class("dim-label")
        oem_box.append(oem_label)
        self._oem_dropdown = self._create_oem_dropdown()
        oem_box.append(self._oem_dropdown)
        action_bar.pack_start(oem_box)

        # Apply Settings Button (was Redo OCR - renamed to be more intuitive)
        self._redo_button = Gtk.Button(icon_name="emblem-synchronizing-symbolic", label=_("Apply"))
        self._redo_button.set_tooltip_text(_("Apply current settings and reprocess the image"))
        self._redo_button.connect("clicked", self._on_redo_clicked)
        self._redo_button.set_sensitive(False)  # Disabled until image is loaded
        action_bar.pack_start(self._redo_button)

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

    def _create_language_dropdown(self) -> Gtk.DropDown:
        """Create the language dropdown with available OCR languages.

        Returns:
            A Gtk.DropDown widget configured with available languages.
        """
        dropdown = Gtk.DropDown()
        dropdown.set_can_focus(False)  # Prevent focus during construction
        string_list = Gtk.StringList()

        # Get available languages from tesseract
        languages = self._ocr_processor.get_available_ocr_languages()
        self._lang_codes: list[str] = []  # Store codes for lookup
        default_index = 0

        for i, (lang_code, lang_name) in enumerate(languages):
            string_list.append(lang_name)
            self._lang_codes.append(lang_code)
            # Set default based on current setting
            if lang_code == self._settings.lang:
                default_index = i

        dropdown.set_model(string_list)
        dropdown.set_tooltip_text(_("Select OCR language"))

        # Defer set_selected and re-enable focus after widget is mapped
        def on_map(_widget):
            dropdown.set_can_focus(True)
            dropdown.set_selected(default_index)
            # Connect signal only after initial selection to avoid triggering save
            dropdown.connect("notify::selected", self._on_lang_changed)

        dropdown.connect("map", on_map)
        return dropdown

    def _on_lang_changed(self, dropdown: Gtk.DropDown, _pspec) -> None:
        """Handle language selection change."""
        selected = dropdown.get_selected()
        if 0 <= selected < len(self._lang_codes):
            new_lang = self._lang_codes[selected]
            self._settings.lang = new_lang
            # Save to persistent config
            config = get_config_manager()
            config.set("ocr.language", new_lang, save_immediately=True)
            logger.info(f"OCR language changed to: {new_lang}")

    def _create_psm_dropdown(self) -> Gtk.DropDown:
        """Create the PSM (Page Segmentation Mode) dropdown."""
        dropdown = Gtk.DropDown()
        dropdown.set_can_focus(False)
        string_list = Gtk.StringList()

        # PSM options - Auto first, then ordered by use case
        self._psm_options = [
            (3, _("Auto")),  # Default - first
            (6, _("Block")),
            (7, _("Line")),
            (8, _("Word")),
            (11, _("Sparse")),
            (4, _("Column")),
            (10, _("Char")),
            (1, _("Auto + OSD")),
            (0, _("OSD only")),
            (13, _("Raw line")),
        ]

        # Find index for current setting
        self._default_psm_index = 0  # Default: Auto (index 0)
        for i, (val, name) in enumerate(self._psm_options):
            if val == self._current_psm:
                self._default_psm_index = i
                break

        for psm_value, psm_name in self._psm_options:
            string_list.append(psm_name)

        dropdown.set_model(string_list)
        dropdown.set_tooltip_text(_("Page Segmentation Mode"))

        def on_map(_widget):
            dropdown.set_can_focus(True)
            dropdown.set_selected(self._default_psm_index)
            dropdown.connect("notify::selected", self._on_psm_changed)

        dropdown.connect("map", on_map)
        return dropdown

    def _on_psm_changed(self, dropdown: Gtk.DropDown, _pspec) -> None:
        """Handle PSM selection change."""
        selected = dropdown.get_selected()
        if 0 <= selected < len(self._psm_options):
            self._current_psm = self._psm_options[selected][0]
            # Save to config
            config = get_config_manager()
            config.set("ocr.psm", self._current_psm, save_immediately=True)
            logger.info(f"PSM changed to: {self._current_psm}")

    def _create_oem_dropdown(self) -> Gtk.DropDown:
        """Create the OEM (OCR Engine Mode) dropdown."""
        dropdown = Gtk.DropDown()
        dropdown.set_can_focus(False)
        string_list = Gtk.StringList()

        # OEM options - Auto first
        self._oem_options = [
            (3, _("Auto")),  # Default - first
            (1, _("LSTM")),
            (0, _("Legacy")),
            (2, _("Both")),
        ]

        # Find index for current setting
        self._default_oem_index = 0  # Default: Auto (index 0)
        for i, (val, name) in enumerate(self._oem_options):
            if val == self._current_oem:
                self._default_oem_index = i
                break

        for oem_value, oem_name in self._oem_options:
            string_list.append(oem_name)

        dropdown.set_model(string_list)
        dropdown.set_tooltip_text(_("OCR Engine Mode"))

        def on_map(_widget):
            dropdown.set_can_focus(True)
            dropdown.set_selected(self._default_oem_index)
            dropdown.connect("notify::selected", self._on_oem_changed)

        dropdown.connect("map", on_map)
        return dropdown

    def _on_oem_changed(self, dropdown: Gtk.DropDown, _pspec) -> None:
        """Handle OEM selection change."""
        selected = dropdown.get_selected()
        if 0 <= selected < len(self._oem_options):
            self._current_oem = self._oem_options[selected][0]
            # Save to config
            config = get_config_manager()
            config.set("ocr.oem", self._current_oem, save_immediately=True)
            logger.info(f"OEM changed to: {self._current_oem}")

    def _on_redo_clicked(self, *args) -> None:
        """Redo OCR on the current image with current settings."""
        if self._current_image_path:
            logger.info(f"Redoing OCR with PSM={self._current_psm}, OEM={self._current_oem}")
            self._start_processing(self._current_image_path)

    def _start_processing(self, image_path: str) -> None:
        """Start OCR processing for a file."""
        self._stack.set_visible_child_name("loading")
        self._copy_button.set_sensitive(False)
        if self._redo_button:
            self._redo_button.set_sensitive(False)

        # Store current image path for redo
        self._current_image_path = image_path

        # Use the currently selected settings
        lang = self._settings.lang
        psm = self._current_psm
        oem = self._current_oem

        self._screen_capture_service.process_image_file(
            image_path,
            callback=self._on_processing_complete,
            lang=lang,
            psm=psm,
            oem=oem,
        )

    def _on_processing_complete(self, text: str | None, error: str | None) -> None:
        """Handle processing completion."""
        if error:
            self._show_error(error)
            self._stack.set_visible_child_name("results")
            # Enable redo if we have an image
            if self._redo_button and self._current_image_path:
                self._redo_button.set_sensitive(True)
            return

        if text:
            self._text = text
            self._text_buffer.set_text(text)
            self._copy_button.set_sensitive(True)
        else:
            self._text_buffer.set_text(_("No text extracted."))
            self._copy_button.set_sensitive(False)

        # Enable redo button if we have an image
        if self._redo_button and self._current_image_path:
            self._redo_button.set_sensitive(True)

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
            callback=self._on_capture_complete,
            lang=self._settings.lang,
            psm=self._current_psm,
            oem=self._current_oem,
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
