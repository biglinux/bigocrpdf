"""
BigOcrPdf - Settings Page Module

This module handles the creation and management of the settings page UI.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
import os
from typing import TYPE_CHECKING

from gi.repository import Adw, Gdk, Gio, GLib, Gtk

if TYPE_CHECKING:
    from window import BigOcrPdfWindow

from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class SettingsPageManager:
    """Manages the settings page UI and interactions"""

    def __init__(self, window: "BigOcrPdfWindow"):
        """Initialize the settings page manager

        Args:
            window: Reference to the main application window
        """
        self.window = window

        # UI component references
        self.lang_dropdown = None
        self.dest_entry = None
        self.folder_combo = None
        self.folder_entry_box = None
        self.file_list_box = None
        self.placeholder = None

        # Preprocessing UI components
        self.deskew_switch = None
        self.perspective_switch = None
        self.orientation_switch = None
        self.scanner_switch = None
        self.replace_ocr_switch = None
        self._preprocessing_signal_connected = False

    def create_settings_page(self) -> Gtk.Widget:
        """Create the settings page (file queue only).

        The sidebar with OCR settings is created separately
        via create_sidebar_content() for the window's left pane.

        Returns:
            A widget containing the file queue UI
        """
        return self._create_file_queue_panel()

    def create_sidebar_content(self) -> Gtk.Widget:
        """Create the sidebar content with OCR settings.

        Returns:
            A widget containing OCR configuration options
        """
        return self._create_config_panel()

    def _create_config_panel(self) -> Gtk.Widget:
        """Create the configuration panel for the left sidebar

        Returns:
            A widget containing OCR settings and destination configuration
        """
        # Scrolled content - no frame wrapper to keep clean sidebar look
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled_window.set_vexpand(True)

        # Main settings container
        settings_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        settings_box.set_spacing(24)
        settings_box.set_margin_start(12)
        settings_box.set_margin_end(12)
        settings_box.set_margin_top(3)
        settings_box.set_margin_bottom(24)

        # OCR Settings Group (includes Image Export options)
        self._add_ocr_settings(settings_box)

        # Preprocessing Group
        self._add_preprocessing_options(settings_box)

        scrolled_window.set_child(settings_box)

        return scrolled_window

    def _add_ocr_settings(self, container: Gtk.Box) -> None:
        """Add OCR settings dropdowns to the container

        Args:
            container: Container to add the OCR settings to
        """
        from bigocrpdf.utils.tooltip_helper import get_tooltip_helper

        tooltip = get_tooltip_helper()

        # OCR settings group
        ocr_group = Adw.PreferencesGroup()

        # Language selection as ComboRow
        self.lang_combo = Adw.ComboRow(title=_("Text Language"))
        lang_icon = Gtk.Image.new_from_icon_name("preferences-desktop-locale-symbolic")
        self.lang_combo.add_prefix(lang_icon)

        # Help button for language details
        lang_help_btn = Gtk.Button(
            icon_name="help-about-symbolic",
            valign=Gtk.Align.CENTER,
            css_classes=["flat", "circular"],
        )
        lang_help_btn.connect("clicked", self._on_language_help_clicked)
        self.lang_combo.add_suffix(lang_help_btn)

        # Populate language dropdown
        languages = self.window.ocr_processor.get_available_ocr_languages()
        self._available_languages = languages  # Store for index lookups
        lang_model = Gtk.StringList()
        for _i, (_lang_code, lang_name) in enumerate(languages):
            lang_model.append(lang_name)
        self.lang_combo.set_model(lang_model)
        self.lang_combo.set_can_focus(False)

        # Use a flag to ensure we only connect the signal once
        self._lang_signal_connected = False

        def on_lang_map(_widget):
            self.lang_combo.set_can_focus(True)
            # Read CURRENT settings value on each map (not stale closure)
            current_lang = self.window.settings.lang
            for i, (code, _name) in enumerate(self._available_languages):
                if code == current_lang:
                    self.lang_combo.set_selected(i)
                    break
            # Only connect the change handler once
            if not self._lang_signal_connected:
                self._lang_signal_connected = True
                self.lang_combo.connect("notify::selected", self._on_language_changed)

        self.lang_combo.connect("map", on_lang_map)
        ocr_group.add(self.lang_combo)

        # Compatibility alias for lang_dropdown
        self.lang_dropdown = self.lang_combo

        tooltip.add_tooltip(
            self.lang_combo,
            _(
                "Choose the language of your document's text.\n"
                "The correct language helps recognize text more accurately."
            ),
        )

        # OCR Precision selector
        self.ocr_precision_combo = Adw.ComboRow(title=_("OCR Precision"))
        precision_icon = Gtk.Image.new_from_icon_name("speedometer-symbolic")
        self.ocr_precision_combo.add_prefix(precision_icon)

        precision_model = Gtk.StringList.new(
            [
                _("Low Precision"),
                _("Standard"),
                _("Precise"),
                _("Very Precise"),
            ]
        )
        self.ocr_precision_combo.set_model(precision_model)
        self.ocr_precision_combo.set_can_focus(False)
        ocr_group.add(self.ocr_precision_combo)

        tooltip.add_tooltip(
            self.ocr_precision_combo,
            _(
                "How carefully the program reads text from your documents.\n\n"
                "• Low Precision: Finds more text, good for blurry or faded pages.\n"
                "• Standard: Works well for most documents.\n"
                "• Precise: Fewer mistakes, may miss some faint text.\n"
                "• Very Precise: Only keeps text it is very sure about."
            ),
        )

        # Flags to avoid reconnecting signals on repeated map events
        self._precision_signal_connected = False
        self._format_signal_connected = False
        self._quality_signal_connected = False
        self._pdfa_signal_connected = False

        def on_precision_map(_widget):
            self._load_advanced_ocr_settings()

        self.ocr_precision_combo.connect("map", on_precision_map)

        # --- Image Export options (merged into this group) ---

        # Image format selector
        self.image_format_combo = Adw.ComboRow(title=_("Export Format"))
        format_icon = Gtk.Image.new_from_icon_name("image-x-generic-symbolic")
        self.image_format_combo.add_prefix(format_icon)
        format_model = Gtk.StringList.new([_("Original"), _("Custom Quality")])
        self.image_format_combo.set_model(format_model)
        self.image_format_combo.set_can_focus(False)
        ocr_group.add(self.image_format_combo)
        tooltip.add_tooltip(
            self.image_format_combo,
            _(
                "Original: Keeps images unchanged (best quality, larger files)\n"
                "Custom Quality: Adjusts image quality to reduce file size"
            ),
        )

        # Quality preset
        self.image_quality_combo = Adw.ComboRow(title=_("Quality"))
        quality_icon = Gtk.Image.new_from_icon_name("applications-graphics-symbolic")
        self.image_quality_combo.add_prefix(quality_icon)
        quality_model = Gtk.StringList.new(
            [
                _("Very Low (30%)"),
                _("Low (50%)"),
                _("Medium (70%)"),
                _("High (85%)"),
                _("Maximum (95%)"),
            ]
        )
        self.image_quality_combo.set_model(quality_model)
        self.image_quality_combo.set_can_focus(False)
        ocr_group.add(self.image_quality_combo)
        tooltip.add_tooltip(
            self.image_quality_combo,
            _(
                "Very Low: Smallest files, lower image quality\n"
                "Low: Small files, some quality loss\n"
                "Medium: Good balance between quality and file size\n"
                "High: Recommended for most documents\n"
                "Maximum: Best image quality, larger files"
            ),
        )

        # PDF/A toggle
        self.pdfa_switch_row = Adw.SwitchRow(title=_("Export as PDF/A"))
        pdfa_icon = Gtk.Image.new_from_icon_name("document-save-symbolic")
        self.pdfa_switch_row.add_prefix(pdfa_icon)
        self.pdfa_switch_row.set_can_focus(False)
        ocr_group.add(self.pdfa_switch_row)
        tooltip.add_tooltip(
            self.pdfa_switch_row,
            _(
                "Creates a PDF designed for long-term storage.\n"
                "Recommended when you need the file to open\n"
                "correctly on any device, now and in the future."
            ),
        )

        # Maximum output file size selector
        self.max_size_combo = Adw.ComboRow(title=_("Maximum Output Size"))
        max_size_icon = Gtk.Image.new_from_icon_name("drive-harddisk-symbolic")
        self.max_size_combo.add_prefix(max_size_icon)
        self._max_size_values = [0, 5, 10, 15, 20, 25, 50, 100]
        max_size_model = Gtk.StringList.new(
            [
                _("No limit"),
                "5 MB",
                "10 MB",
                "15 MB",
                "20 MB",
                "25 MB",
                "50 MB",
                "100 MB",
            ]
        )
        self.max_size_combo.set_model(max_size_model)
        self.max_size_combo.set_can_focus(False)
        ocr_group.add(self.max_size_combo)
        tooltip.add_tooltip(
            self.max_size_combo,
            _(
                "Sets a maximum size for the final file.\n"
                "If the result is too large, it is automatically split\n"
                "into smaller numbered files (e.g. document-01.pdf).\n\n"
                "Useful when sending by email or uploading online."
            ),
        )

        # Connect signal immediately (like other combos) to avoid map-timing issues
        self.max_size_combo.connect("notify::selected", self._on_max_size_changed)

        container.append(ocr_group)

        # Load image export settings after widget map
        def on_export_map(_widget):
            self._load_image_export_settings()
            self._load_max_size_setting()

        ocr_group.connect("map", on_export_map)

    def _create_file_queue_panel(self) -> Gtk.Widget:
        """Create the file queue panel for the right side

        Returns:
            A widget containing the file queue UI matching video-converter style
        """
        # Create main container
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        main_box.set_vexpand(True)
        main_box.add_css_class("view")

        # Queue scroll area
        queue_scroll = Gtk.ScrolledWindow()
        queue_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        queue_scroll.set_vexpand(True)

        # Create the list box for queue items
        self.file_list_box = Gtk.ListBox()
        self.file_list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        self.file_list_box.add_css_class("boxed-list")
        self.file_list_box.set_margin_start(6)
        self.file_list_box.set_margin_end(6)
        self.file_list_box.set_margin_top(3)
        self.file_list_box.set_margin_bottom(6)

        # Create placeholder for empty queue (video-converter style)
        self.placeholder = Adw.StatusPage()
        self.placeholder.set_icon_name("document-open-symbolic")
        self.placeholder.set_title(_("No PDF Files"))
        self.placeholder.set_description(_("Drag files here or use the Add Files button"))
        self.placeholder.set_vexpand(True)
        self.placeholder.set_hexpand(True)
        self.placeholder.set_margin_top(3)
        self.placeholder.set_margin_bottom(6)
        self.file_list_box.set_placeholder(self.placeholder)

        # Setup drag and drop
        self._setup_drag_and_drop()

        # Add files if there are any
        if self.window.settings.selected_files:
            self._populate_file_list()

        queue_scroll.set_child(self.file_list_box)
        main_box.append(queue_scroll)

        # Bottom options bar
        options_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        options_box.set_spacing(12)
        options_box.set_margin_start(12)
        options_box.set_margin_end(12)
        options_box.set_margin_top(12)
        options_box.set_margin_bottom(12)
        options_box.set_vexpand(False)

        # Folder selection combo
        folder_options_store = Gtk.StringList()
        folder_options_store.append(_("Save in the same folder as the original file"))
        folder_options_store.append(_("Custom folder"))

        self.folder_combo = Gtk.DropDown()
        self.folder_combo.set_model(folder_options_store)
        self.folder_combo.set_selected(0 if self.window.settings.save_in_same_folder else 1)
        self.folder_combo.set_valign(Gtk.Align.CENTER)
        options_box.append(self.folder_combo)

        # Folder entry box (shown when custom folder is selected)
        self.folder_entry_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.folder_entry_box.set_spacing(4)
        self.folder_entry_box.set_visible(not self.window.settings.save_in_same_folder)
        self.folder_entry_box.set_hexpand(True)

        # Output folder entry
        self.dest_entry = Gtk.Entry()
        self.dest_entry.set_hexpand(True)
        self.dest_entry.set_placeholder_text(_("Select folder"))
        self.dest_entry.set_text(self.window.settings.destination_folder or "")
        self.folder_entry_box.append(self.dest_entry)

        # Browse folder button
        folder_button = Gtk.Button()
        folder_button.set_icon_name("folder-symbolic")
        folder_button.connect("clicked", self.window.on_browse_clicked)
        folder_button.add_css_class("flat")
        folder_button.add_css_class("circular")
        folder_button.set_valign(Gtk.Align.CENTER)
        self.folder_entry_box.append(folder_button)

        options_box.append(self.folder_entry_box)

        # Flexible spacer to push button right
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        options_box.append(spacer)

        # Output options button
        options_button = Gtk.Button(label=_("Output options"))
        options_button.connect("clicked", lambda _: self._show_pdf_options_dialog())
        options_box.append(options_button)

        # Connect combo box signal
        self.folder_combo.connect("notify::selected", self._on_folder_type_changed)

        main_box.append(options_box)

        return main_box

    def _on_folder_type_changed(self, combo, _param) -> None:
        """Handle folder type combo change"""
        selected = combo.get_selected()
        use_custom_folder = selected == 1  # "Custom folder" option

        # Show/hide folder entry when selection changes
        self.folder_entry_box.set_visible(use_custom_folder)

        # Save setting
        self.window.settings.save_in_same_folder = not use_custom_folder

    def _on_language_changed(self, combo, _param) -> None:
        """Handle language selection change"""
        selected = combo.get_selected()
        languages = self.window.ocr_processor.get_available_ocr_languages()
        if selected < len(languages):
            lang_code, _ = languages[selected]
            self.window.settings.lang = lang_code
            logger.info(f"Language changed to: {lang_code}")
            self.window.settings._save_all_settings()

    def _on_language_help_clicked(self, _button: Gtk.Button) -> None:
        """Show a modern dialog with supported languages for each OCR model."""
        from bigocrpdf.services.rapidocr_service.discovery import ModelDiscovery

        available_languages = self.window.ocr_processor.get_available_ocr_languages()

        # Main container
        content_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=16,
            margin_top=12,
            margin_bottom=24,
            margin_start=16,
            margin_end=16,
        )

        # Description banner
        desc_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
        )
        desc_box.add_css_class("card")

        info_icon = Gtk.Image.new_from_icon_name("dialog-information-symbolic")
        info_icon.set_pixel_size(24)
        info_icon.set_margin_start(16)
        info_icon.set_margin_top(12)
        info_icon.set_margin_bottom(12)
        info_icon.set_valign(Gtk.Align.CENTER)
        info_icon.add_css_class("accent")
        desc_box.append(info_icon)

        desc_label = Gtk.Label(
            label=_(
                "Choose the model that matches the language of your documents for the best results."
            ),
            wrap=True,
            wrap_mode=2,
            xalign=0,
        )
        desc_label.set_margin_top(12)
        desc_label.set_margin_bottom(12)
        desc_label.set_margin_end(16)
        desc_box.append(desc_label)

        content_box.append(desc_box)

        # Build a section for each model
        for lang_code, lang_name in available_languages:
            details = ModelDiscovery.LANGUAGE_DETAILS.get(lang_code, "")
            if not details:
                continue

            # Parse and sort languages alphabetically
            languages = sorted([lang.strip() for lang in details.split(",") if lang.strip()])
            lang_count = len(languages)

            # Model section header
            section_group = Adw.PreferencesGroup(
                title=f"{lang_name}",
                description=_("{count} languages").format(count=lang_count),
            )

            # Create a grid-style layout with 3 columns
            num_cols = 3
            grid = Gtk.Grid()
            grid.set_row_spacing(2)
            grid.set_column_spacing(8)
            grid.set_column_homogeneous(True)
            grid.set_margin_start(12)
            grid.set_margin_end(12)
            grid.set_margin_top(8)
            grid.set_margin_bottom(8)

            for idx, language in enumerate(languages):
                row_idx = idx // num_cols
                col_idx = idx % num_cols

                label = Gtk.Label(label=language)
                label.set_xalign(0)
                label.set_margin_top(4)
                label.set_margin_bottom(4)
                label.set_margin_start(8)
                grid.attach(label, col_idx, row_idx, 1, 1)

            # Wrap grid in a listbox-style row
            grid_row = Gtk.ListBoxRow()
            grid_row.set_activatable(False)
            grid_row.set_child(grid)

            grid_listbox = Gtk.ListBox()
            grid_listbox.set_selection_mode(Gtk.SelectionMode.NONE)
            grid_listbox.add_css_class("boxed-list")
            grid_listbox.append(grid_row)

            section_group.add(grid_listbox)
            content_box.append(section_group)

        scrolled = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            vexpand=True,
        )
        scrolled.set_child(content_box)

        # Dialog with toolbar view
        header = Adw.HeaderBar()

        toolbar_view = Adw.ToolbarView()
        toolbar_view.add_top_bar(header)
        toolbar_view.set_content(scrolled)

        dialog = Adw.Dialog(
            title=_("Supported Languages"),
            content_width=550,
            content_height=600,
        )
        dialog.set_child(toolbar_view)
        dialog.present(self.window)

    def _setup_drag_and_drop(self) -> None:
        """Set up drag and drop functionality for the file list"""
        drop_target = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        drop_target.set_gtypes([Gdk.FileList])
        drop_target.connect("drop", self._on_drop)
        self.file_list_box.add_controller(drop_target)

    def _populate_file_list(self) -> None:
        """Populate the file list box with the selected files"""
        # Check if the file list box is ready
        if not self.file_list_box:
            return

        # Remove existing items
        while True:
            child = self.file_list_box.get_first_child()
            if child:
                self.file_list_box.remove(child)
            else:
                break

        # Re-set placeholder after clearing
        if hasattr(self, "placeholder") and self.placeholder:
            self.file_list_box.set_placeholder(self.placeholder)

        # Add each file as a row
        for idx, file_path in enumerate(self.window.settings.selected_files):
            self._create_file_row(file_path, idx)

    def _add_preprocessing_options(self, container: Gtk.Box) -> None:
        """Add image preprocessing options to the container.

        Args:
            container: Container to add the preprocessing options to
        """
        from bigocrpdf.utils.tooltip_helper import get_tooltip_helper

        tooltip = get_tooltip_helper()

        # Preprocessing group
        preprocessing_group = Adw.PreferencesGroup(title=_("Image Preprocessing"))

        # Auto Detect
        self.auto_detect_switch = Adw.SwitchRow(title=_("Auto Detect"))
        self.auto_detect_switch.set_can_focus(False)
        auto_detect_icon = Gtk.Image.new_from_icon_name("system-search-symbolic")
        self.auto_detect_switch.add_prefix(auto_detect_icon)
        preprocessing_group.add(self.auto_detect_switch)
        tooltip.add_tooltip(
            self.auto_detect_switch,
            _(
                "When ON: The program checks each page and only fixes\n"
                "problems it finds (recommended for most documents).\n"
                "When OFF: All corrections below are always applied."
            ),
        )

        # Deskew
        self.deskew_switch = Adw.SwitchRow(title=_("Deskew"))
        self.deskew_switch.set_can_focus(False)
        deskew_icon = Gtk.Image.new_from_icon_name("object-rotate-right-symbolic")
        self.deskew_switch.add_prefix(deskew_icon)
        preprocessing_group.add(self.deskew_switch)
        tooltip.add_tooltip(
            self.deskew_switch,
            _("Straightens pages that were scanned at a slight angle."),
        )

        # Perspective Correction
        self.perspective_switch = Adw.SwitchRow(title=_("Perspective Correction"))
        self.perspective_switch.set_can_focus(False)
        perspective_icon = Gtk.Image.new_from_icon_name("view-wrapped-symbolic")
        self.perspective_switch.add_prefix(perspective_icon)
        preprocessing_group.add(self.perspective_switch)
        tooltip.add_tooltip(
            self.perspective_switch,
            _(
                "Fixes pages that look tilted or warped, for example\nwhen a document was photographed instead of scanned."
            ),
        )

        # Auto-rotate
        self.orientation_switch = Adw.SwitchRow(title=_("Auto-rotate"))
        self.orientation_switch.set_can_focus(False)
        orientation_icon = Gtk.Image.new_from_icon_name("object-flip-horizontal-symbolic")
        self.orientation_switch.add_prefix(orientation_icon)
        preprocessing_group.add(self.orientation_switch)
        tooltip.add_tooltip(
            self.orientation_switch,
            _("Detects and fixes pages that are upside-down or sideways."),
        )

        # Scanner effect
        self.scanner_switch = Adw.SwitchRow(title=_("Scanner Effect"))
        self.scanner_switch.set_can_focus(False)
        scanner_icon = Gtk.Image.new_from_icon_name("scanner-symbolic")
        self.scanner_switch.add_prefix(scanner_icon)
        preprocessing_group.add(self.scanner_switch)
        tooltip.add_tooltip(
            self.scanner_switch,
            _("Makes text darker and the background lighter,\nlike a clean scanner copy."),
        )

        # Replace existing OCR
        self.replace_ocr_switch = Adw.SwitchRow(title=_("Replace Existing OCR"))
        self.replace_ocr_switch.set_can_focus(False)
        replace_ocr_icon = Gtk.Image.new_from_icon_name("edit-clear-all-symbolic")
        self.replace_ocr_switch.add_prefix(replace_ocr_icon)
        preprocessing_group.add(self.replace_ocr_switch)
        tooltip.add_tooltip(
            self.replace_ocr_switch,
            _(
                "When ON: Redoes the text recognition even if the PDF\n"
                "already has searchable text. Use this when the existing\n"
                "text is incorrect or of poor quality.\n"
                "When OFF: Keeps existing text and only processes pages without it."
            ),
        )

        container.append(preprocessing_group)

        # Load settings after widget map
        def on_map(_widget):
            self._load_preprocessing_settings()

        preprocessing_group.connect("map", on_map)

    def _load_preprocessing_settings(self) -> None:
        """Load preprocessing settings from OcrSettings."""
        settings = self.window.settings
        try:
            if hasattr(self, "auto_detect_switch"):
                self.auto_detect_switch.set_can_focus(True)
                self.auto_detect_switch.set_active(getattr(settings, "enable_auto_detect", True))

            if hasattr(self, "deskew_switch"):
                self.deskew_switch.set_can_focus(True)
                self.deskew_switch.set_active(settings.enable_deskew)

            if hasattr(self, "perspective_switch"):
                self.perspective_switch.set_can_focus(True)
                self.perspective_switch.set_active(
                    getattr(settings, "enable_perspective_correction", False)
                )

            if hasattr(self, "orientation_switch"):
                self.orientation_switch.set_can_focus(True)
                self.orientation_switch.set_active(settings.enable_orientation_detection)

            if hasattr(self, "scanner_switch"):
                self.scanner_switch.set_can_focus(True)
                self.scanner_switch.set_active(getattr(settings, "enable_scanner_effect", False))

            if hasattr(self, "replace_ocr_switch"):
                self.replace_ocr_switch.set_can_focus(True)
                self.replace_ocr_switch.set_active(getattr(settings, "replace_existing_ocr", False))

            # Connect signals only once to avoid duplicate handlers on re-map
            if not self._preprocessing_signal_connected:
                self._preprocessing_signal_connected = True
                if hasattr(self, "auto_detect_switch"):
                    self.auto_detect_switch.connect(
                        "notify::active", self._on_preprocessing_changed
                    )
                if hasattr(self, "deskew_switch"):
                    self.deskew_switch.connect("notify::active", self._on_preprocessing_changed)
                if hasattr(self, "perspective_switch"):
                    self.perspective_switch.connect(
                        "notify::active", self._on_preprocessing_changed
                    )
                if hasattr(self, "orientation_switch"):
                    self.orientation_switch.connect(
                        "notify::active", self._on_preprocessing_changed
                    )
                if hasattr(self, "scanner_switch"):
                    self.scanner_switch.connect("notify::active", self._on_preprocessing_changed)
                if hasattr(self, "replace_ocr_switch"):
                    self.replace_ocr_switch.connect(
                        "notify::active", self._on_preprocessing_changed
                    )

        except Exception as e:
            logger.error(f"Error loading preprocessing settings: {e}")

    # Precision presets: (text_score, box_thresh)
    # Wider range ensures visible differences in OCR filtering behavior.
    # - text_score: RapidOCR discards recognized text regions below this confidence
    # - box_thresh: detection model discards regions with detection score below this
    PRECISION_PRESETS: list[tuple[float, float]] = [
        (0.1, 0.3),  # Low Precision — catches faded/degraded text
        (0.3, 0.5),  # Standard — balanced (default)
        (0.5, 0.6),  # Precise — selective, reduces noise
        (0.7, 0.7),  # Very Precise — strict, only high-confidence text
    ]

    def _get_precision_index_from_settings(self, settings) -> int:
        """Get dropdown index based on current text_score and box_thresh values.

        Returns:
            int: Index for the dropdown (0=Low, 1=Standard, 2=Precise, 3=Very Precise)
        """
        text_score = getattr(settings, "text_score_threshold", 0.3)
        box_thresh = getattr(settings, "box_thresh", 0.5)

        for idx, (ts, bt) in enumerate(self.PRECISION_PRESETS):
            if abs(text_score - ts) < 0.05 and abs(box_thresh - bt) < 0.05:
                return idx

        # Default to Standard if no match
        return 1

    def _load_advanced_ocr_settings(self) -> None:
        """Load advanced OCR settings from OcrSettings."""
        settings = self.window.settings
        try:
            if hasattr(self, "ocr_precision_combo"):
                self.ocr_precision_combo.set_can_focus(True)
                # Map stored values to dropdown index
                precision_idx = self._get_precision_index_from_settings(settings)
                self.ocr_precision_combo.set_selected(precision_idx)
                # Connect signal only once to avoid duplicate handlers
                if not self._precision_signal_connected:
                    self.ocr_precision_combo.connect(
                        "notify::selected", self._on_ocr_precision_changed
                    )
                    self._precision_signal_connected = True

        except Exception as e:
            logger.error(f"Error loading advanced OCR settings: {e}")

    def _on_ocr_precision_changed(self, combo: Adw.ComboRow, _pspec) -> None:
        """Handle OCR precision preset changes."""
        selected = combo.get_selected()
        if selected < 0 or selected >= len(self.PRECISION_PRESETS):
            return
        text_score, box_thresh = self.PRECISION_PRESETS[selected]

        self.window.settings.text_score_threshold = text_score
        self.window.settings.box_thresh = box_thresh

        precision_names = ["low", "standard", "precise", "very_precise"]
        logger.info(
            f"OCR precision changed to: {precision_names[selected]} "
            f"(text_score={text_score}, box_thresh={box_thresh})"
        )
        self.window.settings._save_all_settings()

    def _load_image_export_settings(self) -> None:
        """Load image export settings from OcrSettings."""
        settings = self.window.settings
        try:
            if hasattr(self, "image_format_combo"):
                self.image_format_combo.set_can_focus(True)
                # Map format string to dropdown index
                # Any non-"original" value maps to index 1 (Custom Quality)
                fmt = getattr(settings, "image_export_format", "original").lower()
                idx = 0 if fmt == "original" else 1
                self.image_format_combo.set_selected(idx)
                if not self._format_signal_connected:
                    self.image_format_combo.connect(
                        "notify::selected", self._on_image_format_changed
                    )
                    self._format_signal_connected = True

            if hasattr(self, "image_quality_combo"):
                self.image_quality_combo.set_can_focus(True)
                quality = getattr(settings, "image_export_quality", 85)
                # Map quality value to dropdown index
                idx = self._get_quality_index_from_value(quality)
                self.image_quality_combo.set_selected(idx)
                if not self._quality_signal_connected:
                    self.image_quality_combo.connect(
                        "notify::selected", self._on_image_quality_changed
                    )
                    self._quality_signal_connected = True

            # Load PDF/A setting
            if hasattr(self, "pdfa_switch_row"):
                self.pdfa_switch_row.set_can_focus(True)
                pdfa_enabled = getattr(settings, "convert_to_pdfa", False)
                self.pdfa_switch_row.set_active(pdfa_enabled)
                if not self._pdfa_signal_connected:
                    self.pdfa_switch_row.connect("notify::active", self._on_pdfa_changed)
                    self._pdfa_signal_connected = True

            # Update quality dropdown sensitivity based on format selection
            self._update_quality_sensitivity()

        except Exception as e:
            logger.error(f"Error loading image export settings: {e}")

    def _get_quality_index_from_value(self, quality: int) -> int:
        """Map quality percentage to dropdown index."""
        if quality <= 35:
            return 0  # Very Low
        elif quality <= 55:
            return 1  # Low
        elif quality <= 75:
            return 2  # Medium
        elif quality <= 90:
            return 3  # High
        else:
            return 4  # Maximum

    def _on_image_format_changed(self, combo: Adw.ComboRow, _pspec) -> None:
        """Handle image export format changes."""
        formats = ["original", "jpeg"]
        selected = combo.get_selected()
        fmt = formats[selected] if selected < len(formats) else "original"

        self.window.settings.image_export_format = fmt
        logger.info(f"Image export format changed to: {fmt}")

        # Update quality dropdown sensitivity
        self._update_quality_sensitivity()
        self.window.settings._save_all_settings()

    def _on_pdfa_changed(self, switch_row: Adw.SwitchRow, _pspec) -> None:
        """Handle PDF/A toggle changes."""
        pdfa_enabled = switch_row.get_active()
        self.window.settings.convert_to_pdfa = pdfa_enabled
        logger.info(f"PDF/A export changed to: {pdfa_enabled}")
        self.window.settings._save_all_settings()

    def _load_max_size_setting(self) -> None:
        """Load maximum output size setting from OcrSettings."""
        try:
            if not hasattr(self, "max_size_combo"):
                return

            self.max_size_combo.set_can_focus(True)
            current_val = getattr(self.window.settings, "max_file_size_mb", 0)
            selected_idx = 0
            for idx, val in enumerate(self._max_size_values):
                if val == current_val:
                    selected_idx = idx
                    break
            self.max_size_combo.set_selected(selected_idx)
        except Exception as e:
            logger.error(f"Error loading max size setting: {e}")

    def _on_max_size_changed(self, combo: Adw.ComboRow, _pspec) -> None:
        """Handle maximum output size changes."""
        selected = combo.get_selected()
        if 0 <= selected < len(self._max_size_values):
            size_mb = self._max_size_values[selected]
            self.window.settings.max_file_size_mb = size_mb
            logger.info(f"Maximum output size changed to: {size_mb} MB (0=no limit)")
            self.window.settings._save_all_settings()

    def _on_image_quality_changed(self, combo: Adw.ComboRow, _pspec) -> None:
        """Handle image quality preset changes."""
        presets = [30, 50, 70, 85, 95]
        selected = combo.get_selected()
        quality = presets[selected] if selected < len(presets) else 85

        self.window.settings.image_export_quality = quality
        logger.info(f"Image export quality changed to: {quality}%")
        self.window.settings._save_all_settings()

    def _update_quality_sensitivity(self) -> None:
        """Update quality dropdown visibility.

        Format combo is always enabled (backend handles standalone mode
        automatically when a non-original format is selected).
        Quality is hidden when format is 'Original' since it has no effect.
        """
        if hasattr(self, "image_format_combo"):
            self.image_format_combo.set_sensitive(True)

        if hasattr(self, "image_quality_combo"):
            is_original = (
                hasattr(self, "image_format_combo") and self.image_format_combo.get_selected() == 0
            )
            self.image_quality_combo.set_visible(not is_original)

    def _on_preprocessing_changed(self, switch_row: Adw.SwitchRow, _pspec) -> None:
        """Handle preprocessing option changes."""
        settings = self.window.settings
        try:
            if switch_row == self.auto_detect_switch:
                settings.enable_auto_detect = switch_row.get_active()
            elif switch_row == self.deskew_switch:
                settings.enable_deskew = switch_row.get_active()
            elif switch_row == self.perspective_switch:
                settings.enable_perspective_correction = switch_row.get_active()
            elif switch_row == self.orientation_switch:
                settings.enable_orientation_detection = switch_row.get_active()
            elif switch_row == self.scanner_switch:
                settings.enable_scanner_effect = switch_row.get_active()
            elif hasattr(self, "replace_ocr_switch") and switch_row == self.replace_ocr_switch:
                settings.replace_existing_ocr = switch_row.get_active()
            settings._save_all_settings()
        except Exception as e:
            logger.error(f"Error saving preprocessing setting: {e}")

    def _create_file_row(self, file_path: str, idx: int) -> None:
        """Create a row for a single file in the list

        Args:
            file_path: Path to the file
            idx: Index of the file in the list
        """
        row = Adw.ActionRow()
        row.set_can_focus(False)
        row.set_activatable(False)

        # Set file name as title
        file_name = os.path.basename(file_path)
        row.set_title(file_name)

        # Add directory and size as subtitle
        try:
            dir_name = os.path.dirname(file_path)
            from bigocrpdf.constants import BYTES_PER_MB

            file_size = os.path.getsize(file_path) / BYTES_PER_MB
            subtitle = f"{dir_name}  •  {file_size:.1f} MB"
            row.set_subtitle(subtitle)
        except (OSError, FileNotFoundError):
            row.set_subtitle(os.path.dirname(file_path))

        # Add page count if available (suffix)
        self._add_page_count_to_row(row, file_path)

        # Action buttons as prefix (left side) - order: remove, open, edit
        # Note: prefix buttons appear in reverse order (last added = first)

        # Edit button (appears rightmost in prefix area)
        edit_button = Gtk.Button.new_from_icon_name("document-edit-symbolic")
        edit_button.set_tooltip_text(_("Edit pages of this file"))
        edit_button.add_css_class("flat")
        edit_button.set_valign(Gtk.Align.CENTER)
        edit_button.connect("clicked", lambda _b, fp=file_path: self._on_edit_file(fp))
        row.add_prefix(edit_button)

        # Open button (appears in middle of prefix area)
        open_button = Gtk.Button.new_from_icon_name("document-open-symbolic")
        open_button.set_tooltip_text(_("Open this file in your default application"))
        open_button.add_css_class("flat")
        open_button.set_valign(Gtk.Align.CENTER)
        open_button.connect("clicked", lambda _b, fp=file_path: self._on_open_file(fp))
        row.add_prefix(open_button)

        # Remove button (appears leftmost in prefix area)
        remove_button = Gtk.Button.new_from_icon_name("trash-symbolic")
        remove_button.set_tooltip_text(_("Remove this file from the list"))
        remove_button.add_css_class("flat")
        remove_button.set_valign(Gtk.Align.CENTER)
        remove_button.connect("clicked", lambda _b, i=idx: self._remove_single_file(i))
        row.add_prefix(remove_button)

        self.file_list_box.append(row)

        def enable_row_focus():
            if row.get_parent():
                row.set_can_focus(True)
            return False

        GLib.timeout_add(100, enable_row_focus)

    def _on_open_file(self, file_path: str) -> None:
        """Open file with the default application.

        Args:
            file_path: Path to the file to open
        """
        from bigocrpdf.utils.pdf_utils import open_file_with_default_app

        if not open_file_with_default_app(file_path):
            self.window.show_toast(_("Failed to open file"))

    def _on_edit_file(self, file_path: str) -> None:
        """Open the PDF editor for the file.

        Args:
            file_path: Path to the file to edit
        """
        try:
            from bigocrpdf.ui.pdf_editor import PDFEditorWindow

            def on_editor_save(document):
                """Handle editor save callback.

                Two modes:
                1. Same file (rotation/deletion only) — saves state as metadata
                   to file_modifications for the OCR processor to apply.
                2. Different file (merge from multiple sources) — replaces the
                   file path and cleans up old temp files.
                """
                # Check if file path changed (merged output)
                if document.path != file_path:
                    try:
                        if file_path in self.window.settings.selected_files:
                            idx = self.window.settings.selected_files.index(file_path)
                            self.window.settings.selected_files[idx] = document.path

                            # Resolve chained temp paths to the true original
                            # On re-edit, file_path may be a previous temp path
                            true_original = self.window.settings.original_file_paths.get(
                                file_path, file_path
                            )
                            self.window.settings.original_file_paths[document.path] = true_original

                            # Clean up stale mapping for previous temp path
                            if file_path in self.window.settings.original_file_paths:
                                del self.window.settings.original_file_paths[file_path]

                            # Clean up old modification entries — the merged file
                            # already incorporates all editor changes
                            if file_path in self.window.settings.file_modifications:
                                del self.window.settings.file_modifications[file_path]

                            # Clean up previous temp file from disk
                            if file_path != true_original and os.path.exists(file_path):
                                try:
                                    os.remove(file_path)
                                    logger.info(f"Removed previous temp file: {file_path}")
                                except OSError as rm_err:
                                    logger.warning(f"Could not remove temp file: {rm_err}")

                            logger.info(
                                f"Replaced original file with merged output: {document.path}"
                            )
                    except ValueError:
                        pass
                else:
                    # Same file — save modification state for OCR processor
                    state = document.to_dict()
                    self.window.settings.file_modifications[document.path] = state

                # Do NOT save page_modifications for merged files — they are
                # already baked in. Only save for same-file edits above.

                # Refresh the file queue display
                self._populate_file_list()
                self.window.update_file_info()

                if hasattr(self, "refresh_queue_status"):
                    self.refresh_queue_status()

                self.window.show_toast(_("Changes saved"))
                logger.info(f"Editor saved changes to: {document.path}")

            # Retrieve saved state if available (use original path to load state)
            # If we replaced the file, logic above would have cleared state for old path
            # and new path has "clean" state.
            initial_state = self.window.settings.file_modifications.get(file_path)

            editor = PDFEditorWindow(
                application=self.window.get_application(),
                pdf_path=file_path,
                on_save_callback=on_editor_save,
                parent_window=self.window,
                initial_state=initial_state,
            )
            editor.present()

            logger.info(f"Opened PDF editor for: {file_path}")
        except Exception as e:
            logger.error(f"Failed to open PDF editor: {e}")
            self.window.show_toast(_("Failed to open PDF editor"))

    def _add_page_count_to_row(self, row: Adw.ActionRow, file_path: str) -> None:
        """Add page count to a file row if available"""
        from bigocrpdf.utils.pdf_utils import get_pdf_page_count

        pages = get_pdf_page_count(file_path)
        if pages > 0:
            page_label = Gtk.Label()
            page_label.set_markup(f"<small>{pages} pg.</small>")
            row.add_suffix(page_label)

    def refresh_queue_status(self) -> None:
        """Update the queue status without rebuilding the entire settings page"""
        file_count = len(self.window.settings.selected_files)

        # Update folder entry visibility based on combo
        if self.folder_combo and self.folder_entry_box:
            use_custom = self.folder_combo.get_selected() == 1
            self.folder_entry_box.set_visible(use_custom)

        # Update header bar queue size
        if hasattr(self.window, "custom_header_bar") and self.window.custom_header_bar:
            self.window.custom_header_bar.update_queue_size(file_count)

        # Update the file list
        if self.file_list_box:
            self._populate_file_list()

    def _on_drop(self, drop_target: Gtk.DropTarget, value, _x: float, _y: float) -> bool:
        """Handle file drop events for both single and multiple files

        Args:
            drop_target: The drop target widget
            value: The dropped file(s) - could be Gio.File or Gdk.FileList
            _x: X coordinate of the drop
            _y: Y coordinate of the drop

        Returns:
            True if the drop was handled successfully, False otherwise
        """
        from bigocrpdf.utils.pdf_utils import images_to_pdf, is_image_file

        try:
            file_paths = self._extract_file_paths_from_drop(value)
            if not file_paths:
                return False

            # Filter for supported files (PDF and images)
            valid_file_paths = self._filter_supported_files(file_paths)
            if not valid_file_paths:
                logger.warning("No valid files in drop data")
                return False

            logger.info(f"{len(valid_file_paths)} files dropped")

            # Separate images from PDFs
            image_files = [p for p in valid_file_paths if is_image_file(p)]
            pdf_files = [p for p in valid_file_paths if not is_image_file(p)]

            # Add PDFs directly
            if pdf_files:
                self.window.settings.add_files(pdf_files)

            if len(image_files) > 1:
                # Multiple images: show merge dialog
                self._show_drop_image_merge_dialog(image_files)
            elif len(image_files) == 1:
                # Single image: convert to PDF and add
                try:
                    pdf_path = images_to_pdf(image_files)
                    self.window.settings.original_file_paths[pdf_path] = image_files[0]
                    self.window.settings.add_files([pdf_path])
                except Exception as e:
                    logger.error(f"Failed to convert dropped image to PDF: {e}")

            # Update UI
            self._populate_file_list()
            file_count = len(self.window.settings.selected_files)
            if hasattr(self.window, "custom_header_bar") and self.window.custom_header_bar:
                self.window.custom_header_bar.update_queue_size(file_count)

            return True
        except Exception as e:
            logger.error(f"Error handling dropped file(s): {e}")
            return False

    def _show_drop_image_merge_dialog(self, image_files: list[str]) -> None:
        """Show merge dialog for dropped images.

        Args:
            image_files: List of image file paths
        """
        from bigocrpdf.utils.pdf_utils import images_to_pdf

        dialog = Adw.AlertDialog()
        dialog.set_heading(_("Multiple Images Dropped"))
        dialog.set_body(
            _("You dropped {} images. How would you like to add them?").format(len(image_files))
        )

        dialog.add_response("separate", _("Separate PDFs"))
        dialog.add_response("merge", _("Merge into One PDF"))
        dialog.set_response_appearance("merge", Adw.ResponseAppearance.SUGGESTED)
        dialog.set_default_response("merge")

        def on_response(_dialog: Adw.AlertDialog, response: str) -> None:
            if response == "merge":
                try:
                    pdf_path = images_to_pdf(image_files)
                    self.window.settings.original_file_paths[pdf_path] = image_files[0]
                    self.window.settings.add_files([pdf_path])
                    self.window.show_toast(
                        _("Merged {} images into one PDF").format(len(image_files))
                    )
                except Exception as e:
                    logger.error(f"Failed to merge dropped images: {e}")
                    self.window.show_toast(_("Error merging images"))
            elif response == "separate":
                for img_path in image_files:
                    try:
                        pdf_path = images_to_pdf([img_path])
                        self.window.settings.original_file_paths[pdf_path] = img_path
                        self.window.settings.add_files([pdf_path])
                    except Exception as e:
                        logger.error(f"Failed to convert dropped image to PDF: {e}")

            # Update UI after dialog response
            self._populate_file_list()
            file_count = len(self.window.settings.selected_files)
            if hasattr(self.window, "custom_header_bar") and self.window.custom_header_bar:
                self.window.custom_header_bar.update_queue_size(file_count)

        dialog.connect("response", on_response)
        dialog.present(self.window)

    def _extract_file_paths_from_drop(self, value) -> list[str]:
        """Extract file paths from drop value

        Args:
            value: The dropped value from GTK

        Returns:
            List of file paths
        """
        file_paths = []

        # Handle different types of drop values
        if isinstance(value, Gio.File):
            # Single file case
            file_path = value.get_path()
            if file_path:
                file_paths.append(file_path)
        elif isinstance(value, list) or hasattr(value, "__iter__"):
            # Multiple files case
            for file in value:
                if isinstance(file, Gio.File):
                    file_path = file.get_path()
                    if file_path:
                        file_paths.append(file_path)
        else:
            logger.warning(f"Unsupported drop value type: {type(value)}")

        return file_paths

    def _filter_supported_files(self, file_paths: list[str]) -> list[str]:
        """Filter file paths to only include valid PDF and image files.

        Args:
            file_paths: List of file paths to filter

        Returns:
            List of valid file paths (PDFs and supported images)
        """
        supported_extensions = (
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".tiff",
            ".tif",
            ".bmp",
            ".webp",
            ".avif",
        )
        valid_paths = []
        for file_path in file_paths:
            if not file_path.lower().endswith(supported_extensions):
                logger.warning(f"Ignoring unsupported file: {file_path}")
                continue

            if not os.path.exists(file_path):
                logger.warning(f"Ignoring nonexistent file: {file_path}")
                continue

            valid_paths.append(file_path)

        return valid_paths

    def _remove_single_file(self, idx: int) -> None:
        """Remove a single file from the list

        Args:
            idx: Index of the file to remove
        """
        # Check if the index is valid
        if idx < 0 or idx >= len(self.window.settings.selected_files):
            return

        # Remove the file from the settings
        file_path = self.window.settings.selected_files.pop(idx)
        logger.info(f"Removed file: {file_path}")

        # Refresh the list
        self._populate_file_list()

        # Update the status information with the new file and page counts
        self.refresh_queue_status()

    def _remove_all_files(self) -> None:
        """Remove all files from the queue"""
        # Check if there are any files
        if not self.window.settings.selected_files:
            return

        # Log the action
        logger.info(f"Removing all {len(self.window.settings.selected_files)} files from queue")

        # Clear all files
        self.window.settings.selected_files.clear()

        # Refresh the list
        self._populate_file_list()

        # Update the status information with the new file and page counts
        self.refresh_queue_status()

    def _show_pdf_options_dialog(self) -> None:
        """Show PDF options dialog - placeholder for now"""
        # This will be implemented when we create the dialogs module
        # For now, import and call the method from ui_manager
        if hasattr(self.window, "ui") and hasattr(self.window.ui, "show_pdf_options_dialog"):
            self.window.ui.show_pdf_options_dialog(lambda _: None)
        else:
            logger.warning("PDF options dialog not yet implemented")
