"""Sidebar UI creation and settings callbacks for SettingsPageManager."""
# Host attributes are supplied by SettingsPageManager's explicit mixin composition.
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gtk

from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class SettingsSidebarMixin:
    """Mixin providing sidebar config panel creation and settings callbacks."""

    # Precision presets: (text_score, box_thresh)
    PRECISION_PRESETS: list[tuple[float, float]] = [
        (0.1, 0.3),  # Low Precision
        (0.3, 0.5),  # Standard
        (0.5, 0.6),  # Precise
        (0.7, 0.7),  # Very Precise
    ]

    def _create_config_panel(self) -> Gtk.Widget:
        """Create the configuration panel for the left sidebar."""
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled_window.set_vexpand(True)

        settings_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        settings_box.set_spacing(24)
        settings_box.set_margin_start(12)
        settings_box.set_margin_end(12)
        settings_box.set_margin_top(3)
        settings_box.set_margin_bottom(24)

        group = Adw.PreferencesGroup()

        self._proc_row = self._dialog_row(
            _("Image Corrections"),
            "applications-graphics-symbolic",
            self._show_corrections_dialog,
        )
        self._create_preprocessing_widgets()
        group.add(self._proc_row)

        self._out_row = self._dialog_row(
            _("Output Settings"),
            "document-save-symbolic",
            self._show_output_dialog,
        )
        self._create_output_widgets()
        group.add(self._out_row)

        self._adv_row = self._dialog_row(
            _("Advanced"),
            "preferences-system-symbolic",
            self._show_advanced_dialog,
        )
        self._create_advanced_widgets()
        group.add(self._adv_row)

        settings_box.append(group)
        group.connect("map", lambda _w: self._load_all_sidebar_settings())
        scrolled_window.set_child(settings_box)
        return scrolled_window

    @staticmethod
    def _dialog_row(title: str, icon_name: str, callback: Callable[[], None]) -> Adw.ActionRow:
        row = Adw.ActionRow(title=title)
        row.set_use_markup(False)
        row.add_prefix(Gtk.Image.new_from_icon_name(icon_name))

        button = Gtk.Button.new_from_icon_name("go-next-symbolic")
        button.add_css_class("flat")
        button.set_tooltip_text(title)
        set_a11y_label(button, title)
        button.connect("clicked", lambda _button: callback())
        row.add_suffix(button)
        row.set_activatable_widget(button)
        return row

    def _create_preprocessing_widgets(self) -> None:
        """Create image preprocessing switch widgets (not added to any parent)."""
        self.deskew_switch = Adw.SwitchRow(title=_("Deskew"))
        self.dewarp_switch = Adw.SwitchRow(title=_("Dewarp"))
        self.perspective_switch = Adw.SwitchRow(title=_("Perspective Correction"))
        self.orientation_switch = Adw.SwitchRow(title=_("Auto-rotate"))
        self.scanner_switch = Adw.SwitchRow(title=_("Scanner Effect"))
        self.enhance_embedded_switch = Adw.SwitchRow(title=_("Enhance Embedded Images"))

        self._correction_switches: dict[str, Adw.SwitchRow] = {
            "deskew": self.deskew_switch,
            "dewarp": self.dewarp_switch,
            "perspective": self.perspective_switch,
            "orientation": self.orientation_switch,
            "scanner": self.scanner_switch,
            "enhance_embedded": self.enhance_embedded_switch,
        }

        # Update subtitle when any switch changes
        for sw in self._correction_switches.values():
            sw.connect("notify::active", lambda *_: self._update_corrections_subtitle())

    def _update_corrections_subtitle(self) -> None:
        """Update the Image Corrections row subtitle with active count."""
        active = sum(1 for s in self._correction_switches.values() if s.get_active())
        total = len(self._correction_switches)
        self._proc_row.set_subtitle(
            _("{active} of {total} enabled").format(active=active, total=total)
        )

    def _show_corrections_dialog(self) -> None:
        """Open the image corrections configuration dialog."""
        from bigocrpdf.ui.corrections_dialog import show_image_corrections_dialog

        show_image_corrections_dialog(self.window, self._correction_switches)

    def _show_output_dialog(self) -> None:
        """Open the output settings configuration dialog."""
        from bigocrpdf.ui.output_dialog import show_output_settings_dialog

        show_output_settings_dialog(self.window, self._output_widgets)

    def _update_output_subtitle(self) -> None:
        """Update the Output Settings row subtitle with summary."""
        parts = []
        idx = self.image_quality_combo.get_selected()
        model = self.image_quality_combo.get_model()
        if model and idx < model.get_n_items():
            parts.append(model.get_string(idx))
        active = sum(
            1
            for key in ("pdfa",)
            if self._output_widgets.get(key) and self._output_widgets[key].get_active()
        )
        if active:
            parts.append(_("{n} options enabled").format(n=active))
        self._out_row.set_subtitle(", ".join(parts))

    def _create_output_widgets(self) -> None:
        """Create output settings widgets (not added to any parent)."""
        self._quality_signal_connected = False
        self._pdfa_signal_connected = False

        self.image_quality_combo = Adw.ComboRow(title=_("Image Quality"))
        set_a11y_label(self.image_quality_combo, _("Image Quality"))
        quality_model = Gtk.StringList.new(
            [
                _("Keep Original"),
                _("Very Low (30%)"),
                _("Low (50%)"),
                _("Medium (70%)"),
                _("High (85%)"),
                _("Maximum (95%)"),
                _("Black & White (JBIG2)"),
            ]
        )
        self.image_quality_combo.set_model(quality_model)

        self.pdfa_switch_row = Adw.SwitchRow(title=_("Export as PDF/A"))

        self._page_layout_signal_connected = False
        self._page_layout_values = ["default", "single", "continuous", "two_page"]
        self.page_layout_combo = Adw.ComboRow(title=_("Page Layout"))
        set_a11y_label(self.page_layout_combo, _("Page Layout"))
        page_layout_model = Gtk.StringList.new(
            [
                _("Default (viewer decides)"),
                _("Single page"),
                _("Continuous scroll"),
                _("Two pages"),
            ]
        )
        self.page_layout_combo.set_model(page_layout_model)

        self.max_size_combo = Adw.ComboRow(title=_("Maximum Output Size"))
        set_a11y_label(self.max_size_combo, _("Maximum Output Size"))
        self._max_size_values = [0, 5, 10, 15, 20, 25, 50, 100]
        max_size_model = Gtk.StringList.new(
            [
                _("No limit"),
                _("5 MB"),
                _("10 MB"),
                _("15 MB"),
                _("20 MB"),
                _("25 MB"),
                _("50 MB"),
                _("100 MB"),
            ]
        )
        self.max_size_combo.set_model(max_size_model)
        self.max_size_combo.connect("notify::selected", self._on_max_size_changed)

        self._output_widgets: dict[str, Gtk.Widget] = {
            "image_quality": self.image_quality_combo,
            "pdfa": self.pdfa_switch_row,
            "page_layout": self.page_layout_combo,
            "max_size": self.max_size_combo,
        }

        # Update subtitle when any switch/combo changes
        for w in self._output_widgets.values():
            if isinstance(w, Adw.SwitchRow):
                w.connect("notify::active", lambda *_: self._update_output_subtitle())
            elif isinstance(w, Adw.ComboRow):
                w.connect("notify::selected", lambda *_: self._update_output_subtitle())

    def _show_advanced_dialog(self) -> None:
        """Open the advanced settings configuration dialog."""
        from bigocrpdf.ui.advanced_dialog import show_advanced_settings_dialog

        show_advanced_settings_dialog(self.window, self._advanced_widgets)

    def _update_advanced_subtitle(self) -> None:
        """Update the Advanced row subtitle with summary."""
        parts = []
        idx = self.ocr_precision_combo.get_selected()
        model = self.ocr_precision_combo.get_model()
        if model and idx < model.get_n_items():
            parts.append(model.get_string(idx))
        active = sum(
            1
            for k in ("replace_ocr", "full_resolution")
            if self._advanced_widgets.get(k)
            and isinstance(self._advanced_widgets[k], Adw.SwitchRow)
            and self._advanced_widgets[k].get_active()
        )
        if active:
            parts.append(_("{n} options enabled").format(n=active))
        self._adv_row.set_subtitle(", ".join(parts) if parts else "")

    def _create_advanced_widgets(self) -> None:
        """Create advanced settings widgets (not added to any parent)."""
        self._precision_signal_connected = False
        self._replace_ocr_signal_connected = False

        self.ocr_precision_combo = Adw.ComboRow(title=_("OCR Precision"))
        set_a11y_label(self.ocr_precision_combo, _("OCR Precision"))
        precision_model = Gtk.StringList.new(
            [
                _("Low Precision"),
                _("Standard"),
                _("Precise"),
                _("Very Precise"),
            ]
        )
        self.ocr_precision_combo.set_model(precision_model)

        self.replace_ocr_switch = Adw.SwitchRow(title=_("Replace Existing OCR"))

        self._full_res_signal_connected = False
        self.full_resolution_switch = Adw.SwitchRow(title=_("Full Resolution Detection"))

        self._advanced_widgets: dict[str, Gtk.Widget] = {
            "ocr_precision": self.ocr_precision_combo,
            "replace_ocr": self.replace_ocr_switch,
            "full_resolution": self.full_resolution_switch,
        }

        # Update subtitle when any widget changes
        for w in self._advanced_widgets.values():
            if isinstance(w, Adw.SwitchRow):
                w.connect("notify::active", lambda *_: self._update_advanced_subtitle())
            elif isinstance(w, Adw.ComboRow):
                w.connect("notify::selected", lambda *_: self._update_advanced_subtitle())

    # ── Settings Load & Callbacks ──

    def _load_all_sidebar_settings(self) -> None:
        """Load all sidebar settings into UI widgets on map."""
        self._load_preprocessing_settings()
        self._load_advanced_ocr_settings()
        self._load_image_export_settings()
        self._load_max_size_setting()
        self._load_replace_ocr_setting()
        self._load_full_resolution_setting()

    def _on_folder_type_changed(self, combo, _param) -> None:
        """Handle folder type combo change."""
        selected = combo.get_selected()
        use_custom_folder = selected == 1
        self.folder_entry_box.set_visible(use_custom_folder)
        self.window.settings.save_in_same_folder = not use_custom_folder
        self.window.settings._save_all_settings()

    def _load_preprocessing_settings(self) -> None:
        """Load preprocessing settings from OcrSettings."""
        settings = self.window.settings
        try:
            self._load_preprocessing_switch_values(settings)
            self._connect_preprocessing_switches()
        except Exception as e:
            logger.error(f"Error loading preprocessing settings: {e}")

        self._update_corrections_subtitle()

    def _load_preprocessing_switch_values(self, settings) -> None:
        switch_settings = (
            ("deskew_switch", "enable_deskew"),
            ("dewarp_switch", "enable_baseline_dewarp"),
            ("perspective_switch", "enable_perspective_correction"),
            ("orientation_switch", "enable_orientation_detection"),
            ("scanner_switch", "enable_scanner_effect"),
            ("enhance_embedded_switch", "enhance_embedded_images"),
        )
        for widget_name, setting_name in switch_settings:
            if hasattr(self, widget_name):
                switch = getattr(self, widget_name)
                switch.set_can_focus(True)
                switch.set_active(getattr(settings, setting_name))

    def _connect_preprocessing_switches(self) -> None:
        if self._preprocessing_signal_connected:
            return

        self._preprocessing_signal_connected = True
        for widget_name in (
            "deskew_switch",
            "dewarp_switch",
            "perspective_switch",
            "orientation_switch",
            "scanner_switch",
            "enhance_embedded_switch",
        ):
            getattr(self, widget_name).connect(
                "notify::active",
                self._on_preprocessing_changed,
            )

    def _get_precision_index_from_settings(self, settings) -> int:
        """Get dropdown index based on current text_score and box_thresh values."""
        text_score = settings.text_score_threshold
        box_thresh = settings.box_thresh
        for idx, (ts, bt) in enumerate(self.PRECISION_PRESETS):
            if abs(text_score - ts) < 0.05 and abs(box_thresh - bt) < 0.05:
                return idx
        return 1

    def _load_advanced_ocr_settings(self) -> None:
        """Load advanced OCR settings from OcrSettings."""
        settings = self.window.settings
        try:
            self.ocr_precision_combo.set_can_focus(True)
            precision_idx = self._get_precision_index_from_settings(settings)
            self.ocr_precision_combo.set_selected(precision_idx)
            if not self._precision_signal_connected:
                self.ocr_precision_combo.connect("notify::selected", self._on_ocr_precision_changed)
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
            self._load_image_quality_combo(settings)
            self._load_pdfa_switch(settings)
            self._load_page_layout_combo(settings)
        except Exception as e:
            logger.error(f"Error loading image export settings: {e}")
        self._update_output_subtitle()

    def _load_image_quality_combo(self, settings) -> None:
        self.image_quality_combo.set_can_focus(True)
        self.image_quality_combo.set_selected(self._image_quality_index_from_settings(settings))
        if not self._quality_signal_connected:
            self.image_quality_combo.connect("notify::selected", self._on_image_quality_changed)
            self._quality_signal_connected = True

    def _image_quality_index_from_settings(self, settings) -> int:
        if settings.force_bilevel_compression:
            return 6

        fmt = settings.image_export_format.lower()
        if fmt == "original":
            return 0

        quality = settings.image_export_quality
        return self._get_quality_index_from_value(quality)

    def _load_pdfa_switch(self, settings) -> None:
        self.pdfa_switch_row.set_can_focus(True)
        self.pdfa_switch_row.set_active(settings.convert_to_pdfa)
        if not self._pdfa_signal_connected:
            self.pdfa_switch_row.connect("notify::active", self._on_pdfa_changed)
            self._pdfa_signal_connected = True

    def _load_page_layout_combo(self, settings) -> None:
        self.page_layout_combo.set_can_focus(True)
        self.page_layout_combo.set_selected(self._page_layout_index_from_settings(settings))
        if not self._page_layout_signal_connected:
            self.page_layout_combo.connect("notify::selected", self._on_page_layout_changed)
            self._page_layout_signal_connected = True

    def _page_layout_index_from_settings(self, settings) -> int:
        layout = settings.page_layout
        try:
            return self._page_layout_values.index(layout)
        except ValueError:
            return 0

    def _get_quality_index_from_value(self, quality: int) -> int:
        """Map quality percentage to dropdown index."""
        if quality <= 35:
            return 1
        elif quality <= 55:
            return 2
        elif quality <= 75:
            return 3
        elif quality <= 90:
            return 4
        else:
            return 5

    def _on_image_quality_changed(self, combo: Adw.ComboRow, _pspec) -> None:
        """Handle unified quality selector changes."""
        selected = combo.get_selected()
        if selected == Gtk.INVALID_LIST_POSITION:
            return
        if selected == 0:
            self.window.settings.image_export_format = "original"
            self.window.settings.force_bilevel_compression = False
            logger.info("Image quality changed to: Keep Original")
        elif selected == 6:
            self.window.settings.force_bilevel_compression = True
            logger.info("Image quality changed to: Black & White (JBIG2)")
        else:
            presets = [30, 50, 70, 85, 95]
            quality = presets[selected - 1] if (selected - 1) < len(presets) else 85
            self.window.settings.image_export_format = "jpeg"
            self.window.settings.image_export_quality = quality
            self.window.settings.force_bilevel_compression = False
            logger.info(f"Image quality changed to: {quality}%")
        self.window.settings._save_all_settings()

    def _on_pdfa_changed(self, switch_row: Adw.SwitchRow, _pspec) -> None:
        """Handle PDF/A toggle changes."""
        pdfa_enabled = switch_row.get_active()
        self.window.settings.convert_to_pdfa = pdfa_enabled
        logger.info(f"PDF/A export changed to: {pdfa_enabled}")
        self.window.settings._save_all_settings()

    def _on_page_layout_changed(self, combo: Adw.ComboRow, _pspec) -> None:
        """Handle page-layout (viewer /PageLayout) selection changes."""
        selected = combo.get_selected()
        if selected < 0 or selected >= len(self._page_layout_values):
            return
        layout = self._page_layout_values[selected]
        self.window.settings.page_layout = layout
        logger.info(f"Page layout changed to: {layout}")
        self.window.settings._save_all_settings()

    def _load_max_size_setting(self) -> None:
        """Load maximum output size setting from OcrSettings."""
        try:
            self.max_size_combo.set_can_focus(True)
            current_val = self.window.settings.max_file_size_mb
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

    def _load_replace_ocr_setting(self) -> None:
        """Load replace existing OCR setting from OcrSettings."""
        try:
            self.replace_ocr_switch.set_can_focus(True)
            self.replace_ocr_switch.set_active(self.window.settings.replace_existing_ocr)
            if not self._replace_ocr_signal_connected:
                self._replace_ocr_signal_connected = True
                self.replace_ocr_switch.connect("notify::active", self._on_replace_ocr_changed)
        except Exception as e:
            logger.error(f"Error loading replace OCR setting: {e}")

    def _on_replace_ocr_changed(self, switch_row: Adw.SwitchRow, _pspec) -> None:
        """Handle replace existing OCR toggle."""
        self.window.settings.replace_existing_ocr = switch_row.get_active()
        self.window.settings._save_all_settings()

    def _load_full_resolution_setting(self) -> None:
        """Load full resolution detection setting from OcrSettings."""
        try:
            self.full_resolution_switch.set_can_focus(True)
            self.full_resolution_switch.set_active(self.window.settings.detection_full_resolution)
            if not self._full_res_signal_connected:
                self._full_res_signal_connected = True
                self.full_resolution_switch.connect(
                    "notify::active", self._on_full_resolution_changed
                )
        except Exception as e:
            logger.error(f"Error loading full resolution setting: {e}")

    def _on_full_resolution_changed(self, switch_row: Adw.SwitchRow, _pspec) -> None:
        """Handle full resolution detection toggle."""
        self.window.settings.detection_full_resolution = switch_row.get_active()
        self.window.settings._save_all_settings()

    def _on_preprocessing_changed(self, switch_row: Adw.SwitchRow, _pspec) -> None:
        """Handle preprocessing option changes."""
        settings = self.window.settings
        try:
            if switch_row == self.deskew_switch:
                settings.enable_deskew = switch_row.get_active()
            elif switch_row == self.dewarp_switch:
                settings.enable_baseline_dewarp = switch_row.get_active()
            elif switch_row == self.perspective_switch:
                settings.enable_perspective_correction = switch_row.get_active()
            elif switch_row == self.orientation_switch:
                settings.enable_orientation_detection = switch_row.get_active()
            elif switch_row == self.scanner_switch:
                settings.enable_scanner_effect = switch_row.get_active()
            elif switch_row == self.enhance_embedded_switch:
                settings.enhance_embedded_images = switch_row.get_active()
            settings._save_all_settings()
        except Exception as e:
            logger.error(f"Error saving preprocessing setting: {e}")
