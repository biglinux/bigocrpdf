"""Sidebar UI creation and settings callbacks for SettingsPageManager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk

from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger

if TYPE_CHECKING:
    pass


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

        self._create_language_widgets(group)

        self._proc_row = Adw.ActionRow(title=_("Image Corrections"))
        self._proc_row.add_prefix(Gtk.Image.new_from_icon_name("applications-graphics-symbolic"))
        self._proc_row.add_suffix(Gtk.Image.new_from_icon_name("go-next-symbolic"))
        self._proc_row.set_activatable(True)
        self._proc_row.connect("activated", self._on_corrections_row_activated)
        self._create_preprocessing_widgets()
        group.add(self._proc_row)

        self._out_row = Adw.ActionRow(title=_("Output Settings"))
        self._out_row.add_prefix(Gtk.Image.new_from_icon_name("document-save-symbolic"))
        self._out_row.add_suffix(Gtk.Image.new_from_icon_name("go-next-symbolic"))
        self._out_row.set_activatable(True)
        self._out_row.connect("activated", self._on_output_row_activated)
        self._create_output_widgets()
        group.add(self._out_row)

        self._adv_row = Adw.ActionRow(title=_("Advanced"))
        self._adv_row.add_prefix(Gtk.Image.new_from_icon_name("preferences-system-symbolic"))
        self._adv_row.add_suffix(Gtk.Image.new_from_icon_name("go-next-symbolic"))
        self._adv_row.set_activatable(True)
        self._adv_row.connect("activated", self._on_advanced_row_activated)
        self._create_advanced_widgets()
        group.add(self._adv_row)

        settings_box.append(group)
        group.connect("map", lambda _w: self._load_all_sidebar_settings())
        scrolled_window.set_child(settings_box)
        return scrolled_window

    def _create_language_widgets(self, group: Adw.PreferencesGroup) -> None:
        """Create language selection widgets and add to group."""
        from bigocrpdf.utils.tooltip_helper import get_tooltip_helper

        tooltip = get_tooltip_helper()

        self.lang_combo = Adw.ComboRow(title=_("Language"))
        set_a11y_label(self.lang_combo, _("Language"))
        lang_icon = Gtk.Image.new_from_icon_name("preferences-desktop-locale-symbolic")
        self.lang_combo.add_prefix(lang_icon)

        lang_help_btn = Gtk.Button(
            icon_name="help-about-symbolic",
            valign=Gtk.Align.CENTER,
            css_classes=["flat", "circular"],
        )
        lang_help_btn.set_tooltip_text(_("Language help"))
        set_a11y_label(lang_help_btn, _("Language help"))
        lang_help_btn.connect("clicked", self._on_language_help_clicked)
        self.lang_combo.add_suffix(lang_help_btn)

        languages = self.window.ocr_processor.get_available_ocr_languages()
        self._available_languages = languages
        lang_model = Gtk.StringList()
        for _i, (_lang_code, lang_name) in enumerate(languages):
            lang_model.append(lang_name)
        self.lang_combo.set_model(lang_model)
        self._lang_signal_connected = False

        group.add(self.lang_combo)
        self.lang_dropdown = self.lang_combo

        tooltip.add_tooltip(
            self.lang_combo,
            _(
                "Choose the language of your document's text.\n"
                "The correct language helps recognize text more accurately."
            ),
        )

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

    def _on_corrections_row_activated(self, _row: Adw.ActionRow) -> None:
        """Open the image corrections configuration dialog."""
        from bigocrpdf.ui.corrections_dialog import show_image_corrections_dialog

        show_image_corrections_dialog(self.window, self._correction_switches)

    def _on_output_row_activated(self, _row: Adw.ActionRow) -> None:
        """Open the output settings configuration dialog."""
        from bigocrpdf.ui.output_dialog import show_output_settings_dialog

        show_output_settings_dialog(self.window, self._output_widgets)

    def _update_output_subtitle(self) -> None:
        """Update the Output Settings row subtitle with summary."""
        parts = []
        try:
            if hasattr(self, "image_quality_combo"):
                idx = self.image_quality_combo.get_selected()
                model = self.image_quality_combo.get_model()
                if model and idx < model.get_n_items():
                    parts.append(model.get_string(idx))
            active = sum(
                1
                for k in ("pdfa",)
                if self._output_widgets.get(k) and self._output_widgets[k].get_active()
            )
            if active:
                parts.append(_("{n} options enabled").format(n=active))
        except Exception as e:
            logger.debug(f"Could not update output subtitle: {e}")
        text = ", ".join(parts) if parts else ""
        self._out_row.set_subtitle(GLib.markup_escape_text(text, -1) if text else "")

    def _create_output_widgets(self) -> None:
        """Create output settings widgets (not added to any parent)."""
        self._quality_signal_connected = False
        self._pdfa_signal_connected = False

        self.image_quality_combo = Adw.ComboRow(title=_("Image Quality"))
        set_a11y_label(self.image_quality_combo, _("Image Quality"))
        quality_model = Gtk.StringList.new([
            _("Keep Original"),
            _("Very Low (30%)"),
            _("Low (50%)"),
            _("Medium (70%)"),
            _("High (85%)"),
            _("Maximum (95%)"),
            _("Black & White (JBIG2)"),
        ])
        self.image_quality_combo.set_model(quality_model)

        self.pdfa_switch_row = Adw.SwitchRow(title=_("Export as PDF/A"))

        self.max_size_combo = Adw.ComboRow(title=_("Maximum Output Size"))
        set_a11y_label(self.max_size_combo, _("Maximum Output Size"))
        self._max_size_values = [0, 5, 10, 15, 20, 25, 50, 100]
        max_size_model = Gtk.StringList.new([
            _("No limit"),
            _("5 MB"),
            _("10 MB"),
            _("15 MB"),
            _("20 MB"),
            _("25 MB"),
            _("50 MB"),
            _("100 MB"),
        ])
        self.max_size_combo.set_model(max_size_model)
        self.max_size_combo.connect("notify::selected", self._on_max_size_changed)

        self._output_widgets: dict[str, Gtk.Widget] = {
            "image_quality": self.image_quality_combo,
            "pdfa": self.pdfa_switch_row,
            "max_size": self.max_size_combo,
        }

        # Update subtitle when any switch/combo changes
        for w in self._output_widgets.values():
            if isinstance(w, Adw.SwitchRow):
                w.connect("notify::active", lambda *_: self._update_output_subtitle())
            elif isinstance(w, Adw.ComboRow):
                w.connect("notify::selected", lambda *_: self._update_output_subtitle())

    def _on_advanced_row_activated(self, _row: Adw.ActionRow) -> None:
        """Open the advanced settings configuration dialog."""
        from bigocrpdf.ui.advanced_dialog import show_advanced_settings_dialog

        show_advanced_settings_dialog(self.window, self._advanced_widgets)

    def _update_advanced_subtitle(self) -> None:
        """Update the Advanced row subtitle with summary."""
        parts = []
        if hasattr(self, "ocr_precision_combo"):
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
        precision_model = Gtk.StringList.new([
            _("Low Precision"),
            _("Standard"),
            _("Precise"),
            _("Very Precise"),
        ])
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
        if self.lang_combo:
            self.lang_combo.set_can_focus(True)
            current_lang = self.window.settings.lang
            for i, (code, _name) in enumerate(self._available_languages):
                if code == current_lang:
                    self.lang_combo.set_selected(i)
                    break
            if not self._lang_signal_connected:
                self._lang_signal_connected = True
                self.lang_combo.connect("notify::selected", self._on_language_changed)

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

    def _on_language_changed(self, combo, _param) -> None:
        """Handle language selection change."""
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

        content_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=16,
            margin_top=12,
            margin_bottom=24,
            margin_start=16,
            margin_end=16,
        )

        desc_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
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

        for lang_code, lang_name in available_languages:
            details = ModelDiscovery.LANGUAGE_DETAILS.get(lang_code, "")
            if not details:
                continue
            languages = sorted([lang.strip() for lang in details.split(",") if lang.strip()])
            lang_count = len(languages)

            section_group = Adw.PreferencesGroup(
                title=f"{lang_name}",
                description=_("{count} languages").format(count=lang_count),
            )

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

    def _load_preprocessing_settings(self) -> None:
        """Load preprocessing settings from OcrSettings."""
        settings = self.window.settings
        try:
            if hasattr(self, "deskew_switch"):
                self.deskew_switch.set_can_focus(True)
                self.deskew_switch.set_active(settings.enable_deskew)
            if hasattr(self, "dewarp_switch"):
                self.dewarp_switch.set_can_focus(True)
                self.dewarp_switch.set_active(getattr(settings, "enable_baseline_dewarp", True))
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
                self.scanner_switch.set_active(getattr(settings, "enable_scanner_effect", True))
            if hasattr(self, "enhance_embedded_switch"):
                self.enhance_embedded_switch.set_can_focus(True)
                self.enhance_embedded_switch.set_active(
                    getattr(settings, "enhance_embedded_images", False)
                )

            if not self._preprocessing_signal_connected:
                self._preprocessing_signal_connected = True
                if hasattr(self, "deskew_switch"):
                    self.deskew_switch.connect("notify::active", self._on_preprocessing_changed)
                if hasattr(self, "dewarp_switch"):
                    self.dewarp_switch.connect("notify::active", self._on_preprocessing_changed)
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
                if hasattr(self, "enhance_embedded_switch"):
                    self.enhance_embedded_switch.connect(
                        "notify::active", self._on_preprocessing_changed
                    )
        except Exception as e:
            logger.error(f"Error loading preprocessing settings: {e}")

        self._update_corrections_subtitle()

    def _get_precision_index_from_settings(self, settings) -> int:
        """Get dropdown index based on current text_score and box_thresh values."""
        text_score = getattr(settings, "text_score_threshold", 0.3)
        box_thresh = getattr(settings, "box_thresh", 0.5)
        for idx, (ts, bt) in enumerate(self.PRECISION_PRESETS):
            if abs(text_score - ts) < 0.05 and abs(box_thresh - bt) < 0.05:
                return idx
        return 1

    def _load_advanced_ocr_settings(self) -> None:
        """Load advanced OCR settings from OcrSettings."""
        settings = self.window.settings
        try:
            if hasattr(self, "ocr_precision_combo"):
                self.ocr_precision_combo.set_can_focus(True)
                precision_idx = self._get_precision_index_from_settings(settings)
                self.ocr_precision_combo.set_selected(precision_idx)
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
            if hasattr(self, "image_quality_combo"):
                self.image_quality_combo.set_can_focus(True)
                if getattr(settings, "force_bilevel_compression", False):
                    self.image_quality_combo.set_selected(6)
                else:
                    fmt = getattr(settings, "image_export_format", "original").lower()
                    if fmt == "original":
                        self.image_quality_combo.set_selected(0)
                    else:
                        quality = getattr(settings, "image_export_quality", 85)
                        idx = self._get_quality_index_from_value(quality)
                        self.image_quality_combo.set_selected(idx)
                if not self._quality_signal_connected:
                    self.image_quality_combo.connect(
                        "notify::selected", self._on_image_quality_changed
                    )
                    self._quality_signal_connected = True

            if hasattr(self, "pdfa_switch_row"):
                self.pdfa_switch_row.set_can_focus(True)
                pdfa_enabled = getattr(settings, "convert_to_pdfa", False)
                self.pdfa_switch_row.set_active(pdfa_enabled)
                if not self._pdfa_signal_connected:
                    self.pdfa_switch_row.connect("notify::active", self._on_pdfa_changed)
                    self._pdfa_signal_connected = True
        except Exception as e:
            logger.error(f"Error loading image export settings: {e}")
        self._update_output_subtitle()

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

    def _load_replace_ocr_setting(self) -> None:
        """Load replace existing OCR setting from OcrSettings."""
        try:
            if not hasattr(self, "replace_ocr_switch"):
                return
            self.replace_ocr_switch.set_can_focus(True)
            self.replace_ocr_switch.set_active(
                getattr(self.window.settings, "replace_existing_ocr", False)
            )
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
            if not hasattr(self, "full_resolution_switch"):
                return
            self.full_resolution_switch.set_can_focus(True)
            self.full_resolution_switch.set_active(
                getattr(self.window.settings, "detection_full_resolution", False)
            )
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
