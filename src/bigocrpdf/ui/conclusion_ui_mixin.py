"""Conclusion page statistics and generated-file list."""
# Host attributes are supplied by the conclusion page's explicit mixin composition.
# pyright: reportAttributeAccessIssue=false

import os
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, Gtk

if TYPE_CHECKING:
    from bigocrpdf.window import BigOcrPdfWindow

from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.comparison import PDFComparisonResult, get_batch_statistics
from bigocrpdf.utils.format_utils import format_file_size
from bigocrpdf.utils.i18n import _, ngettext
from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.pdf_utils import get_pdf_page_count

OutputFile = tuple[str, int, int, PDFComparisonResult | None]


def _icon_button(icon_name: str, tooltip: str, on_click: Callable[[], None]) -> Gtk.Button:
    button = Gtk.Button.new_from_icon_name(icon_name)
    button.add_css_class("circular")
    button.add_css_class("flat")
    button.set_tooltip_text(tooltip)
    set_a11y_label(button, tooltip)
    button.connect("clicked", lambda _button: on_click())
    return button


class ConclusionStatsFileListMixin:
    """Mixin providing conclusion page statistics update and file list management."""

    def __init__(self, window: "BigOcrPdfWindow"):
        """Initialize the conclusion page manager

        Args:
            window: Reference to the main application window
        """
        self.window = window

        # UI component references for results display
        self.result_file_count = None
        self.result_page_count = None
        self.result_time = None
        self.result_file_size = None
        self.result_size_change = None
        self.output_list_box = None

        # Bulk-selection state (set by the page builder).
        self._selection_toggle_btn: Gtk.ToggleButton | None = None
        self._selection_action_bar: Gtk.Box | None = None
        self._selection_count_label: Gtk.Label | None = None
        self._bulk_export_button: Gtk.MenuButton | None = None
        self._selection_mode: bool = False
        self._selected_files: set[str] = set()
        self._output_files: list[OutputFile] = []

    def update_conclusion_page(self) -> None:
        """Update statistics and rows from one consistent output snapshot."""
        comparisons = {
            result.output_path: result for result in self.window.settings.comparison_results
        }
        self._output_files = []
        for output_file in self.window.settings.processed_files:
            try:
                file_size = os.path.getsize(output_file)
            except OSError as error:
                logger.warning("Could not read output file %s: %s", output_file, error)
                continue
            self._output_files.append(
                (
                    output_file,
                    get_pdf_page_count(output_file),
                    file_size,
                    comparisons.get(output_file),
                )
            )

        self._update_statistics()
        self._update_file_list()

    def _update_statistics(self) -> None:
        """Update result totals from the collected output snapshot."""
        file_count = self.window.processing.ocr_processor.get_successful_input_count()
        self.result_file_count.set_text(str(file_count))

        total_pages = sum(pages for _path, pages, _size, _comparison in self._output_files)
        total_size = sum(size for _path, _pages, size, _comparison in self._output_files)
        self.result_page_count.set_text(str(total_pages))
        self.result_file_size.set_text(format_file_size(total_size))
        self._update_processing_time()
        self._update_size_change(total_size)

    def _update_size_change(self, total_output: int) -> None:
        """Update the aggregate input/output size comparison."""
        if not self.result_size_change:
            return

        self.result_size_change.remove_css_class("success")
        self.result_size_change.remove_css_class("warning")
        results = self.window.settings.comparison_results
        if not results:
            self.result_size_change.set_text("--")
            return

        total_input = get_batch_statistics(results)["total_input_size_bytes"]
        if total_input <= 0:
            self.result_size_change.set_text("--")
            return

        change_percent = round(((total_output - total_input) / total_input) * 100, 1)
        sign = "+" if change_percent >= 0 else ""
        self.result_size_change.set_text(
            f"{format_file_size(total_input)} → {format_file_size(total_output)} "
            f"({sign}{change_percent:.1f}%)"
        )
        if change_percent < 0:
            self.result_size_change.add_css_class("success")
        elif change_percent > 50:
            self.result_size_change.add_css_class("warning")

    def _update_processing_time(self) -> None:
        """Update the processing time display"""
        if self.window.processing.process_start_time:
            elapsed_time = time.time() - self.window.processing.process_start_time
            minutes = int(elapsed_time / 60)
            seconds = int(elapsed_time % 60)
            self.result_time.set_text(f"{minutes:02d}:{seconds:02d}")
        else:
            self.result_time.set_text("--:--")

    def _update_file_list(self) -> None:
        """Rebuild the generated-file list from the collected output snapshot."""
        self._clear_output_list()
        visible_files = {path for path, _pages, _size, _comparison in self._output_files}
        self._selected_files.intersection_update(visible_files)

        for output_file, pages, file_size, comparison in self._output_files:
            self.output_list_box.append(
                self._create_file_row(output_file, pages, file_size, comparison)
            )
        self._refresh_selection_ui()

    def _clear_output_list(self) -> None:
        """Remove all generated-file rows."""
        while child := self.output_list_box.get_first_child():
            self.output_list_box.remove(child)

    def _create_file_row(
        self,
        output_file: str,
        pages: int,
        file_size: int,
        comparison: PDFComparisonResult | None = None,
    ) -> Adw.ActionRow:
        """Create a row for a processed file

        Args:
            output_file: Path to the output file
            pages: Number of pages
            file_size: File size in bytes
            comparison: Optional comparison result for size change display

        Returns:
            An Adw.ActionRow for the file
        """
        row = Adw.ActionRow()
        row.set_title(os.path.basename(output_file))
        row.set_subtitle(os.path.dirname(output_file))

        # Add file statistics
        self._add_file_statistics_to_row(row, pages, file_size, comparison)

        if self._selection_mode:
            # In selection mode the row is dedicated to picking files; per-row
            # action buttons would only get in the way.
            check = Gtk.CheckButton()
            check.set_active(output_file in self._selected_files)
            check.connect("toggled", self._on_row_check_toggled, output_file)
            row.add_prefix(check)
        else:
            file_icon = Gtk.Image.new_from_icon_name("x-office-document-symbolic")
            row.add_prefix(file_icon)
            button_container = self._create_file_action_buttons(output_file)
            row.add_suffix(button_container)

        return row

    # ── Selection mode ────────────────────────────────────────────────

    def _on_selection_toggle_clicked(self, button: Gtk.ToggleButton) -> None:
        """Toggle selection mode on/off and rebuild the file list accordingly."""
        self._selection_mode = button.get_active()
        if not self._selection_mode:
            self._selected_files.clear()
        if self._selection_action_bar is not None:
            self._selection_action_bar.set_visible(self._selection_mode)
        self._update_file_list()

    def _on_row_check_toggled(self, check: Gtk.CheckButton, file_path: str) -> None:
        """Track selection set as individual rows are toggled."""
        if check.get_active():
            self._selected_files.add(file_path)
        else:
            self._selected_files.discard(file_path)
        self._refresh_selection_ui()

    def _on_select_all_clicked(self) -> None:
        """Mark every visible file as selected."""
        self._selected_files.update(path for path, _pages, _size, _comparison in self._output_files)
        self._update_file_list()
        self._refresh_selection_ui()

    def _on_clear_selection_clicked(self) -> None:
        """Drop all selections without leaving selection mode."""
        self._selected_files.clear()
        self._update_file_list()

    def _refresh_selection_ui(self) -> None:
        """Sync the action bar label and bulk-export button sensitivity."""
        if self._selection_count_label is not None:
            count = len(self._selected_files)
            self._selection_count_label.set_text(_("Selected: {count}").format(count=count))
        if self._bulk_export_button is not None:
            self._bulk_export_button.set_sensitive(bool(self._selected_files))

    def _add_file_statistics_to_row(
        self,
        row: Adw.ActionRow,
        pages: int,
        file_size: int,
        comparison: PDFComparisonResult | None = None,
    ) -> None:
        """Add file statistics to a file row

        Args:
            row: The row to add statistics to
            pages: Number of pages
            file_size: File size in bytes
            comparison: Optional comparison result for size change
        """
        # Add page count
        page_label = Gtk.Label(
            label=ngettext("{count} page", "{count} pages", pages).format(count=pages)
        )
        page_label.add_css_class("caption")
        row.add_suffix(page_label)

        # Add size label
        size_label = Gtk.Label(label=format_file_size(file_size))
        size_label.add_css_class("caption")
        row.add_suffix(size_label)

        # Add size change indicator with theme-aware CSS classes
        if comparison and comparison.input_size_bytes > 0:
            change_pct = comparison.size_change_percent
            sign = "+" if change_pct >= 0 else ""
            change_label = Gtk.Label(label=f"({sign}{change_pct:.0f}%)")
            change_label.add_css_class("caption")
            if change_pct < 0:
                change_label.add_css_class("success")
            elif change_pct > 50:
                change_label.add_css_class("warning")
            row.add_suffix(change_label)

    def _create_file_action_buttons(self, output_file: str) -> Gtk.Box:
        """Create the actions shown beside one generated file."""
        buttons = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=4,
            halign=Gtk.Align.END,
        )

        open_button = _icon_button(
            "document-open-symbolic",
            _("Open the processed file"),
            lambda: self._open_file(output_file),
        )
        open_button.set_margin_start(12)
        open_button.set_margin_end(12)
        buttons.append(open_button)
        buttons.append(
            _icon_button(
                "folder-open-symbolic",
                _("Show in file manager"),
                lambda: self._reveal_in_file_manager(output_file),
            )
        )
        buttons.append(
            _icon_button(
                "format-text-uppercase-symbolic",
                _("View the text found in this document"),
                lambda: self._show_extracted_text(output_file),
            )
        )
        buttons.append(self._create_export_menu_button(output_file))
        return buttons

    def _create_export_menu_button(self, output_file: str) -> Gtk.MenuButton:
        """Unified export menu for a single OCR'd file.

        Backed by ``Gio.Menu`` + a popover-menu so keyboard navigation
        (Up/Down/Enter) and screen-reader semantics come for free.
        """
        menu_model = Gio.Menu()
        menu_model.append(_("OpenDocument (.odt)"), "row.odt")
        menu_model.append(_("Markdown (.md)"), "row.md")

        button = Gtk.MenuButton()
        button.set_icon_name("document-save-as-symbolic")
        button.set_tooltip_text(_("Export to other formats"))
        button.add_css_class("flat")
        set_a11y_label(button, _("Export to other formats"))
        button.set_menu_model(menu_model)

        group = Gio.SimpleActionGroup()
        odt_action = Gio.SimpleAction.new("odt", None)
        odt_action.connect(
            "activate", lambda *_a: self._show_odf_export_options_dialog(output_file)
        )
        group.add_action(odt_action)
        md_action = Gio.SimpleAction.new("md", None)
        md_action.connect(
            "activate", lambda *_a: self._show_markdown_export_options_dialog(output_file)
        )
        group.add_action(md_action)
        button.insert_action_group("row", group)
        return button
