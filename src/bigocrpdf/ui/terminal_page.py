"""Processing page UI and progress updates."""

import time
from typing import TYPE_CHECKING

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

if TYPE_CHECKING:
    from bigocrpdf.window import BigOcrPdfWindow

from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.format_utils import format_elapsed_time
from bigocrpdf.utils.i18n import _, ngettext
from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.progress_state import ProgressState
from bigocrpdf.utils.timer import safe_remove_source

PROGRESS_UPDATE_INTERVAL_MS = 800


class TerminalPageManager:
    """Manage the processing page and its progress timer."""

    def __init__(self, window: "BigOcrPdfWindow"):
        self.window = window
        self.terminal_progress_bar: Gtk.ProgressBar | None = None
        self.terminal_status_bar: Gtk.Label | None = None
        self._summary_box: Gtk.Box | None = None
        self._summary_parent: Gtk.Box | None = None
        self.progress_timer_id: int | None = None
        self._progress_state = ProgressState()

    def create_terminal_page(self) -> Gtk.Box:
        """Create the processing page."""
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        main_box.set_vexpand(True)

        progress_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        progress_card.set_margin_bottom(8)
        progress_card.set_vexpand(True)
        main_box.append(progress_card)

        progress_area = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=24)
        progress_area.set_valign(Gtk.Align.CENTER)
        progress_area.set_vexpand(True)
        progress_area.set_margin_start(24)
        progress_area.set_margin_end(24)
        progress_area.set_margin_bottom(24)
        progress_card.append(progress_area)

        pdf_icon = Gtk.Image.new_from_icon_name("x-office-document-symbolic")
        pdf_icon.set_pixel_size(48)
        pdf_icon.set_margin_bottom(16)
        pdf_icon.set_halign(Gtk.Align.CENTER)
        progress_area.append(pdf_icon)

        current_file_label = Gtk.Label(label=_("Processing PDF files..."))
        current_file_label.add_css_class("title-3")
        current_file_label.set_halign(Gtk.Align.CENTER)
        current_file_label.set_margin_bottom(24)
        progress_area.append(current_file_label)

        self.terminal_progress_bar = Gtk.ProgressBar()
        self.terminal_progress_bar.set_show_text(True)
        self.terminal_progress_bar.set_text(_("0%"))
        self.terminal_progress_bar.set_margin_bottom(8)
        set_a11y_label(self.terminal_progress_bar, _("OCR processing progress"))
        progress_area.append(self.terminal_progress_bar)

        self.terminal_status_bar = Gtk.Label(label=_("Preparing processing..."))
        self.terminal_status_bar.add_css_class("body")
        self.terminal_status_bar.set_halign(Gtk.Align.CENTER)
        self.terminal_status_bar.set_margin_bottom(8)
        self.terminal_status_bar.set_accessible_role(Gtk.AccessibleRole.STATUS)
        progress_area.append(self.terminal_status_bar)

        cancel_button = Gtk.Button(label=_("Cancel"))
        cancel_button.add_css_class("destructive-action")
        cancel_button.connect("clicked", lambda _button: self.window.processing.cancel())
        cancel_button.set_margin_top(16)
        cancel_button.set_halign(Gtk.Align.CENTER)
        set_a11y_label(cancel_button, _("Cancel"))
        progress_area.append(cancel_button)

        self._summary_parent = progress_card
        self._add_active_settings_summary(progress_card)
        return main_box

    def _rebuild_settings_summary(self) -> None:
        """Rebuild the summary with the current settings."""
        if self._summary_box is not None and self._summary_parent is not None:
            self._summary_parent.remove(self._summary_box)
            self._summary_box = None
        if self._summary_parent is not None:
            self._add_active_settings_summary(self._summary_parent)

    def _add_active_settings_summary(self, container: Gtk.Box) -> None:
        """Add a compact summary of the active processing settings."""
        settings = self.window.settings
        all_effects = (
            (_("Deskew"), settings.enable_deskew),
            (_("Dewarp"), settings.enable_baseline_dewarp),
            (_("Perspective"), settings.enable_perspective_correction),
            (_("Auto-rotate"), settings.enable_orientation_detection),
            (_("Scanner Effect"), settings.enable_scanner_effect),
        )
        active = [name for name, enabled in all_effects if enabled]
        inactive = [name for name, enabled in all_effects if not enabled]

        image_format = settings.image_export_format.lower()
        quality_label = (
            _("Keep Original")
            if image_format == "original"
            else _("JPEG {quality}%").format(quality=settings.image_export_quality)
        )

        summary = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        summary.set_halign(Gtk.Align.CENTER)
        summary.set_valign(Gtk.Align.END)
        summary.set_margin_start(32)
        summary.set_margin_end(32)
        summary.set_margin_bottom(16)

        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator.set_margin_start(64)
        separator.set_margin_end(64)
        separator.set_margin_bottom(4)
        summary.append(separator)

        grid = Gtk.Grid(column_spacing=12, row_spacing=2, halign=Gtk.Align.CENTER)
        row = 0

        def add_row(label_text: str, value_text: str) -> None:
            nonlocal row
            label = Gtk.Label(label=label_text, halign=Gtk.Align.END)
            label.add_css_class("dim-label")
            label.add_css_class("caption")
            value = Gtk.Label(label=value_text, halign=Gtk.Align.START)
            value.add_css_class("dim-label")
            value.add_css_class("caption")
            grid.attach(label, 0, row, 1, 1)
            grid.attach(value, 1, row, 1, 1)
            row += 1

        add_row(_("Recognition"), _("Automatic multilingual (PP-OCRv6)"))
        add_row(_("Quality"), quality_label)
        if settings.convert_to_pdfa:
            add_row(_("Format"), "PDF/A")
        if settings.max_file_size_mb > 0:
            add_row(_("Max Size"), _("{mb} MB").format(mb=settings.max_file_size_mb))
        if settings.replace_existing_ocr:
            add_row(_("Mode"), _("Replace OCR"))
        if active:
            add_row(_("Active"), ", ".join(active))
        if inactive:
            add_row(_("Inactive"), ", ".join(inactive))

        summary.append(grid)
        self._summary_box = summary
        container.append(summary)

    def start_progress_monitor(self) -> None:
        """Reset the page and start its progress timer."""
        self.stop_progress_monitor()
        self._rebuild_settings_summary()
        self.reset_progress()
        self.progress_timer_id = GLib.timeout_add(
            PROGRESS_UPDATE_INTERVAL_MS,
            self._update_ocr_progress,
        )

    def stop_progress_monitor(self) -> None:
        """Stop the progress timer."""
        if self.progress_timer_id is not None:
            safe_remove_source(self.progress_timer_id)
            self.progress_timer_id = None

    def update_processing_status(self, input_file: str | None = None) -> None:
        """Update the completed and remaining input counts."""
        processor = self.window.processing.ocr_processor
        completed = processor.get_completed_input_count()
        total = processor.get_total_count()

        if self.terminal_status_bar is not None:
            self.terminal_status_bar.set_markup(
                _("<b>Completed: {completed}/{total}</b> · Remaining: {remaining}").format(
                    completed=completed,
                    total=total,
                    remaining=max(0, total - completed),
                )
            )

        if input_file:
            logger.info(
                _("Processed file {current}/{total}: {filename}").format(
                    current=completed,
                    total=total,
                    filename=self.window.settings.display_name(input_file),
                )
            )

    def update_terminal_progress(self, fraction: float, text: str | None = None) -> None:
        """Update the progress bar without redundant redraws."""
        if self.terminal_progress_bar is None:
            return

        fraction = max(0.0, min(1.0, fraction))
        if self._progress_state.update_fraction(fraction):
            self.terminal_progress_bar.set_fraction(fraction)
        if text is not None and self._progress_state.update_text(text):
            self.terminal_progress_bar.set_text(text)

    def _update_ocr_progress(self) -> bool:
        """Read processor state and update the page."""
        if self.window.ui.main_stack.get_visible_child_name() != "terminal":
            self.progress_timer_id = None
            return False

        processor = self.window.processing.ocr_processor
        progress_data = {
            "progress": processor.get_progress(),
            "processed_files": processor.get_completed_input_count(),
            "total_files": processor.get_total_count(),
            "current_file_info": processor.get_current_file_info(),
            "is_processing": processor.is_processing(),
        }
        self._update_progress_bar_incremental(progress_data["progress"])

        keep_monitoring = self._update_status_text_incremental(progress_data)
        if not keep_monitoring:
            self.progress_timer_id = None
        return keep_monitoring

    def _update_progress_bar_incremental(self, progress: float) -> None:
        """Update the displayed percentage when it changes meaningfully."""
        if self.terminal_progress_bar is None:
            return

        progress = max(0.0, min(1.0, progress))
        if self._progress_state.update_fraction(progress):
            self.terminal_progress_bar.set_fraction(progress)
            progress_text = f"{self._progress_state.get_percentage()}%"
            if self._progress_state.update_text(progress_text):
                self.terminal_progress_bar.set_text(progress_text)

    def _update_status_text_incremental(self, progress_data: dict) -> bool:
        """Update the current processing status and return timer ownership."""
        if self.terminal_status_bar is None:
            return True

        processed_files = progress_data.get("processed_files", 0)
        total_files = progress_data.get("total_files", 0)
        current_file_info = progress_data.get("current_file_info", {})
        is_processing = progress_data.get("is_processing", True)
        progress = progress_data.get("progress", 0.0)

        elapsed_time = 0
        if self.window.processing.process_start_time:
            elapsed_time = max(
                0,
                int(time.time() - self.window.processing.process_start_time),
            )
        time_str = format_elapsed_time(elapsed_time)

        if not is_processing and progress >= 1.0:
            self._show_completion_status(total_files, time_str)
            return False
        if current_file_info and current_file_info.get("filename"):
            self._show_processing_status(current_file_info, time_str)
        elif processed_files > 0:
            self._show_simple_progress_status(processed_files, total_files, time_str)
        else:
            self._show_initial_status(total_files, time_str)
        return True

    def _show_completion_status(self, total_files: int, time_str: str) -> None:
        status_bar = self.terminal_status_bar
        if status_bar is None:
            return

        status_text = ngettext(
            "<b>OCR processing complete!</b> {total} file processed · Total time: {time}",
            "<b>OCR processing complete!</b> {total} files processed · Total time: {time}",
            total_files,
        ).format(total=total_files, time=time_str)
        if self._progress_state.update_status(status_text):
            status_bar.set_markup(status_text)
            self.window.announce_status(
                ngettext(
                    "OCR processing complete. {total} file processed.",
                    "OCR processing complete. {total} files processed.",
                    total_files,
                ).format(total=total_files)
            )

    def _show_processing_status(self, current_file_info: dict, time_str: str) -> None:
        status_bar = self.terminal_status_bar
        if status_bar is None:
            return

        filename = current_file_info.get("filename", "")
        file_number = current_file_info.get("file_number", 1)
        total_files = current_file_info.get("total_files", 1)
        status_message = current_file_info.get("status_message", "")
        escaped_filename = GLib.markup_escape_text(str(filename))
        escaped_time = GLib.markup_escape_text(time_str)

        if status_message:
            status_text = _(
                "File {current}/{total}: <b>{filename}</b> - {status} • Time: {time}"
            ).format(
                current=file_number,
                total=total_files,
                filename=escaped_filename,
                status=GLib.markup_escape_text(str(status_message)),
                time=escaped_time,
            )
        else:
            status_text = _(
                "Processing file {current}/{total}: <b>{filename}</b> • Time: {time}"
            ).format(
                current=file_number,
                total=total_files,
                filename=escaped_filename,
                time=escaped_time,
            )

        if self._progress_state.update_status(status_text):
            status_bar.set_markup(status_text)
            self.window.announce_status(
                _("Processing file {current} of {total}: {filename}").format(
                    current=file_number,
                    total=total_files,
                    filename=filename,
                )
            )

    def _show_simple_progress_status(
        self,
        processed_files: int,
        total_files: int,
        time_str: str,
    ) -> None:
        status_bar = self.terminal_status_bar
        if status_bar is None:
            return

        status_text = _("Completed: {processed}/{total} · Time: {time}").format(
            processed=processed_files,
            total=total_files,
            time=time_str,
        )
        if self._progress_state.update_status(status_text):
            status_bar.set_text(status_text)

    def _show_initial_status(self, total_files: int, time_str: str) -> None:
        status_bar = self.terminal_status_bar
        if status_bar is None:
            return

        status_text = ngettext(
            "Starting processing of {total} file... · Time: {time}",
            "Starting processing of {total} files... · Time: {time}",
            total_files,
        ).format(total=total_files, time=time_str)
        if self._progress_state.update_status(status_text):
            status_bar.set_text(status_text)

    def reset_progress(self) -> None:
        """Reset the progress widgets and cached display state."""
        if self.terminal_progress_bar is not None:
            self.terminal_progress_bar.set_fraction(0.0)
            self.terminal_progress_bar.set_text(_("{percent}%").format(percent=0))
        if self.terminal_status_bar is not None:
            self.terminal_status_bar.set_text(_("Preparing processing..."))
        self._progress_state.reset()

    def cleanup(self) -> None:
        """Release resources owned by the page."""
        self.stop_progress_monitor()
