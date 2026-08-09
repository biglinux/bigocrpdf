"""Conclusion page manager."""

from bigocrpdf.ui.conclusion_export_mixin import ConclusionExportMixin
from bigocrpdf.ui.conclusion_page_builder import ConclusionPageBuilderMixin
from bigocrpdf.ui.conclusion_ui_mixin import ConclusionStatsFileListMixin


class ConclusionPageManager(
    ConclusionPageBuilderMixin,
    ConclusionStatsFileListMixin,
    ConclusionExportMixin,
):
    """Manage the conclusion page, its results, and export actions."""

    def reset_page(self) -> None:
        """Reset result widgets and bulk-selection state."""
        for label, text in (
            (self.result_file_count, "0"),
            (self.result_page_count, "0"),
            (self.result_time, "00:00"),
            (self.result_file_size, "0 KB"),
            (self.result_size_change, "--"),
        ):
            if label is not None:
                label.set_text(text)

        if self.result_size_change is not None:
            self.result_size_change.remove_css_class("success")
            self.result_size_change.remove_css_class("warning")

        self._selection_mode = False
        self._selected_files.clear()
        if self._selection_toggle_btn is not None:
            self._selection_toggle_btn.set_active(False)
        if self._selection_action_bar is not None:
            self._selection_action_bar.set_visible(False)
        self._refresh_selection_ui()

        if self.output_list_box is not None:
            self._clear_output_list()
