"""
BigOcrPdf - Conclusion Page Module

This module handles the creation and management of the conclusion/results page UI.
"""

from bigocrpdf.ui.conclusion_export_mixin import ConclusionExportMixin
from bigocrpdf.ui.conclusion_page_builder import ConclusionPageBuilderMixin
from bigocrpdf.ui.conclusion_ui_mixin import ConclusionStatsFileListMixin


class ConclusionPageManager(
    ConclusionPageBuilderMixin,
    ConclusionStatsFileListMixin,
    ConclusionExportMixin,
):
    """Manages the conclusion/results page UI and interactions"""

    pass
