"""
BigOcrPdf - Utils Package

Utility modules for the application.
"""

from bigocrpdf.utils.checkpoint_manager import (
    CheckpointData,
    CheckpointManager,
    get_checkpoint_manager,
)
from bigocrpdf.utils.comparison import (
    PDFComparisonResult,
    compare_pdfs,
    get_batch_statistics,
)
from bigocrpdf.utils.history_manager import HistoryEntry, HistoryManager, get_history_manager
from bigocrpdf.utils.i18n import _, setup_i18n
from bigocrpdf.utils.logger import logger

__all__ = [
    "logger",
    "_",
    "setup_i18n",
    "CheckpointData",
    "CheckpointManager",
    "get_checkpoint_manager",
    "PDFComparisonResult",
    "compare_pdfs",
    "get_batch_statistics",
    "HistoryEntry",
    "HistoryManager",
    "get_history_manager",
]
