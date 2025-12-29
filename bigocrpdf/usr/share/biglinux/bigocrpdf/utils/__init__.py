"""
BigOcrPdf - Utils Package

This package contains utility functions and helpers used throughout the application.
"""

# Import core utilities
# Import format utilities
# Import configuration manager
from .config_manager import (
    ConfigManager,
    get_config,
    get_config_manager,
    set_config,
)

# Import custom exceptions
from .exceptions import (
    BigOcrPdfError,
    ConfigurationError,
    DependencyError,
    InvalidPdfError,
    LanguageNotAvailableError,
    OcrProcessingError,
    OutputPathError,
    PermissionDeniedError,
    ProcessTimeoutError,
    QueueError,
    ValidationError,
)
from .format_utils import (
    format_elapsed_time,
    format_file_size,
    format_progress_percentage,
    format_time_mmss,
)
from .i18n import _, setup_i18n
from .logger import logger

# Import PDF utilities
from .pdf_utils import (
    clear_page_count_cache,
    get_pdf_file_info,
    get_pdf_page_count,
    validate_pdf_file,
)

# Import progress state
from .progress_state import ProgressState

# Import signal manager
from .signal_manager import SignalManager

# Import text utilities
from .text_utils import (
    cleanup_temp_sidecar,
    extract_text_from_pdf,
    find_extracted_text,
    get_sidecar_path,
    read_text_from_sidecar,
)
from .timer import safe_remove_source

# Export everything
__all__ = [
    # Core
    "logger",
    "_",
    "setup_i18n",
    "safe_remove_source",
    # Config
    "ConfigManager",
    "get_config_manager",
    "get_config",
    "set_config",
    # PDF
    "get_pdf_page_count",
    "validate_pdf_file",
    "get_pdf_file_info",
    "clear_page_count_cache",
    # Text
    "extract_text_from_pdf",
    "read_text_from_sidecar",
    "find_extracted_text",
    "get_sidecar_path",
    "cleanup_temp_sidecar",
    # Format
    "format_file_size",
    "format_elapsed_time",
    "format_time_mmss",
    "format_progress_percentage",
    # Progress
    "ProgressState",
    # Signal
    "SignalManager",
    # Exceptions
    "BigOcrPdfError",
    "ConfigurationError",
    "DependencyError",
    "InvalidPdfError",
    "LanguageNotAvailableError",
    "OcrProcessingError",
    "OutputPathError",
    "PermissionDeniedError",
    "ProcessTimeoutError",
    "QueueError",
    "ValidationError",
]