"""
BigOcrPdf - Utils Package

This package contains utility functions and helpers used throughout the application.
"""

# Import only what is necessary
from .i18n import _, setup_i18n
from .logger import logger
from .timer import safe_remove_source

# Export everything
__all__ = ['logger', '_', 'setup_i18n', 'safe_remove_source']