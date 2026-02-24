"""
BigOcrPdf - UI Package

This package contains all UI-related components for the application.
"""

from .file_selection_manager import FileSelectionManager
from .navigation_manager import NavigationManager, NavigationState
from .widgets import load_css

__all__ = [
    "load_css",
    "FileSelectionManager",
    "NavigationManager",
    "NavigationState",
]
