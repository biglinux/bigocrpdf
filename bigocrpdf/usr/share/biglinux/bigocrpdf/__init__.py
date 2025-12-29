"""
BigOcrPdf - Python package for adding OCR to PDF files

This package provides a GTK4 application for adding OCR to PDF files,
making them searchable and their text selectable.
"""

import locale

__version__ = "1.0.0"
__author__ = "Biglinux"
__license__ = "GPL-3.0"


def setup_i18n():
    """Initialize internationalization"""
    try:
        locale.setlocale(locale.LC_ALL, '')
    except locale.Error:
        # Fallback to C locale if system locale is not properly configured
        locale.setlocale(locale.LC_ALL, "C")