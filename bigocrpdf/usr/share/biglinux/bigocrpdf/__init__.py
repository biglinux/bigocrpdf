"""
BigOcrPdf - Python package for adding OCR to PDF files

This package provides a GTK4 application for adding OCR to PDF files,
making them searchable and their text selectable.
"""

import os
import sys
import locale
import logging

__version__ = "1.0.0"
__author__ = "Biglinux"
__license__ = "GPL-3.0"

# Initialize i18n
def setup_i18n():
    """Initialize internationalization"""
    try:
        locale.setlocale(locale.LC_ALL, '')
    except locale.Error:
        # Fallback to C locale if system locale is not properly configured
        locale.setlocale(locale.LC_ALL, 'C')

# Initialize logging
def setup_logging(log_level=logging.INFO):
    """Initialize logging configuration
    
    Args:
        log_level: Logging level to use
    """
    from utils.logger import setup_logger
    return setup_logger(log_level)

# Add package directory to sys.path
def add_package_to_path():
    """Add the package directory to Python path"""
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)

# Simple version function
def get_version():
    """Get application version
    
    Returns:
        Application version string
    """
    return __version__