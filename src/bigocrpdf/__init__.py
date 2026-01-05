"""
BigOcrPdf - Python package for adding OCR to PDF files

This package provides a GTK4 application for adding OCR to PDF files,
making them searchable and their text selectable.
"""

import locale
import sys

# Handle direct execution from source directory
if __package__ is None:
    import pathlib

    parent_dir = pathlib.Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    __package__ = "bigocrpdf"

__version__ = "2.0.0"
__author__ = "BigLinux Team"
__license__ = "GPL-3.0"


def setup_i18n() -> None:
    """Initialize internationalization."""
    try:
        locale.setlocale(locale.LC_ALL, "")
    except locale.Error:
        # Fallback to C locale if system locale is not properly configured
        locale.setlocale(locale.LC_ALL, "C")


def main() -> int:
    """Main entry point for the application.

    Returns:
        The application exit code.
    """
    from bigocrpdf.main import main as _main

    return _main()


__all__ = ["main", "__version__", "__author__", "__license__", "setup_i18n"]


if __name__ == "__main__":
    sys.exit(main())
