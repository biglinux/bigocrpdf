"""
BigOcrPdf - Python package for adding OCR to PDF files

This package provides a GTK4 application for adding OCR to PDF files,
making them searchable and their text selectable.
"""

import locale
import os
import sys

# Handle direct execution from source directory
if __package__ is None:
    import pathlib

    parent_dir = pathlib.Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    __package__ = "bigocrpdf"

__version__ = "3.0.0"
__author__ = "BigLinux Team"
__license__ = "GPL-3.0"


def setup_i18n() -> None:
    """Initialize internationalization."""
    try:
        locale.setlocale(locale.LC_ALL, "")
        # Keep LC_NUMERIC as C to avoid breaking onnxruntime/RapidOCR
        # which expects dot as decimal separator (not comma as in pt_BR)
        locale.setlocale(locale.LC_NUMERIC, "C")
    except locale.Error:
        # Fallback to C locale if system locale is not properly configured
        locale.setlocale(locale.LC_ALL, "C")


def _setup_python_compatibility() -> bool:
    """Setup Python version compatibility for OCR modules.

    Returns:
        True if setup succeeded, False if there are compatibility issues.
    """
    try:
        from bigocrpdf.utils.python_compat import setup_python_compatibility

        setup_python_compatibility()
        return True
    except Exception as e:
        print(f"Warning: Python compatibility setup failed: {e}", file=sys.stderr)
        return False


def _check_ocr_dependencies() -> tuple[bool, str]:
    """Check if OCR dependencies are available and compatible.

    Returns:
        Tuple of (success, error_message). If success is True, error_message is empty.
    """
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    # Try to import rapidocr
    try:
        from rapidocr import RapidOCR  # noqa: F401

        return True, ""
    except ImportError as e:
        error_msg = str(e)

        # Check for common compatibility errors
        if "openvino" in error_msg.lower() or "_pyopenvino" in error_msg:
            return False, (
                f"OpenVINO is not compatible with Python {python_version}.\n"
                f"The OCR packages were compiled for a different Python version.\n\n"
                f"Solution:\n"
                f"  Install ONNX Runtime as fallback engine:\n"
                f"    sudo pacman -S python-onnxruntime-cuda\n"
                f"  Or for CPU-only:\n"
                f"    sudo pacman -S python-onnxruntime-cpu"
            )
        elif "onnxruntime" in error_msg.lower():
            return False, (
                f"ONNX Runtime is not compatible with Python {python_version}.\n\n"
                f"Solution:\n"
                f"  sudo pacman -S python-onnxruntime-cuda\n"
                f"  Or for CPU-only:\n"
                f"    sudo pacman -S python-onnxruntime-cpu"
            )
        else:
            return False, (
                f"Failed to import rapidocr: {error_msg}\n\n"
                f"Please ensure rapidocr is installed for Python {python_version}."
            )


def _check_gtk_dependencies() -> bool:
    """Check if GTK dependencies are available.

    Returns:
        True if dependencies are met, False otherwise
    """
    try:
        import gi

        gi.require_version("Gtk", "4.0")
        gi.require_version("Adw", "1")

        from gi.repository import (
            Adw,  # noqa: F401
            Gtk,  # noqa: F401
        )

        return True
    except (ImportError, ValueError) as e:
        # We can't use translations yet as dependencies are missing
        print(f"Error: Missing dependencies: {e}", file=sys.stderr)
        print("Please make sure GTK4 and libadwaita are installed", file=sys.stderr)
        return False


def main() -> int:
    """Main entry point for the application.

    Returns:
        The application exit code.
    """
    # 1. Setup Python compatibility first
    _setup_python_compatibility()

    # 2. Setup Locale
    setup_i18n()

    # 3. Import modules that might depend on compatibility/locale
    from bigocrpdf.application import BigOcrPdfApp
    from bigocrpdf.config import (
        CONFIG_DIR,
        SELECTED_FILE_PATH,
        setup_environment,
    )
    from bigocrpdf.utils.logger import logger

    # 4. Setup environment and parse command line arguments
    args = setup_environment()

    # 5. Check dependencies
    # Check GTK first as we need it for UI
    if not _check_gtk_dependencies():
        return 1

    # Check OCR dependencies next
    ocr_ok, ocr_error = _check_ocr_dependencies()
    if not ocr_ok:
        logger.error(f"OCR Dependency Error: {ocr_error}")
        # Continue anyway - the GUI can still show the error gracefully or run in limited mode
        print(f"\n*** OCR Dependency Error ***\n{ocr_error}\n", file=sys.stderr)

    # 6. Set up configuration directory
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # 7. Always start with a clean file queue
    try:
        if os.path.exists(SELECTED_FILE_PATH):
            os.remove(SELECTED_FILE_PATH)
            logger.info("Cleared file queue at startup")
    except Exception as e:
        logger.error(f"Error clearing file queue: {e}")

    # 8. Run application
    try:
        # Initialize the GTK application
        app = BigOcrPdfApp()

        # Add files from command line if provided
        if hasattr(args, "files") and args.files:
            logger.debug(f"Files provided in arguments: {args.files}")

        # Run the application
        return app.run(sys.argv)
    except Exception as e:
        logger.error(f"Critical error starting application: {e}")
        return 1


__all__ = ["main", "__version__", "__author__", "__license__", "setup_i18n"]


if __name__ == "__main__":
    sys.exit(main())
