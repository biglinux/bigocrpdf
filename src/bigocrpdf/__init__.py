"""
BigOcrPdf - Python package for adding OCR to PDF files

allow-noisy-log: startup dependency diagnostics are user-facing stderr output.

This package provides a GTK4 application for adding OCR to PDF files,
making them searchable and their text selectable.
"""

import os
import sys
from dataclasses import dataclass

from bigocrpdf.utils.i18n import setup_i18n

__version__ = "3.0.0"

# The real floor, not the version we develop against.  Widgets introduced after
# this point are used through bigocrpdf.utils.adw_compat, which falls back when
# they are absent -- an AppImage runs on whatever its build container shipped,
# and Ubuntu 24.04 carries GTK 4.14 with libadwaita 1.5.
_MIN_GTK_VERSION = (4, 14)
_MIN_ADW_VERSION = (1, 5)


@dataclass(frozen=True)
class OcrDependencyState:
    """Resolved startup availability of the OCR engine."""

    is_available: bool
    error: str = ""


def _get_install_cmd(package: str) -> str:
    """Return a distro-appropriate install command for the given package."""
    import shutil

    if shutil.which("pacman"):
        return f"sudo pacman -S {package}"
    if shutil.which("apt"):
        return f"sudo apt install {package}"
    if shutil.which("dnf"):
        return f"sudo dnf install {package}"
    return f"pip install {package}"


def _check_ocr_dependencies() -> tuple[bool, str]:
    """Check if OCR dependencies are available and compatible.

    Returns:
        Tuple of (success, error_message). If success is True, error_message is empty.
    """
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    # Try to import rapidocr
    try:
        from rapidocr import OCRVersion, RapidOCR  # noqa: F401

        if not hasattr(OCRVersion, "PPOCRV6"):
            return False, "Installed RapidOCR does not support PP-OCRv6. Upgrade python-rapidocr."

        from bigocrpdf.services.rapidocr_service.config import OCRConfig

        config = OCRConfig()
        if config.get_det_model_path() is None or config.get_rec_model_path() is None:
            # Naming the directory that was actually searched turns an opaque
            # failure into something a user or packager can act on, which
            # matters most in relocatable builds where it is not /usr/share.
            return False, (
                "Required PP-OCRv6 small models are not installed.\n\n"
                f"Looked in: {config.model_base_path}\n\n"
                "Install the rapidocr-models-v6-small package, or point "
                "BIGOCRPDF_RAPIDOCR_DIR at a directory containing models/ and "
                "fonts/, then restart the application."
            )

        return True, ""
    except ImportError as e:
        error_msg = str(e)

        # Check for common compatibility errors
        if "openvino" in error_msg.lower() or "_pyopenvino" in error_msg:
            cuda_cmd = _get_install_cmd("python-onnxruntime-cuda")
            cpu_cmd = _get_install_cmd("python-onnxruntime-cpu")
            return False, (
                f"OpenVINO is not compatible with Python {python_version}.\n"
                f"The OCR packages were compiled for a different Python version.\n\n"
                f"Solution:\n"
                f"  Install ONNX Runtime as fallback engine:\n"
                f"    {cuda_cmd}\n"
                f"  Or for CPU-only:\n"
                f"    {cpu_cmd}"
            )
        elif "onnxruntime" in error_msg.lower():
            cuda_cmd = _get_install_cmd("python-onnxruntime-cuda")
            cpu_cmd = _get_install_cmd("python-onnxruntime-cpu")
            return False, (
                f"ONNX Runtime is not compatible with Python {python_version}.\n\n"
                f"Solution:\n"
                f"  {cuda_cmd}\n"
                f"  Or for CPU-only:\n"
                f"    {cpu_cmd}"
            )
        else:
            return False, (
                f"Failed to import rapidocr: {error_msg}\n\n"
                f"Please ensure rapidocr is installed for Python {python_version}."
            )


def _check_gtk_dependencies() -> bool:
    """Check if supported GTK dependencies are available.

    Returns:
        True if dependencies are met, False otherwise
    """
    try:
        import gi

        gi.require_version("Gtk", "4.0")
        gi.require_version("Adw", "1")

        from gi.repository import Adw, Gtk

        gtk_version = (
            Gtk.get_major_version(),
            Gtk.get_minor_version(),
            Gtk.get_micro_version(),
        )
        adw_version = (
            Adw.get_major_version(),
            Adw.get_minor_version(),
            Adw.get_micro_version(),
        )

        errors = []
        if gtk_version[:2] < _MIN_GTK_VERSION:
            errors.append(
                f"GTK {_MIN_GTK_VERSION[0]}.{_MIN_GTK_VERSION[1]} or newer is required "
                f"(found {'.'.join(map(str, gtk_version))})"
            )
        if adw_version[:2] < _MIN_ADW_VERSION:
            errors.append(
                f"libadwaita {_MIN_ADW_VERSION[0]}.{_MIN_ADW_VERSION[1]} or newer is required "
                f"(found {'.'.join(map(str, adw_version))})"
            )

        if errors:
            print("Error: Unsupported graphical runtime:", file=sys.stderr)
            for error in errors:
                print(f"  {error}", file=sys.stderr)
            return False

        return True
    except (ImportError, ValueError) as e:
        # We can't use translations yet as dependencies are missing
        print(f"Error: Missing dependencies: {e}", file=sys.stderr)
        print(
            f"Please install GTK {_MIN_GTK_VERSION[0]}.{_MIN_GTK_VERSION[1]} or newer "
            f"and libadwaita {_MIN_ADW_VERSION[0]}.{_MIN_ADW_VERSION[1]} or newer",
            file=sys.stderr,
        )
        return False


def main() -> int:
    """Main entry point for the application.

    Returns:
        The application exit code.
    """
    # Configure locale before building translated UI.
    setup_i18n()

    from bigocrpdf.config import (
        CONFIG_DIR,
        SELECTED_FILE_PATH,
        setup_environment,
    )
    from bigocrpdf.utils.logger import logger, setup_logger

    # Parse arguments before configuring the requested log level.
    args = setup_environment()
    setup_logger()

    # Check for image mode flag
    if getattr(args, "image_mode", False):
        if "--image-mode" in sys.argv:
            sys.argv.remove("--image-mode")
        return main_image()

    # Check GTK first as we need it for UI
    if not _check_gtk_dependencies():
        return 1

    from bigocrpdf.application import BigOcrPdfApp

    # Check OCR dependencies next
    ocr_ok, ocr_error = _check_ocr_dependencies()
    ocr_dependency = OcrDependencyState(is_available=ocr_ok, error=ocr_error)
    if not ocr_ok:
        logger.error(f"OCR Dependency Error: {ocr_error}")
        # Continue anyway - the GUI can still show the error gracefully or run in limited mode
        print(f"\n*** OCR Dependency Error ***\n{ocr_error}\n", file=sys.stderr)

    # Set up configuration directory
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # Always start with a clean file queue
    try:
        if os.path.exists(SELECTED_FILE_PATH):
            os.remove(SELECTED_FILE_PATH)
            logger.info("Cleared file queue at startup")
    except Exception as e:
        logger.error(f"Error clearing file queue: {e}")

    # Run application
    try:
        # Initialize the GTK application
        app = BigOcrPdfApp(ocr_dependency=ocr_dependency)

        # Add files from command line if provided
        if hasattr(args, "files") and args.files:
            logger.debug(f"Files provided in arguments: {args.files}")

        # Run the application
        return app.run(sys.argv)
    except Exception as e:
        logger.error(f"Critical error starting application: {e}")
        return 1


def main_image() -> int:
    """Entry point for the Image OCR application (standalone).

    This uses a separate application_id for proper Wayland taskbar grouping.

    Returns:
        The application exit code.
    """
    setup_i18n()

    from bigocrpdf.utils.logger import logger, setup_logger

    setup_logger()
    if not _check_gtk_dependencies():
        return 1

    from bigocrpdf.image_application import ImageOcrApp

    # Check OCR dependencies
    ocr_ok, ocr_error = _check_ocr_dependencies()
    ocr_dependency = OcrDependencyState(is_available=ocr_ok, error=ocr_error)
    if not ocr_ok:
        logger.error(f"OCR Dependency Error: {ocr_error}")
        print(f"\n*** OCR Dependency Error ***\n{ocr_error}\n", file=sys.stderr)

    try:
        app = ImageOcrApp(ocr_dependency=ocr_dependency)
        return app.run(sys.argv)
    except Exception as e:
        logger.error(f"Critical error starting Image OCR: {e}")
        return 1


__all__ = [
    "OcrDependencyState",
    "main",
    "main_image",
    "__version__",
    "setup_i18n",
]
