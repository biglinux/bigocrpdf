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
        from rapidocr import RapidOCR  # noqa: F401

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

    # Check for image mode flag
    if getattr(args, "image_mode", False):
        if "--image-mode" in sys.argv:
            sys.argv.remove("--image-mode")
        from bigocrpdf.__init__ import main_image

        return main_image()

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
        # Detect --edit / -e early so the app can use a separate D-Bus name
        edit_mode = "--edit" in sys.argv or "-e" in sys.argv

        # Initialize the GTK application
        app = BigOcrPdfApp(edit_mode=edit_mode)

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
    # 1. Setup Python compatibility first
    _setup_python_compatibility()

    # 2. Setup Locale
    setup_i18n()

    # 3. Import the standalone image application
    from bigocrpdf.image_application import ImageOcrApp
    from bigocrpdf.utils.logger import logger

    # 4. Check dependencies
    if not _check_gtk_dependencies():
        return 1

    # Check OCR dependencies
    ocr_ok, ocr_error = _check_ocr_dependencies()
    if not ocr_ok:
        logger.error(f"OCR Dependency Error: {ocr_error}")
        print(f"\n*** OCR Dependency Error ***\n{ocr_error}\n", file=sys.stderr)

    # 5. Run application with its own application_id
    try:
        app = ImageOcrApp()
        return app.run(sys.argv)
    except Exception as e:
        logger.error(f"Critical error starting Image OCR: {e}")
        return 1


__all__ = ["main", "main_image", "__version__", "__author__", "__license__", "setup_i18n"]


if __name__ == "__main__":
    sys.exit(main())
