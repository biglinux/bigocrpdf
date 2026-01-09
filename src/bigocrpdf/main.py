#!/usr/bin/env python3
"""
BigOcrPdf - Main Module

This is the main entry point for the BigOcrPdf application.
"""

import locale
import os
import sys

# After i18n is set up, import other modules that may use translations
from bigocrpdf.application import BigOcrPdfApp
from bigocrpdf.config import (
    APP_ID,
    CONFIG_DIR,
    SELECTED_FILE_PATH,
    parse_command_line,
    setup_environment,
)

# Then import and initialize i18n before any other module that uses translations
from bigocrpdf.utils.i18n import _

# First import and configure logger
from bigocrpdf.utils.logger import logger


def check_dependencies() -> bool:
    """Check if all required dependencies are installed.

    Returns:
        True if all dependencies are met, False otherwise
    """
    try:
        # Check for GTK - import only inside this function
        import gi

        gi.require_version("Gtk", "4.0")
        gi.require_version("Adw", "1")

        # Just check that these can be imported
        # Check for OCRmyPDF - import only inside this function
        import ocrmypdf  # noqa: F401
        from gi.repository import (
            Adw,  # noqa: F401
            Gtk,  # noqa: F401
        )

        return True
    except (ImportError, ValueError) as e:
        print(f"{_('Error: Missing dependencies')}: {e}")
        print(_("Please make sure GTK4, libadwaita, and OCRmyPDF are installed"))
        return False


def setup_locale() -> None:
    """Setup localization."""
    try:
        locale.setlocale(locale.LC_ALL, "")
        logger.info(f"Set locale to: {locale.getlocale()}")
    except locale.Error:
        # Fallback to C locale
        locale.setlocale(locale.LC_ALL, "C")
        logger.warning(_("Failed to set system locale, falling back to C locale"))


def main() -> int:
    """Main function.

    Returns:
        The application exit code
    """
    # Setup locale first, before other initialization
    setup_locale()

    # Check if we should run in image mode (must be done BEFORE argument parsing)
    # 1. Explicit flag (from wrapper script)
    image_mode = "--image-mode" in sys.argv
    if image_mode:
        sys.argv.remove("--image-mode")

    # Setup environment and parse command line arguments
    setup_environment()
    args = parse_command_line()

    # Check dependencies
    if not check_dependencies():
        return 1

    # Set up configuration directory
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # Always start with a clean file queue
    try:
        # Clear the selected files at startup
        if os.path.exists(SELECTED_FILE_PATH):
            os.remove(SELECTED_FILE_PATH)
            logger.info(_("Cleared file queue at startup"))
    except Exception as e:
        logger.error(f"{_('Error clearing file queue')}: {e}")

    # 2. Heuristic: Check if arguments contain images and no PDFs
    if not image_mode and len(sys.argv) > 1:
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        has_image = False
        has_pdf = False

        for arg in sys.argv[1:]:
            # Simple extension check, not perfect but fast
            lower_arg = arg.lower()
            if lower_arg.endswith(".pdf"):
                has_pdf = True
                break
            elif any(lower_arg.endswith(ext) for ext in image_exts):
                has_image = True

        if has_image and not has_pdf:
            image_mode = True

    # Determine Application ID based on context
    app_id = APP_ID
    if image_mode:
        from bigocrpdf.config import IMAGE_APP_ID

        app_id = IMAGE_APP_ID
        logger.info(f"Starting with Image App ID: {app_id}")

    try:
        # Initialize the GTK application
        app = BigOcrPdfApp(application_id=app_id)

        # Run the application
        return app.run(sys.argv)
    except Exception as e:
        logger.error(f"{_('Critical error starting application')}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
