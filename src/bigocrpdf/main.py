#!/usr/bin/env python3
"""
BigOcrPdf - Main Module

This is the main entry point for the BigOcrPdf application.
"""

import locale
import os
import sys


# Try to import ocrmypdf, and if it fails, look for it in other python versions
# This is a resilience fix for when system python is updated but packages are still in old version folders
try:
    import ocrmypdf
except ImportError:
    import glob

    # Common path pattern: /usr/lib/python3.*/site-packages/ocrmypdf
    # Sort reverse to try newer versions first (works for 3.14 vs 3.13)
    for path in sorted(
        glob.glob("/usr/lib/python3.*/site-packages/ocrmypdf"), reverse=True
    ):
        site_pkg = os.path.dirname(path)
        if site_pkg not in sys.path:
            sys.path.append(site_pkg)
            try:
                import ocrmypdf

                # If we are here, we found it
                break
            except ImportError:
                # If import still fails (e.g. missing dependencies), remove path and continue
                if site_pkg in sys.path:
                    sys.path.remove(site_pkg)

# After i18n is set up, import other modules that may use translations
from bigocrpdf.application import BigOcrPdfApp
from bigocrpdf.config import (
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

    try:
        # Initialize the GTK application
        app = BigOcrPdfApp()

        # Add files from command line if provided
        if hasattr(args, "files") and args.files:
            # Files are passed to the running instance via Gio.Application default handling
            # or on_open if this is the primary instance.
            logger.debug(f"Files provided in arguments: {args.files}")

        # Run the application
        return app.run(sys.argv)
    except Exception as e:
        logger.error(f"{_('Critical error starting application')}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
