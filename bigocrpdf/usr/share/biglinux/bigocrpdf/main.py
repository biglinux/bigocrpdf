#!/usr/bin/env python3
"""
BigOcrPdf - Main Module

This is the main entry point for the BigOcrPdf application.
"""

import os
import sys
import locale

# Add the parent directory to Python path to make imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Then import and initialize i18n before any other module that uses translations
from utils.i18n import _

# First import and configure logger
from utils.logger import logger

# After i18n is set up, import other modules that may use translations
from config import CONFIG_DIR, SELECTED_FILE_PATH, setup_environment, parse_command_line
from app import BigOcrPdfApp


def check_dependencies() -> bool:
    """Check if all required dependencies are installed
    
    Returns:
        True if all dependencies are met, False otherwise
    """
    try:
        # Check for GTK - import only inside this function
        import gi
        gi.require_version("Gtk", "4.0")
        gi.require_version("Adw", "1")
        
        # Just check that these can be imported
        from gi.repository import Gtk  # noqa
        from gi.repository import Adw  # noqa
        
        # Check for OCRmyPDF - import only inside this function
        import ocrmypdf  # noqa
        
        return True
    except (ImportError, ValueError) as e:
        print(f"{_('Error: Missing dependencies')}: {e}")
        print(_("Please make sure GTK4, libadwaita, and OCRmyPDF are installed"))
        return False


def setup_locale() -> None:
    """Setup localization"""
    try:
        locale.setlocale(locale.LC_ALL, '')
        logger.info(f"Set locale to: {locale.getlocale()}")
    except locale.Error:
        # Fallback to C locale
        locale.setlocale(locale.LC_ALL, 'C')
        logger.warning(_("Failed to set system locale, falling back to C locale"))


def main() -> int:
    """Main function

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
        if hasattr(args, 'files') and args.files:
            # We'll handle this in the activate signal by passing
            # the files to the application
            pass
        
        # Run the application
        return app.run(sys.argv)
    except Exception as e:
        logger.error(f"{_('Critical error starting application')}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())