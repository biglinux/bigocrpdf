#!/usr/bin/env python3
"""
BigOcrPdf - Main Module

This is the main entry point for the BigOcrPdf application.
"""

import os
import sys
import locale
import logging

# Add the parent directory to Python path to make imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core modules
from utils.i18n import _
from app import BigOcrPdfApp
from config import CONFIG_DIR, SELECTED_FILE_PATH, setup_environment, parse_command_line
from utils.logger import logger


def check_dependencies() -> bool:
    """Check if all required dependencies are installed
    
    Returns:
        True if all dependencies are met, False otherwise
    """
    try:
        # Check for GTK
        import gi
        gi.require_version("Gtk", "4.0")
        gi.require_version("Adw", "1")
        from gi.repository import Gtk, Adw
        
        # Check for OCRmyPDF
        import ocrmypdf
        
        return True
    except (ImportError, ValueError) as e:
        print(f"{_('Error: Missing dependencies')}: {e}")
        print(_("Please make sure GTK4, libadwaita, and OCRmyPDF are installed"))
        return False


def setup_locale() -> None:
    """Setup localization"""
    try:
        locale.setlocale(locale.LC_ALL, '')
    except locale.Error:
        # Fallback to C locale
        locale.setlocale(locale.LC_ALL, 'C')
        logger.warning(_("Failed to set system locale, falling back to C locale"))


def main() -> int:
    """Main function

    Returns:
        The application exit code
    """
    # Setup environment and parse command line arguments
    setup_environment()
    args = parse_command_line()
    
    # Set up localization
    setup_locale()
    
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