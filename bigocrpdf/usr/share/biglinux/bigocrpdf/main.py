#!/usr/bin/env python3
"""
BigOcrPdf - Main Module

This is the main entry point for the BigOcrPdf application.
"""

import os
import sys
from utils.i18n import _
from utils.logger import logger
from config import CONFIG_DIR, SELECTED_FILE_PATH, setup_environment, parse_command_line
from app import BigOcrPdfApp


def main() -> int:
    """Main function

    Returns:
        The application exit code
    """

    # Setup environment and parse command line arguments
    setup_environment()
    args = parse_command_line()

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

    # Initialize the GTK application
    app = BigOcrPdfApp()

    # Add files from command line if provided
    if hasattr(args, "files") and args.files:
        # We'll handle this in the activate signal by passing
        # the files to the application
        pass

    # Run the application
    return app.run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
