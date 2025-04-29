#!/usr/bin/env python3
"""
BigOcrPdf - Main Module

This is the main entry point for the BigOcrPdf application.
"""

import os
import sys

# Add the parent directory to Python path to make imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute imports instead of relative imports
from bigocrpdf.utils.i18n import _
from bigocrpdf.app import BigOcrPdfApp
from bigocrpdf.config import CONFIG_DIR, SELECTED_FILE_PATH
from bigocrpdf.utils.logger import logger


def main() -> int:
    """Main function

    Returns:
        The application exit code
    """
    # Set up configuration directory
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # Always start with a clean file queue
    try:
        # Clear the selected files at startup
        if os.path.exists(SELECTED_FILE_PATH):
            os.remove(SELECTED_FILE_PATH)
            logger.info("Cleared file queue at startup")
    except Exception as e:
        logger.error(f"Error clearing file queue: {e}")

    # Initialize the GTK application
    app = BigOcrPdfApp()

    # Run the application
    return app.run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
