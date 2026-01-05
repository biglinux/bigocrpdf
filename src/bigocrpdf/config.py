#!/usr/bin/env python3
"""
BigOcrPdf - Configuration Module

This module contains all configuration constants and paths used by the application.
"""

import argparse
import logging
import os
import sys
from typing import Final

# Import gettext for translatable strings (lazy import to avoid circular)
from bigocrpdf.utils.i18n import _

# ============================================================================
# Application Constants
# ============================================================================

APP_NAME: Final[str] = "Big OCR PDF"
APP_ID: Final[str] = "br.com.biglinux.bigocrpdf"
APP_VERSION: Final[str] = "2.0.0"
APP_DESCRIPTION: Final[str] = _("Add OCR to your PDF documents to make them searchable")
APP_WEBSITE: Final[str] = "https://www.biglinux.com.br"
APP_ISSUES: Final[str] = "https://github.com/biglinux/bigocrpdf/issues"
APP_DEVELOPERS: Final[list[str]] = ["BigLinux <biglinux.com.br>"]
APP_ICON_NAME: Final[str] = "bigocrpdf"


# ============================================================================
# Environment Detection
# ============================================================================

IS_DEVELOPMENT: Final[bool] = not getattr(sys, "frozen", False)


# ============================================================================
# Paths
# ============================================================================

if IS_DEVELOPMENT:
    BASE_DIR: Final[str] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESOURCES_DIR: Final[str] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "resources"
    )
else:
    # When running as installed application
    BASE_DIR: Final[str] = os.path.join(sys.prefix, "share", "bigocrpdf")
    RESOURCES_DIR: Final[str] = os.path.join(BASE_DIR, "resources")


# ============================================================================
# Configuration Directory
# ============================================================================

CONFIG_DIR: Final[str] = os.path.expanduser("~/.config/bigocrpdf")
SELECTED_FILE_PATH: Final[str] = os.path.join(CONFIG_DIR, "selected-file")
LANG_FILE_PATH: Final[str] = os.path.join(CONFIG_DIR, "lang")
QUALITY_FILE_PATH: Final[str] = os.path.join(CONFIG_DIR, "quality")
ALIGN_FILE_PATH: Final[str] = os.path.join(CONFIG_DIR, "align")
SAVEFILE_PATH: Final[str] = os.path.join(CONFIG_DIR, "savefile")
SAME_FOLDER_PATH: Final[str] = os.path.join(CONFIG_DIR, "same-folder")

# Ensure configuration directory exists
os.makedirs(CONFIG_DIR, exist_ok=True)


# ============================================================================
# Logging Configuration
# ============================================================================

LOG_LEVEL: int = logging.INFO
LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOGGER_NAME: Final[str] = "BigOcrPdf"


# ============================================================================
# Window Configuration
# ============================================================================

DEFAULT_WINDOW_WIDTH: Final[int] = 820
DEFAULT_WINDOW_HEIGHT: Final[int] = 600
WINDOW_STATE_KEY: Final[str] = "window"


# ============================================================================
# UI Constants
# ============================================================================

# Markup formatting constants
MARKUP_SPAN_SMALL_OPEN: Final[str] = "<span font_size='small'>"
MARKUP_SPAN_CLOSE: Final[str] = "</span>"

# Signal name constants
SIGNAL_NOTIFY_ACTIVE: Final[str] = "notify::active"

# Default label for unset values
LABEL_NOT_SET: Final[str] = "Not set"


# ============================================================================
# Global Settings (can be overridden via command line)
# ============================================================================

DEBUG_MODE: bool = False
VERBOSE_MODE: bool = False


def parse_command_line() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description=APP_DESCRIPTION)
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        help=_("Print version information and exit"),
    )
    parser.add_argument("-d", "--debug", action="store_true", help=_("Enable debug mode"))
    parser.add_argument("--verbose", action="store_true", help=_("Enable verbose output"))
    parser.add_argument("files", nargs="*", help=_("PDF files to process"))

    return parser.parse_args()


def setup_environment() -> None:
    """Configure environment variables and settings."""
    global DEBUG_MODE, VERBOSE_MODE, LOG_LEVEL

    # Parse command line arguments
    args = parse_command_line()

    # Handle version flag
    if args.version:
        print(f"{APP_NAME} {APP_VERSION}")
        sys.exit(0)

    # Set debug mode
    if args.debug:
        DEBUG_MODE = True
        LOG_LEVEL = logging.DEBUG

    # Set verbose mode
    if args.verbose:
        VERBOSE_MODE = True

    # Create config directory if it doesn't exist
    os.makedirs(CONFIG_DIR, exist_ok=True)
