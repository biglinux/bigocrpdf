#!/usr/bin/env python3
"""
BigOcrPdf - Configuration Module

This module contains all configuration constants and paths used by the application.
"""

import os
import logging
import argparse
import sys

# Import gettext for translatable strings
from utils.i18n import _

# Application information
APP_NAME = "Big OCR PDF"
APP_ID = "br.com.biglinux.bigocrpdf"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = _("Add OCR to your PDF documents to make them searchable")
APP_WEBSITE = "https://www.biglinux.com.br"
APP_ISSUES = "https://github.com/biglinux/bigocrpdf/issues"
APP_DEVELOPERS = ["Biglinux <biglinux.com.br>"]

# Determine if we're running in a development environment
IS_DEVELOPMENT = not getattr(sys, 'frozen', False)

# Paths
if IS_DEVELOPMENT:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESOURCES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")
else:
    # When running as installed application
    BASE_DIR = os.path.join(sys.prefix, "share", "bigocrpdf")
    RESOURCES_DIR = os.path.join(BASE_DIR, "resources")

APP_ICON_NAME = "bigocrpdf"

# Configuration directory
CONFIG_DIR = os.path.expanduser("~/.config/bigocrpdf")
SELECTED_FILE_PATH = os.path.join(CONFIG_DIR, "selected-file")
LANG_FILE_PATH = os.path.join(CONFIG_DIR, "lang")
QUALITY_FILE_PATH = os.path.join(CONFIG_DIR, "quality")
ALIGN_FILE_PATH = os.path.join(CONFIG_DIR, "align")
SAVEFILE_PATH = os.path.join(CONFIG_DIR, "savefile")
SUFFIX_FILE_PATH = os.path.join(CONFIG_DIR, "suffix")
SAME_FOLDER_PATH = os.path.join(CONFIG_DIR, "same-folder")

# Ensure configuration directory exists
os.makedirs(CONFIG_DIR, exist_ok=True)

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOGGER_NAME = "BigOcrPdf"

# Global settings (can be overridden via command line)
DEBUG_MODE = False
VERBOSE_MODE = False


def parse_command_line() -> argparse.Namespace:
    """Parse command line arguments
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description=APP_DESCRIPTION)
    parser.add_argument('-v', '--version', action='store_true', 
                        help=_('Print version information and exit'))
    parser.add_argument('-d', '--debug', action='store_true', 
                        help=_('Enable debug mode'))
    parser.add_argument('--verbose', action='store_true', 
                        help=_('Enable verbose output'))
    parser.add_argument('files', nargs='*', 
                        help=_('PDF files to process'))
    
    return parser.parse_args()


def setup_environment() -> None:
    """Configure environment variables and settings"""
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