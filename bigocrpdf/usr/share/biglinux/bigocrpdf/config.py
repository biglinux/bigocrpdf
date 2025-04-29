#!/usr/bin/env python3
"""
BigOcrPdf - Configuration Module

This module contains all configuration constants and paths used by the application.
"""

import os
import logging

# Import gettext for translatable strings
from .utils.i18n import _

# Application information
APP_NAME = _("Big OCR PDF")
APP_ID = "org.biglinux.bigocrpdf"
APP_VERSION = "1.0"
APP_DESCRIPTION = _("Add OCR to your PDF documents to make them searchable")
APP_WEBSITE = "https://www.biglinux.com.br"
APP_ISSUES = "https://github.com/biglinux/bigocrpdf/issues"
APP_DEVELOPERS = ["Big Linux Team <team@biglinux.com.br>"]

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")
ICON_PATH = os.path.join(BASE_DIR, "icon-big-ocr-pdf.svg")

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
