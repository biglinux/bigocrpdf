#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BigOcrPdf - Internationalization Module

This module initializes gettext for internationalization support
"""

import gettext
import os
import locale
import sys
from utils.logger import logger

def setup_i18n():
    """Initialize the internationalization system
    
    Returns:
        The configured translation function
    """
    try:
        # Try to set the system locale
        try:
            locale.setlocale(locale.LC_ALL, '')
            current_locale = locale.getlocale()
            logger.info(f"Current locale: {current_locale}")
        except Exception as e:
            logger.warning(f"Error setting locale: {e}")
            # Fallback to C locale
            locale.setlocale(locale.LC_ALL, 'C')
        
        # Determine the locale directory
        # Check multiple locations where translation files might be
        locale_dirs = [
            # System-wide location
            "/usr/share/locale",
            
            # App-specific location when installed
            os.path.join(sys.prefix, "share", "locale"),
            
            # Development location
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "locale"),
            
            # BigLinux specific location (from your tree structure)
            "/usr/share/biglinux/bigocrpdf/locale"
        ]
        
        # Log where we're looking for translations
        for locale_dir in locale_dirs:
            if os.path.exists(locale_dir):
                logger.info(f"Found locale directory: {locale_dir}")
                # Check if our translation domain exists in this directory
                if os.path.exists(os.path.join(locale_dir, "pt", "LC_MESSAGES", "bigocrpdf.mo")):
                    logger.info(f"Found Portuguese translation in: {locale_dir}")
                # Configure gettext to look for translations in this directory
                gettext.bindtextdomain("bigocrpdf", locale_dir)
        
        # Set the text domain
        gettext.textdomain("bigocrpdf")
        
        # Return the translation function
        return gettext.gettext
    
    except Exception as e:
        logger.error(f"Error setting up i18n: {e}")
        # Return a dummy translation function in case of error
        return lambda x: x

# Initialize the translation function
_ = setup_i18n()

# Test the translation
try:
    test_msg = _("Test message")
    logger.info(f"Translation test: '{test_msg}'")
    if test_msg == "Test message":
        logger.warning("Translation may not be working - test message was not translated")
except Exception as e:
    logger.error(f"Error testing translation: {e}")