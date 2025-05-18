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

# NÃO IMPORTE logger aqui!

# Primeiro, crie uma função simples para fallback
def _dummy_translate(text):
    return text

# Inicialize _ com a função dummy
_ = _dummy_translate

# Configure gettext
try:
    # Try to set the system locale
    try:
        locale.setlocale(locale.LC_ALL, '')
    except Exception:
        # Fallback to C locale
        locale.setlocale(locale.LC_ALL, 'C')
    
    # Determine the locale directory
    # Check multiple locations where translation files might be
    locale_dirs = [
        "/usr/share/locale",
        os.path.join(sys.prefix, "share", "locale"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "locale"),
        "/usr/share/biglinux/bigocrpdf/locale"
    ]
    
    for locale_dir in locale_dirs:
        if os.path.exists(locale_dir):
            # Binds the domain to the locale dir
            gettext.bindtextdomain("bigocrpdf", locale_dir)
    
    # Set the text domain
    gettext.textdomain("bigocrpdf")
    
    # Update _ to use the real gettext
    _ = gettext.gettext
    
except Exception:
    # Keep using the dummy function if there's an error
    pass

def setup_i18n():
    """Reinitialize the internationalization system if needed
    
    Returns:
        The translation function
    """
    return _