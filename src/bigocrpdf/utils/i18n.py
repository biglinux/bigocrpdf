#!/usr/bin/env python3
"""
BigOcrPdf - Internationalization Module

This module initializes gettext for internationalization support.
"""

import gettext
import locale
import os
import sys
from collections.abc import Callable


def _dummy_translate(text: str) -> str:
    """Fallback translation function that returns the original text.

    Args:
        text: The text to translate.

    Returns:
        The original text unchanged.
    """
    return text


# Initialize _ with the fallback function
_: Callable[[str], str] = _dummy_translate

# Configure gettext
try:
    # Try to set the system locale
    try:
        locale.setlocale(locale.LC_ALL, "")
        # Keep LC_NUMERIC as C to avoid breaking libraries that expect
        # dot as decimal separator (e.g., onnxruntime used by RapidOCR).
        # Locales like pt_BR use comma, which causes ONNX model parsing failures.
        locale.setlocale(locale.LC_NUMERIC, "C")
    except Exception:
        # Fallback to C locale
        locale.setlocale(locale.LC_ALL, "C")

    # Determine the locale directory
    # Check multiple locations where translation files might be
    locale_dirs = [
        "/usr/share/locale",
        os.path.join(sys.prefix, "share", "locale"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "locale"),
        "/usr/share/biglinux/bigocrpdf/locale",
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


def N_(text: str) -> str:
    """Mark a string for extraction without translating it at definition time.

    Use this for strings that are defined as constants but translated later
    via ``_()``.  Extraction tools (xgettext / pybabel) will pick up the
    string, but it is returned unchanged so the runtime ``_()`` call can
    do the actual translation.
    """
    return text


def setup_i18n() -> Callable[[str], str]:
    """Reinitialize the internationalization system if needed.

    Returns:
        The translation function.
    """
    return _
