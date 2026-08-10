"""Pytest configuration for bigocrpdf tests.

Captures references to real numpy and cv2 modules BEFORE any test module
mocks them via sys.modules.  The references are registered as pytest
fixtures for tests that need real numerical computation.

Also pins the process to untranslated messages -- see ``_force_untranslated``.
"""

import os

# Tests assert on the untranslated msgids, so no catalog may be selected.
# Without this the suite passes or fails according to the developer's desktop
# language, which is how 15 tests came to fail on a pt_BR machine and nowhere
# else.
#
# Only LANGUAGE is set, and only to "C".  LANG and LC_ALL are deliberately left
# alone: they also drive the text encoding, and forcing them to C breaks the
# tests that round-trip non-ASCII through pdftotext.  "C" rather than "en"
# because an English catalog is installed and would be selected.
#
# This must run before any bigocrpdf import -- gettext caches its translation
# objects, so a module that resolves a string first would keep the catalog.
os.environ["LANGUAGE"] = "C"

import cv2 as _real_cv2  # noqa: E402
import numpy as _real_numpy  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def real_numpy():
    """Provide the real numpy module (not MagicMock)."""
    return _real_numpy


@pytest.fixture
def real_cv2():
    """Provide the real cv2 module (not MagicMock)."""
    return _real_cv2


# Also store as module-level attributes for setUpClass access
REAL_NUMPY = _real_numpy
REAL_CV2 = _real_cv2
