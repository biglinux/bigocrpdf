"""Pytest configuration for bigocrpdf tests.

Pins the process to untranslated messages, and keeps an installed OCR engine
out of the tests that do not ask for it.
"""

import os

# Tests assert on the untranslated msgids, so no catalog may be selected.
# Without this the suite passes or fails according to the developer's desktop
# language.
#
# Only LANGUAGE is set, and only to "C".  LANG and LC_ALL stay as they are:
# they also drive the text encoding, and forcing them to C breaks the
# tests that round-trip non-ASCII through pdftotext.  "C" rather than "en"
# because an English catalog is installed and would be selected.
#
# This must run before any bigocrpdf import -- gettext caches its translation
# objects, so a module that resolves a string first would keep the catalog.
os.environ["LANGUAGE"] = "C"

import pytest


@pytest.fixture(autouse=True)
def _no_installed_ocr_engine(request, monkeypatch):
    """Keep a ppocr-native installed on the test host out of every test but
    the ``real_ocr`` ones.

    A test that needs the text detector, deskew and photo perspective among
    them, carries ``real_ocr``; other tests that exercise the engine patch
    ``find_worker`` themselves.
    """
    if request.node.get_closest_marker("real_ocr"):
        return
    from bigocrpdf.services.ocr import ppocr_native

    monkeypatch.setattr(ppocr_native, "find_worker", lambda: None)
