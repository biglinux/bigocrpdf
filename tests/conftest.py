"""Pytest configuration for bigocrpdf tests.

Captures references to real numpy and cv2 modules BEFORE any test module
mocks them via sys.modules.  The references are registered as pytest
fixtures for tests that need real numerical computation.
"""

import cv2 as _real_cv2
import numpy as _real_numpy
import pytest


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
