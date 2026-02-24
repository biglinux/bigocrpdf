"""Bi-level image analysis and binarization for JBIG2/CCITT compression.

Detects whether a processed image is suitable for 1-bit compression
(JBIG2 or CCITT Group 4) and performs binarization when appropriate.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Minimum ratio of near-black + near-white pixels to consider bilevel
_DEFAULT_BILEVEL_THRESHOLD = 0.95

# Pixel intensity margins for "near black" and "near white"
_DARK_CUTOFF = 30
_LIGHT_CUTOFF = 225


def is_bilevel_candidate(
    img: np.ndarray,
    threshold: float = _DEFAULT_BILEVEL_THRESHOLD,
) -> bool:
    """Check if an image is essentially bi-level (suitable for JBIG2/CCITT).

    An image is considered bilevel when the vast majority of its pixels
    are near-black or near-white, which is typical of scanned text
    documents, forms, and line art.

    Args:
        img: OpenCV image (BGR or grayscale).
        threshold: Minimum ratio of near-black + near-white pixels (0-1).
            Default 0.95 is conservative to avoid quality loss.

    Returns:
        True if the image is a good bilevel candidate.
    """
    gray = _to_grayscale(img)

    near_black = int(np.count_nonzero(gray < _DARK_CUTOFF))
    near_white = int(np.count_nonzero(gray > _LIGHT_CUTOFF))
    total = gray.size

    if total == 0:
        return False

    bilevel_ratio = (near_black + near_white) / total
    logger.debug(
        "Bilevel analysis: %.1f%% near-black/white (threshold=%.0f%%)",
        bilevel_ratio * 100,
        threshold * 100,
    )
    return bilevel_ratio >= threshold


def binarize(img: np.ndarray) -> np.ndarray:
    """Convert image to binary (0 or 255) using Otsu's method.

    Otsu's global thresholding works well for cleanly scanned documents.
    The result is a single-channel uint8 image with only 0 and 255 values,
    ready for JBIG2 or CCITT encoding.

    Args:
        img: OpenCV image (BGR or grayscale).

    Returns:
        Binary uint8 image (single channel, values 0 or 255).
    """
    gray = _to_grayscale(img)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale if needed, otherwise return as-is."""
    if len(img.shape) == 3 and img.shape[2] >= 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
