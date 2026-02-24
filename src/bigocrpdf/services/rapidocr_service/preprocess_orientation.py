"""
Orientation detection and correction for document images.

Detects whether a document is rotated 0°, 90°, 180°, or 270° and provides
correction. Uses a multi-signal approach: Hough lines, edge energy, and
aspect ratio analysis.

Extracted from preprocessor.py to follow single-responsibility principle.
"""

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from bigocrpdf.services.rapidocr_service.config import OCRConfig

logger = logging.getLogger(__name__)


def detect_orientation(img: np.ndarray, config: "OCRConfig") -> int:
    """Detect document orientation (0, 90, 180, or 270 degrees).

    Uses a multi-signal approach:
    1. Aspect ratio check — documents are typically portrait
    2. Hough line analysis — text lines indicate reading direction
    3. Edge energy ratio — fallback for simple cases

    Args:
        img: Input image in BGR format
        config: OCR configuration object

    Returns:
        Angle to rotate to correct orientation (CW rotation needed)
    """
    if not config.enable_orientation_detection:
        return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    is_landscape = w > h

    # Signal 1: Hough line direction analysis
    # Text lines in a correctly-oriented document run horizontally
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    hough_vote = 0  # positive = needs rotation, negative = correct
    angles = None
    if lines is not None and len(lines) > 20:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
        angles = np.array(angles)
        # Count near-horizontal (text lines) vs near-vertical
        n_horizontal = int(np.sum(np.abs(angles) < 30) + np.sum(np.abs(angles) > 150))
        n_vertical = int(np.sum((np.abs(angles) > 60) & (np.abs(angles) < 120)))
        total = n_horizontal + n_vertical
        if total > 0:
            vert_ratio = n_vertical / total
            # If > 55% of lines are vertical, text is probably rotated
            if vert_ratio > 0.55:
                hough_vote = 1
            elif vert_ratio < 0.45:
                hough_vote = -1

    # Signal 2: Edge energy ratio (existing approach, relaxed threshold)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    horizontal_energy = np.sum(np.abs(sobely))
    vertical_energy = np.sum(np.abs(sobelx))
    energy_ratio = vertical_energy / (horizontal_energy + 1e-10)

    energy_vote = 0
    if energy_ratio > 1.15:
        energy_vote = 1  # More vertical edges → text is rotated
    elif energy_ratio < 0.85:
        energy_vote = -1  # More horizontal edges → text is correct

    # Signal 3: Aspect ratio (documents are usually portrait)
    aspect_vote = 1 if is_landscape else -1

    # Combine signals: need at least 2 out of 3 agreeing for rotation
    rotation_score = hough_vote + energy_vote + aspect_vote

    if rotation_score >= 2:
        # Needs rotation — determine direction (90 or 270)
        if lines is not None and len(lines) > 20 and angles is not None:
            vert_mask = (np.abs(angles) > 60) & (np.abs(angles) < 120)
            vert_angles = angles[vert_mask]
            if len(vert_angles) > 10:
                n_positive = int(np.sum(vert_angles > 0))
                n_negative = int(np.sum(vert_angles < 0))
                if n_negative > n_positive:
                    logger.info(f"Orientation detected: 270° CW (score={rotation_score})")
                    return 270
                else:
                    logger.info(f"Orientation detected: 90° CW (score={rotation_score})")
                    return 90

        # Fallback: use left/right ink density
        left_half = gray[:, : w // 2]
        right_half = gray[:, w // 2 :]
        left_density = np.mean(left_half < 200)
        right_density = np.mean(right_half < 200)

        if left_density > right_density:
            logger.info(f"Orientation detected: 270° CW (score={rotation_score}, fallback)")
            return 270
        else:
            logger.info(f"Orientation detected: 90° CW (score={rotation_score}, fallback)")
            return 90

    return 0


def correct_orientation(img: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image to correct orientation.

    Args:
        img: Input image
        angle: Rotation angle (0, 90, 180, or 270)

    Returns:
        Rotated image
    """
    if angle == 0:
        return img
    elif angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img
