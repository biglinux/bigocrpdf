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

    hough_vote, angles = _hough_orientation_vote(gray)
    energy_vote = _edge_energy_vote(gray)
    aspect_vote = 1 if w > h else -1
    rotation_score = hough_vote + energy_vote + aspect_vote

    # Aspect ratio and edge energy cannot distinguish a legitimate landscape
    # page from sideways text. Require direct line-orientation evidence before
    # applying a destructive quarter-turn.
    if hough_vote > 0 and rotation_score >= 2:
        return _rotation_direction(gray, angles, rotation_score)

    return 0


def _hough_orientation_vote(gray: np.ndarray) -> tuple[int, np.ndarray | None]:
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    if lines is None or len(lines) <= 20:
        return 0, None

    segments = np.asarray(lines).reshape(-1, 4)
    angles = np.degrees(
        np.arctan2(segments[:, 3] - segments[:, 1], segments[:, 2] - segments[:, 0])
    )
    n_horizontal = int(np.sum(np.abs(angles) < 30) + np.sum(np.abs(angles) > 150))
    n_vertical = int(np.sum((np.abs(angles) > 60) & (np.abs(angles) < 120)))
    total = n_horizontal + n_vertical
    if total == 0:
        return 0, angles

    vert_ratio = n_vertical / total
    if vert_ratio > 0.55:
        return (2 if vert_ratio > 0.80 else 1), angles
    if vert_ratio < 0.45:
        return -1, angles
    return 0, angles


def _edge_energy_vote(gray: np.ndarray) -> int:
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    horizontal_energy = np.sum(np.abs(sobely))
    vertical_energy = np.sum(np.abs(sobelx))
    energy_ratio = vertical_energy / (horizontal_energy + 1e-10)

    if energy_ratio > 1.15:
        return 1
    if energy_ratio < 0.85:
        return -1
    return 0


def _rotation_direction(
    gray: np.ndarray,
    angles: np.ndarray | None,
    rotation_score: int,
) -> int:
    angle_direction = _rotation_direction_from_angles(angles, rotation_score)
    if angle_direction:
        return angle_direction
    return _rotation_direction_from_density(gray, rotation_score)


def _rotation_direction_from_angles(angles: np.ndarray | None, rotation_score: int) -> int:
    if angles is None:
        return 0

    vert_mask = (np.abs(angles) > 60) & (np.abs(angles) < 120)
    vert_angles = angles[vert_mask]
    if len(vert_angles) <= 10:
        return 0

    n_positive = int(np.sum(vert_angles > 0))
    n_negative = int(np.sum(vert_angles < 0))
    if n_negative > n_positive:
        logger.info(f"Orientation detected: 270° CW (score={rotation_score})")
        return 270

    logger.info(f"Orientation detected: 90° CW (score={rotation_score})")
    return 90


def _rotation_direction_from_density(gray: np.ndarray, rotation_score: int) -> int:
    _, w = gray.shape
    left_half = gray[:, : w // 2]
    right_half = gray[:, w // 2 :]
    left_density = np.mean(left_half < 200)
    right_density = np.mean(right_half < 200)

    if left_density > right_density:
        logger.info(f"Orientation detected: 270° CW (score={rotation_score}, fallback)")
        return 270

    logger.info(f"Orientation detected: 90° CW (score={rotation_score}, fallback)")
    return 90


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
