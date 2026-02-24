"""Skew detection and correction.

Functions for detecting uniform and regional text skew in document images
and applying rotation or mesh-based dewarping to correct it.
"""

import logging

import cv2
import numpy as np

from bigocrpdf.services.perspective_margins import trim_white_borders

logger = logging.getLogger(__name__)


def _detect_line_angles(image: np.ndarray, min_lines: int = 5) -> list[tuple[int, float]] | None:
    """Detect per-line skew angles via text baseline fitting.

    Shared core for both global and regional skew detection.

    Args:
        image: Input BGR image
        min_lines: Minimum number of valid lines required

    Returns:
        List of (y_position, angle) tuples, or None if too few lines
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    horizontal_proj = np.sum(binary, axis=1)
    smooth_proj = gaussian_filter1d(horizontal_proj.astype(float), sigma=3)
    min_height = w * 0.02
    min_distance = max(10, h // 200)
    peaks, _ = find_peaks(smooth_proj, height=min_height, distance=min_distance)

    if len(peaks) < min_lines:
        logger.debug(f"Only {len(peaks)} text lines detected")
        return None

    line_data: list[tuple[int, float]] = []

    for peak_y in peaks:
        strip_height = max(15, h // 150)
        y_start = max(0, peak_y - strip_height // 2)
        y_end = min(h, peak_y + strip_height // 2)
        strip = binary[y_start:y_end, :]

        contours, _ = cv2.findContours(strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 3:
            continue

        centers = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))

        if len(centers) < 3:
            continue

        centers_arr = np.array(centers, dtype=np.float32)
        vx, vy, _, _ = cv2.fitLine(centers_arr, cv2.DIST_L2, 0, 0.01, 0.01)
        angle = np.degrees(np.arctan2(vy[0], vx[0]))

        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90

        if abs(angle) < 15:
            line_data.append((peak_y, angle))

    if len(line_data) < min_lines:
        logger.debug(f"Only {len(line_data)} valid line angles")
        return None

    return line_data


def detect_skew_angle(image: np.ndarray) -> float | None:
    """
    Detect the overall skew angle of a document using text baseline detection.

    Uses horizontal projection to find text lines, then fits lines through
    character centers to determine angles. Returns the median angle.

    Args:
        image: Input BGR image

    Returns:
        Skew angle in degrees (negative = clockwise rotation needed),
        or None if not detected
    """
    line_data = _detect_line_angles(image)
    if line_data is None:
        return None

    skew_angle = float(np.median([a for _, a in line_data]))
    logger.debug(f"Detected skew angle: {skew_angle:.2f}° (from {len(line_data)} lines)")
    return skew_angle


def detect_regional_skew(image: np.ndarray, n_regions: int = 5) -> list[tuple[int, float]] | None:
    """
    Detect skew angles at different vertical regions using text baseline detection.

    Uses horizontal projection to find text lines, then fits a line through
    character centers to determine each line's angle.

    Args:
        image: Input BGR image
        n_regions: Number of vertical regions to aggregate results into

    Returns:
        List of (y_center, angle) tuples, or None if detection fails
    """
    line_data = _detect_line_angles(image)
    if line_data is None:
        return None

    h = image.shape[0]
    region_height = h // n_regions
    region_angles: dict[int, list[float]] = {i: [] for i in range(n_regions)}

    for y, angle in line_data:
        region_idx = min(int(y // region_height), n_regions - 1)
        region_angles[region_idx].append(angle)

    results = []
    for i in range(n_regions):
        y_center = int(region_height * (i + 0.5))
        if len(region_angles[i]) >= 2:
            angle = np.median(region_angles[i])
            results.append((y_center, angle))

    if len(results) < 2:
        return None

    logger.debug(f"Detected regional skew from {len(line_data)} text lines")
    return results


def mesh_perspective_correction(
    image: np.ndarray, regional_skew: list[tuple[int, float]]
) -> np.ndarray:
    """
    Apply mesh-based correction to correct varying skew across the page.

    For each region, the text has a certain angle. To straighten it, we need
    to vertically shift pixels so that slanted text becomes horizontal.

    For a line with angle θ:
    - Right side is lower by width * tan(θ)
    - To correct: shift pixel's y position based on its x position

    Args:
        image: Input BGR image
        regional_skew: List of (y_center, angle) tuples from detect_regional_skew

    Returns:
        Dewarped image
    """
    h, w = image.shape[:2]

    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    # Interpolate skew angles across all y positions
    y_positions = np.array([y for y, _ in regional_skew])
    angles = np.array([a for _, a in regional_skew])

    # Extend to cover full image
    if y_positions[0] > 0:
        y_positions = np.insert(y_positions, 0, 0)
        angles = np.insert(angles, 0, angles[0])
    if y_positions[-1] < h - 1:
        y_positions = np.append(y_positions, h - 1)
        angles = np.append(angles, angles[-1])

    # Interpolate angles for all y values
    all_angles = np.interp(np.arange(h), y_positions, angles)

    # Calculate y displacement for each pixel
    # For angle θ, pixel at x needs y shift of: (x - w/2) * tan(θ)
    # Negative angle = right side lower, so right pixels move UP (negative y shift)
    y_displacement = np.zeros((h, w), dtype=np.float32)

    # Vectorized computation
    x_offset = np.arange(w, dtype=np.float32) - w / 2

    for y in range(h):
        angle_rad = np.radians(all_angles[y])
        y_displacement[y, :] = x_offset * np.tan(angle_rad)

    # Create the remapping coordinates
    # To correct, we sample from shifted position
    map_x = x_coords
    map_y = y_coords + y_displacement  # Add displacement to move pixels up/down

    # Apply remapping
    result = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    return result


def correct_skew(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate the image to correct skew.

    Args:
        image: Input BGR image
        angle: Skew angle in degrees (negative = clockwise rotation needed)

    Returns:
        Rotated image with skew corrected
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Get rotation matrix (negative angle because we're correcting)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new bounding box size
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix for the new center
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    # Apply rotation with white background
    rotated = cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    # Trim white borders to reduce image size
    rotated = trim_white_borders(rotated)

    return rotated
