"""
OpenCV-based Document Perspective Correction and Skew Detection

Simple and reliable perspective/skew correction for documents using OpenCV.
Four modes:
1. Photo perspective: Detects document borders in photos and applies 4-point transform
2. Perspective correction from margins: Detects margin variation (keystone)
3. Perspective correction from boundaries: Detects document edges
4. Skew correction: Detects text line angles (regional or uniform) and corrects

Detection priority:
1. First checks if image is a photo (has dark borders around document)
2. Then checks for perspective distortion via margin analysis
3. Then checks for document contour (boundary-based perspective)
4. Then checks for varying regional skew (curved pages)
5. Finally checks for simple uniform skew
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_photo_document_borders(
    image: np.ndarray, margin_px: int = 20, min_area_ratio: float = 0.15
) -> np.ndarray | None:
    """
    Detect document borders in a photographed document using brightness analysis.

    Detects bright paper on a dark surface by:
    1. Checking if borders are significantly darker than the center (photo indicator)
    2. Thresholding on brightness to isolate the document region
    3. Finding the largest bright contour and approximating it to a quadrilateral
    4. Validating that the detected region shows actual perspective distortion

    Args:
        image: Input BGR image
        margin_px: Safety margin (pixels) to add around detected corners
        min_area_ratio: Minimum area of detected region as ratio of image area

    Returns:
        4x2 numpy array of ordered corner points [top-left, top-right,
        bottom-right, bottom-left], or None if document not detected
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ========== STEP 1: Check if this looks like a photo (dark borders, bright center) ==========
    border_w = max(w // 10, 20)
    border_h = max(h // 10, 20)

    border_means = [
        gray[:border_h, :].mean(),  # top
        gray[-border_h:, :].mean(),  # bottom
        gray[:, :border_w].mean(),  # left
        gray[:, -border_w:].mean(),  # right
    ]
    center_brightness = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean()

    # For a true photo (document on desk), at least 3 of 4 borders must be
    # significantly darker than center. Scans of bound documents often have
    # only 1 dark border (the binding shadow), which should NOT trigger this.
    dark_border_threshold = 30
    dark_borders = sum(1 for b in border_means if center_brightness - b > dark_border_threshold)

    if dark_borders < 3:
        logger.debug(
            f"Not a photo: only {dark_borders}/4 borders are dark "
            f"(borders={[f'{b:.0f}' for b in border_means]}, center={center_brightness:.0f})"
        )
        return None

    border_brightness = np.mean(border_means)

    # ========== STEP 2: Find document region using brightness thresholding ==========
    # Threshold halfway between border and center brightness
    thresh_val = int((border_brightness + center_brightness) / 2)
    _, doc_mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    # Morphological cleanup to remove noise and fill gaps
    kernel_close = np.ones((15, 15), np.uint8)
    doc_mask = cv2.morphologyEx(doc_mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = np.ones((10, 10), np.uint8)
    doc_mask = cv2.morphologyEx(doc_mask, cv2.MORPH_OPEN, kernel_open)

    # Find contours in the brightness mask
    contours, _ = cv2.findContours(doc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.debug("No bright region found in photo")
        return None

    # Get the largest contour (should be the document)
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    image_area = h * w

    if area < image_area * min_area_ratio:
        logger.debug(
            f"Bright region too small: {area / image_area:.1%} of image (need {min_area_ratio:.0%})"
        )
        return None

    # If the bright region fills almost the entire image, this is a scan, not a photo
    if area > image_area * 0.92:
        logger.debug("Bright region fills entire image — this is a scan, not a photo")
        return None

    # ========== STEP 3: Approximate contour to a quadrilateral ==========
    peri = cv2.arcLength(largest, True)
    corners = None

    # Try progressively looser approximation to find a quadrilateral
    for eps_mult in [0.02, 0.04, 0.06, 0.08, 0.10]:
        approx = cv2.approxPolyDP(largest, eps_mult * peri, True)
        if len(approx) == 4:
            corners = order_points(approx.reshape(4, 2).astype(np.float32))
            break

    if corners is None:
        # Fallback: use minimum area rotated rectangle
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        corners = order_points(box.astype(np.float32))

    # ========== STEP 4: Add safety margin (expand corners outward) ==========
    center = corners.mean(axis=0)
    for i in range(4):
        direction = corners[i] - center
        norm = np.linalg.norm(direction)
        if norm > 0:
            corners[i] += (direction / norm) * margin_px

    # Clamp to image bounds
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)

    logger.debug(
        f"Photo document detection: "
        f"TL=({corners[0, 0]:.0f}, {corners[0, 1]:.0f}), "
        f"TR=({corners[1, 0]:.0f}, {corners[1, 1]:.0f}), "
        f"BR=({corners[2, 0]:.0f}, {corners[2, 1]:.0f}), "
        f"BL=({corners[3, 0]:.0f}, {corners[3, 1]:.0f}), "
        f"area={area / image_area:.1%}"
    )

    return corners


def correct_photo_perspective(
    image: np.ndarray, corners: np.ndarray, preserve_aspect: bool = True
) -> np.ndarray:
    """
    Apply perspective correction to a photographed document.

    Args:
        image: Input BGR image
        corners: 4x2 array of corners [top-left, top-right, bottom-right, bottom-left]
        preserve_aspect: If True, try to preserve A4/Letter aspect ratio

    Returns:
        Corrected image
    """
    # Calculate dimensions
    top_w = np.linalg.norm(corners[1] - corners[0])
    bot_w = np.linalg.norm(corners[2] - corners[3])
    left_h = np.linalg.norm(corners[3] - corners[0])
    right_h = np.linalg.norm(corners[2] - corners[1])

    new_w = int(max(top_w, bot_w))
    new_h = int(max(left_h, right_h))

    # Adjust to standard paper aspect ratio.
    # Photos taken at steep angles distort the aspect ratio severely, so we
    # use a wide tolerance to catch documents that look nearly square due to
    # extreme perspective foreshortening.
    if preserve_aspect:
        aspect = new_h / new_w if new_w > 0 else 1.0
        A4_ASPECT = 297 / 210  # ~1.414
        LETTER_ASPECT = 11 / 8.5  # ~1.294
        tolerance = 0.5  # wide tolerance for photo distortion

        if abs(aspect - A4_ASPECT) < tolerance:
            new_h = int(new_w * A4_ASPECT)
        elif abs(aspect - LETTER_ASPECT) < tolerance and abs(aspect - LETTER_ASPECT) < abs(
            aspect - A4_ASPECT
        ):
            new_h = int(new_w * LETTER_ASPECT)
        elif abs(aspect - 1 / A4_ASPECT) < tolerance:
            # Landscape A4
            new_w = int(new_h * A4_ASPECT)
        elif abs(aspect - 1 / LETTER_ASPECT) < tolerance:
            # Landscape Letter
            new_w = int(new_h * LETTER_ASPECT)

    # Destination rectangle
    dst = np.array(
        [[0, 0], [new_w - 1, 0], [new_w - 1, new_h - 1], [0, new_h - 1]], dtype=np.float32
    )

    # Perspective transform
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(
        image,
        M,
        (new_w, new_h),
        borderValue=(255, 255, 255),  # White background
    )

    logger.info(f"Photo perspective correction applied: {new_w}x{new_h}")
    return warped


def detect_perspective_distortion(image: np.ndarray) -> dict | None:
    """
    Detect perspective/keystone distortion by analyzing text line margins.

    Analyzes how the left/right margins vary across the vertical extent of the
    page for UNIFORM-WIDTH text lines only. This filters out variable content
    (tables, short lines, headers) which would cause false positives.

    Args:
        image: Input BGR image

    Returns:
        Dictionary with distortion metrics and source corners, or None if no
        significant distortion detected
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Use Otsu's method for more reliable binarization
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find text lines by horizontal projection
    horizontal_proj = np.sum(binary, axis=1)
    smooth_proj = gaussian_filter1d(horizontal_proj.astype(float), sigma=5)
    peaks, _ = find_peaks(smooth_proj, height=w * 0.01, distance=10)

    if len(peaks) < 10:
        return None

    # Collect margin data for each line
    all_lines = []
    for peak_y in peaks:
        strip = binary[max(0, peak_y - 15) : min(h, peak_y + 15), :]
        row_sum = np.sum(strip, axis=0)
        threshold = row_sum.max() * 0.05
        text_cols = np.where(row_sum > threshold)[0]

        if len(text_cols) > w * 0.1:  # At least 10% of page width
            left = text_cols[0]
            right = text_cols[-1]
            width = right - left
            # Skip lines touching page borders (likely noise/artifacts)
            if left > 10 and right < w - 10:
                all_lines.append((peak_y, left, right, width))

    if len(all_lines) < 10:
        return None

    # Find the most common line width (mode) to identify "full-width" paragraphs
    widths = [line[3] for line in all_lines]
    width_median = np.median(widths)
    width_std = np.std(widths)

    # Only keep lines within 1.5 std of median width (similar-width lines)
    min_width = width_median - 1.5 * width_std
    max_width = width_median + 1.5 * width_std

    filtered_lines = [line for line in all_lines if min_width <= line[3] <= max_width]

    if len(filtered_lines) < 10:
        logger.debug("Not enough consistent-width lines for perspective detection")
        return None

    y_pos = np.array([line[0] for line in filtered_lines])
    left_m = np.array([line[1] for line in filtered_lines])
    right_m = np.array([line[2] for line in filtered_lines])

    # Fit linear regression to margins using robust estimator (RANSAC-like)
    # Use median-based slope estimation to reduce outlier influence
    y_norm = y_pos - np.mean(y_pos)
    left_slope = np.median((left_m - np.median(left_m)) / (y_norm + 1e-10))
    right_slope = np.median((right_m - np.median(right_m)) / (y_norm + 1e-10))

    # Also compute standard polyfit
    left_coeffs = np.polyfit(y_pos, left_m, 1)
    right_coeffs = np.polyfit(y_pos, right_m, 1)

    left_slope_lsq = left_coeffs[0] * 1000  # pixels per 1000 rows
    right_slope_lsq = right_coeffs[0] * 1000

    # Calculate distortion: use the smaller of median and LSQ (more conservative)
    margin_change_left = min(abs(left_slope * h), abs(left_slope_lsq * h / 1000))
    margin_change_right = min(abs(right_slope * h), abs(right_slope_lsq * h / 1000))

    # Conservative threshold: at least 80 pixels AND 3% of page width
    # Both margins must show distortion to avoid false positives from
    # forms/documents with naturally uneven margins.
    min_threshold = max(80, w * 0.03)
    if margin_change_left < min_threshold or margin_change_right < min_threshold:
        logger.debug(
            f"No significant perspective distortion detected "
            f"(left={margin_change_left:.1f}, right={margin_change_right:.1f}, threshold={min_threshold:.1f})"
        )
        return None

    # Check that margins are moving in consistent (parallel) manner
    # For true perspective, both margins should shift same direction
    # If they're moving opposite directions, that's convergence/divergence
    left_direction = np.sign(left_coeffs[0])
    right_direction = np.sign(right_coeffs[0])

    # If both move same direction: rotation, handle with skew correction
    # If opposite directions: keystone, handle with perspective correction
    if left_direction == right_direction:
        logger.debug("Margins moving parallel - this is skew, not perspective")
        return None

    # Estimate source corners (the distorted document corners)
    top_left = left_coeffs[0] * 0 + left_coeffs[1]
    top_right = right_coeffs[0] * 0 + right_coeffs[1]
    bot_left = left_coeffs[0] * h + left_coeffs[1]
    bot_right = right_coeffs[0] * h + right_coeffs[1]

    # Add some padding
    padding = 10
    src_corners = np.array(
        [
            [max(0, top_left - padding), 0],
            [min(w - 1, top_right + padding), 0],
            [min(w - 1, bot_right + padding), h - 1],
            [max(0, bot_left - padding), h - 1],
        ],
        dtype=np.float32,
    )

    logger.info(
        f"Detected perspective distortion: "
        f"left margin change={margin_change_left:.1f}px, "
        f"right margin change={margin_change_right:.1f}px "
        f"(from {len(filtered_lines)} lines)"
    )

    return {
        "left_slope": left_slope_lsq,
        "right_slope": right_slope_lsq,
        "margin_change_left": margin_change_left,
        "margin_change_right": margin_change_right,
        "src_corners": src_corners,
    }


def correct_perspective_from_margins(image: np.ndarray, distortion: dict) -> np.ndarray:
    """
    Correct perspective distortion by normalizing text margins across the page.

    Uses vectorized numpy operations for efficient remapping instead of
    per-pixel Python loops.

    Args:
        image: Input BGR image
        distortion: Distortion data from detect_perspective_distortion()

    Returns:
        Corrected image
    """
    h, w = image.shape[:2]

    # Calculate the target margins (median of detected margins)
    left_slope = distortion["left_slope"]  # px per 1000 rows
    right_slope = distortion["right_slope"]

    # Get base values at y = 0 (top of page)
    base_left = distortion["src_corners"][0, 0]  # left margin at top
    base_right = distortion["src_corners"][1, 0]  # right margin at top

    # Calculate median margins at y = h/2 (center of page)
    mid_y = h / 2
    median_left = base_left + (left_slope / 1000) * mid_y
    median_right = base_right + (right_slope / 1000) * mid_y
    target_width = median_right - median_left
    target_left = median_left

    if target_width <= 0:
        logger.warning("Invalid target width for perspective correction")
        return image

    # Build remapping coordinates using vectorized numpy operations
    # For each destination pixel (x, y), compute where to sample from source
    y_coords = np.arange(h, dtype=np.float32)
    x_coords = np.arange(w, dtype=np.float32)

    # Current margins at each row (vectorized over y)
    current_left = base_left + (left_slope / 1000) * y_coords  # shape (h,)
    current_right = base_right + (right_slope / 1000) * y_coords  # shape (h,)
    current_width = current_right - current_left  # shape (h,)

    # Avoid division by zero
    current_width = np.maximum(current_width, 1.0)

    # For each destination (x, y):
    #   x_rel = (x - target_left) / target_width
    #   source_x = current_left[y] + x_rel * current_width[y]
    # Vectorized: map_x[y, x] = current_left[y] + ((x - target_left) / target_width) * current_width[y]

    x_rel = (x_coords - target_left) / target_width  # shape (w,)

    # Outer product: map_x[y, x] = current_left[y] + x_rel[x] * current_width[y]
    map_x = current_left[:, np.newaxis] + x_rel[np.newaxis, :] * current_width[:, np.newaxis]
    map_y = np.repeat(y_coords[:, np.newaxis], w, axis=1)

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # Apply remapping
    result = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    logger.info("Perspective correction (margin normalization) applied successfully")
    return result


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
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Adaptive thresholding to get binary image
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    # Horizontal projection - sum of black pixels per row
    horizontal_proj = np.sum(binary, axis=1)

    # Smooth and find peaks (text lines)
    smooth_proj = gaussian_filter1d(horizontal_proj.astype(float), sigma=3)
    min_height = w * 0.02
    min_distance = max(10, h // 200)

    peaks, _ = find_peaks(smooth_proj, height=min_height, distance=min_distance)

    if len(peaks) < 5:
        logger.debug(f"Only {len(peaks)} text lines detected")
        return None

    # Detect angle for each text line
    angles = []

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

        centers = np.array(centers, dtype=np.float32)
        vx, vy, _, _ = cv2.fitLine(centers, cv2.DIST_L2, 0, 0.01, 0.01)
        angle = np.degrees(np.arctan2(vy[0], vx[0]))

        if angle > 45:
            angle = angle - 90
        elif angle < -45:
            angle = angle + 90

        if abs(angle) < 15:
            angles.append(angle)

    if len(angles) < 5:
        logger.debug(f"Only {len(angles)} valid line angles")
        return None

    skew_angle = np.median(angles)
    logger.debug(f"Detected skew angle: {skew_angle:.2f}° (from {len(angles)} lines)")

    return skew_angle


def detect_regional_skew(image: np.ndarray, n_regions: int = 5) -> list[tuple[int, float]] | None:
    """
    Detect skew angles at different vertical regions using text baseline detection.

    Uses horizontal projection to find text lines, then fits a line through
    character centers to determine each line's angle. Much more accurate than
    Hough-based detection for documents with tables, borders, or graphics.

    Args:
        image: Input BGR image
        n_regions: Number of vertical regions to aggregate results into

    Returns:
        List of (y_center, angle) tuples, or None if detection fails
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Adaptive thresholding to get binary image
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    # Horizontal projection - sum of black pixels per row
    horizontal_proj = np.sum(binary, axis=1)

    # Smooth and find peaks (text lines)
    smooth_proj = gaussian_filter1d(horizontal_proj.astype(float), sigma=3)
    min_height = w * 0.02  # Minimum projection height
    min_distance = max(10, h // 200)  # Minimum distance between lines

    peaks, _ = find_peaks(smooth_proj, height=min_height, distance=min_distance)

    if len(peaks) < 5:
        logger.debug(f"Only {len(peaks)} text lines detected, insufficient for analysis")
        return None

    # Detect angle for each text line
    line_data = []  # (y_position, angle)

    for peak_y in peaks:
        # Extract a strip around this line
        strip_height = max(15, h // 150)
        y_start = max(0, peak_y - strip_height // 2)
        y_end = min(h, peak_y + strip_height // 2)
        strip = binary[y_start:y_end, :]

        # Find connected components (characters/words)
        contours, _ = cv2.findContours(strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 3:
            continue

        # Get center of each contour
        centers = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))

        if len(centers) < 3:
            continue

        # Fit a line through the centers
        centers = np.array(centers, dtype=np.float32)
        vx, vy, _, _ = cv2.fitLine(centers, cv2.DIST_L2, 0, 0.01, 0.01)
        angle = np.degrees(np.arctan2(vy[0], vx[0]))

        # Normalize angle to be relative to horizontal
        if angle > 45:
            angle = angle - 90
        elif angle < -45:
            angle = angle + 90

        # Filter outliers (reasonable text skew range)
        if abs(angle) < 15:
            line_data.append((peak_y, angle))

    if len(line_data) < 5:
        logger.debug(f"Only {len(line_data)} valid line angles, insufficient")
        return None

    # Aggregate into regions
    region_height = h // n_regions
    region_angles = {i: [] for i in range(n_regions)}

    for y, angle in line_data:
        region_idx = min(int(y // region_height), n_regions - 1)
        region_angles[region_idx].append(angle)

    # Calculate median angle per region
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


def trim_white_borders(image: np.ndarray, threshold: int = 250) -> np.ndarray:
    """
    Remove white borders from an image.

    Args:
        image: Input BGR image
        threshold: Pixel intensity threshold for white (0-255)

    Returns:
        Cropped image without white borders
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find rows and columns that are not all white
    mask = gray < threshold

    # Find bounding box of non-white content
    rows_with_content = np.any(mask, axis=1)
    cols_with_content = np.any(mask, axis=0)

    if not np.any(rows_with_content) or not np.any(cols_with_content):
        # No content found, return original
        return image

    # Get bounds
    y_min = np.argmax(rows_with_content)
    y_max = len(rows_with_content) - np.argmax(rows_with_content[::-1])
    x_min = np.argmax(cols_with_content)
    x_max = len(cols_with_content) - np.argmax(cols_with_content[::-1])

    # Add small margin (5 pixels)
    margin = 5
    y_min = max(0, y_min - margin)
    y_max = min(image.shape[0], y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(image.shape[1], x_max + margin)

    return image[y_min:y_max, x_min:x_max]


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points in clockwise order: top-left, top-right, bottom-right, bottom-left.

    Args:
        pts: Array of 4 points

    Returns:
        Ordered points array
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has smallest diff, bottom-left has largest diff
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply perspective transform to warp the image based on four points.

    Args:
        image: Input image
        pts: Four corner points of the document

    Returns:
        Warped image with corrected perspective
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width (max of top and bottom width)
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # Compute height (max of left and right height)
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # Destination points
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )

    # Compute perspective transform matrix and apply
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped


def detect_document_contour(image: np.ndarray) -> np.ndarray | None:
    """
    Detect the document contour in an image.

    Uses edge detection and contour finding to locate the document boundaries.

    Args:
        image: Input BGR image

    Returns:
        Four corner points of the document, or None if not found
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate to connect edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Sort by area, largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find the largest contour that can be approximated to a quadrilateral
    for contour in contours[:5]:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # If it has 4 points, it's likely the document
        if len(approx) == 4:
            # Check if it's large enough (at least 20% of image area)
            area = cv2.contourArea(approx)
            image_area = image.shape[0] * image.shape[1]
            if area > image_area * 0.2:
                return approx.reshape(4, 2).astype("float32")

    return None


def needs_perspective_correction(image: np.ndarray, threshold: float = 0.03) -> bool:
    """
    Check if the image needs perspective correction.

    Analyzes the document edges to determine if they are significantly
    non-perpendicular.

    Args:
        image: Input BGR image
        threshold: Minimum skew angle (in radians) to trigger correction

    Returns:
        True if perspective correction is needed
    """
    contour = detect_document_contour(image)

    if contour is None:
        # No document contour found - assume it's already flat
        return False

    pts = order_points(contour)
    (tl, tr, br, bl) = pts

    # Calculate angles of each edge
    angles = []

    # Top edge angle
    top_angle = abs(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))
    angles.append(top_angle)

    # Bottom edge angle
    bottom_angle = abs(np.arctan2(br[1] - bl[1], br[0] - bl[0]))
    angles.append(bottom_angle)

    # Left edge angle (should be ~90 degrees = pi/2)
    left_angle = abs(np.arctan2(bl[1] - tl[1], bl[0] - tl[0]) - np.pi / 2)
    angles.append(left_angle)

    # Right edge angle
    right_angle = abs(np.arctan2(br[1] - tr[1], br[0] - tr[0]) - np.pi / 2)
    angles.append(right_angle)

    max_skew = max(angles)

    logger.debug(f"Perspective analysis: max_skew={max_skew:.4f}, threshold={threshold}")

    return max_skew > threshold


def gentle_margin_perspective_correction(
    image: np.ndarray,
    max_correction_pct: float = 15.0,
    min_convergence_pct: float = 8.0,
    max_convergence_pct: float = 25.0,
) -> np.ndarray | None:
    """Detect and correct perspective using text margin convergence.

    Analyses horizontal text-density projections in vertical strips to
    estimate left/right margin trend lines.  When the margins *converge*
    (i.e. opposite-sign slopes — a hallmark of true perspective rather
    than rotational skew), a per-row horizontal remap is applied.

    Header/footer strips (top and bottom ~15 %) are excluded from the
    regression to avoid contamination from logos, page numbers, and
    signatures.

    The correction uses a **centre-pivot remap**: the row at the vertical
    centre of the image stays untouched, while rows above and below are
    gently scaled horizontally about their text-area centre.  This avoids
    the corner distortion that ``warpPerspective`` introduces.

    Args:
        image:               Input BGR image.
        max_correction_pct:  Maximum perspective correction strength
                             (caps the applied transform).
        min_convergence_pct: Minimum detected width-change percentage to
                             trigger correction.
        max_convergence_pct: Maximum detected width-change percentage
                             (rejects extreme values likely caused by
                             layout artifacts).

    Returns:
        Corrected image, or ``None`` if no correction was applied.
    """
    if image is None or image.size == 0:
        return None

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape[:2]
    if h < 200 or w < 200:
        return None

    # ── Binarize ────────────────────────────────────────────────────
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        20,
    )

    # ── Horizontal text projection in vertical strips ───────────────
    n_strips = 20
    strip_h = h // n_strips

    left_margins: list[float] = []
    right_margins: list[float] = []
    strip_ys: list[float] = []

    for i in range(n_strips):
        y_start = i * strip_h
        y_end = min((i + 1) * strip_h, h)
        strip = binary[y_start:y_end, :]

        h_proj = np.sum(strip > 0, axis=0)
        text_cols = np.where(h_proj > strip_h * 0.05)[0]

        if len(text_cols) > 10:
            left = float(np.percentile(text_cols, 2))
            right = float(np.percentile(text_cols, 98))
            # Reject strips with very narrow text (noise / logos)
            if (right - left) > w * 0.15:
                left_margins.append(left)
                right_margins.append(right)
                strip_ys.append((y_start + y_end) / 2.0)

    if len(strip_ys) < 8:
        return None

    # ── Filter out header/footer strips ─────────────────────────────
    n_total = len(strip_ys)
    skip = max(2, n_total // 7)  # Skip ~15% on each end
    body_slice = slice(skip, n_total - skip)

    ys_body = np.array(strip_ys[body_slice])
    lm_body = np.array(left_margins[body_slice])
    rm_body = np.array(right_margins[body_slice])

    if len(ys_body) < 5:
        # Fall back to all strips if body slice is too small
        ys_body = np.array(strip_ys)
        lm_body = np.array(left_margins)
        rm_body = np.array(right_margins)

    # ── Regression-based outlier removal on each margin ──────────────
    def _robust_mask(ys_arr: np.ndarray, vals: np.ndarray) -> np.ndarray:
        coeffs = np.polyfit(ys_arr, vals, 1)
        residuals = np.abs(vals - np.polyval(coeffs, ys_arr))
        mad = float(np.median(residuals))
        if mad > 0:
            return residuals < 2.5 * 1.4826 * mad
        return np.ones(len(ys_arr), dtype=bool)

    mask = _robust_mask(ys_body, lm_body) & _robust_mask(ys_body, rm_body)
    ys_body, lm_body, rm_body = ys_body[mask], lm_body[mask], rm_body[mask]

    if len(ys_body) < 4:
        return None

    # ── Fit left/right margin regression ─────────────────────────────
    lc = np.polyfit(ys_body, lm_body, 1)
    rc = np.polyfit(ys_body, rm_body, 1)

    # Require opposite-sign slopes: true perspective makes margins
    # converge, while rotational skew shifts both in the same direction.
    if lc[0] * rc[0] > 0:
        logger.debug(
            "Gentle perspective: same-sign slopes "
            f"(left={np.degrees(np.arctan(lc[0])):.1f}°, "
            f"right={np.degrees(np.arctan(rc[0])):.1f}°) → skew, not perspective"
        )
        return None

    # Extrapolate to full image height
    left_top = float(np.polyval(lc, 0))
    left_bot = float(np.polyval(lc, h))
    right_top = float(np.polyval(rc, 0))
    right_bot = float(np.polyval(rc, h))

    width_top = right_top - left_top
    width_bot = right_bot - left_bot

    if min(width_top, width_bot) <= 0:
        return None

    convergence_pct = abs(width_bot - width_top) / min(width_top, width_bot) * 100

    if convergence_pct < min_convergence_pct or convergence_pct > max_convergence_pct:
        logger.debug(
            f"Gentle perspective: convergence {convergence_pct:.1f}% "
            f"outside [{min_convergence_pct}, {max_convergence_pct}]%"
        )
        return None

    # ── Centre-pivot per-row horizontal remap ───────────────────────
    # Instead of a full projective warp (which distorts corners), each
    # row is independently rescaled about its text-area centre.  The
    # centre row (y = h/2) serves as the reference and stays unchanged.
    correction_pct = min(convergence_pct, max_correction_pct)
    ratio = correction_pct / convergence_pct

    # Reference: width at the vertical centre of the image
    y_mid = h / 2.0
    ref_width = float(np.polyval(rc, y_mid)) - float(np.polyval(lc, y_mid))

    if ref_width <= 0:
        return None

    # Build remap arrays
    mapx = np.empty((h, w), dtype=np.float32)
    mapy = np.empty((h, w), dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)

    for y in range(h):
        fy = float(y)
        src_left = np.polyval(lc, fy)
        src_right = np.polyval(rc, fy)
        src_width = src_right - src_left
        src_center = (src_left + src_right) / 2.0

        if src_width <= 0:
            mapx[y, :] = xs
            mapy[y, :] = fy
            continue

        # How much wider/narrower this row is versus the reference
        full_scale = src_width / ref_width
        # Apply partial correction (blend identity ↔ full correction)
        scale = 1.0 + (full_scale - 1.0) * ratio

        # Remap: scale about the text-area centre of this row
        mapx[y, :] = np.float32(src_center) + (xs - np.float32(src_center)) * np.float32(scale)
        mapy[y, :] = np.float32(fy)

    corrected = cv2.remap(
        image,
        mapx,
        mapy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    logger.info(
        f"Gentle perspective correction applied: "
        f"convergence {convergence_pct:.1f}%, "
        f"correction {correction_pct:.1f}%"
    )
    return corrected


class PerspectiveCorrector:
    """
    OpenCV-based document perspective corrector.

    Performs perspective correction or skew correction to straighten documents.

    Four modes (in priority order):
    1. Photo perspective: Detects document on dark background (photographed)
    2. Perspective from margins: Detects keystone distortion from margin variation
    3. Perspective from boundaries: Detects document edges for 4-point transform
    4. Skew correction: Regional mesh or simple rotation for text alignment

    Only applies correction when significant distortion is detected.

    Example:
        corrector = PerspectiveCorrector()
        corrected = corrector(warped_image)
    """

    def __init__(
        self,
        skew_threshold: float = 0.5,
        variance_threshold: float = 0.3,
        skip_skew: bool = False,
    ):
        """
        Initialize perspective corrector.

        Args:
            skew_threshold: Minimum skew angle (degrees) to trigger correction.
                           Default 0.5 degrees. Set to 0 to always apply.
            variance_threshold: Minimum angle variance (degrees) between regions
                               to trigger mesh dewarping vs simple rotation.
                               Default 0.3 degrees (suitable for book scans with
                               subtle binding curvature).
            skip_skew: If True, skip the cascade's own skew correction steps
                      (regional_skew and skew_angle). Use when the caller
                      has its own deskew pass to avoid double correction.
        """
        self.skew_threshold = skew_threshold
        self.variance_threshold = variance_threshold
        self.skip_skew = skip_skew

    @staticmethod
    def _validate_correction(original: np.ndarray, corrected: np.ndarray, method: str) -> bool:
        """Validate that a correction improved text line alignment.

        Compares horizontal projection profile sharpness: straighter text lines
        produce sharper (higher contrast) horizontal projection profiles.

        Args:
            original: Original image
            corrected: Corrected image
            method: Name of correction method (for logging)

        Returns:
            True if correction improved alignment
        """
        try:
            gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            gray_corr = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

            # Horizontal projection: sum of pixel values per row
            proj_orig = np.sum(gray_orig < 128, axis=1).astype(float)
            proj_corr = np.sum(gray_corr < 128, axis=1).astype(float)

            # Profile "sharpness": standard deviation of the derivative
            # Higher values = sharper peaks = better text line alignment
            diff_orig = np.diff(proj_orig)
            diff_corr = np.diff(proj_corr)
            sharpness_orig = float(np.std(diff_orig))
            sharpness_corr = float(np.std(diff_corr))

            # Require at least 100% sharpness retention (no degradation allowed)
            is_valid = sharpness_corr >= sharpness_orig
            logger.debug(
                f"{method} validation: original_sharpness={sharpness_orig:.1f}, "
                f"corrected_sharpness={sharpness_corr:.1f}, valid={is_valid}"
            )
            return is_valid
        except Exception as e:
            logger.debug(f"{method} validation failed: {e}")
            return True  # On error, accept the correction

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Correct perspective or skew distortion in a document image.

        Args:
            image: Input BGR image (numpy array, uint8)

        Returns:
            Corrected image, or original if no correction needed
        """
        # 0. First priority: Check if this is a photo of a document
        # (document on dark background, photographed at angle)
        photo_corners = detect_photo_document_borders(image)

        if photo_corners is not None:
            logger.info("Detected photo of document, applying perspective correction...")
            corrected = correct_photo_perspective(image, photo_corners)
            return corrected

        # 1. Second priority: Check for perspective distortion from margin analysis
        # This catches keystone/trapezoid distortion even when document fills the page
        perspective_distortion = detect_perspective_distortion(image)

        if perspective_distortion is not None:
            logger.info("Applying perspective correction from margin analysis...")
            corrected = correct_perspective_from_margins(image, perspective_distortion)
            return corrected

        # 2. Third priority: Try to detect document contour for perspective correction
        contour = detect_document_contour(image)

        if contour is not None:
            # Check if perspective correction is needed
            if self.skew_threshold > 0:
                threshold_rad = np.radians(self.skew_threshold)
                if not needs_perspective_correction(image, threshold_rad):
                    logger.debug("Document appears flat. Skipping perspective correction.")
                    return image

            logger.info("Applying perspective correction from document boundary...")
            corrected = four_point_transform(image, contour)

            # Validate result
            orig_area = image.shape[0] * image.shape[1]
            new_area = corrected.shape[0] * corrected.shape[1]
            if new_area < orig_area * 0.5 or new_area > orig_area * 1.5:
                logger.warning("Perspective correction produced unexpected results.")
                # Fall through to try other methods
            else:
                logger.info("Perspective correction applied successfully.")
                return corrected

        # 3. Fourth priority: Gentle margin-based perspective correction
        # Catches mild perspective distortion (converging text margins) that
        # heavier detectors miss. Works best in combination with baseline
        # dewarp which runs later in the preprocessor pipeline.
        gentle_result = gentle_margin_perspective_correction(image)
        if gentle_result is not None:
            return gentle_result

        # 4. Fifth priority: Check for regional skew (varying skew across page)
        if not self.skip_skew:
            logger.debug("Checking for regional skew variation...")
            regional_skew = detect_regional_skew(image, n_regions=5)

            if regional_skew is not None and len(regional_skew) >= 3:
                angles = [a for _, a in regional_skew]
                angle_range = max(angles) - min(angles)

                logger.debug(f"Regional angles: {[f'{a:.1f}°' for _, a in regional_skew]}")
                logger.debug(f"Angle range: {angle_range:.2f}°")

                # Check if skew varies significantly across page
                if angle_range > self.variance_threshold:
                    max_skew = max(abs(a) for a in angles)
                    if max_skew > self.skew_threshold:
                        logger.info(
                            f"Detected varying skew (range: {angle_range:.1f}°). "
                            f"Applying mesh dewarping..."
                        )
                        corrected = mesh_perspective_correction(image, regional_skew)
                        if self._validate_correction(
                            image, corrected, "mesh_perspective_correction"
                        ):
                            logger.info("Mesh perspective correction applied successfully.")
                            return corrected
                        else:
                            logger.info(
                                "Mesh perspective correction rejected (did not improve alignment)."
                            )

        # 5. Sixth priority: Fall back to simple rotation for uniform skew
        if not self.skip_skew:
            skew_angle = detect_skew_angle(image)

            if skew_angle is None:
                logger.debug("Could not detect any distortion. Returning original image.")
                return image

            if abs(skew_angle) < self.skew_threshold:
                logger.debug(f"Skew angle ({skew_angle:.2f}°) below threshold. Skipping.")
                return image

            logger.info(f"Applying simple skew correction: {skew_angle:.2f}°")
            corrected = correct_skew(image, skew_angle)

            logger.info("Skew correction applied successfully.")
            return corrected
        else:
            logger.debug("Skipping skew correction steps (handled by caller).")

        logger.debug("No perspective distortion detected.")
        return image
