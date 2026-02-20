"""Margin-based perspective detection and gentle correction.

Functions for detecting perspective/keystone distortion from text margin
convergence and applying correction via horizontal remapping or projective
transforms.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


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


def gentle_margin_perspective_correction(
    image: np.ndarray,
    max_correction_pct: float = 25.0,
    min_convergence_pct: float = 10.0,
    max_convergence_pct: float = 30.0,
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

    # ── Reject noisy fits ────────────────────────────────────────────
    # If either margin regression has high residual standard deviation
    # relative to the image width, the margin data is too noisy for
    # reliable perspective detection (e.g. tables, mixed layouts).
    for coeffs, vals, label in [(lc, lm_body, "left"), (rc, rm_body, "right")]:
        residuals = vals - np.polyval(coeffs, ys_body)
        std_res = float(np.sqrt(np.mean(residuals**2)))
        if std_res > w * 0.05:
            logger.debug(
                f"Gentle perspective: {label} margin fit too noisy "
                f"(std_res={std_res:.0f}px, limit={w * 0.05:.0f}px)"
            )
            return None

    # ── Decompose slopes into rotation + perspective ─────────────────
    # Rotation shifts both margins in the same direction (average slope).
    # Perspective makes them converge/diverge (differential slope).
    # We measure perspective using derotated slopes but build the transform
    # using the ACTUAL margin positions (src) mapped to rotation-only
    # positions (dst) so only perspective is corrected.
    rot_slope = (lc[0] + rc[0]) / 2.0
    persp_slope_l = lc[0] - rot_slope  # perspective component of left
    persp_slope_r = rc[0] - rot_slope  # perspective component of right

    logger.debug(
        f"Gentle perspective: slopes "
        f"(left={np.degrees(np.arctan(lc[0])):.1f}°, "
        f"right={np.degrees(np.arctan(rc[0])):.1f}°), "
        f"rotation={np.degrees(np.arctan(rot_slope)):.1f}°, "
        f"perspective L={persp_slope_l:+.4f} R={persp_slope_r:+.4f}"
    )

    # Measure convergence from perspective-only component (derotated).
    # Use image centre as pivot to avoid intercept issues.
    y_mid = h / 2.0
    mid_left = float(np.polyval(lc, y_mid))
    mid_right = float(np.polyval(rc, y_mid))
    mid_width = mid_right - mid_left
    if mid_width <= 0:
        return None

    # Perspective-only widths at top and bottom (relative to centre row)
    half_h = h / 2.0
    persp_delta_l = persp_slope_l * half_h  # left margin shift from centre
    persp_delta_r = persp_slope_r * half_h  # right margin shift from centre
    width_top = mid_width - persp_delta_l + persp_delta_r  # y=0
    width_bot = mid_width + persp_delta_l - persp_delta_r  # y=h

    if min(width_top, width_bot) <= 0:
        return None

    convergence_pct = abs(width_bot - width_top) / min(width_top, width_bot) * 100

    if convergence_pct < min_convergence_pct or convergence_pct > max_convergence_pct:
        logger.debug(
            f"Gentle perspective: convergence {convergence_pct:.1f}% "
            f"outside [{min_convergence_pct}, {max_convergence_pct}]%"
        )
        return None

    # ── Full projective correction via warpPerspective ─────────────
    # src_pts = actual margin positions (where content IS in the image).
    # dst_pts = rotation-only positions (perspective removed, rotation kept).
    # This corrects ONLY keystone distortion; deskew handles rotation.
    correction_pct = min(convergence_pct, max_correction_pct)
    ratio = correction_pct / convergence_pct

    # Actual margin positions (full regression)
    src_left_top = float(np.polyval(lc, 0))
    src_left_bot = float(np.polyval(lc, h))
    src_right_top = float(np.polyval(rc, 0))
    src_right_bot = float(np.polyval(rc, h))

    # Rotation-only positions: same intercepts at y_mid, but slope = rot_slope
    # dst_left(y) = mid_left + rot_slope * (y - y_mid)
    # dst_right(y) = mid_right + rot_slope * (y - y_mid)
    dst_left_top = mid_left + rot_slope * (0 - y_mid)
    dst_left_bot = mid_left + rot_slope * (h - y_mid)
    dst_right_top = mid_right + rot_slope * (0 - y_mid)
    dst_right_bot = mid_right + rot_slope * (h - y_mid)

    src_pts = np.array(
        [
            [src_left_top, 0.0],
            [src_right_top, 0.0],
            [src_right_bot, float(h)],
            [src_left_bot, float(h)],
        ],
        dtype=np.float32,
    )

    dst_pts = np.array(
        [
            [dst_left_top, 0.0],
            [dst_right_top, 0.0],
            [dst_right_bot, float(h)],
            [dst_left_bot, float(h)],
        ],
        dtype=np.float32,
    )

    # Apply partial correction: interpolate destination from source toward
    # the ideal target.  ratio=0 → identity, ratio=1 → full correction.
    partial_dst = (src_pts + ratio * (dst_pts - src_pts)).astype(np.float32)

    M = cv2.getPerspectiveTransform(src_pts, partial_dst)
    corrected = cv2.warpPerspective(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    logger.info(
        f"Gentle perspective correction applied: "
        f"convergence {convergence_pct:.1f}%, "
        f"correction {correction_pct:.1f}%"
    )
    return corrected


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
