"""
Deskew functions for image preprocessing.

Contains all deskew-related algorithms: probmap-based angle measurement,
OCR-box angle measurement, morphological fallback, angular perspective
correction, and rotation utilities.

Extracted from preprocessor.py to follow single-responsibility principle.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by given angle, keeping original dimensions.

    Uses BORDER_REPLICATE to avoid introducing artificial white borders
    or expanding image bounds that confuse downstream OCR detection.

    Args:
        img: Input image
        angle: Rotation angle in degrees

    Returns:
        Rotated image at original dimensions
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def correct_angular_perspective(
    img: np.ndarray,
    angle_top: float,
    angle_bot: float,
    angle_mean: float,
) -> np.ndarray:
    """Apply projective warp to correct angular perspective gradient.

    When a page is photographed at an angle, text lines near the camera
    (e.g. top of page) appear more tilted than lines farther away
    (bottom).  This creates a linear gradient in text-line angles that
    cannot be corrected by a single rotation.

    The correction applies ``cv2.warpPerspective`` that rotates the top
    edge by ``-angle_top`` and the bottom edge by ``-angle_bot``,
    effectively removing the differential rotation while preserving
    the overall page geometry.

    The warp is constructed by shifting the top corners vertically by
    ``±tan(angle_top) * w/2`` and bottom corners by ``±tan(angle_bot) * w/2``,
    mapping to a rectangle where all corners have zero angular offset.

    Args:
        img: Input BGR image.
        angle_top: Measured text-line angle at page top (degrees).
        angle_bot: Measured text-line angle at page bottom (degrees).
        angle_mean: Mean text-line angle (degrees).

    Returns:
        Image with angular perspective gradient removed.
    """
    h, w = img.shape[:2]

    # Compute vertical shifts at each corner from the angular gradient.
    half_w = w / 2.0

    dy_top = np.tan(np.radians(angle_top)) * half_w
    dy_bot = np.tan(np.radians(angle_bot)) * half_w

    # Source corners: current positions (identity)
    # Order: TL, TR, BR, BL
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    # Destination corners: shifted to remove angular tilt
    dst = np.array(
        [
            [0, 0 + dy_top],
            [w, 0 - dy_top],
            [w, h - dy_bot],
            [0, h + dy_bot],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(src, dst)
    result = cv2.warpPerspective(
        img,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    logger.info(
        f"Angular perspective correction: "
        f"top={angle_top:+.2f}° → bot={angle_bot:+.2f}° "
        f"(dy_top={dy_top:+.1f}px, dy_bot={dy_bot:+.1f}px)"
    )
    return result


def measure_box_angles(boxes: list, page_width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure text-line angles from detection boxes.

    Args:
        boxes: List of detection box dicts with 'box' key (4×2 array).
        page_width: Page width in pixels for filtering.

    Returns:
        (angles, y_positions, widths) — arrays of filtered measurements.
        Empty arrays if fewer than 5 valid boxes.
    """
    angles = []
    ys = []
    weights = []
    for b in boxes:
        box = b["box"]
        dx = box[1][0] - box[0][0]
        dy = box[1][1] - box[0][1]
        width = float(np.hypot(dx, dy))
        if width < page_width * 0.05:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        if abs(angle) > 15.0:
            continue
        angles.append(angle)
        ys.append(float(np.mean(box[:, 1])))
        weights.append(width)
    return np.array(angles), np.array(ys), np.array(weights)


def detect_skew(img: np.ndarray) -> float:
    """Detect skew angle using text-line morphological analysis.

    Primary: dilates binary text image horizontally to form text-line blobs,
    then computes the angle of each blob via minAreaRect.  This isolates
    actual text baselines from structural features (borders, rules) that
    sit at 0° and would otherwise dominate.

    Fallback: Hough line transform, then contour span analysis.

    Sign convention: positive angle = text tilted clockwise (slopes
    down-right), returned as-is for ``cv2.getRotationMatrix2D`` which
    rotates counter-clockwise to correct it.

    Result clamped to ±MAX_DESKEW_ANGLE.

    Args:
        img: Input image in BGR format

    Returns:
        Skew angle in degrees (typically small, < 5°)
    """
    max_deskew = 5.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # === Primary: text-line morphological analysis ===
    inverted = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Horizontal dilation merges characters into text-line blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_angles: list[float] = []
    text_y_positions: list[float] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        # Filter: wide text-line-shaped blobs only
        if bw > w // 6 and bh < h // 10 and bw > bh * 2 and area > 500:
            rect = cv2.minAreaRect(cnt)
            angle = rect[-1]
            # Normalize OpenCV minAreaRect angle to ±45° range
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90
            if abs(angle) < 10:
                text_angles.append(angle)
                text_y_positions.append(float(y + bh / 2))

    if len(text_angles) >= 5:
        # Check for perspective gradient: if text-line angles vary
        # systematically from top to bottom, this is residual
        # perspective, NOT uniform skew. Deskew would make it worse.
        arr_a = np.array(text_angles)
        arr_y = np.array(text_y_positions)
        # Filter non-zero angles for gradient analysis
        nz_mask = np.abs(arr_a) > 0.01
        nz_count = int(nz_mask.sum())
        if nz_count >= 5:
            nz_a = arr_a[nz_mask]
            nz_y = arr_y[nz_mask]
            y_mean = nz_y.mean()
            a_mean = nz_a.mean()
            cov = np.sum((nz_y - y_mean) * (nz_a - a_mean))
            var_y = np.sum((nz_y - y_mean) ** 2)
            if var_y > 0:
                slope = cov / var_y
                angle_span = abs(slope * h)
                predicted = a_mean + slope * (nz_y - y_mean)
                residual_std = float(np.std(nz_a - predicted))
                # R² — fraction of angle variance explained by
                # the linear gradient
                ss_res = float(np.sum((nz_a - predicted) ** 2))
                ss_tot = float(np.sum((nz_a - a_mean) ** 2))
                r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                if angle_span > 3.0 and residual_std < 1.5 and r_squared > 0.3:
                    logger.debug(
                        f"Perspective gradient: span={angle_span:.1f}° "
                        f"over {h}px, residual_std={residual_std:.2f}°, "
                        f"R²={r_squared:.3f} — skipping deskew"
                    )
                    return 0.0

        median_angle = float(np.clip(np.median(text_angles), -max_deskew, max_deskew))
        logger.debug(f"Text-line skew: {median_angle:.2f}° ({len(text_angles)} lines)")
        return median_angle

    # === Fallback: Hough line transform ===
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    min_line_len = max(w // 8, 100)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=min_line_len,
        maxLineGap=10,
    )

    hough_angles: list[float] = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx > dy * 3 and dx > 0:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 10:
                    hough_angles.append(angle)

    if len(hough_angles) >= 3:
        median_angle = float(np.clip(np.median(hough_angles), -max_deskew, max_deskew))
        logger.debug(f"Hough skew: {median_angle:.2f}° ({len(hough_angles)} lines)")
        return median_angle

    # === Last resort: contour span analysis ===
    try:
        from bigocrpdf.services.contour_analysis import detect_skew_from_contours

        contour_angle = detect_skew_from_contours(img)
        if abs(contour_angle) > 0.5:
            clamped = float(np.clip(contour_angle, -max_deskew, max_deskew))
            logger.debug(
                f"Contour fallback skew: {clamped:.2f}° "
                f"(text lines: {len(text_angles)}, hough: {len(hough_angles)})"
            )
            return clamped
    except Exception as e:
        logger.debug(f"Contour skew fallback failed: {e}")

    return 0.0


def fallback_deskew(img: np.ndarray, enable_auto_detect: bool = True) -> np.ndarray:
    """Fallback deskew using morphological analysis when OCR boxes unavailable.

    Uses the classical ``detect_skew`` method and applies rotation
    with optional refinement pass.

    Args:
        img: Input BGR image.
        enable_auto_detect: Whether auto-detection is enabled.

    Returns:
        Deskewed image.
    """
    skew_angle = detect_skew(img)
    if not enable_auto_detect or abs(skew_angle) > 0.5:
        if abs(skew_angle) > 0.01:
            logger.debug(f"Fallback deskew: correcting {skew_angle:.2f}°")
            result = rotate_image(img, skew_angle)
            skew2 = detect_skew(result)
            if abs(skew2) > 1.0:
                logger.debug(f"Fallback deskew: refinement {skew2:.2f}°")
                result = rotate_image(result, skew2)
            return result
    return img


def ocr_box_deskew(img: np.ndarray, enable_auto_detect: bool = True) -> np.ndarray:
    """Deskew using OCR detection boxes for precise text-line angle measurement.

    Uses DBNet text detection boxes (quadrilaterals) instead of morphological
    blobs.  Detection boxes give precise text-line orientation because the
    neural network directly predicts tight-fitting quadrilaterals around
    each text region, whereas morphological dilation creates imprecise blobs
    whose angles are biased by character spacing and font metrics.

    Two correction modes:

    **Angular gradient** (R² > 0.3, span > 2°):
        Text-line angles vary linearly from top to bottom, indicating
        residual perspective distortion.  A projective warp is applied
        that rotates the top and bottom of the page by different amounts,
        eliminating the gradient.  Any residual uniform skew is then
        corrected by a simple rotation.

    **Uniform skew** (R² ≤ 0.3):
        Text-line angles are roughly constant across the page.  A simple
        affine rotation by the weighted median angle is applied.

    Falls back to the classical morphological ``detect_skew`` method if
    OCR box detection fails or returns too few boxes.

    Args:
        img: Input BGR image.
        enable_auto_detect: Whether auto-detection is enabled.

    Returns:
        Deskewed image.
    """
    try:
        from bigocrpdf.services.rapidocr_service.dewarp_detection import (
            detect_text_boxes,
        )
    except ImportError:
        logger.debug("OCR box detection unavailable, falling back to morphological")
        return fallback_deskew(img, enable_auto_detect)

    h, w = img.shape[:2]
    boxes = detect_text_boxes(img)
    if len(boxes) < 5:
        logger.debug(f"OCR box deskew: only {len(boxes)} boxes, falling back")
        return fallback_deskew(img, enable_auto_detect)

    # Measure angle of each text box from its top edge (box[0]→box[1])
    angles = []
    ys = []
    weights = []
    for b in boxes:
        box = b["box"]  # 4×2 array: [TL, TR, BR, BL]
        dx = box[1][0] - box[0][0]
        dy = box[1][1] - box[0][1]
        width = float(np.hypot(dx, dy))

        # Filter: wide text regions only (>5% page width, <50% for full lines)
        if width < w * 0.05:
            continue

        angle = float(np.degrees(np.arctan2(dy, dx)))
        if abs(angle) > 15.0:
            continue

        y_center = float(np.mean(box[:, 1]))
        angles.append(angle)
        ys.append(y_center)
        weights.append(width)  # Weight by box width (longer = more reliable)

    if len(angles) < 5:
        logger.debug(f"OCR box deskew: only {len(angles)} valid boxes, falling back")
        return fallback_deskew(img, enable_auto_detect)

    arr_a = np.array(angles)
    arr_y = np.array(ys)
    arr_w = np.array(weights)

    # Weighted linear regression: angle = slope * y + intercept
    w_sum = arr_w.sum()
    y_mean = float(np.sum(arr_w * arr_y) / w_sum)
    a_mean = float(np.sum(arr_w * arr_a) / w_sum)
    cov = float(np.sum(arr_w * (arr_y - y_mean) * (arr_a - a_mean)))
    var_y = float(np.sum(arr_w * (arr_y - y_mean) ** 2))

    if var_y < 1e-6:
        # All boxes at same Y — just do uniform deskew
        if abs(a_mean) > 0.5:
            logger.debug(f"OCR box deskew: uniform {a_mean:.2f}°")
            return rotate_image(img, a_mean)
        return img

    slope = cov / var_y
    angle_top = a_mean + slope * (0 - y_mean)
    angle_bot = a_mean + slope * (h - y_mean)
    angle_span = abs(angle_bot - angle_top)

    # R² — how much of angle variance is explained by linear gradient
    predicted = a_mean + slope * (arr_y - y_mean)
    residuals = arr_a - predicted
    ss_res = float(np.sum(arr_w * residuals**2))
    ss_tot = float(np.sum(arr_w * (arr_a - a_mean) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-6 else 0.0

    logger.debug(
        f"OCR box deskew: {len(angles)} boxes, mean={a_mean:.2f}°, "
        f"top={angle_top:.2f}°→bot={angle_bot:.2f}°, "
        f"span={angle_span:.2f}°, R²={r_squared:.3f}"
    )

    result = img

    if angle_span > 3.0 and r_squared > 0.4 and len(angles) >= 20:
        # Angular gradient detected — apply perspective angular correction.
        result = correct_angular_perspective(result, angle_top, angle_bot, a_mean)

        # Re-measure after correction for refinement pass
        boxes2 = detect_text_boxes(result)
        a2, y2, _ = measure_box_angles(boxes2, w)
        if len(a2) >= 5:
            coeffs2 = np.polyfit(y2, a2, 1)
            at2 = float(np.polyval(coeffs2, 0))
            ab2 = float(np.polyval(coeffs2, h))
            span2 = abs(ab2 - at2)
            mean2 = float(np.mean(a2))

            if span2 > 1.5:
                # Still has angular gradient — apply refinement
                logger.debug(f"OCR box deskew: refinement pass span={span2:.2f}° mean={mean2:.2f}°")
                result = correct_angular_perspective(result, at2, ab2, mean2)
                # Re-measure for uniform residual
                boxes3 = detect_text_boxes(result)
                a3, _, _ = measure_box_angles(boxes3, w)
                if len(a3) >= 3:
                    mean3 = float(np.median(a3))
                    if abs(mean3) > 0.3:
                        logger.debug(f"OCR box deskew: uniform residual {mean3:.2f}°")
                        result = rotate_image(result, mean3)
            elif abs(mean2) > 0.3:
                logger.debug(f"OCR box deskew: uniform residual {mean2:.2f}°")
                result = rotate_image(result, mean2)
        else:
            # Fall back: estimate residual from regression
            residual_median = float(np.median(residuals))
            total_residual = a_mean - (angle_top + angle_bot) / 2.0 + residual_median
            if abs(total_residual) > 0.3:
                result = rotate_image(result, total_residual)

    elif abs(a_mean) > 0.5:
        # Uniform skew — simple rotation using weighted median
        sorted_idx = np.argsort(arr_a)
        cum_w = np.cumsum(arr_w[sorted_idx])
        median_idx = int(np.searchsorted(cum_w, w_sum / 2.0))
        median_idx = min(median_idx, len(arr_a) - 1)
        w_median = float(arr_a[sorted_idx[median_idx]])

        skew_angle = float(np.clip(w_median, -5.0, 5.0))
        logger.debug(f"OCR box deskew: uniform skew correction {skew_angle:.2f}°")
        result = rotate_image(result, skew_angle)
    else:
        logger.debug("OCR box deskew: no correction needed")

    return result


def probmap_angle_deskew(img: np.ndarray, probmap_max_side: int = 0) -> np.ndarray:
    """Deskew using DBNet probability-map baselines for angle measurement.

    Replaces the heavier ``ocr_box_deskew`` for the default pipeline.
    Instead of running full OCR text detection (``detect_text_boxes``,
    ~500 ms, ~120 MB peak), this method runs lightweight DBNet inference
    (~180 ms) and extracts text-line angles directly from spline-fitted
    baselines (~25 ms).

    The DBNet model is the same one used in Step 1 (curvature correction),
    so it is already warm in the OpenVINO cache after the first page.

    Accuracy is comparable to OCR-box angles: both measure text-line
    orientation from neural network features, but probmap baselines span
    the full width of each line (giving more accurate slope estimates for
    long lines), while OCR boxes give tighter per-word measurements
    (better for short text fragments).

    Falls back to ``ocr_box_deskew`` if probmap extraction fails.

    Args:
        img: Input BGR image.
        probmap_max_side: Max side for DBNet inference (0 = default).

    Returns:
        Deskewed image.
    """
    try:
        from bigocrpdf.services.rapidocr_service.dewarp_probmap import (
            _extract_baselines,
            _get_probmap,
            _scale_baselines,
        )
    except ImportError:
        logger.debug("Probmap not available for deskew, falling back to OCR boxes")
        return ocr_box_deskew(img)

    h, w = img.shape[:2]

    try:
        prob, scale_x, scale_y = _get_probmap(img, max_side=probmap_max_side)
        inf_h, inf_w = prob.shape[:2]
        baselines = _extract_baselines(prob, inf_h, inf_w)
        del prob  # Free probmap early (~4.5 MB float32)
    except Exception as exc:
        logger.debug(f"Probmap deskew failed: {exc}, falling back to OCR boxes")
        return ocr_box_deskew(img)

    if len(baselines) < 3:
        logger.debug(f"Probmap deskew: only {len(baselines)} baselines, falling back to OCR boxes")
        return ocr_box_deskew(img)

    # Scale baselines from inference to full image resolution
    baselines = _scale_baselines(baselines, scale_x, scale_y)
    if len(baselines) < 3:
        return ocr_box_deskew(img)

    # Measure slope of each baseline from its spline endpoints
    angles = []
    ys = []
    widths = []
    for yc, spl, xs, xe in baselines:
        span = xe - xs
        if span < w * 0.20:  # Skip short baselines (< 20% page width)
            continue

        # Evaluate spline at endpoints to get the baseline angle
        y_start = float(spl(xs))
        y_end = float(spl(xe))
        dx = xe - xs
        dy = y_end - y_start
        angle = float(np.degrees(np.arctan2(dy, dx)))

        if abs(angle) > 15.0:
            continue

        angles.append(angle)
        ys.append(yc)
        widths.append(span)

    if len(angles) < 3:
        logger.debug(
            f"Probmap deskew: only {len(angles)} valid baselines, falling back to OCR boxes"
        )
        return ocr_box_deskew(img)

    arr_a = np.array(angles)
    arr_y = np.array(ys)
    arr_w = np.array(widths)

    # Weighted statistics (weight by baseline span — longer = more accurate)
    w_sum = arr_w.sum()
    y_mean = float(np.sum(arr_w * arr_y) / w_sum)
    a_mean = float(np.sum(arr_w * arr_a) / w_sum)

    # Weighted linear regression: angle = slope * y + intercept
    cov = float(np.sum(arr_w * (arr_y - y_mean) * (arr_a - a_mean)))
    var_y = float(np.sum(arr_w * (arr_y - y_mean) ** 2))

    if var_y < 1e-6:
        if abs(a_mean) > 0.5:
            logger.debug(f"Probmap deskew: uniform {a_mean:.2f}° (all same Y)")
            return rotate_image(img, a_mean)
        return img

    slope = cov / var_y
    angle_top = a_mean + slope * (0 - y_mean)
    angle_bot = a_mean + slope * (h - y_mean)
    angle_span = abs(angle_bot - angle_top)

    # R² — proportion of angle variance explained by linear gradient
    predicted = a_mean + slope * (arr_y - y_mean)
    residuals = arr_a - predicted
    ss_res = float(np.sum(arr_w * residuals**2))
    ss_tot = float(np.sum(arr_w * (arr_a - a_mean) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-6 else 0.0

    logger.debug(
        f"Probmap deskew: {len(angles)} baselines, mean={a_mean:.2f}°, "
        f"top={angle_top:.2f}°→bot={angle_bot:.2f}°, "
        f"span={angle_span:.2f}°, R²={r_squared:.3f}"
    )

    result = img

    if angle_span > 3.0 and r_squared > 0.4 and len(angles) >= 10:
        # Angular gradient — apply perspective correction
        result = correct_angular_perspective(result, angle_top, angle_bot, a_mean)

        # Skip refinement passes for small corrections (single pass is sufficient)
        if angle_span < 5.0:
            # Apply uniform residual from original measurement
            if abs(a_mean) > 0.3:
                result = rotate_image(result, a_mean)
        else:
            # Re-measure for refinement using probmap (much lighter than OCR boxes)
            try:
                prob2, sx2, sy2 = _get_probmap(result, max_side=probmap_max_side)
                bl2 = _extract_baselines(prob2, prob2.shape[0], prob2.shape[1])
                del prob2
                bl2 = _scale_baselines(bl2, sx2, sy2)
            except Exception:
                bl2 = []

            if len(bl2) >= 3:
                a2, y2 = [], []
                for yc, spl, xs, xe in bl2:
                    if xe - xs < w * 0.20:
                        continue
                    dy2 = float(spl(xe)) - float(spl(xs))
                    ang2 = float(np.degrees(np.arctan2(dy2, xe - xs)))
                    if abs(ang2) > 15:
                        continue
                    a2.append(ang2)
                    y2.append(yc)

                if len(a2) >= 3:
                    a2_arr = np.array(a2)
                    y2_arr = np.array(y2)
                    coeffs2 = np.polyfit(y2_arr, a2_arr, 1)
                    at2 = float(np.polyval(coeffs2, 0))
                    ab2 = float(np.polyval(coeffs2, h))
                    span2 = abs(ab2 - at2)
                    mean2 = float(np.median(a2))

                    if span2 > 1.5:
                        # Still has angular gradient — apply refinement
                        logger.debug(
                            f"Probmap deskew: refinement pass span={span2:.2f}° mean={mean2:.2f}°"
                        )
                        result = correct_angular_perspective(result, at2, ab2, mean2)
                        # Final uniform residual check
                        try:
                            prob3, sx3, sy3 = _get_probmap(result, max_side=probmap_max_side)
                            bl3 = _extract_baselines(prob3, prob3.shape[0], prob3.shape[1])
                            del prob3
                            bl3 = _scale_baselines(bl3, sx3, sy3)
                            a3 = []
                            for _yc, spl, xs, xe in bl3:
                                if xe - xs < w * 0.20:
                                    continue
                                dy3 = float(spl(xe)) - float(spl(xs))
                                ang3 = float(np.degrees(np.arctan2(dy3, xe - xs)))
                                if abs(ang3) > 15:
                                    continue
                                a3.append(ang3)
                            if len(a3) >= 3:
                                mean3 = float(np.median(a3))
                                if abs(mean3) > 0.3:
                                    logger.debug(f"Probmap deskew: uniform residual {mean3:.2f}°")
                                    result = rotate_image(result, mean3)
                        except Exception:
                            pass
                    elif abs(mean2) > 0.3:
                        logger.debug(f"Probmap deskew: uniform residual {mean2:.2f}°")
                        result = rotate_image(result, mean2)

    elif abs(a_mean) > 0.5:
        # Uniform skew — simple rotation using weighted median
        sorted_idx = np.argsort(arr_a)
        cum_w = np.cumsum(arr_w[sorted_idx])
        median_idx = int(np.searchsorted(cum_w, w_sum / 2.0))
        median_idx = min(median_idx, len(arr_a) - 1)
        w_median = float(arr_a[sorted_idx[median_idx]])

        skew_angle = float(np.clip(w_median, -5.0, 5.0))
        logger.debug(f"Probmap deskew: uniform skew correction {skew_angle:.2f}°")
        result = rotate_image(result, skew_angle)
    else:
        logger.debug("Probmap deskew: no correction needed")

    return result
