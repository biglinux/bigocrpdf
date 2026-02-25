"""Document detection and 4-point perspective transforms.

Functions for detecting documents in photos (bright paper on dark background)
and applying perspective transforms using detected corner points.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _refine_corners_with_edges(
    gray: np.ndarray,
    corners: np.ndarray,
    doc_mask: np.ndarray,
    contour: np.ndarray | None = None,
) -> np.ndarray:
    """Refine document corners by fitting lines to the brightness-contour sides.

    When brightness-based corner detection clips to image edges (because the
    document extends to or beyond the frame), we can still recover precise
    corners.  The brightness mask contour already follows the document border
    — we just need to compute where its sides actually intersect instead of
    using the approximated polygon vertices.

    Algorithm:
      1. Split the contour into 4 sides using the initial corners as delimiters.
      2. Fit a line to each side using cv2.fitLine (robust L2 fit).
      3. Compute corners as intersections of adjacent side-lines.
      4. Validate the result (convex, reasonable area).

    Args:
        gray:     Grayscale image
        corners:  4×2 initial corners [TL, TR, BR, BL]
        doc_mask: Binary mask of the bright document region
        contour:  Optional full contour from the brightness detection.  If
                  None, the contour is re-extracted from *doc_mask*.

    Returns:
        Refined 4×2 corners, or the original corners if refinement fails.
    """
    h, w = gray.shape[:2]
    edge_margin = 10  # pixels from image edge = "touching boundary"

    # Only refine when at least one corner touches the image boundary
    on_edge = (
        (corners[:, 0] < edge_margin)
        | (corners[:, 0] > w - edge_margin)
        | (corners[:, 1] < edge_margin)
        | (corners[:, 1] > h - edge_margin)
    )
    if not np.any(on_edge):
        return corners  # All corners are well inside the image

    # ---- 1. Get the contour ----
    if contour is None:
        contours, _ = cv2.findContours(doc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return corners
        contour = max(contours, key=cv2.contourArea)

    # Flatten to Nx2
    pts = contour.reshape(-1, 2).astype(np.float64)
    n_pts = len(pts)
    if n_pts < 20:
        return corners

    # ---- 2. Find contour-index of points closest to each corner ----
    corner_indices = []
    for ci in range(4):
        dists = np.linalg.norm(pts - corners[ci].astype(np.float64), axis=1)
        corner_indices.append(int(np.argmin(dists)))

    # Sort indices so we walk around contour in order
    # Contour may be CW or CCW; we just need consistent side assignment
    sorted_ci = sorted(range(4), key=lambda k: corner_indices[k])
    sorted_corner_idx = [corner_indices[k] for k in sorted_ci]

    # ---- 3. Split contour into 4 sides and fit lines ----
    def _extract_side(idx_start: int, idx_end: int) -> np.ndarray:
        """Extract contour points between two indices (wrapping around)."""
        if idx_end > idx_start:
            return pts[idx_start : idx_end + 1]
        else:
            return np.vstack([pts[idx_start:], pts[: idx_end + 1]])

    # Build 4 sides between adjacent sorted corner indices
    sides: list[np.ndarray] = []
    for i in range(4):
        s = sorted_corner_idx[i]
        e = sorted_corner_idx[(i + 1) % 4]
        side_pts = _extract_side(s, e)
        sides.append(side_pts)

    # Fit a line to each side: cv2.fitLine returns (vx, vy, x0, y0)
    # We need at least 5 points per side for a meaningful fit
    fitted_lines: list[tuple[float, float, float, float] | None] = []
    for side_pts in sides:
        if len(side_pts) < 5:
            fitted_lines.append(None)
            continue
        # Use CHAIN_APPROX_NONE to get dense points but subsample if too many
        if len(side_pts) > 200:
            step = len(side_pts) // 200
            side_pts = side_pts[::step]
        line = cv2.fitLine(side_pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        vx = float(line[0][0])
        vy = float(line[1][0])
        x0 = float(line[2][0])
        y0 = float(line[3][0])
        fitted_lines.append((vx, vy, x0, y0))

    # ---- 4. Compute corners as intersections of adjacent lines ----
    def _line_intersect_from_fit(
        l1: tuple[float, float, float, float],
        l2: tuple[float, float, float, float],
    ) -> tuple[float, float] | None:
        """Intersection of two lines given as (vx, vy, x0, y0)."""
        vx1, vy1, x01, y01 = l1
        vx2, vy2, x02, y02 = l2
        # Line 1: P1 = (x01, y01) + t*(vx1, vy1)
        # Line 2: P2 = (x02, y02) + s*(vx2, vy2)
        denom = vx1 * vy2 - vy1 * vx2
        if abs(denom) < 1e-9:
            return None  # Parallel lines
        dx = x02 - x01
        dy = y02 - y01
        t = (dx * vy2 - dy * vx2) / denom
        ix = x01 + t * vx1
        iy = y01 + t * vy1
        # Reject intersections far outside image
        if -w * 0.15 <= ix <= w * 1.15 and -h * 0.15 <= iy <= h * 1.15:
            return (ix, iy)
        return None

    refined = corners.copy()
    n_refined = 0

    # Each corner is the intersection of the line before and after it
    for i in range(4):
        # Which corner (in original TL/TR/BR/BL order) does sorted_ci[i] map to?
        orig_ci = sorted_ci[i]
        if not on_edge[orig_ci]:
            continue

        # Sides meeting at this corner: side that ENDS here and side that STARTS here
        side_before = (i - 1) % 4
        side_after = i

        l_before = fitted_lines[side_before]
        l_after = fitted_lines[side_after]
        if l_before is None or l_after is None:
            continue

        pt = _line_intersect_from_fit(l_before, l_after)
        if pt is not None:
            refined[orig_ci] = np.array(pt, dtype=np.float32)
            n_refined += 1

    if n_refined > 0:
        # Validate: refined quad must be convex and area > 15% of image
        def _is_convex_quad(q: np.ndarray) -> bool:
            for i in range(4):
                o = q[i]
                a = q[(i + 1) % 4] - o
                b = q[(i + 2) % 4] - o
                if np.cross(a, b) <= 0:
                    return False
            return True

        if _is_convex_quad(refined):
            refined_area = cv2.contourArea(refined)
            if refined_area > h * w * 0.15:
                logger.info(
                    f"Contour-refined {n_refined} corner(s): "
                    f"TL=({refined[0, 0]:.0f},{refined[0, 1]:.0f}), "
                    f"TR=({refined[1, 0]:.0f},{refined[1, 1]:.0f}), "
                    f"BR=({refined[2, 0]:.0f},{refined[2, 1]:.0f}), "
                    f"BL=({refined[3, 0]:.0f},{refined[3, 1]:.0f})"
                )
                return refined
            else:
                logger.debug("Refined quad too small — keeping original corners")
        else:
            logger.debug("Refined quad is non-convex — keeping original corners")

    return corners


def detect_photo_document_borders(
    image: np.ndarray, margin_px: int = 20, min_area_ratio: float = 0.15
) -> np.ndarray | None:
    """
    Detect document borders in a photographed document using brightness analysis.

    Detects bright paper on a dark surface by:
    1. Checking if borders are significantly darker than the center (photo indicator)
    2. Thresholding on brightness to isolate the document region
    3. Finding the largest bright contour and approximating it to a quadrilateral
    4. Refining corners with edge detection when they clip to image boundaries
    5. Adding a safety margin around the detected corners

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

    # ========== STEP 4: Refine corners that clip to image edges ==========
    corners = _refine_corners_with_edges(gray, corners, doc_mask)

    # ========== STEP 5: Add safety margin (expand corners outward) ==========
    # Use a proportional margin: 1.5% of the larger dimension, minimum 20px.
    # This ensures enough padding for high-resolution photos where 20px is tiny.
    proportional_margin = max(margin_px, int(max(h, w) * 0.015))
    center = corners.mean(axis=0)
    for i in range(4):
        direction = corners[i] - center
        norm = np.linalg.norm(direction)
        if norm > 0:
            corners[i] += (direction / norm) * proportional_margin

    # NOTE: We intentionally do NOT clamp corners to image bounds.
    # warpPerspective handles out-of-bounds source coordinates by filling
    # with borderValue (white).  Clamping would shift the source quad
    # inward, cropping document content near edges.

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
    # Photos taken at steep angles distort the aspect ratio, so we
    # snap to a standard size when the raw aspect is close.
    # Tolerance 0.15 avoids over-stretching documents whose raw aspect
    # is far from any standard (e.g. landscape photos, receipts).
    if preserve_aspect:
        aspect = new_h / new_w if new_w > 0 else 1.0
        A4_ASPECT = 297 / 210  # ~1.414
        LETTER_ASPECT = 11 / 8.5  # ~1.294
        tolerance = 0.15

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

    # Perspective transform — Lanczos4 produces ~50% sharper text than bilinear
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LANCZOS4,
        borderValue=(255, 255, 255),  # White background
    )

    logger.info(f"Photo perspective correction applied: {new_w}x{new_h}")
    return warped


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
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height), flags=cv2.INTER_LANCZOS4)

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
