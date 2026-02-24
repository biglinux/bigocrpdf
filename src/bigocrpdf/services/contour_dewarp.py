"""Dewarp algorithms and skew detection using contour analysis.

Primary entry point: **dewarp_3d**
  Two-pass 3D dewarp that models the page as a curved surface:
  Pass 1 — Perspective correction via homography from detected page
           corners (cv2.warpPerspective — preserves text sharpness).
  Pass 2 — Text baseline refinement (handles residual interior curvature
           from spine binding).

Fallback: **dewarp_baseline**
  Per-line approach that straightens detected text baselines.
  Used when page boundaries cannot be reliably detected.

Also provides: **detect_skew_from_contours**
  Text contour span analysis for page skew angle detection.

Signals used for curvature estimation:
  1. Page boundary edges — cv2 contour detection + Otsu thresholding
  2. Text baselines — morphological ops + polynomial baseline fitting
  3. Illumination gradient — column-wise brightness for spine detection

The perspective transform uses 4 detected corner points to map the
page to a rectangular output. This preserves text sharpness because
cv2.warpPerspective is a single projective transform with minimal
interpolation artifacts — unlike cv2.remap which introduces more
quality loss due to per-pixel coordinate lookups.

Based on techniques from:
- Matt Zucker's page_dewarp (2016)
  https://mzucker.github.io/2016/08/15/page-dewarping.html
- lmmx/page-dewarp library
  https://github.com/lmmx/page-dewarp
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from bigocrpdf.services.contour_spans import (
    SPAN_MIN_WIDTH,
    _assemble_spans,
    _detect_text_contours,
    _sample_spans,
)

logger = logging.getLogger(__name__)

# ── Algorithm constants ──────────────────────────────────────────────────────

PAGE_MARGIN_X: int = 50
PAGE_MARGIN_Y: int = 20

# Our tuning
MAX_SCREEN_DIM: int = 2048
MIN_SPANS: int = 3

# Constants for baseline dewarp
_BL_MAX_DIM: int = 2048  # Downscale to this max dimension for detection
_BL_MIN_CURVED_LINES: int = 8  # Minimum curved lines for dewarp
_BL_MIN_DISPLACEMENT_PX: float = 8.0  # Minimum P95 displacement (full-res px)
_BL_REMAP_DECIMATE: int = 8  # Decimation for remap grid
_BL_MIN_LAPLACIAN_RATIO: float = 0.65  # Quality gate
_BL_MAX_DY_RATIO: float = 0.08  # Max displacement as fraction of height
_BL_POLY_DEGREE: int = 4  # 2D polynomial surface degree


# ── 3D Dewarp constants ─────────────────────────────────────────────────────

_3D_MAX_DIM: int = 2048
_3D_MIN_CURVED_LINES: int = 10  # Minimum curved baselines for pass 2
_3D_MIN_DISPLACEMENT_PX: float = 3.0
_3D_MAX_DY_RATIO: float = 0.08
_3D_POLY_DEGREE: int = 4
_3D_MIN_LAPLACIAN_RATIO: float = 0.55
_3D_MIN_EDGE_AREA_RATIO: float = 0.25  # Min contour area / image area
_3D_MIN_EDGE_CURVATURE_PX: float = 5.0  # Min edge curvature for perspective


# ── 3D Dewarp: page boundary detection ───────────────────────────────────────


def _detect_page_boundary(
    gray: np.ndarray,
    h: int,
    w: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    """Detect page outline and extract 4 oriented edge curves.

    Uses Otsu thresholding to separate bright page from dark background/
    shadow, finds the largest contour, identifies 4 corners (closest to
    image corners), and traces 4 edge curves between them.

    Edge orientation convention:
        top:    left → right  (T(0)=TL, T(1)=TR)
        bottom: left → right  (B(0)=BL, B(1)=BR)
        left:   top → bottom  (L(0)=TL, L(1)=BL)
        right:  top → bottom  (R(0)=TR, R(1)=BR)

    Returns:
        (corners_4×2, edge_dict) or None.
    """
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None

    page_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(page_contour)
    if area < _3D_MIN_EDGE_AREA_RATIO * h * w:
        return None

    pts = page_contour.reshape(-1, 2).astype(np.float64)
    n_contour = len(pts)

    # Find 4 corners closest to image corners
    image_corners = np.array(
        [[0, 0], [w, 0], [w, h], [0, h]],
        dtype=np.float64,
    )
    corner_indices: list[int] = []
    corners = np.zeros((4, 2), dtype=np.float64)
    for ci, ic in enumerate(image_corners):
        dists = np.sum((pts - ic) ** 2, axis=1)
        idx = int(np.argmin(dists))
        corner_indices.append(idx)
        corners[ci] = pts[idx]

    # Extract the shorter path along the contour between corners
    def _extract_edge(i1: int, i2: int) -> np.ndarray:
        fwd_len = (i2 - i1) % n_contour + 1
        bwd_len = (i1 - i2) % n_contour + 1
        if fwd_len <= bwd_len:
            if i2 >= i1:
                return pts[i1 : i2 + 1].copy()
            return np.vstack([pts[i1:], pts[: i2 + 1]]).copy()
        # Backward path, reversed to maintain i1→i2 direction
        if i1 >= i2:
            return pts[i2 : i1 + 1].copy()[::-1]
        return np.vstack([pts[i2:], pts[: i1 + 1]]).copy()[::-1]

    raw_top = _extract_edge(corner_indices[0], corner_indices[1])
    raw_right = _extract_edge(corner_indices[1], corner_indices[2])
    raw_bottom = _extract_edge(corner_indices[2], corner_indices[3])
    raw_left = _extract_edge(corner_indices[3], corner_indices[0])

    # Ensure correct orientation
    # bottom: left-to-right (x increasing)
    if raw_bottom[0][0] > raw_bottom[-1][0]:
        raw_bottom = raw_bottom[::-1]
    # left: top-to-bottom (y increasing)
    if raw_left[0][1] > raw_left[-1][1]:
        raw_left = raw_left[::-1]

    edge_dict = {
        "top": raw_top,
        "right": raw_right,
        "bottom": raw_bottom,
        "left": raw_left,
    }
    return corners, edge_dict


def _measure_edge_curvature(edge_pts: np.ndarray) -> float:
    """Max perpendicular deviation of an edge from its chord."""
    if len(edge_pts) < 5:
        return 0.0
    p1, p2 = edge_pts[0], edge_pts[-1]
    line_vec = p2 - p1
    line_len = float(np.linalg.norm(line_vec))
    if line_len < 10:
        return 0.0
    normal = np.array([-line_vec[1], line_vec[0]]) / line_len
    deviations = np.dot(edge_pts - p1, normal)
    return float(np.max(np.abs(deviations)))


# ── 3D Dewarp: text baseline extraction (shared with dewarp_baseline) ───────


def _extract_baseline_controls(
    gray: np.ndarray,
    h: int,
    w: int,
    scale: float,
) -> tuple[list[float], list[float], list[float], int, int]:
    """Extract curvature control points from text baselines.

    Uses morphological close to merge characters into text line blobs,
    then extracts the bottom edge (baseline) of each blob and fits
    polynomials. Curvature-only displacement (linear slope subtracted)
    preserves layout while measuring genuine page curvature.

    Returns:
        (control_x, control_y, control_dy, n_lines, n_curved)
        All coordinates in full-resolution pixels.
    """
    from scipy.ndimage import median_filter

    sh, sw = int(h * scale), int(w * scale)
    if scale < 1.0:
        small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        small = gray.copy()
        sh, sw = h, w

    binary = cv2.adaptiveThreshold(
        small,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        15,
    )

    kw = max(sw // 8, 60)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    line_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    line_mask = cv2.dilate(line_mask, kernel_v)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
    line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(
        line_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    min_line_w = sw * 0.08
    max_line_h = sh * 0.04

    control_x: list[float] = []
    control_y: list[float] = []
    control_dy: list[float] = []
    n_lines = 0
    n_curved = 0

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < min_line_w or ch > max_line_h or cw < 2 * ch:
            continue
        n_lines += 1

        points = c.reshape(-1, 2)
        x_to_max_y: dict[int, int] = {}
        for px, py in points:
            if px not in x_to_max_y or py > x_to_max_y[px]:
                x_to_max_y[px] = py

        xs_raw = sorted(x_to_max_y.keys())
        if len(xs_raw) < 15:
            continue

        xs = np.array(xs_raw, dtype=np.float64)
        ys = np.array([x_to_max_y[int(xv)] for xv in xs], dtype=np.float64)

        win = min(11, max(3, len(ys) // 6 * 2 + 1))
        ys_smooth = median_filter(ys, size=win).astype(np.float64)

        deg = min(3, len(xs) - 1)
        try:
            coeffs = np.polyfit(xs, ys_smooth, deg)
            linear_coeffs = np.polyfit(xs, ys_smooth, 1)
        except (np.linalg.LinAlgError, ValueError):
            continue

        n_samp = max(int((xs[-1] - xs[0]) / 10), 8)
        x_eval = np.linspace(xs[0], xs[-1], n_samp)
        y_cubic = np.polyval(coeffs, x_eval)
        y_linear = np.polyval(linear_coeffs, x_eval)
        dy = y_linear - y_cubic

        max_line_dy = float(np.max(np.abs(dy)))
        if max_line_dy < 1.5:
            continue

        n_curved += 1
        for xv, yv, dv in zip(x_eval, y_cubic, dy, strict=False):
            control_x.append(float(xv) / scale)
            control_y.append(float(yv) / scale)
            control_dy.append(float(dv) / scale)

    return control_x, control_y, control_dy, n_lines, n_curved


# ── 3D Dewarp: baseline refinement remap ─────────────────────────────────────


def _baseline_refinement_remap(
    bl_cx: list[float],
    bl_cy: list[float],
    bl_cdy: list[float],
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build a refinement remap from text baseline controls.

    Fits a 2D polynomial surface to baseline-derived control points
    and generates a vertical displacement field for residual curvature
    correction after the Coons patch pass.

    Returns:
        (map_x, map_y) float32 arrays, or None if insufficient displacement.
    """
    from scipy.ndimage import gaussian_filter

    control_x = np.array(bl_cx)
    control_y = np.array(bl_cy)
    control_dy = np.array(bl_cdy)

    max_allowed_dy = h * _3D_MAX_DY_RATIO
    control_dy = np.clip(control_dy, -max_allowed_dy, max_allowed_dy)

    disp_95 = float(np.percentile(np.abs(control_dy), 95))
    if disp_95 < _3D_MIN_DISPLACEMENT_PX:
        return None

    # Top/bottom anchors (zero displacement)
    edge_n = 25
    anchor_x: list[float] = []
    anchor_y: list[float] = []
    anchor_dy: list[float] = []
    for xv in np.linspace(0, w, edge_n):
        anchor_x.extend([float(xv), float(xv)])
        anchor_y.extend([0.0, float(h)])
        anchor_dy.extend([0.0, 0.0])

    all_x = np.concatenate([control_x, np.array(anchor_x)])
    all_y = np.concatenate([control_y, np.array(anchor_y)])
    all_dy = np.concatenate([control_dy, np.array(anchor_dy)])

    # Fit degree-4 polynomial surface
    xn, yn = all_x / w, all_y / h
    cols = []
    for i in range(_3D_POLY_DEGREE + 1):
        for j in range(_3D_POLY_DEGREE + 1 - i):
            cols.append(xn**i * yn**j)
    vander = np.column_stack(cols)

    try:
        coeffs_2d, _, _, _ = np.linalg.lstsq(vander, all_dy, rcond=None)
    except np.linalg.LinAlgError:
        return None

    dec = 8
    grid_y, grid_x = np.mgrid[0:h:dec, 0:w:dec]
    gx_flat = grid_x.ravel().astype(np.float64) / w
    gy_flat = grid_y.ravel().astype(np.float64) / h

    grid_cols = []
    for i in range(_3D_POLY_DEGREE + 1):
        for j in range(_3D_POLY_DEGREE + 1 - i):
            grid_cols.append(gx_flat**i * gy_flat**j)
    grid_vander = np.column_stack(grid_cols)

    dy_flat = grid_vander @ coeffs_2d
    dy_grid = dy_flat.reshape(grid_y.shape)
    dy_grid = gaussian_filter(dy_grid, sigma=2.0)
    dy_grid = np.clip(dy_grid, -max_allowed_dy, max_allowed_dy)

    map_x_s = grid_x.astype(np.float32)
    map_y_s = (grid_y.astype(np.float64) - dy_grid).astype(np.float32)
    map_x = cv2.resize(map_x_s, (w, h), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(map_y_s, (w, h), interpolation=cv2.INTER_LINEAR)

    max_disp = float(np.max(np.abs(control_dy)))
    logger.debug(f"Baseline refinement: max Δy={max_disp:.1f}px, P95={disp_95:.1f}px")
    return map_x, map_y


# ── 3D Dewarp: main entry point ─────────────────────────────────────────────


def _apply_perspective_pass(
    image: np.ndarray,
    boundary: tuple,
    scale: float,
    h: int,
    w: int,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """Pass 1: apply perspective correction from page boundary corners.

    Returns (corrected_image, page_roi) or (original_image, None).
    """
    corners_s, edge_dict_s = boundary
    curvatures = {k: _measure_edge_curvature(v) for k, v in edge_dict_s.items()}
    max_curv = max(curvatures.values())

    if max_curv < _3D_MIN_EDGE_CURVATURE_PX:
        return image, None

    corners_full = corners_s / scale
    centroid = corners_full.mean(axis=0)
    expand_ratio = 0.04
    for i in range(4):
        corners_full[i] += (corners_full[i] - centroid) * expand_ratio
    corners_full[:, 0] = np.clip(corners_full[:, 0], 0, w - 1)
    corners_full[:, 1] = np.clip(corners_full[:, 1], 0, h - 1)

    top_len = float(np.linalg.norm(corners_full[1] - corners_full[0]))
    bot_len = float(np.linalg.norm(corners_full[2] - corners_full[3]))
    left_len = float(np.linalg.norm(corners_full[3] - corners_full[0]))
    right_len = float(np.linalg.norm(corners_full[2] - corners_full[1]))
    nat_w = int((top_len + bot_len) / 2)
    nat_h = int((left_len + right_len) / 2)
    src_pts = corners_full.astype(np.float32)

    if nat_w / w >= 0.85 and nat_h / h >= 0.85:
        result, page_roi, out_w, out_h = _perspective_tight_crop(
            image,
            src_pts,
            nat_w,
            nat_h,
            w,
            h,
        )
    else:
        result, page_roi, out_w, out_h = _perspective_full_bounds(
            image,
            src_pts,
            nat_w,
            nat_h,
            w,
            h,
        )

    logger.info(
        f"3D dewarp pass 1: perspective, "
        f"max edge curvature {max_curv / scale:.1f}px, "
        f"output {out_w}×{out_h}"
    )
    return result, page_roi


def _perspective_tight_crop(
    image: np.ndarray,
    src_pts: np.ndarray,
    nat_w: int,
    nat_h: int,
    w: int,
    h: int,
) -> tuple[np.ndarray, tuple[int, int, int, int], int, int]:
    """Standard tight crop when natural dims >= 85% of original."""
    out_w = max(int(w * 0.9), min(int(w * 1.1), nat_w))
    out_h = max(int(h * 0.9), min(int(h * 1.1), nat_h))
    dst_pts = np.array(
        [[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(
        image,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return result, (0, 0, out_w, out_h), out_w, out_h


def _perspective_full_bounds(
    image: np.ndarray,
    src_pts: np.ndarray,
    nat_w: int,
    nat_h: int,
    w: int,
    h: int,
) -> tuple[np.ndarray, tuple[int, int, int, int], int, int]:
    """Full-bounds expansion when boundary detection missed significant content."""
    dst_rect = np.array(
        [[0, 0], [nat_w, 0], [nat_w, nat_h], [0, nat_h]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src_pts, dst_rect)
    img_corners = np.array(
        [[0, 0], [w, 0], [w, h], [0, h]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(img_corners, M).reshape(-1, 2)

    all_pts = np.vstack([dst_rect, warped])
    x_min = float(np.min(all_pts[:, 0]))
    y_min = float(np.min(all_pts[:, 1]))
    x_max = float(np.max(all_pts[:, 0]))
    y_max = float(np.max(all_pts[:, 1]))

    tx = -min(0.0, x_min)
    ty = -min(0.0, y_min)
    out_w = min(int(np.ceil(x_max - x_min)), int(w * 1.15))
    out_h = min(int(np.ceil(y_max - y_min)), int(h * 1.15))

    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    M_final = T @ M.astype(np.float64)
    result = cv2.warpPerspective(
        image,
        M_final,
        (out_w, out_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return result, (int(tx), int(ty), nat_w, nat_h), out_w, out_h


def _apply_baseline_pass(
    result: np.ndarray,
    perspective_applied: bool,
    page_roi: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, bool, int]:
    """Pass 2: text baseline curvature refinement.

    Returns (image, baseline_applied, n_curved).
    """
    result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    rh, rw = result_gray.shape

    roi_x = roi_y = 0
    if perspective_applied and page_roi is not None:
        rx, ry, roi_w, roi_h = page_roi
        rx = max(0, min(rx, rw - 1))
        ry = max(0, min(ry, rh - 1))
        roi_w = min(roi_w, rw - rx)
        roi_h = min(roi_h, rh - ry)
        analysis_gray = result_gray[ry : ry + roi_h, rx : rx + roi_w]
        roi_x, roi_y = rx, ry
        ah, aw = analysis_gray.shape
    else:
        analysis_gray = result_gray
        ah, aw = rh, rw

    result_scale = min(1.0, _3D_MAX_DIM / max(ah, aw))
    bl_cx, bl_cy, bl_cdy, n_lines, n_curved = _extract_baseline_controls(
        analysis_gray,
        ah,
        aw,
        result_scale,
    )

    if roi_x or roi_y:
        bl_cx = [x + roi_x for x in bl_cx]
        bl_cy = [y + roi_y for y in bl_cy]

    if not perspective_applied and bl_cdy:
        disp_p95 = float(np.percentile(np.abs(np.array(bl_cdy)), 95))
        if disp_p95 < 8.0:
            n_curved = 0

    if n_curved >= _3D_MIN_CURVED_LINES and len(bl_cx) >= 15:
        bl_remap = _baseline_refinement_remap(bl_cx, bl_cy, bl_cdy, rh, rw)
        if bl_remap is not None:
            bl_map_x, bl_map_y = bl_remap
            result = cv2.remap(
                result,
                bl_map_x,
                bl_map_y,
                cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )
            return result, True, n_curved

    return result, False, n_curved


def dewarp_3d(image: np.ndarray) -> np.ndarray | None:
    """Dewarp a document page using 3D surface simulation (two-pass)."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    max_dim = max(h, w)
    scale = min(1.0, _3D_MAX_DIM / max_dim)
    if scale < 1.0:
        small_gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small_gray = gray.copy()
    sh, sw = small_gray.shape

    # Pass 1: Perspective correction
    boundary = _detect_page_boundary(small_gray, sh, sw)
    perspective_applied = False
    page_roi = None
    result = image

    if boundary is not None:
        result, page_roi = _apply_perspective_pass(image, boundary, scale, h, w)
        perspective_applied = page_roi is not None

    # Pass 2: Baseline refinement
    result, baseline_applied, n_curved = _apply_baseline_pass(
        result,
        perspective_applied,
        page_roi,
    )

    if not perspective_applied and not baseline_applied:
        logger.debug(f"3D dewarp: no correction — curved_lines={n_curved}")
        return None

    # Quality validation
    final_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    lap_orig = cv2.Laplacian(gray, cv2.CV_64F).var()
    lap_corr = cv2.Laplacian(final_gray, cv2.CV_64F).var()

    if lap_orig > 0:
        lap_ratio = lap_corr / lap_orig
        if lap_ratio < _3D_MIN_LAPLACIAN_RATIO:
            logger.warning(
                f"3D dewarp rejected: Laplacian ratio {lap_ratio:.3f} < {_3D_MIN_LAPLACIAN_RATIO}"
            )
            return None

    passes = []
    if perspective_applied:
        passes.append("perspective")
    if baseline_applied:
        passes.append(f"baselines ({n_curved} lines)")
    logger.info(f"3D dewarp applied: {' + '.join(passes)}")
    return result


def dewarp_baseline(image: np.ndarray) -> np.ndarray | None:
    """Dewarp a document image by straightening detected text baselines.

    Uses actual text baselines (bottom edges of text line contours) for
    precise curvature detection, combined with a polynomial displacement
    surface for smooth correction.

    Key design:
    - Morphological close merges characters into text line blobs
    - Bottom edge of each blob = baseline (more curvature-sensitive than centroids)
    - Curvature-only correction (linear slope subtracted per line to preserve layout)
    - Degree-4 2D polynomial surface for smooth interpolation
    - Edge anchors only on top/bottom (not left/right) to preserve spine correction
    - Quality gate via Laplacian variance ratio

    Args:
        image: Input BGR image (numpy array, uint8).

    Returns:
        Corrected BGR image, or None if no significant curvature
        is detected or insufficient text lines found.
    """
    from scipy.ndimage import gaussian_filter

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ── Step 1: Compute scale and delegate baseline extraction ───────
    max_dim = max(h, w)
    scale = min(1.0, _BL_MAX_DIM / max_dim)

    control_x, control_y, control_dy, n_lines, n_curved = _extract_baseline_controls(
        gray, h, w, scale
    )

    if n_curved < _BL_MIN_CURVED_LINES or len(control_x) < 20:
        logger.debug(f"Baseline dewarp: {n_lines} lines, only {n_curved} curved, skipping")
        return None

    logger.debug(
        f"Baseline dewarp: {n_lines} lines detected, "
        f"{n_curved} with curvature, {len(control_x)} control points"
    )

    # ── Step 4: Check displacement significance ─────────────────────
    control_x_arr = np.array(control_x)
    control_y_arr = np.array(control_y)
    control_dy_arr = np.array(control_dy)

    # Clamp extreme displacements
    max_allowed_dy = h * _BL_MAX_DY_RATIO
    control_dy_arr = np.clip(control_dy_arr, -max_allowed_dy, max_allowed_dy)

    max_displacement = float(np.max(np.abs(control_dy_arr)))
    disp_95 = float(np.percentile(np.abs(control_dy_arr), 95))

    if disp_95 < _BL_MIN_DISPLACEMENT_PX:
        logger.debug(
            f"Baseline dewarp: P95 displacement {disp_95:.1f}px "
            f"< {_BL_MIN_DISPLACEMENT_PX}px, skipping"
        )
        return None

    # ── Step 5: Add edge anchors and build surface ───────────────────
    # Only anchor top and bottom edges (NOT left/right) to preserve
    # curvature correction at the spine edge where correction is needed most
    edge_n = 20
    edge_x_list: list[float] = []
    edge_y_list: list[float] = []
    edge_dy_list: list[float] = []

    for xv in np.linspace(0, w, edge_n):
        edge_x_list.extend([float(xv), float(xv)])
        edge_y_list.extend([0.0, float(h)])
        edge_dy_list.extend([0.0, 0.0])

    all_x = np.concatenate([control_x_arr, np.array(edge_x_list)])
    all_y = np.concatenate([control_y_arr, np.array(edge_y_list)])
    all_dy = np.concatenate([control_dy_arr, np.array(edge_dy_list)])

    # ── Step 6: Fit 2D polynomial surface ────────────────────────────
    # Degree-4 polynomial captures varying curvature across the page
    # 15 terms: 1, x, y, x², xy, y², x³, x²y, xy², y³, x⁴, x³y, x²y², xy³, y⁴
    xn = all_x / w
    yn = all_y / h

    poly_deg = _BL_POLY_DEGREE
    cols = []
    for i in range(poly_deg + 1):
        for j in range(poly_deg + 1 - i):
            cols.append(xn**i * yn**j)
    vander = np.column_stack(cols)

    try:
        coeffs_2d, _, _, _ = np.linalg.lstsq(vander, all_dy, rcond=None)
    except np.linalg.LinAlgError:
        logger.debug("Baseline dewarp: 2D polynomial fit failed, skipping")
        return None

    # ── Step 7: Evaluate on decimated grid ───────────────────────────
    dec = _BL_REMAP_DECIMATE
    grid_y, grid_x = np.mgrid[0:h:dec, 0:w:dec]

    gx_flat = grid_x.ravel().astype(np.float64) / w
    gy_flat = grid_y.ravel().astype(np.float64) / h

    grid_cols = []
    for i in range(poly_deg + 1):
        for j in range(poly_deg + 1 - i):
            grid_cols.append(gx_flat**i * gy_flat**j)
    grid_vander = np.column_stack(grid_cols)

    dy_flat = grid_vander @ coeffs_2d
    dy_grid = dy_flat.reshape(grid_y.shape)

    # Smooth and clamp
    dy_grid = gaussian_filter(dy_grid, sigma=2.0)
    dy_grid = np.clip(dy_grid, -max_allowed_dy, max_allowed_dy)

    # ── Step 8: Build remap maps and apply ───────────────────────────
    map_x_small = grid_x.astype(np.float32)
    map_y_small = (grid_y.astype(np.float64) - dy_grid).astype(np.float32)

    map_x = cv2.resize(map_x_small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    map_y = cv2.resize(map_y_small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    result = cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    # ── Step 9: Quality validation ───────────────────────────────────
    gray_corr = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    lap_orig = cv2.Laplacian(gray, cv2.CV_64F).var()
    lap_corr = cv2.Laplacian(gray_corr, cv2.CV_64F).var()

    if lap_orig > 0:
        lap_ratio = lap_corr / lap_orig
        if lap_ratio < _BL_MIN_LAPLACIAN_RATIO:
            logger.debug(
                f"Baseline dewarp rejected: Laplacian ratio {lap_ratio:.3f} "
                f"< {_BL_MIN_LAPLACIAN_RATIO} (too much blur)"
            )
            return None

    logger.info(
        f"Baseline dewarp applied: {n_curved} curved lines, "
        f"{len(control_x)} control points, "
        f"max displacement {max_displacement:.1f}px, P95 {disp_95:.1f}px"
    )
    return result


# ── Skew detection from spans ────────────────────────────────────────────────


def _detect_skew_from_spans(span_points: list[np.ndarray]) -> float:
    """Compute the dominant text angle from span baselines.

    Uses a weighted median of per-span linear fit slopes, where longer spans
    get more weight.

    Args:
        span_points: List of (N, 2) arrays of (x, y) points per span.

    Returns:
        Skew angle in degrees (positive = counterclockwise).
    """
    angles: list[float] = []
    weights: list[float] = []

    for pts in span_points:
        x, y = pts[:, 0], pts[:, 1]
        x_range = float(x.max() - x.min())
        if x_range < SPAN_MIN_WIDTH or len(x) < 3:
            continue

        # Linear fit: y = mx + b
        slope, _ = np.polyfit(x, y, 1)
        angle = float(np.degrees(np.arctan(slope)))

        # Only consider near-horizontal lines (skip vertical/diagonal elements)
        if abs(angle) < 15:
            angles.append(angle)
            weights.append(x_range)

    if len(angles) < 3:
        return 0.0

    # Weighted median — sort by angle and find the weight-median
    sorted_pairs = sorted(zip(angles, weights, strict=True), key=lambda p: p[0])
    total_weight = sum(weights)
    cumulative = 0.0
    for angle, weight in sorted_pairs:
        cumulative += weight
        if cumulative >= total_weight / 2:
            return angle

    return float(np.median(angles))


def detect_skew_from_contours(image: np.ndarray) -> float:
    """Detect page skew angle using text contour span analysis.

    Uses adaptive threshold to find text contours, assembles them into
    horizontal spans, and computes the dominant baseline slope as a
    weighted median of per-span slopes.

    This is more robust than Hough-based detection for images with short
    text lines, poor contrast, or few long horizontal features.

    Args:
        image: Input BGR image (numpy array, uint8).

    Returns:
        Skew angle in degrees (positive = clockwise). Returns 0.0 if
        insufficient text spans or keypoints are found for reliable
        detection.
    """
    h, w = image.shape[:2]

    # Downscale for detection
    max_dim = max(h, w)
    if max_dim > MAX_SCREEN_DIM:
        scale = MAX_SCREEN_DIM / max_dim
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = image.copy()

    sh, sw = small.shape[:2]

    # Create page mask (exclude margins)
    pagemask = np.zeros((sh, sw), dtype=np.uint8)
    mx = min(PAGE_MARGIN_X, sw // 4)
    my = min(PAGE_MARGIN_Y, sh // 4)
    pagemask[my : sh - my, mx : sw - mx] = 255

    # Detect text contours and assemble spans
    cinfo_list = _detect_text_contours(small, pagemask, text=True)

    spans = _assemble_spans(cinfo_list)
    if len(spans) < MIN_SPANS:
        # Try line detection as fallback
        line_cinfo = _detect_text_contours(small, pagemask, text=False)
        line_spans = _assemble_spans(line_cinfo)
        if len(line_spans) > len(spans):
            spans = line_spans

    if len(spans) < MIN_SPANS:
        return 0.0

    # Sample keypoints
    span_points = _sample_spans(small.shape, spans)
    n_pts = sum(len(sp) for sp in span_points)
    if n_pts < 20:
        return 0.0

    angle = _detect_skew_from_spans(span_points)
    logger.debug(f"Contour skew detection: {angle:.2f}° ({len(span_points)} spans, {n_pts} points)")
    return angle
