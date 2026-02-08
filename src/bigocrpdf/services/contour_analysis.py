"""Contour-based page dewarp using text span detection and baseline correction.

Implements a **baseline-detection dewarp** (dewarp_baseline):
   Per-line approach that straightens detected text baselines.

Text contour detection and span assembly pipeline:
- Adaptive threshold + morphological ops detect text contours
- PCA-based contour orientation (blob_mean_and_tangent via SVD)
- Greedy graph matching assembles contours into horizontal text spans
- Keypoints sampled along spans at dense intervals

Based on techniques from:
- Matt Zucker's page_dewarp (2016)
  https://mzucker.github.io/2016/08/15/page-dewarping.html
- lmmx/page-dewarp library
  https://github.com/lmmx/page-dewarp

No external library dependency — uses only numpy, scipy, cv2.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Algorithm constants (from page-dewarp defaults) ──────────────────────────

ADAPTIVE_WINSZ: int = 55

TEXT_MIN_WIDTH: int = 15
TEXT_MIN_HEIGHT: int = 2
TEXT_MIN_ASPECT: float = 1.5
TEXT_MAX_THICKNESS: int = 15

SPAN_MIN_WIDTH: int = 30
SPAN_PX_PER_STEP: int = 20

EDGE_MAX_OVERLAP: float = 1.0
EDGE_MAX_LENGTH: float = 160.0
EDGE_ANGLE_COST: float = 10.0
EDGE_MAX_ANGLE: float = 7.5

PAGE_MARGIN_X: int = 50
PAGE_MARGIN_Y: int = 20

# Our tuning
MAX_SCREEN_DIM: int = 2048
MIN_SPANS: int = 3


# ── Contour info ─────────────────────────────────────────────────────────────


class _ContourInfo:
    """Geometric and orientation data about a single text contour."""

    __slots__ = (
        "contour",
        "rect",
        "mask",
        "center",
        "tangent",
        "angle",
        "local_xrng",
        "point0",
        "point1",
        "pred",
        "succ",
    )

    def __init__(
        self,
        contour: np.ndarray,
        rect: tuple[int, int, int, int],
        mask: np.ndarray,
        center: np.ndarray,
        tangent: np.ndarray,
    ) -> None:
        self.contour = contour
        self.rect = rect
        self.mask = mask
        self.center = center
        self.tangent = tangent
        self.angle: float = float(np.arctan2(tangent[1], tangent[0]))

        # Project contour points onto tangent axis to find extent
        clx = [float(np.dot(self.tangent, pt.flatten() - self.center)) for pt in contour]
        lxmin, lxmax = min(clx), max(clx)
        self.local_xrng: tuple[float, float] = (lxmin, lxmax)
        self.point0: np.ndarray = self.center + self.tangent * lxmin
        self.point1: np.ndarray = self.center + self.tangent * lxmax
        self.pred: _ContourInfo | None = None
        self.succ: _ContourInfo | None = None

    def proj_x(self, point: np.ndarray) -> float:
        """Scalar projection of a point onto this contour's tangent axis."""
        return float(np.dot(self.tangent, point.flatten() - self.center))

    def local_overlap(self, other: _ContourInfo) -> float:
        """Measure horizontal overlap in local tangent coordinates."""
        xmin = self.proj_x(other.point0)
        xmax = self.proj_x(other.point1)
        return min(self.local_xrng[1], xmax) - max(self.local_xrng[0], xmin)


# ── Contour detection ────────────────────────────────────────────────────────


def _blob_mean_and_tangent(
    contour: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute centroid and principal orientation of a contour using moments + SVD."""
    moments = cv2.moments(contour)
    area = moments["m00"]
    if not area:
        return None
    mean_x = moments["m10"] / area
    mean_y = moments["m01"] / area
    cov = np.array([[moments["mu20"], moments["mu11"]], [moments["mu11"], moments["mu02"]]]) / area
    _, svd_u, _ = cv2.SVDecomp(cov)
    center = np.array([mean_x, mean_y])
    tangent = svd_u[:, 0].flatten().copy()
    return center, tangent


def _make_tight_mask(
    contour: np.ndarray, xmin: int, ymin: int, width: int, height: int
) -> np.ndarray:
    """Create a tight binary mask of a contour within its bounding box."""
    mask = np.zeros((height, width), dtype=np.uint8)
    tight_contour = contour - np.array((xmin, ymin)).reshape((-1, 1, 2))
    cv2.drawContours(mask, [tight_contour], contourIdx=0, color=1, thickness=-1)
    return mask


def _detect_text_contours(
    small: np.ndarray,
    pagemask: np.ndarray,
    text: bool = True,
) -> list[_ContourInfo]:
    """Detect text contours using adaptive threshold + morphological ops.

    Args:
        small: Downscaled BGR image.
        pagemask: Binary mask of the page region (excluding margins).
        text: If True, detect text blobs (dilate horizontally);
              if False, detect lines (erode to remove thin elements).

    Returns:
        List of _ContourInfo objects for valid text contours.
    """
    sgray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(
        src=sgray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=ADAPTIVE_WINSZ,
        C=25 if text else 7,
    )

    if text:
        # Dilate horizontally to connect characters into word blobs
        kernel = np.ones((1, 9), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel)
    else:
        # Erode to remove thin lines, keep thick table borders
        kernel = np.ones((7, 1), dtype=np.uint8)
        mask = cv2.erode(mask, kernel)

    # AND with page mask to exclude margins
    mask = np.minimum(mask, pagemask)

    # Find and filter contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cinfo_list: list[_ContourInfo] = []

    for contour in contours:
        rect = cv2.boundingRect(contour)
        xmin, ymin, width, height = rect

        if width < TEXT_MIN_WIDTH or height < TEXT_MIN_HEIGHT or width < TEXT_MIN_ASPECT * height:
            continue

        tight_mask = _make_tight_mask(contour, xmin, ymin, width, height)
        if tight_mask.sum(axis=0).max() > TEXT_MAX_THICKNESS:
            continue

        result = _blob_mean_and_tangent(contour)
        if result is None:
            continue

        center, tangent = result
        cinfo_list.append(_ContourInfo(contour, rect, tight_mask, center, tangent))

    return cinfo_list


# ── Span assembly ────────────────────────────────────────────────────────────


def _angle_dist(angle_b: float, angle_a: float) -> float:
    """Compute angular distance between two angles."""
    diff = angle_b - angle_a
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return abs(diff)


def _generate_candidate_edge(
    cinfo_a: _ContourInfo, cinfo_b: _ContourInfo
) -> tuple[float, _ContourInfo, _ContourInfo] | None:
    """Generate a candidate edge between two contours, scored by proximity + angle.

    Returns (score, left_contour, right_contour) or None if the pair is invalid.
    """
    # Ensure a is to the left of b
    if cinfo_a.point0[0] > cinfo_b.point1[0]:
        cinfo_a, cinfo_b = cinfo_b, cinfo_a

    x_overlap_a = cinfo_a.local_overlap(cinfo_b)
    x_overlap_b = cinfo_b.local_overlap(cinfo_a)

    overall_tangent = cinfo_b.center - cinfo_a.center
    overall_angle = float(np.arctan2(overall_tangent[1], overall_tangent[0]))

    delta_angle = (
        max(
            _angle_dist(cinfo_a.angle, overall_angle),
            _angle_dist(cinfo_b.angle, overall_angle),
        )
        * 180
        / np.pi
    )

    x_overlap = max(x_overlap_a, x_overlap_b)
    dist = float(np.linalg.norm(cinfo_b.point0 - cinfo_a.point1))

    if dist > EDGE_MAX_LENGTH or x_overlap > EDGE_MAX_OVERLAP or delta_angle > EDGE_MAX_ANGLE:
        return None

    score = dist + delta_angle * EDGE_ANGLE_COST
    return (score, cinfo_a, cinfo_b)


def _assemble_spans(cinfo_list: list[_ContourInfo]) -> list[list[_ContourInfo]]:
    """Assemble contours into horizontal text spans using greedy graph matching.

    A 'span' is a left-to-right chain of contours forming a text line or
    partial text line.
    """
    cinfo_list = sorted(cinfo_list, key=lambda ci: ci.rect[1])

    # Generate all candidate edges
    candidate_edges: list[tuple[float, _ContourInfo, _ContourInfo]] = []
    for i, ci_i in enumerate(cinfo_list):
        for j in range(i):
            edge = _generate_candidate_edge(ci_i, cinfo_list[j])
            if edge is not None:
                candidate_edges.append(edge)

    # Sort by score (lower is better)
    candidate_edges.sort(key=lambda e: e[0])

    # Link contours: each contour can have at most one predecessor and one successor
    for _, cinfo_a, cinfo_b in candidate_edges:
        if cinfo_a.succ is None and cinfo_b.pred is None:
            cinfo_a.succ = cinfo_b
            cinfo_b.pred = cinfo_a

    # Build spans by walking from each head (no predecessor) to tail
    spans: list[list[_ContourInfo]] = []
    remaining = list(cinfo_list)

    while remaining:
        cinfo = remaining[0]
        # Walk to head of chain
        while cinfo.pred:
            cinfo = cinfo.pred

        cur_span: list[_ContourInfo] = []
        width = 0.0

        while cinfo:
            if cinfo in remaining:
                remaining.remove(cinfo)
            cur_span.append(cinfo)
            width += cinfo.local_xrng[1] - cinfo.local_xrng[0]
            cinfo = cinfo.succ

        if width > SPAN_MIN_WIDTH:
            spans.append(cur_span)

    return spans


# ── Span sampling ────────────────────────────────────────────────────────────


def _sample_spans(shape: tuple[int, ...], spans: list[list[_ContourInfo]]) -> list[np.ndarray]:
    """Sample keypoints along spans at regular intervals.

    Within each contour's bounding rectangle, measures the vertical centroid
    of the mask at horizontal steps.

    Returns:
        List of arrays, each containing sampled points for one span
        in pixel coordinates, shape (N, 2) where columns are (x, y).
    """
    span_points: list[np.ndarray] = []
    for span in spans:
        contour_points: list[tuple[float, float]] = []
        for cinfo in span:
            yvals = np.arange(cinfo.mask.shape[0]).reshape((-1, 1))
            totals = (yvals * cinfo.mask).sum(axis=0)
            col_sums = cinfo.mask.sum(axis=0)
            # Skip columns with no contour pixels
            valid = col_sums > 0
            if not np.any(valid):
                continue
            means = np.zeros_like(totals, dtype=float)
            means[valid] = totals[valid] / col_sums[valid]

            xmin, ymin = cinfo.rect[:2]
            step = SPAN_PX_PER_STEP
            start = int(np.floor_divide(np.mod(len(means) - 1, step), 2))
            contour_points.extend(
                (x + xmin, means[x] + ymin) for x in range(start, len(means), step) if valid[x]
            )

        if len(contour_points) >= 2:
            pts = np.array(contour_points, dtype=np.float64)
            span_points.append(pts)

    return span_points


# ── Baseline-detection dewarp (Leptonica-style) ──────────────────────────────

# Constants for baseline dewarp
_BL_MAX_DIM: int = 2048  # Downscale to this max dimension for detection
_BL_MIN_LINES: int = 4  # Minimum text lines for dewarp
_BL_MIN_DISPLACEMENT_PX: float = 3.0  # Minimum max displacement (small-image px)
_BL_REMAP_DECIMATE: int = 8  # Decimation for remap grid
_BL_MIN_LAPLACIAN_RATIO: float = 0.70  # Quality gate


def dewarp_baseline(image: np.ndarray) -> np.ndarray | None:
    """Dewarp a document image by straightening detected text baselines.

    Uses a per-line approach instead of a global 3D projection model:
    1. Detect text components and cluster into horizontal text lines
    2. Fit a quadratic polynomial to each baseline
    3. Compute a vertical displacement field to straighten all lines
    4. Interpolate the sparse displacements to a smooth dense field
    5. Apply via cv2.remap with white border fill

    This approach is more robust than the 3D model (page_dewarp_3d):
    - Handles varying curvature across the page (per-line fitting)
    - Fast: O(N) in text lines, no iterative optimization (< 2s)
    - Safe for flat pages: early exit if max displacement < 3px
    - No BORDER_REPLICATE artifacts: uses white fill

    Args:
        image: Input BGR image (numpy array, uint8).

    Returns:
        Corrected BGR image, or None if no significant curvature
        is detected or insufficient text lines found.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ── Step 1: Downscale for detection ──────────────────────────────
    max_dim = max(h, w)
    if max_dim > _BL_MAX_DIM:
        scale = _BL_MAX_DIM / max_dim
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = gray.copy()
        scale = 1.0

    sh, sw = small.shape

    # ── Step 2: Detect text components ───────────────────────────────
    binary = cv2.adaptiveThreshold(
        small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15
    )

    # Horizontal close to merge characters into word/line blobs
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)

    # Remove thin vertical noise
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_v)

    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )

    # Filter: keep components that look like text line segments
    min_width = sw * 0.03  # at least 3% of image width
    max_height = sh * 0.05  # text line height < 5% of image
    min_area = 60

    components: list[tuple[float, float, float, float]] = []
    for i in range(1, num_labels):  # skip background label 0
        x, y, cw, ch, area = stats[i]
        if cw > min_width and ch < max_height and area > min_area and cw > 2 * ch:
            _cx, cy = centroids[i]
            components.append((float(x), float(cy), float(cw), float(x + cw)))

    if len(components) < 10:
        logger.debug(f"Baseline dewarp: only {len(components)} text components, skipping")
        return None

    # ── Step 3: Cluster components into text lines ───────────────────
    components.sort(key=lambda c: c[1])  # sort by y-centroid

    # Estimate typical line spacing from component height distribution
    heights = [
        stats[i][3]
        for i in range(1, num_labels)
        if stats[i][2] > min_width and stats[i][4] > min_area
    ]
    if heights:
        median_h = float(np.median(heights))
        merge_threshold = max(median_h * 0.8, sh * 0.008)
    else:
        median_h = sh * 0.012
        merge_threshold = median_h

    # Max vertical span for a single text line (prevents chain-effect
    # merging where gradual y-shifts accumulate across many components)
    max_line_span = median_h * 2.5

    lines: list[list[tuple[float, float, float, float]]] = []
    current_line = [components[0]]
    line_sum_y = components[0][1]
    line_count = 1
    line_min_y = components[0][1]
    line_max_y = components[0][1]

    for i in range(1, len(components)):
        comp_y = components[i][1]
        line_mean_y = line_sum_y / line_count

        # Compare against line mean (not last component) to avoid
        # chain-effect merging on curved text
        within_threshold = abs(comp_y - line_mean_y) < merge_threshold
        new_span = max(line_max_y, comp_y) - min(line_min_y, comp_y)
        within_span = new_span < max_line_span

        if within_threshold and within_span:
            current_line.append(components[i])
            line_sum_y += comp_y
            line_count += 1
            line_min_y = min(line_min_y, comp_y)
            line_max_y = max(line_max_y, comp_y)
        else:
            if len(current_line) >= 2:
                lines.append(current_line)
            current_line = [components[i]]
            line_sum_y = comp_y
            line_count = 1
            line_min_y = comp_y
            line_max_y = comp_y
    if len(current_line) >= 2:
        lines.append(current_line)

    if len(lines) < _BL_MIN_LINES:
        logger.debug(f"Baseline dewarp: only {len(lines)} text lines, skipping")
        return None

    logger.debug(
        f"Baseline dewarp: detected {len(lines)} text lines from {len(components)} components"
    )

    # ── Step 4: Fit quadratic polynomial to each baseline ────────────
    # Each control point stores: (x_fullres, y_fullres, dy_fullres)
    control_x: list[float] = []
    control_y: list[float] = []
    control_dy: list[float] = []

    # Minimum components per line for a meaningful curvature fit.
    # With only 3 points a quadratic is a perfect interpolation (zero
    # residuals), so any noise looks like curvature.  Require ≥5 for
    # quadratic; otherwise fit linear (slope only, no curvature).
    _MIN_POINTS_QUADRATIC = 5
    # Per-line curvature minimum: skip lines whose max |dy| (in full-
    # resolution pixels) is below this — they are essentially straight.
    _MIN_LINE_CURVATURE_PX = 3.0

    for line in lines:
        if len(line) < 3:
            continue

        # Component centroids
        xs = np.array([(c[0] + c[2] / 2) for c in line])  # center x
        ys = np.array([c[1] for c in line])  # centroid y

        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]

        # Fit quadratic only when enough points for a reliable fit
        deg = min(2, len(xs) - 1)
        if deg == 2 and len(xs) < _MIN_POINTS_QUADRATIC:
            deg = 1  # downgrade to linear — too few points for curvature
        try:
            coeffs = np.polyfit(xs, ys, deg)
        except (np.linalg.LinAlgError, ValueError):
            continue

        # Sample the fitted curve at regular intervals
        n_samples = max(int((xs[-1] - xs[0]) / 20), 5)
        x_range = np.linspace(xs[0], xs[-1], n_samples)
        y_fitted = np.polyval(coeffs, x_range)

        # Target: straighten to a line (preserve slope, remove only
        # curvature).  Fit a separate linear model to the original
        # data so that dy captures only the quadratic deviation.
        # This prevents dewarp from fighting perspective distortion
        # (which produces straight but tilted lines) while still
        # correcting actual curvature from book spines or paper folds.
        if deg == 2:
            try:
                linear_coeffs = np.polyfit(xs, ys, 1)
            except (np.linalg.LinAlgError, ValueError):
                continue
            y_target = np.polyval(linear_coeffs, x_range)
        else:
            # deg < 2: baseline is already linear, nothing to correct
            y_target = y_fitted.copy()

        # Displacement: curvature-only vertical shift
        dy = y_target - y_fitted

        # Skip lines with negligible curvature (essentially straight)
        max_line_dy = float(np.max(np.abs(dy)) / scale)
        if max_line_dy < _MIN_LINE_CURVATURE_PX:
            continue

        for xv, yv, dv in zip(x_range, y_fitted, dy, strict=False):
            control_x.append(xv / scale)
            control_y.append(yv / scale)
            control_dy.append(dv / scale)

    if len(control_x) < 20:
        logger.debug("Baseline dewarp: too few control points, skipping")
        return None

    control_x_arr = np.array(control_x)
    control_y_arr = np.array(control_y)
    control_dy_arr = np.array(control_dy)

    # ── Step 5: Check if curvature is significant ────────────────────
    # Use the 95th percentile to be robust to outlier points
    disp_95 = float(np.percentile(np.abs(control_dy_arr), 95) * scale)
    if disp_95 < _BL_MIN_DISPLACEMENT_PX:
        logger.debug(
            f"Baseline dewarp: 95th pctl displacement {disp_95:.1f}px "
            f"< {_BL_MIN_DISPLACEMENT_PX}px, skipping"
        )
        return None

    # Clamp individual displacements to prevent wild corrections
    # Max reasonable displacement is ~5% of image height
    max_allowed_dy = h * 0.05
    control_dy_arr = np.clip(control_dy_arr, -max_allowed_dy, max_allowed_dy)

    max_displacement = float(np.max(np.abs(control_dy_arr)) * scale)
    logger.debug(
        f"Baseline dewarp: max displacement {max_displacement:.1f}px "
        f"(clamped to ±{max_allowed_dy:.0f}px), "
        f"{len(control_x)} control points"
    )

    # ── Step 6: Build dense disparity field ──────────────────────────
    # Fit a low-order 2D polynomial surface to all control points.
    # Book curvature is a slowly-varying smooth effect, so a low-order
    # polynomial captures it without the oscillations that per-line
    # column-wise interpolation creates. Edge anchors are included
    # to prevent extrapolation beyond the text region.
    dec = _BL_REMAP_DECIMATE
    grid_y_coords, grid_x_coords = np.mgrid[0:h:dec, 0:w:dec]
    grid_h, grid_w = grid_y_coords.shape

    # Add zero-displacement anchors at image edges
    edge_n = 10
    edge_xs: list[float] = []
    edge_ys: list[float] = []
    edge_dys: list[float] = []

    for x in np.linspace(0, w, edge_n):
        edge_xs.append(x)
        edge_ys.append(0.0)
        edge_dys.append(0.0)
    for x in np.linspace(0, w, edge_n):
        edge_xs.append(x)
        edge_ys.append(float(h))
        edge_dys.append(0.0)
    for y in np.linspace(0, h, edge_n):
        edge_xs.append(0.0)
        edge_ys.append(y)
        edge_dys.append(0.0)
    for y in np.linspace(0, h, edge_n):
        edge_xs.append(float(w))
        edge_ys.append(y)
        edge_dys.append(0.0)

    all_x = np.concatenate([control_x_arr, np.array(edge_xs)])
    all_y = np.concatenate([control_y_arr, np.array(edge_ys)])
    all_dy = np.concatenate([control_dy_arr, np.array(edge_dys)])

    # Normalize coordinates to [0, 1] for numerical stability
    xn = all_x / w
    yn = all_y / h

    # Build 2D polynomial Vandermonde matrix (degree 3)
    # Basis: 1, x, y, x², xy, y², x³, x²y, xy², y³ (10 terms)
    poly_deg = 3
    cols = []
    for i in range(poly_deg + 1):
        for j in range(poly_deg + 1 - i):
            cols.append(xn**i * yn**j)
    vander = np.column_stack(cols)

    # Least-squares fit of the smooth surface
    try:
        coeffs_2d, _, _, _ = np.linalg.lstsq(vander, all_dy, rcond=None)
    except np.linalg.LinAlgError:
        logger.debug("Baseline dewarp: 2D polynomial fit failed, skipping")
        return None

    # Evaluate the smooth surface on the decimated grid
    gx_flat = grid_x_coords.ravel().astype(np.float64) / w
    gy_flat = grid_y_coords.ravel().astype(np.float64) / h

    grid_cols = []
    for i in range(poly_deg + 1):
        for j in range(poly_deg + 1 - i):
            grid_cols.append(gx_flat**i * gy_flat**j)
    grid_vander = np.column_stack(grid_cols)

    dy_flat = grid_vander @ coeffs_2d
    dy_grid = dy_flat.reshape(grid_h, grid_w)

    # Final clamp after surface evaluation
    dy_grid = np.clip(dy_grid, -max_allowed_dy, max_allowed_dy)

    # ── Step 7: Build remap maps ─────────────────────────────────────
    map_x_small = grid_x_coords.astype(np.float32)
    map_y_small = (grid_y_coords.astype(np.float64) - dy_grid).astype(np.float32)

    # Upsample to full resolution
    map_x = cv2.resize(map_x_small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    map_y = cv2.resize(map_y_small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    # ── Step 8: Apply remap ──────────────────────────────────────────
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
        f"Baseline dewarp applied: {len(lines)} lines, "
        f"{len(control_x)} control points, "
        f"max displacement {max_displacement:.1f}px"
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
