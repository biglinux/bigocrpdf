"""Page curvature correction from the text detector's probability map.

The detector's probability map is thresholded and eroded into text lines, and
each line's baseline is traced column by column through probability-weighted
centroids. A spline fitted to each baseline gives its vertical displacement;
removing each baseline's linear trend keeps the curvature and drops any
rotation. The displacements are interpolated between baselines into a smooth
field, and the page is remapped when the largest displacement reaches
``_MIN_CURVATURE_PX``.
"""

import logging
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from scipy import ndimage
from scipy.interpolate import UnivariateSpline

from bigocrpdf.services.ocr.ppocr_native import Detector

logger = logging.getLogger(__name__)


_DBNET_MAX_SIDE = 1536
_DBNET_PROB_THRESHOLD = 0.3

# Curvature thresholds
_MIN_CURVATURE_PX = 5.0  # pixels: skip remap if max curvature below this
_MAX_CURVATURE_PX = 60.0  # pixels: cap per-baseline displacement to avoid distortion
_MIN_BASELINE_SPAN = 0.40  # fraction of page width: reject shorter baselines

# Baseline extraction constants
_BASELINE_ERODE_KERN_H = 3  # vertical erode kernel height
_BASELINE_MIN_W_RATIO = 0.05  # min component width as fraction of page width
_BASELINE_MIN_H_PX = 5  # pixels: minimum component height
_BASELINE_MIN_VALID_PTS = 4  # min valid column points for spline fitting
_SPLINE_MIN_POINTS = 4  # min points to attempt spline
_SPLINE_SMOOTH_MULT = 5.0  # smoothing factor = this × len(points)
_CENTROID_OUTLIER_SIGMA = 2.5  # × std for centroid outlier rejection
_BASELINE_SAMPLE_DIV = 10  # n_samples = comp_w / this (min clamped)
_BASELINE_PROB_EPS = 0.01  # min column probability to count as valid

# Curvature remap constants
_CURVATURE_CHUNK = 128  # pixels: row chunk size for vectorised remap
_FADE_MARGIN_GAP_MULT = 3.0  # fade margin = typical_gap × this
_SMOOTH_DOWNSCALE = 4  # downscale factor for Gaussian smoothing
_SMOOTH_SIGMA_Y_MULT = 0.4  # σ_y = downscaled_height × this
_SMOOTH_SIGMA_X_RATIO = 0.03  # σ_x = σ_y × this ratio (σ_x/σ_y≈0.075)

# DBNet inference
_INFERENCE_ALIGN_STRIDE = 32  # resize dimensions must be multiples of this


def _get_probmap(
    detector: Detector, img_bgr: np.ndarray, max_side: int = 0
) -> tuple[np.ndarray, float, float]:
    """Run DBNet inference and return the text probability map.

    The page is resized to multiples of 32 within ``_DBNET_MAX_SIDE``; the
    detector normalizes it the PaddleOCR way.

    Args:
        img_bgr: Input image in BGR format (OpenCV).
        max_side: Maximum inference side length, 0 for ``_DBNET_MAX_SIDE``.
                  Lower values reduce RAM on constrained systems.

    Returns:
        Probability map plus the horizontal and vertical scale factors needed
        to map inference coordinates back to the source image.

    Raises:
        RuntimeError: When the detector is not installed or fails.
    """
    h, w = img_bgr.shape[:2]

    effective_max = max_side if max_side > 0 else _DBNET_MAX_SIDE
    scale = min(effective_max / max(h, w), 1.0)
    new_h = int(h * scale / _INFERENCE_ALIGN_STRIDE) * _INFERENCE_ALIGN_STRIDE
    new_w = int(w * scale / _INFERENCE_ALIGN_STRIDE) * _INFERENCE_ALIGN_STRIDE

    prob = detector.probability_map(cv2.resize(img_bgr, (new_w, new_h)))

    # Kept at inference resolution with scale factors instead of a
    # full-resolution float32 copy; baselines are scaled afterwards.
    scale_x = w / prob.shape[1]
    scale_y = h / prob.shape[0]
    return prob, scale_x, scale_y


@dataclass(frozen=True)
class Baselines:
    """The text baselines the detector found on one image, at inference
    resolution, with the scales that map them back to the image.

    Holding the image keeps the identity test in ``measure_baselines`` safe.
    """

    image: np.ndarray
    max_side: int
    lines: list
    scale_x: float
    scale_y: float

    def crop(self, image: np.ndarray, x: int, y: int) -> "Baselines":
        """These baselines on ``image``, the crop of this image whose top-left
        corner is (x, y). Translation leaves every angle as it was."""
        dx, dy = x / self.scale_x, y / self.scale_y
        lines = [
            (yc - dy, lambda v, spl=spl: spl(v + dx) - dy, xs - dx, xe - dx)
            for yc, spl, xs, xe in self.lines
        ]
        return Baselines(image, self.max_side, lines, self.scale_x, self.scale_y)


def measure_baselines(
    detector: Detector, img_bgr: np.ndarray, max_side: int = 0, known: Baselines | None = None
) -> Baselines:
    """The image's baselines; ``known`` stands when it measured this very image at this size."""
    if known is not None and known.image is img_bgr and known.max_side == max_side:
        return known
    prob, scale_x, scale_y = _get_probmap(detector, img_bgr, max_side=max_side)
    return Baselines(img_bgr, max_side, _extract_baselines(prob), scale_x, scale_y)


def _extract_baselines(prob: np.ndarray) -> list:
    """Extract text baselines from the probability map.

    Steps:
    1. Threshold at ``_DBNET_PROB_THRESHOLD`` → binary.
    2. Erode with a (1, 3) kernel to separate vertically adjacent lines.
    3. Connected component labelling via ``ndimage.label``.
    4. For each component (filtered by size):
       a. Sample columns at ``comp_width // 10`` intervals.
       b. At each column, compute probability-weighted centroid.
       c. Reject 2.5σ outlier points.
       d. Fit ``UnivariateSpline(s=5*N, k=3)`` through the centroids.
    5. Return sorted list of (y_centre, spline, x_start, x_end).

    ``ndimage.find_objects`` gives each component's bounding box, so each
    component is read from its own sub-array instead of a full-page scan.

    Args:
        prob: Text probability map, float32 shape (H, W).

    Returns:
        Sorted list of (y_centre, spline, x_start, x_end) tuples.
    """
    binary = (prob > _DBNET_PROB_THRESHOLD).astype(np.uint8)
    kern_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (1, _BASELINE_ERODE_KERN_H))
    eroded = cv2.erode(binary, kern_erode, iterations=1)

    labeled, n_comp = cast(tuple[np.ndarray, int], ndimage.label(eroded))
    if n_comp == 0:
        return []

    comp_slices = ndimage.find_objects(labeled)
    baselines = []
    min_comp_w = int(prob.shape[1] * _BASELINE_MIN_W_RATIO)

    for comp_id, slc in enumerate(comp_slices, start=1):
        if slc is None:
            continue

        y_slc, x_slc = slc
        comp_h = y_slc.stop - y_slc.start
        comp_w = x_slc.stop - x_slc.start

        # Size filters: must be wide enough (≥5% of page width),
        # tall enough (≥5 px), and wider than tall (text-line shape).
        if comp_w < min_comp_w or comp_h < _BASELINE_MIN_H_PX or comp_w < comp_h:
            continue

        # Extract local sub-arrays (fast: no full-image scan)
        local_mask = labeled[y_slc, x_slc] == comp_id
        local_prob = prob[y_slc, x_slc]

        # Sample centroids at regular column intervals (vectorised)
        n_samples = max(comp_w // _BASELINE_SAMPLE_DIV, _BASELINE_SAMPLE_DIV)
        sample_cols = np.linspace(0, comp_w - 1, n_samples).astype(int)
        row_indices = np.arange(comp_h, dtype=np.float64)

        # Batch column extraction and weighted centroid computation
        prob_cols = local_prob[:, sample_cols] * local_mask[:, sample_cols]
        total_prob = prob_cols.sum(axis=0)
        valid = total_prob >= _BASELINE_PROB_EPS

        if np.count_nonzero(valid) < _BASELINE_MIN_VALID_PTS:
            continue

        centroids = row_indices @ prob_cols  # weighted sum per column
        tx = (x_slc.start + sample_cols[valid]).astype(np.float64)
        ty = y_slc.start + centroids[valid] / total_prob[valid]

        # Outlier rejection (2.5σ from median)
        if len(tx) > 5:
            med = np.median(ty)
            std = max(np.std(ty), 1.0)
            keep = np.abs(ty - med) < _CENTROID_OUTLIER_SIGMA * std
            tx, ty = tx[keep], ty[keep]

        if len(tx) < _SPLINE_MIN_POINTS:
            continue

        try:
            spl = UnivariateSpline(tx, ty, s=_SPLINE_SMOOTH_MULT * len(tx), k=3)
            baselines.append((float(np.mean(ty)), spl, float(tx[0]), float(tx[-1])))
        except Exception:
            continue

    baselines.sort(key=lambda b: b[0])
    return baselines


def scale_baselines(baselines: list, scale_x: float, scale_y: float) -> list:
    """Scale baseline coordinates from inference to full image resolution.

    Re-fits splines at full-resolution coordinates using densely sampled
    points from the inference-resolution splines.  The source data is
    already smooth (cubic spline output), so the re-fitting converges
    with minimal smoothing.

    Args:
        baselines: List of (y_centre, spline, x_start, x_end) at inference res.
        scale_x: Horizontal scale factor (full_width / inference_width).
        scale_y: Vertical scale factor (full_height / inference_height).

    Returns:
        List of (y_centre, spline, x_start, x_end) at full resolution.
    """
    if abs(scale_x - 1.0) < 0.001 and abs(scale_y - 1.0) < 0.001:
        return baselines

    scaled = []
    for yc, spl, xs, xe in baselines:
        # Sample the inference-resolution spline at moderate density.
        # The source is already a smooth cubic spline, so 50-100 points
        # are sufficient to reproduce it accurately at any scale.
        n_pts = min(max(int((xe - xs) / 5), 20), 100)
        x_inf = np.linspace(xs, xe, n_pts)
        y_inf = spl(x_inf)

        # Scale to full resolution
        x_full = x_inf * scale_x
        y_full = y_inf * scale_y

        try:
            spl_full = UnivariateSpline(x_full, y_full, s=len(x_full) * 0.01, k=3)
            scaled.append((yc * scale_y, spl_full, xs * scale_x, xe * scale_x))
        except Exception:
            continue

    return scaled


def _build_curvature_remap(
    baselines: list,
    h: int,
    w: int,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """Build curvature-correction remap arrays from baselines.

    For each baseline, evaluates its spline displacement (y_spline(x) - y_centre)
    across the full page width, then **removes the linear trend** to isolate
    genuine curvature from residual rotation.

    Dewarp runs before deskew, so a baseline on a rotated page, or near a
    curved region such as a book spine, keeps a slope. Left in, the field
    would carry that slope to the page edges and distort the text.

    Robustness measures:
    1. **Short baseline filtering**: Baselines spanning less than
       ``_MIN_BASELINE_SPAN`` of the page width (page numbers, signatures)
       are excluded: they carry too little curvature and give noisy estimates.
    2. **Per-column x-range masking**: Each baseline displacement is only
       applied within its actual horizontal span (with small margins),
       preventing hallucinated correction in regions with no detected text.
    3. **Margin fade-out**: The displacement field is faded to zero over
       margins above the first and below the last baseline, preventing
       edge artefacts (text duplication, ghosting).

    The displacement field is interpolated row-by-row using vectorised
    ``np.searchsorted`` to find the two bracketing baselines and linear
    weights.

    Args:
        baselines: List of (y_centre, spline, x_start, x_end) tuples.
        h, w: Image dimensions.

    Returns:
        (map_x, map_y, max_curvature) where maps are float32 arrays for
        ``cv2.remap`` and max_curvature is the peak displacement in pixels.
        The maps are None when the curvature stays below ``_MIN_CURVATURE_PX``.
    """
    if not baselines:
        return None, None, 0.0

    # Filter: discard short baselines (< 40% page width)
    min_span = w * _MIN_BASELINE_SPAN
    baselines = [b for b in baselines if (b[3] - b[2]) >= min_span]
    if len(baselines) < 2:
        return None, None, 0.0

    n_bl = len(baselines)
    bys = np.array([b[0] for b in baselines], dtype=np.float64)

    # Pre-evaluate spline displacement for each baseline,
    # subtracting the linear trend to isolate curvature only.
    ax = np.arange(w, dtype=np.float64)
    bdy = np.zeros((n_bl, w), dtype=np.float32)

    for i, (yc, spl, xs, xe) in enumerate(baselines):
        x_s = int(max(xs, 0))
        x_e = int(min(xe, w - 1))
        if x_e <= x_s:
            continue

        xv = ax[x_s : x_e + 1]
        raw = spl(xv) - yc  # raw displacement (curvature + rotation residual)

        # Remove linear trend: fit y = slope·x + intercept, subtract
        coeffs = np.polyfit(xv, raw, 1)  # [slope, intercept]
        detrended = (raw - np.polyval(coeffs, xv)).astype(np.float32)

        # Cap displacement to avoid visual distortion from extreme curvature
        np.clip(detrended, -_MAX_CURVATURE_PX, _MAX_CURVATURE_PX, out=detrended)

        bdy[i, x_s : x_e + 1] = detrended

        # Fade edge values to zero at page margins (linear ramp).
        # Curvature approaches zero at page edges (page lies flat there).
        if x_s > 0:
            fade = np.linspace(0.0, 1.0, x_s, dtype=np.float32)
            bdy[i, :x_s] = detrended[0] * fade
        if x_e < w - 1:
            count = w - 1 - x_e
            fade = np.linspace(1.0, 0.0, count, dtype=np.float32)
            bdy[i, x_e + 1 :] = detrended[-1] * fade

    # Vectorised per-row interpolation between bracketing baselines
    all_y = np.arange(h, dtype=np.float64)
    idx = np.searchsorted(bys, all_y, side="right") - 1
    idx_above = np.clip(idx, 0, n_bl - 1)
    idx_below = np.clip(idx + 1, 0, n_bl - 1)

    y_above = bys[idx_above]
    y_below = bys[idx_below]
    gap = np.maximum(y_below - y_above, 1.0)
    t = np.clip((all_y - y_above) / gap, 0.0, 1.0)

    # Rows inside baseline span: interpolate normally
    # Rows outside: fade out over a margin (typical baseline gap)
    if n_bl >= 2:
        typical_gap = float(np.median(np.diff(bys)))
    else:
        typical_gap = 60.0
    # Use wider margin to prevent vertical compression of characters
    fade_margin = typical_gap * _FADE_MARGIN_GAP_MULT

    above_all = all_y < bys[0]
    below_all = all_y >= bys[-1]

    # Fade above: 1.0 at bys[0] → 0.0 at bys[0] - fade_margin
    t[above_all] = 0.0
    idx_above[above_all] = 0
    idx_below[above_all] = 0

    # Fade below: 1.0 at bys[-1] → 0.0 at bys[-1] + fade_margin
    t[below_all] = 0.0
    idx_above[below_all] = n_bl - 1
    idx_below[below_all] = n_bl - 1

    # Linear interpolation: dy[y, :] = bdy[above, :] * (1-t) + bdy[below, :] * t
    # Chunked so the gathered rows never hold the whole page twice.
    t_col = t.astype(np.float32)
    dy_field = np.empty((h, w), dtype=np.float32)
    _CHUNK = _CURVATURE_CHUNK
    for r0 in range(0, h, _CHUNK):
        r1 = min(r0 + _CHUNK, h)
        chunk_t = t_col[r0:r1, np.newaxis]
        dy_field[r0:r1] = bdy[idx_above[r0:r1]] * (1.0 - chunk_t) + bdy[idx_below[r0:r1]] * chunk_t
    del bdy  # Free per-baseline displacement array

    # Apply cosine (smoothstep) fade-out at top/bottom margins.
    # Cosine produces zero-gradient at the boundary, preventing
    # vertical compression/stretching of characters near edges.
    fade_top_lin = np.clip((all_y - (bys[0] - fade_margin)) / fade_margin, 0.0, 1.0)
    fade_bot_lin = np.clip(((bys[-1] + fade_margin) - all_y) / fade_margin, 0.0, 1.0)
    # Smoothstep: 3t² - 2t³ (zero derivative at boundaries)
    fade_top = (3.0 * fade_top_lin**2 - 2.0 * fade_top_lin**3).astype(np.float32)
    fade_bot = (3.0 * fade_bot_lin**2 - 2.0 * fade_bot_lin**3).astype(np.float32)
    fade = (fade_top * fade_bot)[:, np.newaxis]
    dy_field *= fade

    # Smooth the displacement field to eliminate discontinuities
    # from partial-width baselines and interpolation transitions.
    # Blurred at a quarter of the size: with a sigma this large the field
    # is low-frequency, so downsampling loses nothing.
    sigma_y = max(typical_gap * _SMOOTH_SIGMA_Y_MULT, 10.0)
    sigma_x = w * _SMOOTH_SIGMA_X_RATIO
    ds = _SMOOTH_DOWNSCALE
    sm_h, sm_w = max(h // ds, 1), max(w // ds, 1)
    dy_small = cv2.resize(dy_field, (sm_w, sm_h), interpolation=cv2.INTER_AREA)
    sy, sx = sigma_y / ds, sigma_x / ds
    ky = max(int(sy * 6) | 1, 3)
    kx = max(int(sx * 6) | 1, 3)
    dy_small = cv2.GaussianBlur(dy_small, (kx, ky), sigmaX=sx, sigmaY=sy)
    # Upsampling interpolates, so the small field already holds the peak.
    max_curvature = float(np.max(np.abs(dy_small)))
    if max_curvature < _MIN_CURVATURE_PX:
        return None, None, max_curvature
    dy_field = cv2.resize(dy_small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Build map_x (identity, contiguous for cv2.remap)
    map_x = np.empty((h, w), dtype=np.float32)
    map_x[:] = np.arange(w, dtype=np.float32)

    # Build map_y: add identity to dy_field in-place to avoid extra copy
    dy_field += np.arange(h, dtype=np.float32)[:, np.newaxis]

    return map_x, dy_field, max_curvature


def probmap_dewarp(
    detector: Detector, img_bgr: np.ndarray, max_side: int = 0
) -> tuple[np.ndarray, Baselines]:
    """Apply curvature correction using DBNet probability map.

    Runs DBNet on the input image, extracts text baselines from the
    probability map, builds a vertical displacement field (with per-baseline
    linear detrending to remove rotation), and applies it via ``cv2.remap``.
    Skipped when the curvature stays below ``_MIN_CURVATURE_PX``.

    There is no global rotation step: the per-baseline linear detrending
    absorbs rotation, and the preprocessor deskews separately afterwards.

    Args:
        img_bgr: Input image in BGR format (OpenCV).
        max_side: Maximum inference side length, 0 for ``_DBNET_MAX_SIDE``.

    Returns:
        Corrected image in BGR format, or the original image if no
        correction was needed, with the baselines measured on the original.
    """
    result = img_bgr
    h, w = result.shape[:2]
    logger.info(f"Probmap dewarp: image {w}×{h}, getting probability map...")

    measured = measure_baselines(detector, result, max_side)
    logger.info(
        f"Probmap dewarp: scale {measured.scale_x:.2f}×{measured.scale_y:.2f}, "
        f"{len(measured.lines)} baselines"
    )
    if len(measured.lines) < 2:
        return result, measured

    # Scale baselines from inference to full image resolution
    baselines = scale_baselines(measured.lines, measured.scale_x, measured.scale_y)
    if len(baselines) < 2:
        logger.info("Probmap dewarp: insufficient baselines after scaling, skipping")
        return result, measured

    logger.info(f"Probmap dewarp: {len(baselines)} baselines found, building remap...")
    map_x, map_y, max_curvature = _build_curvature_remap(baselines, h, w)

    if map_x is None or map_y is None:
        logger.debug(
            f"Probmap dewarp: max curvature {max_curvature:.1f}px "
            f"below threshold {_MIN_CURVATURE_PX}px, skipping remap"
        )
        return result, measured

    logger.info(
        f"Probmap dewarp: applying curvature correction "
        f"({len(baselines)} baselines, max_curv={max_curvature:.1f}px)"
    )

    result = cv2.remap(
        result,
        map_x,
        map_y,
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return result, measured
