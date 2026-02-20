"""
Probability-Map Dewarp — fast deskew + curvature correction via DBNet.

Two-stage pipeline using the DBNet text detection probability map:

1. **Deskew** (~25 ms): Classical CV approach using Otsu binarization +
   horizontal morphological closing to form pseudo text-line segments.
   Each segment's angle is measured via ``cv2.minAreaRect``, and the
   page rotation is estimated as the MAD-filtered median of all segment
   angles.  Applied via ``cv2.warpAffine`` if |angle| >= 0.3°.

2. **Curvature correction** (~450 ms): Runs DBNet inference on the
   deskewed image to obtain a text probability map.  Connected components
   in a thresholded + eroded binary map are traced column-by-column using
   probability-weighted centroids.  A ``UnivariateSpline`` is fitted to
   each baseline, and the vertical displacement field is interpolated
   between baselines using vectorised ``searchsorted`` + linear weights.
   Applied via ``cv2.remap`` if max curvature >= 5 px.

Performance (3723×2632 page at 300 DPI):
    - Deskew:   ~25 ms  (Otsu + morphology + minAreaRect)
    - Rotate:   ~35 ms  (warpAffine, skipped if |angle| < 0.3°)
    - Probmap:  ~180 ms (DBNet OpenVINO inference)
    - Baselines: ~130 ms (ndimage.find_objects + local sub-arrays)
    - Remap:    ~120 ms (vectorised field + cv2.remap)
    - Total:    ~490 ms (vs ~2800 ms for OCR-guided with 2 iterations)

Quality improvement (18-page contrato.pdf, Tesseract word count):
    - P7:  +108% vs original, +136% vs OCR-guided
    - P13: +53% vs original, +38% vs OCR-guided
    - P15: +39% vs original, +45% vs OCR-guided
    - P2:  -1.2% (flat page, within noise)

Architecture:
    The pipeline order (deskew THEN dewarp) is critical.  On rotated +
    curved pages, the probability map on the un-deskewed image absorbs
    rotation into the curvature field, producing incorrect correction.
    Deskewing first removes the linear component so the prob-map measures
    only residual curvature.
"""

import logging
import threading

import cv2
import numpy as np
from scipy import ndimage
from scipy.interpolate import UnivariateSpline

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────

_DBNET_MODEL_PATH = "/usr/share/rapidocr/models/ch_PP-OCRv5_mobile_det.onnx"
_DBNET_MAX_SIDE = 1536
_DBNET_PROB_THRESHOLD = 0.3

# ImageNet normalisation constants (PaddleOCR convention)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Deskew thresholds
_MIN_DESKEW_ANGLE = 0.3  # degrees — skip rotation if below this
_MAX_DESKEW_ANGLE = 15.0  # degrees — reject outlier segments above this

# Curvature thresholds
_MIN_CURVATURE_PX = 5.0  # pixels — skip remap if max curvature below this
_MAX_CURVATURE_PX = 60.0  # pixels — cap per-baseline displacement to avoid distortion
_MIN_BASELINE_SPAN = 0.40  # fraction of page width — reject shorter baselines

# ── Cached OpenVINO model ──────────────────────────────────────────

_ov_model = None
_ov_lock = threading.Lock()


def _get_model():
    """Lazily load and compile the DBNet model via OpenVINO.

    The compiled model is cached globally and protected by a lock for
    thread safety.  OpenVINO compiled models support concurrent infer
    requests, so a single instance is sufficient.

    Returns:
        Compiled OpenVINO model.

    Raises:
        RuntimeError: If the model file is not found or OpenVINO fails.
    """
    global _ov_model
    if _ov_model is not None:
        return _ov_model
    with _ov_lock:
        if _ov_model is not None:
            return _ov_model
        try:
            import openvino as ov

            core = ov.Core()
            _ov_model = core.compile_model(_DBNET_MODEL_PATH, "CPU")
            logger.debug("DBNet model loaded via OpenVINO for probmap dewarp")
        except Exception as exc:
            raise RuntimeError(f"Failed to load DBNet model at {_DBNET_MODEL_PATH}: {exc}") from exc
    return _ov_model


# ── Stage 1: Classical CV Deskew ───────────────────────────────────


def detect_deskew_angle(gray: np.ndarray) -> float:
    """Detect page rotation angle using Otsu pseudo-box morphological analysis.

    Steps:
    1. Otsu binarisation on inverted grayscale image.
    2. Horizontal closing (kernel width = image_width / 80) merges
       characters into text-line segments.
    3. Vertical dilation (1×3) connects vertically adjacent components.
    4. External contours → ``cv2.minAreaRect`` angle for each.
    5. Filter segments by geometry: min height 5 px, max height 5% of
       image, min width 2% of image, max width 15%, aspect ratio ≥ 2:1,
       |angle| < 15°.
    6. MAD outlier rejection (3×MAD from median).
    7. Return median of surviving angles.

    The method agrees with DBNet neural box angles (within ~0.3°) while
    being 100× faster (~25 ms vs ~2500 ms for full OCR detection).

    Args:
        gray: Grayscale image (uint8).

    Returns:
        Detected skew angle in degrees.  Positive = clockwise tilt.
        Returns 0.0 if insufficient segments are found.
    """
    h, w = gray.shape

    # Otsu binarisation on inverted image (text → white)
    _, binary = cv2.threshold(cv2.bitwise_not(gray), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Horizontal closing: merge characters within a line
    kern_w = max(w // 80, 5)
    kern_close = cv2.getStructuringElement(cv2.MORPH_RECT, (kern_w, 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern_close)

    # Vertical dilation: connect broken segments
    kern_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    dilated = cv2.dilate(closed, kern_vert, iterations=1)

    # Find contours and measure angles
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_h_seg = int(h * 0.05)
    min_w_seg = int(w * 0.02)
    max_w_seg = int(w * 0.15)

    angles = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        (_, _), (rw, rh), angle = rect

        # Normalise OpenCV angle convention
        if rw < rh:
            angle = angle + 90.0
            actual_w, actual_h = rh, rw
        else:
            actual_w, actual_h = rw, rh

        # Geometry filters
        if actual_h < 5 or actual_h > max_h_seg:
            continue
        if actual_w < min_w_seg or actual_w > max_w_seg:
            continue
        if actual_w < actual_h * 2:
            continue

        # Normalise to [-90, 90) range
        if angle > 45:
            angle -= 90
        elif angle < -45:
            angle += 90

        if abs(angle) < _MAX_DESKEW_ANGLE:
            angles.append(angle)

    if len(angles) < 3:
        logger.debug(f"Deskew: only {len(angles)} segments, skipping")
        return 0.0

    angles = np.array(angles)

    # Discard exact-zero angles: minAreaRect returns 0° for axis-aligned
    # bounding rects, which biases the median towards zero on pages where
    # many short segments happen to be quantised to 0°.
    nonzero = angles[np.abs(angles) >= 0.01]
    if len(nonzero) >= 3:
        angles = nonzero

    median = np.median(angles)

    # MAD outlier rejection (3×MAD)
    mad = np.median(np.abs(angles - median))
    if mad > 0.01:
        mask = np.abs(angles - median) < 3.0 * mad
        angles = angles[mask]

    if len(angles) < 3:
        return 0.0

    result = float(np.median(angles))
    logger.debug(f"Deskew: {result:.2f}° from {len(angles)} segments")
    return result


# ── Stage 2: Probability-Map Curvature Correction ─────────────────


def _get_probmap(img_bgr: np.ndarray, max_side: int = 0) -> np.ndarray:
    """Run DBNet inference and return the text probability map.

    Preprocessing follows PaddleOCR convention:
    1. Resize to multiple of 32, max side ``_DBNET_MAX_SIDE``.
    2. RGB conversion, (x/255 - mean) / std normalisation.
    3. HWC → CHW → NCHW transpose.

    The output sigmoid probability map is resized back to the original
    image dimensions for pixel-accurate baseline tracing.

    Args:
        img_bgr: Input image in BGR format (OpenCV).
        max_side: Maximum inference side length. 0 = use default (1536).
                  Lower values (e.g. 1024) reduce RAM on constrained systems.

    Returns:
        Probability map as float32 array, shape (H, W), range [0, 1].
    """
    model = _get_model()
    h, w = img_bgr.shape[:2]

    effective_max = max_side if max_side > 0 else _DBNET_MAX_SIDE
    scale = min(effective_max / max(h, w), 1.0)
    new_h = int(h * scale / 32) * 32
    new_w = int(w * scale / 32) * 32

    resized = cv2.resize(img_bgr, (new_w, new_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
    blob = blob.transpose(2, 0, 1)[np.newaxis]

    result = model([blob])
    prob = result[0][0, 0]  # [N,1,H,W] → [H,W]

    # Return at inference resolution with scale factors to avoid
    # allocating a full-resolution float32 array (~37 MB for 3723×2632).
    # Baselines are extracted at inference res and scaled afterwards.
    scale_x = w / prob.shape[1]
    scale_y = h / prob.shape[0]
    return prob.copy(), scale_x, scale_y


def _extract_baselines(prob: np.ndarray, h: int, w: int) -> list:
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

    Uses ``ndimage.find_objects`` for O(1) per-component bounding box
    lookup and local sub-array indexing, achieving 14-16× speedup over
    naive ``np.where(labeled == id)`` on full images.

    Args:
        prob: Text probability map, float32 shape (H, W).
        h, w: Image dimensions.

    Returns:
        Sorted list of (y_centre, spline, x_start, x_end) tuples.
    """
    binary = (prob > _DBNET_PROB_THRESHOLD).astype(np.uint8)
    kern_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    eroded = cv2.erode(binary, kern_erode, iterations=1)

    labeled, n_comp = ndimage.label(eroded)
    if n_comp == 0:
        return []

    comp_slices = ndimage.find_objects(labeled)
    baselines = []
    min_comp_w = int(w * 0.05)

    for comp_id, slc in enumerate(comp_slices, start=1):
        if slc is None:
            continue

        y_slc, x_slc = slc
        comp_h = y_slc.stop - y_slc.start
        comp_w = x_slc.stop - x_slc.start

        # Size filters: must be wide enough (≥5% of page width),
        # tall enough (≥5 px), and wider than tall (text-line shape).
        if comp_w < min_comp_w or comp_h < 5 or comp_w < comp_h:
            continue

        # Extract local sub-arrays (fast — no full-image scan)
        local_mask = labeled[y_slc, x_slc] == comp_id
        local_prob = prob[y_slc, x_slc]

        # Sample centroids at regular column intervals (vectorised)
        n_samples = max(comp_w // 10, 10)
        sample_cols = np.linspace(0, comp_w - 1, n_samples).astype(int)
        row_indices = np.arange(comp_h, dtype=np.float64)

        # Batch column extraction and weighted centroid computation
        prob_cols = local_prob[:, sample_cols] * local_mask[:, sample_cols]
        total_prob = prob_cols.sum(axis=0)
        valid = total_prob >= 0.01

        if np.count_nonzero(valid) < 4:
            continue

        centroids = row_indices @ prob_cols  # weighted sum per column
        tx = (x_slc.start + sample_cols[valid]).astype(np.float64)
        ty = y_slc.start + centroids[valid] / total_prob[valid]

        # Outlier rejection (2.5σ from median)
        if len(tx) > 5:
            med = np.median(ty)
            std = max(np.std(ty), 1.0)
            keep = np.abs(ty - med) < 2.5 * std
            tx, ty = tx[keep], ty[keep]

        if len(tx) < 4:
            continue

        try:
            spl = UnivariateSpline(tx, ty, s=5.0 * len(tx), k=3)
            baselines.append((float(np.mean(ty)), spl, float(tx[0]), float(tx[-1])))
        except Exception:
            continue

    baselines.sort(key=lambda b: b[0])
    return baselines


def _scale_baselines(baselines: list, scale_x: float, scale_y: float) -> list:
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

    Why detrend?
    ~~~~~~~~~~~~
    The deskew stage corrects rotation using a global median angle, but
    individual baselines near curved regions (book spine) may retain a
    slope.  Without detrending, this slope is amplified to ±90 px at
    page edges, causing severe text distortion.

    Robustness measures:
    1. **Short baseline filtering**: Baselines spanning < 20% of the page
       width (e.g. page numbers, signatures) are excluded — they lack
       sufficient curvature information and produce noisy estimates.
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
        Returns (None, None, 0.0) if no correction is needed.
    """
    if not baselines:
        return None, None, 0.0

    # ── Filter: discard short baselines (< 40% page width) ──
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

    # ── Vectorised per-row interpolation between bracketing baselines ──
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
    fade_margin = typical_gap * 3.0

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
    # Chunked to reduce peak memory from ~185 MB to ~4 MB.
    t_col = t.astype(np.float32)
    dy_field = np.empty((h, w), dtype=np.float32)
    _CHUNK = 128
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
    # Downscale 4× → GaussianBlur → upscale for ~16× speedup.
    # Large sigma means the information is low-frequency, so
    # downsampling is effectively lossless.
    sigma_y = max(typical_gap * 0.4, 10.0)
    sigma_x = w * 0.03
    ds = 4
    sm_h, sm_w = max(h // ds, 1), max(w // ds, 1)
    dy_small = cv2.resize(dy_field, (sm_w, sm_h), interpolation=cv2.INTER_AREA)
    sy, sx = sigma_y / ds, sigma_x / ds
    ky = max(int(sy * 6) | 1, 3)
    kx = max(int(sx * 6) | 1, 3)
    dy_small = cv2.GaussianBlur(dy_small, (kx, ky), sigmaX=sx, sigmaY=sy)
    dy_field = cv2.resize(dy_small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Max curvature from displacement field (before adding identity)
    max_curvature = float(np.max(np.abs(dy_field)))

    # Build map_x (identity, contiguous for cv2.remap)
    map_x = np.empty((h, w), dtype=np.float32)
    map_x[:] = np.arange(w, dtype=np.float32)

    # Build map_y: add identity to dy_field in-place to avoid extra copy
    dy_field += np.arange(h, dtype=np.float32)[:, np.newaxis]

    return map_x, dy_field, max_curvature


# ── Main Entry Point ───────────────────────────────────────────────


def probmap_dewarp(img_bgr: np.ndarray, max_side: int = 0) -> np.ndarray:
    """Apply curvature correction using DBNet probability map.

    Runs DBNet on the input image, extracts text baselines from the
    probability map, builds a vertical displacement field (with per-baseline
    linear detrending to remove rotation), and applies it via ``cv2.remap``.
    Skipped if max curvature < 5 px.

    Deskew is intentionally omitted: extensive testing on PDF-extracted images
    showed that all deskew methods (Otsu, baseline-slope, smart-slope) degrade
    OCR quality.  The per-baseline linear detrending already handles rotation
    residuals without a separate global rotation step.

    Args:
        img_bgr: Input image in BGR format (OpenCV).
        max_side: Maximum inference side length. 0 = use default (1536).

    Returns:
        Corrected image in BGR format.  Returns the original image
        if no correction was needed.
    """
    result = img_bgr
    h, w = result.shape[:2]
    logger.info(f"Probmap dewarp: image {w}×{h}, getting probability map...")

    # ── Curvature correction ──
    prob, scale_x, scale_y = _get_probmap(result, max_side=max_side)
    inf_h, inf_w = prob.shape[:2]
    logger.info(
        f"Probmap dewarp: probmap {inf_w}×{inf_h} "
        f"(scale {scale_x:.2f}×{scale_y:.2f}), extracting baselines..."
    )
    baselines = _extract_baselines(prob, inf_h, inf_w)
    del prob  # Free inference-resolution probmap early

    if len(baselines) < 2:
        logger.info(f"Probmap dewarp: only {len(baselines)} baselines found, skipping")
        return result

    # Scale baselines from inference to full image resolution
    baselines = _scale_baselines(baselines, scale_x, scale_y)
    if len(baselines) < 2:
        logger.info("Probmap dewarp: insufficient baselines after scaling, skipping")
        return result

    logger.info(f"Probmap dewarp: {len(baselines)} baselines found, building remap...")
    map_x, map_y, max_curvature = _build_curvature_remap(baselines, h, w)

    if map_x is None or max_curvature < _MIN_CURVATURE_PX:
        logger.debug(
            f"Probmap dewarp: max curvature {max_curvature:.1f}px "
            f"below threshold {_MIN_CURVATURE_PX}px, skipping remap"
        )
        return result

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

    return result


# ── Public API ─────────────────────────────────────────────────────

__all__ = [
    "detect_deskew_angle",
    "probmap_dewarp",
]
