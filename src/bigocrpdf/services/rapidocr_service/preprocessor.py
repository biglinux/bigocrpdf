"""
Image Preprocessing for RapidOCR.

Thin coordinator that orchestrates preprocessing phases by delegating to
specialized modules:

- ``preprocess_deskew``: Deskew, angular perspective correction, rotation
- ``preprocess_enhance``: Illumination, sharpening, scanner effect, color
- ``preprocess_orientation``: Document orientation detection/correction

This is the SINGLE source of truth for ImagePreprocessor class.
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import cv2
import numpy as np

from bigocrpdf.services.rapidocr_service.geometry_trace import (
    REASON_BELOW_THRESHOLD,
    REASON_DISABLED,
    REASON_OK,
    GeometryStep,
    GeometryTrace,
)
from bigocrpdf.services.rapidocr_service.preprocess_deskew import (
    probmap_angle_deskew,
)
from bigocrpdf.services.rapidocr_service.preprocess_enhance import (
    apply_color_enhancements,
    apply_independent_effects,
)
from bigocrpdf.services.rapidocr_service.preprocess_orientation import (
    correct_orientation,
    detect_orientation,
)

if TYPE_CHECKING:
    from bigocrpdf.services.rapidocr_service.config import OCRConfig

logger = logging.getLogger(__name__)

# How much brighter than a dark border the page centre must be for the border
# to count as a border rather than as the page's own tone.
_MIN_BORDER_CONTRAST = 30

# Re-export for backward compatibility
__all__ = ["ImagePreprocessor"]


class ImagePreprocessor:
    """Adaptive image preprocessing for optimal OCR results.

    The preprocessing is split into three independent phases:
    1. Geometric corrections (perspective, deskew) - Runs based on their own flags
    2. Color/Enhancement processing - Runs if enable_preprocessing=True
    3. Scanner effect - Runs INDEPENDENTLY if enable_scanner_effect=True

    Attributes:
        config: OCR configuration object
        probmap_max_side: Max side for DBNet inference (adaptive per resource tier)
    """

    def __init__(self, config: "OCRConfig") -> None:
        """Initialize the preprocessor.

        Args:
            config: OCR configuration object
        """
        self.config = config
        # Color processing requires explicit enable_preprocessing flag
        self.enable_color_processing = config.enable_preprocessing
        # Probmap inference resolution: 0 = use default (1536).
        # On constrained systems, resource_manager sets this to 1024
        # to reduce peak memory by ~30%.
        self.probmap_max_side: int = 0
        # Track whether geometric corrections were actually applied
        # (perspective warp, dewarp, deskew) — even if dimensions stay the same,
        # the coordinate space changes and standalone mode must be used.
        self.geometry_applied: bool = False
        self.crop_applied: bool = False
        self.crop_offset_px: tuple[int, int] = (0, 0)
        self.crop_original_size_px: tuple[int, int] | None = None
        # Which correction actually ran, and why the others did not. Travels to
        # OcrPage.diagnostics; see geometry_trace.
        self.trace = GeometryTrace()

    def process(
        self,
        img: np.ndarray,
        *,
        cancel_check: Callable[[], None] | None = None,
    ) -> np.ndarray:
        """Apply preprocessing to image.

        Geometric corrections run INDEPENDENTLY of enable_preprocessing.
        Color/enhancement processing only runs if enable_preprocessing=True.
        Scanner effect runs INDEPENDENTLY if enable_scanner_effect=True.

        Args:
            img: Input image in BGR format (OpenCV)

        Returns:
            Processed image in BGR format
        """
        # No copy needed: all operations (dewarp, perspective, deskew,
        # illumination, sharpening) return NEW arrays via cv2/scipy.
        # trim_dark_borders returns a view, but downstream steps create
        # new arrays so the original is never mutated in-place.
        result = img

        # Reset per-image geometry tracking
        self.geometry_applied = False
        self.crop_applied = False
        self.crop_offset_px = (0, 0)
        self.crop_original_size_px = None
        self.trace.reset()

        # === PHASE 1: GEOMETRIC CORRECTIONS (INDEPENDENT) ===
        self._check_cancel(cancel_check)
        result = self._apply_geometric_corrections(
            result,
            cancel_check=cancel_check,
        )

        # === PHASE 2: COLOR/ENHANCEMENT PROCESSING ===
        self._check_cancel(cancel_check)
        if self.enable_color_processing:
            result = apply_color_enhancements(result, self.config)

        # === PHASE 3: INDEPENDENT EFFECTS ===
        self._check_cancel(cancel_check)
        result = apply_independent_effects(result, self.config)
        self._check_cancel(cancel_check)

        return result

    @staticmethod
    def _check_cancel(cancel_check: Callable[[], None] | None) -> None:
        if cancel_check is not None:
            cancel_check()

    def detect_orientation(self, img: np.ndarray) -> int:
        """Detect document orientation (0, 90, 180, or 270 degrees).

        Delegates to preprocess_orientation module.
        """
        return detect_orientation(img, self.config)

    def correct_orientation(self, img: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image to correct orientation.

        Delegates to preprocess_orientation module.
        """
        return correct_orientation(img, angle)

    def _apply_geometric_corrections(
        self,
        img: np.ndarray,
        *,
        cancel_check: Callable[[], None] | None = None,
    ) -> np.ndarray:
        """Apply geometric corrections (perspective, deskew, dewarp, illumination).

        Sets ``self.geometry_applied = True`` if any correction changes the
        coordinate space (even without changing dimensions).

        Args:
            img: Input image in BGR format

        Returns:
            Geometrically corrected image
        """
        result = img

        # Step 1: Dewarp — correct page curvature
        # Primary: DBNet probability-map dewarp (~500ms, best quality)
        # Fallback: 3D Coons patch + baseline refinement
        # Must run before deskew/perspective: curved pages confuse deskew.
        if getattr(self.config, "enable_baseline_dewarp", False):
            self._check_cancel(cancel_check)
            with self.trace.stage("dewarp") as step:
                before = result
                result, probmap_analyzed = self._try_probmap_dewarp(result, step)
                self._check_cancel(cancel_check)
                if result is img and not probmap_analyzed:
                    # Probmap couldn't analyze (import/runtime error),
                    # try 3D/baseline fallback
                    result = self._try_3d_dewarp(img, step)
                step.applied = result is not before
                if step.applied:
                    self.geometry_applied = True
                    step.reason = REASON_OK
        else:
            self.trace.steps.append(GeometryStep("dewarp", reason=REASON_DISABLED))

        # Step 2: Perspective correction if enabled (must run BEFORE deskew)
        if self.config.enable_perspective_correction:
            self._check_cancel(cancel_check)
            with self.trace.stage("perspective") as step:
                before = result
                result = self._correct_perspective(result, step)
                self._check_cancel(cancel_check)
                step.applied = result is not before
                if step.applied:
                    self.geometry_applied = True
                    step.reason = REASON_OK
        else:
            self.trace.steps.append(GeometryStep("perspective", reason=REASON_DISABLED))

        # Step 3: Trim dark borders from photographed documents
        # Runs ALWAYS — dark margins from camera photos confuse OCR text detection.
        self._check_cancel(cancel_check)
        with self.trace.stage("trim_dark_borders") as step:
            before = result
            result = self._trim_dark_borders(result)
            step.applied = result is not before
            if step.applied:
                step.reason = REASON_OK
                offset_x, offset_y = self.crop_offset_px
                step.params = {
                    "offset_x": offset_x,
                    "offset_y": offset_y,
                    "width": result.shape[1],
                    "height": result.shape[0],
                }

        # Step 4: Probmap-guided deskew + angular perspective correction
        if self.config.enable_deskew:
            self._check_cancel(cancel_check)
            with self.trace.stage("deskew") as step:
                before = result
                deskew_trace: dict[str, float | str] = {}
                result = probmap_angle_deskew(result, self.probmap_max_side, trace=deskew_trace)
                self._check_cancel(cancel_check)
                step.method = str(deskew_trace.pop("path", "none"))
                step.params = {
                    key: float(value)
                    for key, value in deskew_trace.items()
                    if isinstance(value, (int, float))
                }
                step.applied = result is not before
                if step.applied:
                    self.geometry_applied = True
                    step.reason = REASON_OK
        else:
            self.trace.steps.append(GeometryStep("deskew", reason=REASON_DISABLED))

        return result

    def _trim_dark_borders(self, img: np.ndarray) -> np.ndarray:
        """Remove truly black borders from photographed documents.

        Photographed documents sometimes have black borders from the camera
        capturing the background or from scanning artifacts. These black
        regions confuse OCR text detection.

        Uses per-row/column MEDIAN brightness to detect borders. Median is
        robust against text: a row with dark text on bright paper has a high
        median (paper dominates), while a genuinely dark border row has a
        low median (most pixels are dark).

        Only trims borders where the median < 60 (uniformly dark), and limits
        trimming to max 5% of each dimension to avoid removing content.

        Args:
            img: Input image in BGR format

        Returns:
            Cropped image without black borders, or original if none found
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Median brightness per row/col — robust against text characters
        dark_thresh = 60
        row_median = np.median(gray, axis=1)
        col_median = np.median(gray, axis=0)

        # A border is dark *relative to the page*. Without this, a uniformly
        # dark image -- an underexposed photograph, a dark-mode screenshot, a
        # scan of black card -- has every edge below the threshold and loses 5%
        # of each side for nothing. This is the only geometric step that runs
        # on every page unconditionally, so it needs its own guard.
        center_median = float(np.median(gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]))
        if center_median < dark_thresh + _MIN_BORDER_CONTRAST:
            logger.debug(f"Skipping border trim: page centre is itself dark ({center_median:.0f})")
            return img

        # Max trim: 5% of each dimension
        max_trim_y = h // 20
        max_trim_x = w // 20

        # Find continuous dark strips from each edge (capped)
        y_min = 0
        while y_min < max_trim_y and row_median[y_min] < dark_thresh:
            y_min += 1

        y_max = h
        while h - y_max < max_trim_y and row_median[y_max - 1] < dark_thresh:
            y_max -= 1

        x_min = 0
        while x_min < max_trim_x and col_median[x_min] < dark_thresh:
            x_min += 1

        x_max = w
        while w - x_max < max_trim_x and col_median[x_max - 1] < dark_thresh:
            x_max -= 1

        # Only crop if we're removing a meaningful amount (> 3px)
        if y_min <= 3 and h - y_max <= 3 and x_min <= 3 and w - x_max <= 3:
            return img

        # Add small margin to avoid cutting into content
        margin = 3
        y_min = max(0, y_min - margin)
        y_max = min(h, y_max + margin)
        x_min = max(0, x_min - margin)
        x_max = min(w, x_max + margin)

        result = img[y_min:y_max, x_min:x_max]

        # Guard against zero-size result (e.g., very dark scans)
        if result.shape[0] < 10 or result.shape[1] < 10:
            logger.warning("Trim would produce too-small image, keeping original")
            return img

        logger.debug(
            f"Trimmed dark borders: {img.shape[:2]} -> {result.shape[:2]} "
            f"(top={y_min}, bottom={h - y_max}, left={x_min}, right={w - x_max})"
        )
        self.crop_applied = True
        self.crop_offset_px = (x_min, y_min)
        self.crop_original_size_px = (w, h)
        return result

    def _try_probmap_dewarp(self, image: np.ndarray, step: GeometryStep) -> tuple[np.ndarray, bool]:
        """Apply curvature correction using DBNet probability map.

        Two-stage pipeline:
        1. Classical CV deskew (Otsu pseudo-boxes, ~25 ms)
        2. Curvature correction from prob-map baselines (~450 ms)

        This is the primary dewarp method, ~5× faster and higher
        quality than contour-based dewarp on rotated/curved pages.

        Args:
            image: Input BGR image.

        Returns:
            Tuple of (corrected_image, analyzed) where analyzed is True
            if probability map analysis succeeded (even if no correction
            was applied due to low curvature).
        """
        try:
            from bigocrpdf.services.rapidocr_service.dewarp_probmap import (
                probmap_dewarp,
            )

            logger.info("Probmap dewarp: starting curvature correction")
            step.method = "probmap"
            result = probmap_dewarp(image, max_side=self.probmap_max_side)

            if result is image:
                # Analyzed successfully; curvature was below the remap gate.
                step.reason = REASON_BELOW_THRESHOLD
                return image, True

            return result, True

        except ImportError as exc:
            logger.warning(f"Probmap dewarp not available: {exc}")
            step.reason = f"exception:{type(exc).__name__}"
            return image, False
        except Exception as exc:
            logger.warning(f"Probmap dewarp failed: {exc}")
            import traceback

            logger.debug(traceback.format_exc())
            step.reason = f"exception:{type(exc).__name__}"
            return image, False

    def _try_3d_dewarp(self, image: np.ndarray, step: GeometryStep) -> np.ndarray:
        """Apply 3D page dewarp using Coons patch surface simulation.

        Two-pass approach:
        Pass 1 — Coons patch from detected page boundary curves (handles
                 perspective + gross curvature from edges).
        Pass 2 — Text baseline refinement (handles residual interior curvature).

        Falls back to baseline-only dewarp if the 3D approach is unavailable
        or if page boundaries cannot be detected.

        This runs as the FIRST preprocessing step because it handles the
        fundamental 3D page geometry that confuses subsequent deskew and
        perspective correction.

        Args:
            image: Input BGR image

        Returns:
            Dewarped image, or original image if dewarp not applicable
        """
        try:
            from bigocrpdf.services.contour_dewarp import dewarp_3d

            result = dewarp_3d(image)
            if result is not None:
                step.method = "dewarp_3d"
                return result

            # Fall back to baseline-only dewarp
            return self._try_baseline_dewarp(image, step)

        except ImportError as e:
            logger.debug(f"3D dewarp not available: {e}")
            return self._try_baseline_dewarp(image, step)
        except Exception as e:
            logger.warning(f"3D dewarp failed: {e}")
            return self._try_baseline_dewarp(image, step)

    def _try_baseline_dewarp(self, image: np.ndarray, step: GeometryStep) -> np.ndarray:
        """Apply baseline dewarp to correct per-line text curvature.

        Uses Leptonica-style baseline detection: detects text lines via
        connected components, fits quadratic baselines, and builds a
        displacement field to straighten curved text. Self-regulating —
        returns the original image unchanged if there are insufficient
        text lines or negligible curvature.

        This must run AFTER deskew so text lines are roughly horizontal,
        making baseline detection reliable.

        Args:
            image: Input BGR image (already deskewed)

        Returns:
            Dewarped image, or original image if dewarp not applicable
        """
        try:
            from bigocrpdf.services.contour_dewarp import dewarp_baseline

            result = dewarp_baseline(image)
            if result is not None:
                step.method = "dewarp_baseline"
                return result
            step.method = "dewarp_baseline"
            step.reason = REASON_BELOW_THRESHOLD
            return image

        except ImportError as e:
            logger.debug(f"Baseline dewarp not available: {e}")
            step.reason = f"exception:{type(e).__name__}"
            return image
        except Exception as e:
            logger.warning(f"Baseline dewarp failed: {e}")
            step.reason = f"exception:{type(e).__name__}"
            return image

    def _correct_perspective(self, image: np.ndarray, step: GeometryStep) -> np.ndarray:
        """Correct perspective distortion in document images using OpenCV.

        Detects document boundaries and applies perspective transformation
        to straighten the document. Only applies correction when significant
        distortion is detected.

        When deskew is also enabled (enable_deskew=True), the perspective
        cascade skips its own skew correction steps to avoid double-correction.

        Args:
            image: Input BGR image

        Returns:
            Corrected image (same dimensions if no correction needed)
        """
        try:
            from bigocrpdf.services.perspective_correction import PerspectiveCorrector

            logger.info("Checking document perspective...")
            # Skip cascade's skew steps when preprocessor handles deskew separately
            skip_skew = self.config.enable_deskew
            corrector = PerspectiveCorrector(skew_threshold=0.5, skip_skew=skip_skew)
            result = corrector(image)
            step.method = corrector.last_method
            if corrector.last_rejected:
                step.reason = f"rejected:{','.join(corrector.last_rejected)}"
            return result

        except ImportError as e:
            logger.warning(f"Perspective correction not available: {e}. Returning original image.")
            step.reason = f"exception:{type(e).__name__}"
            return image
        except Exception as e:
            logger.warning(f"Perspective correction failed: {e}. Returning original image.")
            import traceback

            logger.debug(traceback.format_exc())
            step.reason = f"exception:{type(e).__name__}"
            return image
