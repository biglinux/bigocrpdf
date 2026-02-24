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
from typing import TYPE_CHECKING

import cv2
import numpy as np

from bigocrpdf.services.rapidocr_service.preprocess_deskew import (
    probmap_angle_deskew,
)
from bigocrpdf.services.rapidocr_service.preprocess_enhance import (
    apply_color_enhancements,
    apply_independent_effects,
    auto_normalize_illumination,
    sharpen_text,
)
from bigocrpdf.services.rapidocr_service.preprocess_orientation import (
    correct_orientation,
    detect_orientation,
)

if TYPE_CHECKING:
    from bigocrpdf.services.rapidocr_service.config import OCRConfig

logger = logging.getLogger(__name__)

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

    def process(self, img: np.ndarray) -> np.ndarray:
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

        # === PHASE 1: GEOMETRIC CORRECTIONS (INDEPENDENT) ===
        result = self._apply_geometric_corrections(result)

        # === PHASE 2: COLOR/ENHANCEMENT PROCESSING ===
        if self.enable_color_processing:
            result = apply_color_enhancements(result, self.config)

        # === PHASE 3: INDEPENDENT EFFECTS ===
        result = apply_independent_effects(result, self.config)

        return result

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

    def _apply_geometric_corrections(self, img: np.ndarray) -> np.ndarray:
        """Apply geometric corrections (perspective, deskew, dewarp, illumination).

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
            result, probmap_analyzed = self._try_probmap_dewarp(result)
            if result is img and not probmap_analyzed:
                # Probmap couldn't analyze (import/runtime error),
                # try 3D/baseline fallback
                result = self._try_3d_dewarp(img)

        # Step 2: Perspective correction if enabled (must run BEFORE deskew)
        if self.config.enable_perspective_correction:
            result = self._correct_perspective(result)

        # Step 3: Trim dark borders from photographed documents
        # Runs ALWAYS — dark margins from camera photos confuse OCR text detection.
        result = self._trim_dark_borders(result)

        # Step 4: Probmap-guided deskew + angular perspective correction
        # Uses DBNet probability map baselines (reuses the same model from
        # Step 1) to measure text-line angles.  ~3× faster and ~120 MB less
        # peak memory than OCR-box detection because it only runs lightweight
        # connected-component analysis on the probability map instead of the
        # full text detection + recognition pipeline.
        #
        # Distinguishes two cases:
        #   a) Angular gradient (span > 3°, R² > 0.4, n ≥ 10): perspective
        #      makes lines tilt differently at top vs bottom → applies
        #      warpPerspective angular correction + iterative refinement.
        #   b) Uniform skew (R² ≤ 0.4): simple rotation correction.
        if self.config.enable_deskew:
            result = probmap_angle_deskew(result, self.probmap_max_side)

        # Step 5: Normalize illumination + sharpen (scanner effect pipeline)
        # Only runs when scanner effect is explicitly enabled.
        # Previously ran with auto-detect by default, which made scanned
        # pages look like the scanner effect was applied even when disabled.
        if self.config.enable_scanner_effect:
            result = auto_normalize_illumination(result, force=True)

        # Step 6: Sharpen text after illumination normalization
        # Only applied when scanner effect is enabled.
        if self.config.enable_scanner_effect:
            result = sharpen_text(result)

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
        return result

    def _try_probmap_dewarp(self, image: np.ndarray) -> tuple[np.ndarray, bool]:
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
            result = probmap_dewarp(image, max_side=self.probmap_max_side)

            if result is image:
                return image, True  # analyzed successfully, no correction needed

            return result, True

        except ImportError as exc:
            logger.warning(f"Probmap dewarp not available: {exc}")
            return image, False
        except Exception as exc:
            logger.warning(f"Probmap dewarp failed: {exc}")
            import traceback

            logger.debug(traceback.format_exc())
            return image, False

    def _try_3d_dewarp(self, image: np.ndarray) -> np.ndarray:
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
            from bigocrpdf.services.contour_analysis import dewarp_3d

            result = dewarp_3d(image)
            if result is not None:
                return result

            # Fall back to baseline-only dewarp
            return self._try_baseline_dewarp(image)

        except ImportError as e:
            logger.debug(f"3D dewarp not available: {e}")
            return self._try_baseline_dewarp(image)
        except Exception as e:
            logger.warning(f"3D dewarp failed: {e}")
            return self._try_baseline_dewarp(image)

    def _try_baseline_dewarp(self, image: np.ndarray) -> np.ndarray:
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
            from bigocrpdf.services.contour_analysis import dewarp_baseline

            result = dewarp_baseline(image)
            if result is not None:
                return result
            return image

        except ImportError as e:
            logger.debug(f"Baseline dewarp not available: {e}")
            return image
        except Exception as e:
            logger.warning(f"Baseline dewarp failed: {e}")
            return image

    def _correct_perspective(self, image: np.ndarray) -> np.ndarray:
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
            return result

        except ImportError as e:
            logger.warning(f"Perspective correction not available: {e}. Returning original image.")
            return image
        except Exception as e:
            logger.warning(f"Perspective correction failed: {e}. Returning original image.")
            import traceback

            logger.debug(traceback.format_exc())
            return image
