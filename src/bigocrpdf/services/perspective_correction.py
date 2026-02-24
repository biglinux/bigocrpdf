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

from bigocrpdf.services.perspective_document import (
    correct_photo_perspective,
    detect_document_contour,
    detect_photo_document_borders,
    four_point_transform,
    needs_perspective_correction,
)
from bigocrpdf.services.perspective_margins import (
    correct_perspective_from_margins,
    detect_perspective_distortion,
    gentle_margin_perspective_correction,
)
from bigocrpdf.services.perspective_skew import (
    correct_skew,
    detect_regional_skew,
    detect_skew_angle,
    mesh_perspective_correction,
)

logger = logging.getLogger(__name__)


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
