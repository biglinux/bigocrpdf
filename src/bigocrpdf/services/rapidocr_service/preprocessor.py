"""
Image Preprocessing for RapidOCR.

This module provides image preprocessing capabilities for optimal OCR results,
including geometric corrections (deskew, perspective correction) and color enhancements.

This is the SINGLE source of truth for ImagePreprocessor class.
"""

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

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
    """

    def __init__(self, config: "OCRConfig") -> None:
        """Initialize the preprocessor.

        Args:
            config: OCR configuration object
        """
        self.config = config
        # Color processing requires explicit enable_preprocessing flag
        self.enable_color_processing = config.enable_preprocessing

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
        result = img.copy()

        # === PHASE 1: GEOMETRIC CORRECTIONS (INDEPENDENT) ===
        result = self._apply_geometric_corrections(result)

        # === PHASE 2: COLOR/ENHANCEMENT PROCESSING ===
        if self.enable_color_processing:
            result = self._apply_color_enhancements(result)

        # === PHASE 3: INDEPENDENT EFFECTS ===
        result = self._apply_independent_effects(result)

        return result

    def _apply_geometric_corrections(self, img: np.ndarray) -> np.ndarray:
        """Apply geometric corrections (perspective, deskew, dewarp, illumination).

        Args:
            img: Input image in BGR format

        Returns:
            Geometrically corrected image
        """
        result = img

        # Step 1: Perspective correction if enabled (must run BEFORE deskew)
        if self.config.enable_perspective_correction:
            result = self._correct_perspective(result)

        # Step 2: Trim dark borders from photographed documents
        # Runs ALWAYS — dark margins from camera photos confuse OCR text detection.
        result = self._trim_dark_borders(result)

        # Step 3: Deskew if enabled
        if self.config.enable_deskew:
            skew_angle = self._detect_skew(result)
            # Auto-detect ON: only correct if skew exceeds threshold
            # Auto-detect OFF: always apply detected skew angle
            auto_detect = getattr(self.config, "enable_auto_detect", True)
            if not auto_detect or abs(skew_angle) > 0.5:
                if abs(skew_angle) > 0.01:  # Skip truly zero angles
                    logger.debug(f"Correcting skew: {skew_angle:.2f} degrees")
                    result = self._rotate_image(result, skew_angle)

        # Step 4: Baseline dewarp — correct per-line text curvature
        # Runs ALWAYS after deskew. Self-regulating: returns None if the image
        # has insufficient text lines or negligible curvature. Fast (~0.2s).
        result = self._try_baseline_dewarp(result)

        # Step 5: Normalize illumination for photographed documents
        # Runs when scanner effect is enabled OR when auto-detect is ON.
        # Auto-detect ON: only normalize if non-uniform illumination detected.
        # Auto-detect OFF + scanner ON: always normalize.
        auto_detect = getattr(self.config, "enable_auto_detect", True)
        if self.config.enable_scanner_effect or auto_detect:
            result = self._auto_normalize_illumination(result, force=not auto_detect)

        # Step 6: Sharpen text after illumination normalization
        # Only applied when scanner effect is enabled.
        if self.config.enable_scanner_effect:
            result = self._sharpen_text(result)

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

    def _auto_normalize_illumination(self, img: np.ndarray, force: bool = False) -> np.ndarray:
        """Detect and correct non-uniform illumination in document images.

        Photographed or folded documents often have brightness gradients from
        shadows, paper curl, or uneven lighting. This severely degrades OCR
        because text in dark regions has lower contrast.

        Detection: Divides image into a 4x4 grid and measures brightness variance.
        If the coefficient of variation exceeds a threshold, applies Gaussian
        background estimation + division normalization to equalize brightness.

        Args:
            img: Input BGR image
            force: If True, always apply normalization regardless of detection

        Returns:
            Illumination-normalized image, or original if uniform
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Measure brightness across a 4x4 grid (ignoring text-heavy variance)
        grid_size = 4
        region_means = []
        for row in range(grid_size):
            for col in range(grid_size):
                y1 = row * h // grid_size
                y2 = (row + 1) * h // grid_size
                x1 = col * w // grid_size
                x2 = (col + 1) * w // grid_size
                region = gray[y1:y2, x1:x2]
                # Use high percentile (paper brightness) to ignore text pixels
                region_means.append(float(np.percentile(region, 90)))

        mean_val = np.mean(region_means)
        std_val = np.std(region_means)

        if mean_val < 1:
            return img

        # Coefficient of variation: std/mean — measures relative spread
        coeff_var = std_val / mean_val

        # Log illumination variance for diagnostics
        illumination_threshold = 0.04
        if coeff_var < illumination_threshold:
            if not force:
                logger.debug(
                    f"Illumination uniform (CV={coeff_var:.3f}), "
                    "no correction needed (auto-detect mode)"
                )
                return img
            logger.debug(
                f"Illumination uniform (CV={coeff_var:.3f}), "
                "applying background normalization (force mode)"
            )
        else:
            logger.info(
                f"Non-uniform illumination detected (CV={coeff_var:.3f}), "
                "applying background normalization"
            )

        # Use LAB color space to normalize illumination while preserving color
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Gaussian background estimation for smooth illumination correction
        # Kernel size proportional to image size (captures broad gradients)
        kernel_size = max(l_channel.shape) // 8  # Larger kernel for broader gradients
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(kernel_size, 51)  # Minimum kernel size

        background = cv2.GaussianBlur(l_channel, (kernel_size, kernel_size), 0)

        # For severe illumination gradients (CV > 0.15), use morphological
        # background estimation which is more robust against text patterns
        if coeff_var > 0.15:
            logger.debug("Severe gradient detected, using morphological refinement")
            morph_size = max(30, min(l_channel.shape) // 20)
            if morph_size % 2 == 0:
                morph_size += 1
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
            bg_morph = cv2.morphologyEx(l_channel, cv2.MORPH_CLOSE, se)
            bg_morph = cv2.GaussianBlur(bg_morph.astype(np.float32), (0, 0), morph_size / 2)
            # Blend Gaussian and morphological estimates for robustness
            background = (0.5 * background.astype(np.float32) + 0.5 * bg_morph).astype(np.uint8)

        # Division normalization: pixel / background * target_brightness
        # Target 230 instead of 255 to preserve contrast without saturation
        # (255 can blow out near-white text or faint details)
        #
        # Minimum background clamp at 30 to prevent extreme amplification in
        # very dark regions (desk/shadow beyond document edges). Without this,
        # background L=2 yields 115x amplification causing visible stains.
        bg_min = 30.0
        background_f = np.maximum(background.astype(np.float32), bg_min)
        target_brightness = 230.0
        normalized_l = (l_channel.astype(np.float32) / background_f) * target_brightness
        normalized_l = np.clip(normalized_l, 0, 255).astype(np.uint8)

        # Smooth blend between original and normalized based on background
        # brightness. Very dark areas (BG <= bg_min) are non-document regions
        # (desk edges, shadows) that should keep their original appearance.
        # Full normalization applies where BG >= bg_blend_full.
        bg_blend_full = 80.0
        blend_mask = np.clip(
            (background.astype(np.float32) - bg_min) / (bg_blend_full - bg_min),
            0.0,
            1.0,
        )
        normalized_l = blend_mask * normalized_l.astype(np.float32) + (
            1.0 - blend_mask
        ) * l_channel.astype(np.float32)
        normalized_l = np.clip(normalized_l, 0, 255).astype(np.uint8)

        # Merge normalized L channel back with original A/B (preserves color)
        normalized_lab = cv2.merge([normalized_l, a_channel, b_channel])
        return cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)

    def _sharpen_text(self, img: np.ndarray) -> np.ndarray:
        """Apply subtle unsharp mask to enhance text readability.

        After illumination normalization, text edges may be slightly softened
        by the Gaussian background estimation. This light sharpening restores
        crisp text edges without amplifying noise.

        Uses LAB color space to sharpen only luminosity, preserving colors.
        Only applies when the image has characteristics of a photo (lower
        sharpness than a clean scan).

        Args:
            img: Input BGR image

        Returns:
            Sharpened image, or original if already sharp enough
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Only sharpen if blur score indicates the image could benefit
        # Clean scans have blur_score > 1000; photos typically 200-800
        if blur_score > 1200:
            logger.debug(f"Image already sharp (blur_score={blur_score:.0f}), skipping sharpening")
            return img

        # Work on L channel only to preserve colors
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Unsharp mask: original + alpha * (original - blurred)
        # sigma=2 targets fine text detail, alpha=0.3 is subtle enough
        # to avoid halo artifacts around text
        blurred = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=2)
        sharpened = cv2.addWeighted(l_channel, 1.3, blurred, -0.3, 0)

        result_lab = cv2.merge([sharpened, a_channel, b_channel])
        logger.debug(f"Applied text sharpening (blur_score={blur_score:.0f})")
        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    def _apply_color_enhancements(self, img: np.ndarray) -> np.ndarray:
        """Apply color and enhancement processing.

        Args:
            img: Input image in BGR format

        Returns:
            Enhanced image
        """
        result = img

        # Analyze image characteristics
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()
        mean_brightness = gray.mean()

        logger.debug(f"Image analysis: contrast={contrast:.1f}, brightness={mean_brightness:.1f}")

        # Apply CLAHE only if low contrast
        if self.config.enable_auto_contrast and contrast < 50:
            logger.debug("Applying CLAHE for low contrast image")
            result = self._apply_clahe(result, clip_limit=2.0)

        # Brightness adjustment
        if self.config.enable_auto_brightness:
            result = self._auto_adjust_brightness(result, mean_brightness)

        # Denoise
        if self.config.enable_denoise:
            logger.debug("Applying denoising")
            result = self._denoise(result)

        # Clean borders if enabled
        if self.config.enable_border_clean:
            logger.debug("Cleaning borders")
            result = self._clean_borders(result)

        # Vintage look
        if self.config.enable_vintage_look:
            logger.debug("Applying vintage scanner look")
            result = self._apply_vintage_look(result)

        return result

    def _auto_adjust_brightness(self, img: np.ndarray, mean_brightness: float) -> np.ndarray:
        """Adjust brightness based on image analysis.

        Args:
            img: Input image
            mean_brightness: Current mean brightness value

        Returns:
            Brightness adjusted image
        """
        if mean_brightness < 80:
            logger.debug("Adjusting brightness for dark image")
            return self._adjust_brightness(img, factor=1.3)
        elif mean_brightness > 200:
            logger.debug("Adjusting brightness for bright image")
            return self._adjust_brightness(img, factor=0.9)
        return img

    def _apply_independent_effects(self, img: np.ndarray) -> np.ndarray:
        """Apply effects that run independently of enable_preprocessing.

        Args:
            img: Input image

        Returns:
            Image with effects applied
        """
        result = img

        # Scanner effect - creates professional scanner document appearance
        if self.config.enable_scanner_effect:
            logger.debug(
                f"Applying scanner effect (strength={self.config.scanner_effect_strength})"
            )
            result = self._apply_scanner_effect(result, self.config.scanner_effect_strength)

        return result

    def detect_orientation(self, img: np.ndarray) -> int:
        """Detect document orientation (0, 90, 180, or 270 degrees).

        Uses a multi-signal approach:
        1. Aspect ratio check — documents are typically portrait
        2. Hough line analysis — text lines indicate reading direction
        3. Edge energy ratio — fallback for simple cases

        Args:
            img: Input image in BGR format

        Returns:
            Angle to rotate to correct orientation (CW rotation needed)
        """
        if not self.config.enable_orientation_detection:
            return 0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        is_landscape = w > h

        # Signal 1: Hough line direction analysis
        # Text lines in a correctly-oriented document run horizontally
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10
        )

        hough_vote = 0  # positive = needs rotation, negative = correct
        if lines is not None and len(lines) > 20:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(angle)
            angles = np.array(angles)
            # Count near-horizontal (text lines) vs near-vertical
            n_horizontal = int(np.sum(np.abs(angles) < 30) + np.sum(np.abs(angles) > 150))
            n_vertical = int(np.sum((np.abs(angles) > 60) & (np.abs(angles) < 120)))
            total = n_horizontal + n_vertical
            if total > 0:
                vert_ratio = n_vertical / total
                # If > 55% of lines are vertical, text is probably rotated
                if vert_ratio > 0.55:
                    hough_vote = 1
                elif vert_ratio < 0.45:
                    hough_vote = -1

        # Signal 2: Edge energy ratio (existing approach, relaxed threshold)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        horizontal_energy = np.sum(np.abs(sobely))
        vertical_energy = np.sum(np.abs(sobelx))
        energy_ratio = vertical_energy / (horizontal_energy + 1e-10)

        energy_vote = 0
        if energy_ratio > 1.15:
            energy_vote = 1  # More vertical edges → text is rotated
        elif energy_ratio < 0.85:
            energy_vote = -1  # More horizontal edges → text is correct

        # Signal 3: Aspect ratio (documents are usually portrait)
        aspect_vote = 1 if is_landscape else -1

        # Combine signals: need at least 2 out of 3 agreeing for rotation
        rotation_score = hough_vote + energy_vote + aspect_vote

        if rotation_score >= 2:
            # Needs rotation — determine direction (90 or 270)
            # Use ink density in left half vs right half to determine
            # which way the text is oriented.
            # For most photographed documents, the common rotation is CW
            # (phone rotated right), which needs 270° CW correction.
            # We use vertical line angles: if median angle is negative (~-90°),
            # lines go down-left → document was rotated CW → need 270° CW.
            # If median is positive (~+90°), rotated CCW → need 90° CW.
            if lines is not None and len(lines) > 20:
                vert_mask = (np.abs(angles) > 60) & (np.abs(angles) < 120)
                vert_angles = angles[vert_mask]
                if len(vert_angles) > 10:
                    n_positive = int(np.sum(vert_angles > 0))
                    n_negative = int(np.sum(vert_angles < 0))
                    if n_negative > n_positive:
                        logger.info(f"Orientation detected: 270° CW (score={rotation_score})")
                        return 270
                    else:
                        logger.info(f"Orientation detected: 90° CW (score={rotation_score})")
                        return 90

            # Fallback: use left/right ink density
            left_half = gray[:, : w // 2]
            right_half = gray[:, w // 2 :]
            left_density = np.mean(left_half < 200)
            right_density = np.mean(right_half < 200)

            if left_density > right_density:
                logger.info(f"Orientation detected: 270° CW (score={rotation_score}, fallback)")
                return 270
            else:
                logger.info(f"Orientation detected: 90° CW (score={rotation_score}, fallback)")
                return 90

        return 0

    def correct_orientation(self, img: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image to correct orientation.

        Args:
            img: Input image
            angle: Rotation angle (0, 90, 180, or 270)

        Returns:
            Rotated image
        """
        if angle == 0:
            return img
        elif angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def _detect_skew(self, img: np.ndarray) -> float:
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

        if len(text_angles) >= 5:
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

    def _rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
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

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """Remove noise using edge-preserving bilateral filter.

        bilateralFilter smooths noise while preserving text edges, which is
        critical for OCR accuracy. It outperforms medianBlur and GaussianBlur
        for document images:
        - bilateralFilter: blur score 398 (sharp text edges preserved)
        - medianBlur(3):   blur score 267 (some edge softening)
        - GaussianBlur(5): blur score  68 (significant text blurring)

        For very noisy images (photos in low light), falls back to
        fastNlMeansDenoisingColored which provides superior denoising at
        the cost of higher computation time.

        Args:
            img: Input BGR image

        Returns:
            Denoised image with text edges preserved
        """
        # Analyze noise level to choose strategy
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # High-frequency energy indicates noise level
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if laplacian_var > 1000:
            # Very noisy image — use stronger NL-means denoising
            logger.debug(
                f"High noise detected (laplacian_var={laplacian_var:.0f}), "
                "using fastNlMeansDenoisingColored"
            )
            return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # Standard: edge-preserving bilateral filter
        # d=9: filter diameter, sigmaColor=75: color similarity range,
        # sigmaSpace=75: spatial proximity range
        return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    def _clean_borders(self, img: np.ndarray) -> np.ndarray:
        """Remove dark borders typical of scanned documents."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        border_mask = np.zeros((h, w), dtype=np.uint8)

        for i in range(1, num_labels):
            x, y, w_b, h_b, area = stats[i]
            touches_edge = (x <= 2) or (y <= 2) or (x + w_b >= w - 2) or (y + h_b >= h - 2)

            if touches_edge and area <= (h * w * 0.5):
                border_mask[labels == i] = 255

        kernel = np.ones((5, 5), dtype=np.uint8)
        border_mask = cv2.dilate(border_mask, kernel, iterations=2)

        result = img.copy()
        result[border_mask == 255] = [255, 255, 255]
        return result

    def _apply_clahe(self, img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        return cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR)

    def _adjust_brightness(self, img: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _apply_scanner_effect(self, img: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Apply professional scanner emulation preserving colors.

        Uses LAB color space to enhance only luminosity while preserving original
        colors. Optimized for large images using downscaled background estimation.

        Args:
            img: Input BGR image
            strength: Effect intensity (0.5=subtle, 1.0=standard, 1.5=high contrast)

        Returns:
            Processed image with scanner-like appearance (colors preserved)
        """
        # Convert to LAB to work on luminosity only (preserves colors)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        h, w = l_channel.shape

        # Estimate and normalize background
        normalized = self._normalize_background(l_channel, h, w)

        # Apply tone curve for contrast
        enhanced_l = self._apply_tone_curve(normalized, strength)

        # Merge back (preserving original colors)
        result_lab = cv2.merge([enhanced_l, a_channel, b_channel])
        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    def _normalize_background(self, channel: np.ndarray, h: int, w: int) -> np.ndarray:
        """Normalize background illumination.

        Args:
            channel: Luminosity channel
            h: Image height
            w: Image width

        Returns:
            Normalized channel
        """
        max_bg_size = 1000
        scale = min(1.0, max_bg_size / max(h, w))

        if scale < 1.0:
            small_h, small_w = int(h * scale), int(w * scale)
            l_small = cv2.resize(channel, (small_w, small_h), interpolation=cv2.INTER_AREA)
        else:
            l_small = channel
            small_h, small_w = h, w

        # Background estimation using morphological closing
        kernel_size = max(30, min(small_h, small_w) // 20)
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        bg_small = cv2.morphologyEx(l_small, cv2.MORPH_CLOSE, kernel)
        bg_small = cv2.GaussianBlur(bg_small.astype(np.float32), (0, 0), kernel_size / 2)

        # Upscale background if needed
        if scale < 1.0:
            background = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            background = bg_small

        # Division normalization
        background = np.maximum(background, 1).astype(np.float32)
        target_white = 255.0

        normalized = (channel.astype(np.float32) / background) * target_white
        return np.clip(normalized, 0, 255).astype(np.uint8)

    def _apply_tone_curve(self, channel: np.ndarray, strength: float) -> np.ndarray:
        """Apply tone curve for scanner-like contrast.

        Args:
            channel: Normalized luminosity channel
            strength: Effect intensity

        Returns:
            Enhanced channel
        """
        gamma = 1.0 + (0.15 * strength)
        lut = np.array(
            [np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255) for i in range(256)], dtype=np.uint8
        )
        return cv2.LUT(channel, lut)

    def _apply_vintage_look(self, image: np.ndarray) -> np.ndarray:
        """Apply vintage scanner look effects.

        Uses fixed (deterministic) values instead of random to ensure
        reproducible OCR results across multiple runs.
        """
        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            # Slight askew (fixed value)
            pil_img = pil_img.rotate(
                -0.3, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255)
            )

            # Black and white
            if self.config.vintage_bw:
                pil_img = pil_img.convert("L")
                enhancer = ImageEnhance.Contrast(pil_img)
                pil_img = enhancer.enhance(1.35)
                pil_img = pil_img.convert("RGB")

            # Slight blur
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))

            # Brightness/contrast jitter (fixed values)
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(1.0)
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1.0)

            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        except Exception as e:
            logger.warning(f"Vintage look failed: {e}")
            return image

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
