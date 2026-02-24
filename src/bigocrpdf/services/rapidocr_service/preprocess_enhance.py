"""
Image enhancement functions for preprocessing.

Contains all color/enhancement algorithms: illumination normalization,
text sharpening, scanner effect, denoising, border cleaning, CLAHE,
brightness adjustment, vintage look, and tone curve.

Extracted from preprocessor.py to follow single-responsibility principle.
"""

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

if TYPE_CHECKING:
    from bigocrpdf.services.rapidocr_service.config import OCRConfig

logger = logging.getLogger(__name__)


def auto_normalize_illumination(img: np.ndarray, force: bool = False) -> np.ndarray:
    """Detect and correct non-uniform illumination in document images.

    Photographed or folded documents often have brightness gradients from
    shadows, paper curl, or uneven lighting. This severely degrades OCR
    because text in dark regions has lower contrast.

    Detection: Divides image into a 4x4 grid and measures brightness variance.
    If the coefficient of variation exceeds a threshold, applies Gaussian
    background estimation + division normalization to equalize brightness.

    OPTIMIZED: Uses downscaled image for background estimation to avoid
    expensive large-kernel Gaussian blur on full-resolution images.

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

    # Threshold 0.06: photographed documents with moderate shadows have
    # CV ~0.05 but the normalization can slightly degrade PP-OCRv5 accuracy
    # on those images. Only activate for severe gradients (CV > 0.06).
    illumination_threshold = 0.06
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
    del lab  # Free LAB copy early (~105 MB)

    # OPTIMIZATION: Estimate background on downscaled image for performance
    max_bg_size = 1000
    scale = min(1.0, max_bg_size / max(h, w))

    if scale < 1.0:
        small_h, small_w = int(h * scale), int(w * scale)
        l_small = cv2.resize(l_channel, (small_w, small_h), interpolation=cv2.INTER_AREA)
    else:
        l_small = l_channel
        small_h, small_w = h, w

    # Gaussian background estimation for smooth illumination correction
    kernel_size = max(small_h, small_w) // 8
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(kernel_size, 51)  # Minimum kernel size

    background_small = cv2.GaussianBlur(l_small, (kernel_size, kernel_size), 0)

    # For severe illumination gradients (CV > 0.15), use morphological
    # background estimation which is more robust against text patterns
    if coeff_var > 0.15:
        logger.debug("Severe gradient detected, using morphological refinement")
        morph_size = max(30, min(small_h, small_w) // 20)
        if morph_size % 2 == 0:
            morph_size += 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
        bg_morph = cv2.morphologyEx(l_small, cv2.MORPH_CLOSE, se)
        bg_morph = cv2.GaussianBlur(bg_morph.astype(np.float32), (0, 0), morph_size / 2)
        # Blend Gaussian and morphological estimates for robustness
        background_small = (0.5 * background_small.astype(np.float32) + 0.5 * bg_morph).astype(
            np.uint8
        )

    # Upscale background to original size if needed
    if scale < 1.0:
        background = cv2.resize(background_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        background = background_small

    # Division normalization: pixel / background * target_brightness
    bg_min = 30.0
    target_brightness = 230.0

    # OPTIMIZATION: Process in-place to reduce memory allocations
    background_f = np.maximum(background.astype(np.float32), bg_min)
    del background  # Free background array
    normalized_l = (l_channel.astype(np.float32) / background_f) * target_brightness
    np.clip(normalized_l, 0, 255, out=normalized_l)

    # Smooth blend between original and normalized based on background brightness
    bg_blend_full = 80.0
    blend_mask = (background_f - bg_min) / (bg_blend_full - bg_min)
    np.clip(blend_mask, 0.0, 1.0, out=blend_mask)

    # Blend: normalized_l = blend_mask * normalized_l + (1 - blend_mask) * l_channel
    l_float = l_channel.astype(np.float32)
    normalized_l *= blend_mask
    l_float *= 1.0 - blend_mask
    normalized_l += l_float
    np.clip(normalized_l, 0, 255, out=normalized_l)

    # Merge normalized L channel back with original A/B (preserves color)
    normalized_lab = cv2.merge([normalized_l.astype(np.uint8), a_channel, b_channel])
    return cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)


def sharpen_text(img: np.ndarray) -> np.ndarray:
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
    # Work on L channel only to preserve colors
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    blur_score = cv2.Laplacian(l_channel, cv2.CV_64F).var()

    # Only sharpen if blur score indicates the image could benefit
    # Clean scans have blur_score > 1000; photos typically 200-800
    if blur_score > 1200:
        logger.debug(f"Image already sharp (blur_score={blur_score:.0f}), skipping sharpening")
        return img

    # Unsharp mask: original + alpha * (original - blurred)
    blurred = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(l_channel, 1.3, blurred, -0.3, 0)

    lab[:, :, 0] = sharpened  # Modify in-place, avoid split/merge
    logger.debug(f"Applied text sharpening (blur_score={blur_score:.0f})")
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_color_enhancements(img: np.ndarray, config: "OCRConfig") -> np.ndarray:
    """Apply color and enhancement processing.

    Args:
        img: Input image in BGR format
        config: OCR configuration object

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
    if config.enable_auto_contrast and contrast < 50:
        logger.debug("Applying CLAHE for low contrast image")
        result = apply_clahe(result, clip_limit=2.0)

    # Brightness adjustment
    if config.enable_auto_brightness:
        result = auto_adjust_brightness(result, mean_brightness)

    # Denoise
    if config.enable_denoise:
        logger.debug("Applying denoising")
        result = denoise(result)

    # Clean borders if enabled
    if config.enable_border_clean:
        logger.debug("Cleaning borders")
        result = clean_borders(result)

    # Vintage look
    if config.enable_vintage_look:
        logger.debug("Applying vintage scanner look")
        result = apply_vintage_look(result, config)

    return result


def auto_adjust_brightness(img: np.ndarray, mean_brightness: float) -> np.ndarray:
    """Adjust brightness based on image analysis.

    Args:
        img: Input image
        mean_brightness: Current mean brightness value

    Returns:
        Brightness adjusted image
    """
    if mean_brightness < 80:
        logger.debug("Adjusting brightness for dark image")
        return adjust_brightness(img, factor=1.3)
    elif mean_brightness > 200:
        logger.debug("Adjusting brightness for bright image")
        return adjust_brightness(img, factor=0.9)
    return img


def apply_independent_effects(img: np.ndarray, config: "OCRConfig") -> np.ndarray:
    """Apply effects that run independently of enable_preprocessing.

    Args:
        img: Input image
        config: OCR configuration object

    Returns:
        Image with effects applied
    """
    result = img

    # Scanner effect - creates professional scanner document appearance
    if config.enable_scanner_effect:
        logger.debug(f"Applying scanner effect (strength={config.scanner_effect_strength})")
        result = apply_scanner_effect(result, config.scanner_effect_strength)

    return result


def denoise(img: np.ndarray) -> np.ndarray:
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
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if laplacian_var > 1000:
        # Very noisy image — use stronger NL-means denoising
        logger.debug(
            f"High noise detected (laplacian_var={laplacian_var:.0f}), "
            "using fastNlMeansDenoisingColored"
        )
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # Standard: edge-preserving bilateral filter
    return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)


def clean_borders(img: np.ndarray) -> np.ndarray:
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


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR)


def adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """Adjust image brightness."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_scanner_effect(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
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
    normalized = _normalize_background(l_channel, h, w)

    # Apply tone curve for contrast
    enhanced_l = _apply_tone_curve(normalized, strength)

    # Merge back (preserving original colors)
    result_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


def _normalize_background(channel: np.ndarray, h: int, w: int) -> np.ndarray:
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

    # Normalize: avoid division by zero
    bg = background.astype(np.float32)
    bg = np.maximum(bg, 1.0)

    l_float = channel.astype(np.float32)
    result = cv2.divide(l_float, bg, scale=255)

    normalized = np.clip(result, 0, 255).astype(np.uint8)
    return normalized


def _apply_tone_curve(channel: np.ndarray, strength: float) -> np.ndarray:
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


def apply_vintage_look(image: np.ndarray, config: "OCRConfig") -> np.ndarray:
    """Apply vintage scanner look effects.

    Uses fixed (deterministic) values instead of random to ensure
    reproducible OCR results across multiple runs.

    Args:
        image: Input BGR image
        config: OCR configuration object

    Returns:
        Image with vintage scanner appearance
    """
    try:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Slight askew (fixed value)
        pil_img = pil_img.rotate(
            -0.3, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255)
        )

        # Black and white
        if config.vintage_bw:
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
