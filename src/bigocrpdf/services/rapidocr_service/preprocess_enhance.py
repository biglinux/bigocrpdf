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

# ── Illumination Normalization ─────────────────────────────────────
_ILLUMINATION_GRID_SIZE = 4
_PAPER_BRIGHTNESS_PERCENTILE = 90
_ILLUMINATION_CV_THRESHOLD = 0.06
_SEVERE_GRADIENT_CV_THRESHOLD = 0.15
_BG_ESTIMATION_MAX_SIZE = 1000  # pixels — downscale larger images
_BG_MIN_VALUE = 30.0
_TARGET_BRIGHTNESS = 230.0
_BG_BLEND_THRESHOLD = 80.0

# ── Gaussian Kernel (illumination) ─────────────────────────────────
_GAUSS_KERN_DIVISOR = 8
_GAUSS_MIN_KERN_SIZE = 51

# ── Morphological Background ──────────────────────────────────────
_MORPH_KERN_DIVISOR = 20
_MORPH_MIN_KERN_SIZE = 30

# ── Sharpening ────────────────────────────────────────────────────
_BLUR_SCORE_SHARPENING_THRESHOLD = 1200
_UNSHARP_ORIGINAL_WEIGHT = 1.3
_UNSHARP_BLUR_WEIGHT = -0.3
_UNSHARP_BLUR_SIGMA = 2

# ── CLAHE ─────────────────────────────────────────────────────────
_CLAHE_CONTRAST_THRESHOLD = 50
_CLAHE_TILE_GRID_SIZE = 8

# ── Brightness ────────────────────────────────────────────────────
_DARK_BRIGHTNESS_THRESHOLD = 80
_DARK_BRIGHTNESS_FACTOR = 1.3
_BRIGHT_BRIGHTNESS_THRESHOLD = 200
_BRIGHT_BRIGHTNESS_FACTOR = 0.9

# ── Denoising ─────────────────────────────────────────────────────
_NOISE_LAPLACIAN_THRESHOLD = 1000
_NLMEANS_H = 10
_NLMEANS_TEMPLATE_WIN = 7
_NLMEANS_SEARCH_WIN = 21
_BILATERAL_DIAMETER = 9
_BILATERAL_SIGMA = 75

# ── Border Cleaning ───────────────────────────────────────────────
_BORDER_EDGE_PX = 2
_BORDER_MAX_AREA_RATIO = 0.5
_BORDER_DILATE_KERN = 5
_BORDER_DILATE_ITERS = 2

# Background already white: skip scanner effect for born-digital content.
# Physical scans/photos always have L_p90 < 230 due to paper/lighting.
_CLEAN_BACKGROUND_L_P90 = 248


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

    # Measure brightness across a grid (ignoring text-heavy variance)
    grid_size = _ILLUMINATION_GRID_SIZE
    region_means = []
    for row in range(grid_size):
        for col in range(grid_size):
            y1 = row * h // grid_size
            y2 = (row + 1) * h // grid_size
            x1 = col * w // grid_size
            x2 = (col + 1) * w // grid_size
            region = gray[y1:y2, x1:x2]
            # Use high percentile (paper brightness) to ignore text pixels
            region_means.append(float(np.percentile(region, _PAPER_BRIGHTNESS_PERCENTILE)))

    mean_val = np.mean(region_means)
    std_val = np.std(region_means)

    if mean_val < 1:
        return img

    # Coefficient of variation: std/mean — measures relative spread
    coeff_var = std_val / mean_val

    # Threshold 0.06: photographed documents with moderate shadows have
    # CV ~0.05 but the normalization can slightly degrade PP-OCRv5 accuracy
    # on those images. Only activate for severe gradients (CV > 0.06).
    illumination_threshold = _ILLUMINATION_CV_THRESHOLD
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
    scale = min(1.0, _BG_ESTIMATION_MAX_SIZE / max(h, w))

    if scale < 1.0:
        small_h, small_w = int(h * scale), int(w * scale)
        l_small = cv2.resize(l_channel, (small_w, small_h), interpolation=cv2.INTER_AREA)
    else:
        l_small = l_channel
        small_h, small_w = h, w

    # Gaussian background estimation for smooth illumination correction
    kernel_size = max(small_h, small_w) // _GAUSS_KERN_DIVISOR
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(kernel_size, _GAUSS_MIN_KERN_SIZE)

    background_small = cv2.GaussianBlur(l_small, (kernel_size, kernel_size), 0)

    # For severe illumination gradients (CV > 0.15), use morphological
    # background estimation which is more robust against text patterns
    if coeff_var > _SEVERE_GRADIENT_CV_THRESHOLD:
        logger.debug("Severe gradient detected, using morphological refinement")
        morph_size = max(_MORPH_MIN_KERN_SIZE, min(small_h, small_w) // _MORPH_KERN_DIVISOR)
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
    bg_min = _BG_MIN_VALUE
    target_brightness = _TARGET_BRIGHTNESS

    # OPTIMIZATION: Process in-place to reduce memory allocations
    background_f = np.maximum(background.astype(np.float32), bg_min)
    del background  # Free background array
    normalized_l = (l_channel.astype(np.float32) / background_f) * target_brightness
    np.clip(normalized_l, 0, 255, out=normalized_l)

    # Smooth blend between original and normalized based on background brightness
    bg_blend_full = _BG_BLEND_THRESHOLD
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
    if blur_score > _BLUR_SCORE_SHARPENING_THRESHOLD:
        logger.debug(f"Image already sharp (blur_score={blur_score:.0f}), skipping sharpening")
        return img

    # Unsharp mask: original + alpha * (original - blurred)
    blurred = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=_UNSHARP_BLUR_SIGMA)
    sharpened = cv2.addWeighted(
        l_channel,
        _UNSHARP_ORIGINAL_WEIGHT,
        blurred,
        _UNSHARP_BLUR_WEIGHT,
        0,
    )

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
    if config.enable_auto_contrast and contrast < _CLAHE_CONTRAST_THRESHOLD:
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
    if mean_brightness < _DARK_BRIGHTNESS_THRESHOLD:
        logger.debug("Adjusting brightness for dark image")
        return adjust_brightness(img, factor=_DARK_BRIGHTNESS_FACTOR)
    elif mean_brightness > _BRIGHT_BRIGHTNESS_THRESHOLD:
        logger.debug("Adjusting brightness for bright image")
        return adjust_brightness(img, factor=_BRIGHT_BRIGHTNESS_FACTOR)
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
        # Skip on already-clean images (born-digital PDFs): background is
        # already pure white, so normalization+stretching can only distort
        # colors (especially on screenshots and colored content).
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_p90 = float(np.percentile(lab[:, :, 0], 90))
        if l_p90 >= _CLEAN_BACKGROUND_L_P90:
            logger.debug(
                f"Skipping scanner effect: background already clean (L_p90={l_p90:.0f})"
            )
        else:
            logger.debug(f"Applying scanner effect (strength={config.scanner_effect_strength})")
            result = apply_scanner_effect(result, config.scanner_effect_strength)
            # Sharpen text after scanner normalization to restore edge crispness
            result = sharpen_text(result)

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

    if laplacian_var > _NOISE_LAPLACIAN_THRESHOLD:
        # Very noisy image — use stronger NL-means denoising
        logger.debug(
            f"High noise detected (laplacian_var={laplacian_var:.0f}), "
            "using fastNlMeansDenoisingColored"
        )
        return cv2.fastNlMeansDenoisingColored(
            img,
            None,
            _NLMEANS_H,
            _NLMEANS_H,
            _NLMEANS_TEMPLATE_WIN,
            _NLMEANS_SEARCH_WIN,
        )

    # Standard: edge-preserving bilateral filter
    return cv2.bilateralFilter(
        img,
        d=_BILATERAL_DIAMETER,
        sigmaColor=_BILATERAL_SIGMA,
        sigmaSpace=_BILATERAL_SIGMA,
    )


def clean_borders(img: np.ndarray) -> np.ndarray:
    """Remove dark borders typical of scanned documents."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    border_mask = np.zeros((h, w), dtype=np.uint8)

    for i in range(1, num_labels):
        x, y, w_b, h_b, area = stats[i]
        touches_edge = (
            (x <= _BORDER_EDGE_PX)
            or (y <= _BORDER_EDGE_PX)
            or (x + w_b >= w - _BORDER_EDGE_PX)
            or (y + h_b >= h - _BORDER_EDGE_PX)
        )

        if touches_edge and area <= (h * w * _BORDER_MAX_AREA_RATIO):
            border_mask[labels == i] = 255

    kernel = np.ones((_BORDER_DILATE_KERN, _BORDER_DILATE_KERN), dtype=np.uint8)
    border_mask = cv2.dilate(border_mask, kernel, iterations=_BORDER_DILATE_ITERS)

    result = img.copy()
    result[border_mask == 255] = [255, 255, 255]
    return result


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(_CLAHE_TILE_GRID_SIZE, _CLAHE_TILE_GRID_SIZE),
    )
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
    hue and saturation.  Optimized for large images via downscaled background
    estimation.

    Pipeline (on L channel only):
      1. Morphological-closing background estimation → division normalization
         removes illumination gradients (shadows, page curl).
      2. Percentile-based contrast stretching maps paper → 255 (white) and
         ink → 0 (black), maximizing text/background separation.

    This is the established state-of-art for document whitening that also
    benefits OCR engines: PP-OCRv4/v5 text detection (DBNet) and recognition
    (SVTR) work best when the background is uniformly white and text edges
    are sharp.

    Args:
        img: Input BGR image
        strength: Effect intensity (0.5=subtle, 1.0=standard, 1.5=aggressive)

    Returns:
        Processed image with scanner-like appearance (colors preserved)
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    h, w = l_channel.shape

    # Step 1: Division normalization — flatten illumination
    normalized = _normalize_background(l_channel, h, w)

    # Step 2: Percentile-based contrast stretching
    enhanced_l = _stretch_contrast(normalized, strength)

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
    scale = min(1.0, _BG_ESTIMATION_MAX_SIZE / max(h, w))

    if scale < 1.0:
        small_h, small_w = int(h * scale), int(w * scale)
        l_small = cv2.resize(channel, (small_w, small_h), interpolation=cv2.INTER_AREA)
    else:
        l_small = channel
        small_h, small_w = h, w

    # Background estimation using morphological closing
    kernel_size = max(_MORPH_MIN_KERN_SIZE, min(small_h, small_w) // _MORPH_KERN_DIVISOR)
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


def _stretch_contrast(channel: np.ndarray, strength: float) -> np.ndarray:
    """Percentile-based contrast stretching for scanner emulation.

    Maps the paper background (high percentile) to 255 and the text floor
    (low percentile) towards 0, linearly scaling everything in between.
    This maximizes text/background contrast while keeping the image natural.

    The strength parameter controls how aggressively the percentiles are
    chosen: higher strength pushes the mapping harder towards pure white
    and pure black.

    Args:
        channel: Division-normalized luminosity channel
        strength: 0.5=gentle, 1.0=standard, 1.5=aggressive

    Returns:
        Contrast-stretched channel
    """
    # Adaptive percentile bounds based on strength
    #   strength=0.5 → p_low=5, p_high=99  (gentle)
    #   strength=1.0 → p_low=2, p_high=98  (standard)
    #   strength=1.5 → p_low=1, p_high=97  (aggressive)
    p_low_pct = max(1.0, 5.0 - 3.0 * strength)
    p_high_pct = min(99.0, 100.0 - 2.0 * strength)

    p_low = float(np.percentile(channel, p_low_pct))
    p_high = float(np.percentile(channel, p_high_pct))

    if p_high <= p_low:
        return channel

    # Linear stretch: p_low → 0, p_high → 255
    scale = 255.0 / (p_high - p_low)
    result = (channel.astype(np.float32) - p_low) * scale
    return np.clip(result, 0, 255).astype(np.uint8)


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
