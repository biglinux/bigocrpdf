"""
Image enhancement functions for preprocessing.

Contains all color/enhancement algorithms: illumination normalization,
text sharpening, scanner effect, denoising, border cleaning, CLAHE,
brightness adjustment, vintage look, and tone curve.
"""

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

if TYPE_CHECKING:
    from bigocrpdf.services.ocr.config import OCRConfig

logger = logging.getLogger(__name__)

_BG_ESTIMATION_MAX_SIZE = 1000  # pixels; larger images are downscaled


_MORPH_KERN_DIVISOR = 20
_MORPH_MIN_KERN_SIZE = 30

_BLUR_SCORE_SHARPENING_THRESHOLD = 1200
_UNSHARP_ORIGINAL_WEIGHT = 1.3
_UNSHARP_BLUR_WEIGHT = -0.3
_UNSHARP_BLUR_SIGMA = 2

_CLAHE_CONTRAST_THRESHOLD = 50
_CLAHE_TILE_GRID_SIZE = 8

_DARK_BRIGHTNESS_THRESHOLD = 80
_DARK_BRIGHTNESS_FACTOR = 1.3
_BRIGHT_BRIGHTNESS_THRESHOLD = 200
_BRIGHT_BRIGHTNESS_FACTOR = 0.9

_NOISE_LAPLACIAN_THRESHOLD = 1000
_NLMEANS_H = 10
_NLMEANS_TEMPLATE_WIN = 7
_NLMEANS_SEARCH_WIN = 21
_BILATERAL_DIAMETER = 9
_BILATERAL_SIGMA = 75

_BORDER_EDGE_PX = 2
_BORDER_MAX_AREA_RATIO = 0.5
_BORDER_DILATE_KERN = 5
_BORDER_DILATE_ITERS = 2


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
    """Apply the scanner effect to an image that will be shown in the PDF.

    Never feed the result to OCR: the recognizer is trained on natural scans
    and reads textured paper more accurately than any whitening of it.
    """
    result = img

    if config.enable_scanner_effect:
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
        # Very noisy image: use stronger NL-means denoising
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
    return np.asarray(cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR))


def adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    """Adjust image brightness."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255).astype(np.uint8)
    return np.asarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))


def apply_scanner_effect(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Whiten the paper the way a flatbed scanner does, keeping colors.

    On the LAB luminosity only:
      1. Division by a morphological-closing background estimate flattens
         illumination (shadows, page curl).
      2. The white point is set from the paper's own noise: pixels within
         ``3 * strength`` robust standard deviations of the paper level become
         white, and darker pixels keep their contrast to that white point.

    Ink is never stretched toward black: on textured paper the darkest pixels
    are grain, and stretching them turns it into speckle as dark as the text.
    Paper chroma goes neutral with the luminosity so whitened paper has no tint.

    Args:
        img: Input BGR image
        strength: Effect intensity (0.5=subtle, 1.0=standard, 1.5=aggressive)

    Returns:
        Processed image with scanner-like appearance (colors preserved)
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    normalized = _normalize_background(l_channel, *l_channel.shape).astype(np.float32)
    paper = float(np.median(normalized))
    noise = 1.4826 * float(np.median(np.abs(normalized - paper)))
    white = max(paper - 3.0 * strength * max(noise, 1.0), 1.0)
    enhanced_l = np.clip(normalized * (255.0 / white), 0, 255).astype(np.uint8)

    paper_mask = enhanced_l == 255
    a_channel[paper_mask] = 128
    b_channel[paper_mask] = 128

    result_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    return np.asarray(cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR))


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
    return np.asarray(normalized)


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
            -0.3, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(255, 255, 255)
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
        brightness_enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = brightness_enhancer.enhance(1.0)
        contrast_enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = contrast_enhancer.enhance(1.0)

        return np.asarray(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))

    except Exception as e:
        logger.warning(f"Vintage look failed: {e}")
        return image
