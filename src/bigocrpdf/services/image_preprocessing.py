"""
BigOcrPdf - Image Preprocessing Module

This module provides image preprocessing functions to improve OCR accuracy.
Uses PIL/Pillow for image manipulation.
"""

import os
import tempfile

from bigocrpdf.utils.logger import logger

# Try to import PIL, handle gracefully if not available
try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not available. Image preprocessing will be disabled.")


# Constants for image preprocessing
DEFAULT_TARGET_DPI = 300
MIN_WIDTH_FOR_UPSCALE = 1000  # Minimum width before upscaling
CONTRAST_FACTOR = 1.3  # Contrast enhancement factor
SHARPNESS_FACTOR = 1.5  # Sharpness enhancement factor


def is_preprocessing_available() -> bool:
    """Check if image preprocessing is available.

    Returns:
        True if PIL/Pillow is available
    """
    return PIL_AVAILABLE


def preprocess_image_for_ocr(
    image_path: str,
    target_dpi: int = DEFAULT_TARGET_DPI,
    enhance_contrast: bool = True,
    sharpen: bool = True,
    grayscale: bool = True,
    denoise: bool = True,
) -> tuple[str, bool]:
    """Preprocess an image to improve OCR accuracy.

    Applies various image processing techniques:
    - Upscaling: Increases resolution for small images
    - Grayscale conversion: Removes color noise
    - Contrast enhancement: Improves text visibility
    - Sharpening: Makes text edges clearer
    - Denoising: Reduces image noise

    Args:
        image_path: Path to the original image
        target_dpi: Target DPI for upscaling (default: 300)
        enhance_contrast: Whether to enhance contrast
        sharpen: Whether to apply sharpening
        grayscale: Whether to convert to grayscale
        denoise: Whether to apply noise reduction

    Returns:
        Tuple of (processed_image_path, is_temporary)
        - processed_image_path: Path to the processed image
        - is_temporary: Whether the processed image is a temp file that should be deleted
    """
    if not PIL_AVAILABLE:
        logger.debug("PIL not available, returning original image")
        return image_path, False

    try:
        # Open the image
        with Image.open(image_path) as img:
            processed = img.copy()

            # Convert to RGB if necessary (handles RGBA, palette, etc.)
            if processed.mode not in ("RGB", "L"):
                processed = processed.convert("RGB")

            # Step 1: Upscale small images
            processed = _upscale_if_needed(processed, target_dpi)

            # Step 2: Convert to grayscale (reduces color noise)
            if grayscale and processed.mode != "L":
                processed = ImageOps.grayscale(processed)
                logger.debug("Converted image to grayscale")

            # Step 3: Apply denoising (median filter)
            if denoise:
                processed = processed.filter(ImageFilter.MedianFilter(size=3))
                logger.debug("Applied noise reduction filter")

            # Step 4: Enhance contrast
            if enhance_contrast:
                enhancer = ImageEnhance.Contrast(processed)
                processed = enhancer.enhance(CONTRAST_FACTOR)
                logger.debug(f"Enhanced contrast by factor {CONTRAST_FACTOR}")

            # Step 5: Sharpen the image
            if sharpen:
                enhancer = ImageEnhance.Sharpness(processed)
                processed = enhancer.enhance(SHARPNESS_FACTOR)
                logger.debug(f"Enhanced sharpness by factor {SHARPNESS_FACTOR}")

            # Save processed image to temporary file
            fd, temp_path = tempfile.mkstemp(suffix=".png", prefix="bigocrpdf_processed_")
            os.close(fd)

            # Save as PNG for best quality (lossless)
            processed.save(temp_path, format="PNG", optimize=True)
            logger.info(
                f"Preprocessed image saved to {temp_path} "
                f"(size: {processed.size[0]}x{processed.size[1]})"
            )

            return temp_path, True

    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        # Return original image if preprocessing fails
        return image_path, False


def _upscale_if_needed(img: "Image.Image", target_dpi: int) -> "Image.Image":
    """Upscale image if it's too small for good OCR.

    Args:
        img: PIL Image object
        target_dpi: Target DPI for upscaling

    Returns:
        Upscaled image or original if already large enough
    """
    width, height = img.size

    # Only upscale if image is small
    if width < MIN_WIDTH_FOR_UPSCALE:
        # Calculate scale factor to reach target size
        scale_factor = max(2.0, MIN_WIDTH_FOR_UPSCALE / width)

        # Limit scale factor to avoid memory issues
        scale_factor = min(scale_factor, 4.0)

        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Use LANCZOS for high-quality upscaling
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.debug(
            f"Upscaled image from {width}x{height} to {new_width}x{new_height} "
            f"(factor: {scale_factor:.2f})"
        )

    return img


def cleanup_temp_image(image_path: str, is_temporary: bool) -> None:
    """Clean up a temporary processed image.

    Args:
        image_path: Path to the image
        is_temporary: Whether the image is temporary and should be deleted
    """
    if is_temporary and image_path and os.path.exists(image_path):
        try:
            os.unlink(image_path)
            logger.debug(f"Cleaned up temporary image: {image_path}")
        except Exception as e:
            logger.warning(f"Could not delete temporary image {image_path}: {e}")
