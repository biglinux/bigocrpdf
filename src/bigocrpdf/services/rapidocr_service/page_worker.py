"""
Page-level worker functions for parallel PDF processing.

These functions run in threads via ThreadPoolExecutor for image
preprocessing. OCR is handled by a separate persistent subprocess
to keep memory usage under ~600 MB total.
"""

import logging
import os
import signal
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from bigocrpdf.services.rapidocr_service.pdf_extractor import load_image_with_exif_rotation
from bigocrpdf.services.rapidocr_service.preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)


def worker_init() -> None:
    """Initializer for ProcessPoolExecutor worker processes.

    Called once per worker process at startup. Performs:
    - Ignore SIGINT so only the main process handles Ctrl+C
    - Set low CPU priority (nice 19) to avoid impacting the desktop
    - Configure logging for the worker process
    """
    # Let the main process handle keyboard interrupts
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Lower CPU priority so OCR processing doesn't starve the UI
    try:
        os.nice(19)
    except OSError:
        pass  # nice() may fail in some containerised environments

    # Suppress verbose library logging in workers
    logging.getLogger("rapidocr").setLevel(logging.WARNING)
    logging.getLogger("onnxruntime").setLevel(logging.WARNING)


def detect_image_quality(img_path: str) -> int:
    """Detect JPEG quality from quantization tables using IJG algorithm.

    Uses the Independent JPEG Group (IJG) quality formula to reverse-map
    quantization tables back to the original quality setting (1-100).
    This gives precise results for standard JPEG encoders.

    Args:
        img_path: Path to the image file

    Returns:
        Detected quality (1-100) or 0 if not detectable
    """
    # IJG standard luminance quantization table (quality=50 baseline)
    _IJG_LUMA_BASE = [
        16,
        11,
        10,
        16,
        24,
        40,
        51,
        61,
        12,
        12,
        14,
        19,
        26,
        58,
        60,
        55,
        14,
        13,
        16,
        24,
        40,
        57,
        69,
        56,
        14,
        17,
        22,
        29,
        51,
        87,
        80,
        62,
        18,
        22,
        37,
        56,
        68,
        109,
        103,
        77,
        24,
        35,
        55,
        64,
        81,
        104,
        113,
        92,
        49,
        64,
        78,
        87,
        103,
        121,
        120,
        101,
        72,
        92,
        95,
        98,
        112,
        100,
        103,
        99,
    ]

    try:
        with Image.open(img_path) as img:
            # Check for JPEG quantization tables
            if hasattr(img, "quantization") and img.quantization:
                qtables = img.quantization
                if qtables:
                    # Use first quantization table (luminance)
                    first_table = (
                        list(qtables.values())[0] if isinstance(qtables, dict) else qtables[0]
                    )

                    # Estimate scaling factor from quantization table vs IJG base
                    # For each coefficient: scaling = (actual * 100) / base
                    scaling_factors = []
                    for actual, base in zip(first_table, _IJG_LUMA_BASE, strict=False):
                        if base > 0 and actual > 0:
                            scaling_factors.append((actual * 100.0) / base)

                    if scaling_factors:
                        # Use median for robustness against rounding variations
                        scaling_factors.sort()
                        mid = len(scaling_factors) // 2
                        if len(scaling_factors) % 2 == 0:
                            avg_scaling = (scaling_factors[mid - 1] + scaling_factors[mid]) / 2
                        else:
                            avg_scaling = scaling_factors[mid]

                        # Reverse IJG formula: scaling → quality
                        if avg_scaling < 100:
                            quality = int(round((200 - avg_scaling) / 2))
                        else:
                            quality = int(round(5000.0 / avg_scaling))

                        return max(1, min(100, quality))

            # For PNG: lossless, return 100
            if img.format == "PNG":
                return 100

            # For WebP: try to detect quality from file size heuristic
            if img.format == "WEBP":
                return 85  # WebP default quality

            # For JPEG2000: lossless capable
            if img.format in ("JPEG2000", "J2K"):
                return 95

    except Exception as e:
        logger.debug(f"Could not detect image quality: {e}")
    return 0  # Not detectable


def detect_original_format(img_path: str) -> str:
    """Detect the original image format from file content.

    Uses PIL to determine the actual format regardless of file extension.

    Args:
        img_path: Path to the image file

    Returns:
        Format string: 'jpeg', 'png', 'webp', 'jp2', 'tiff', or 'unknown'
    """
    try:
        with Image.open(img_path) as img:
            fmt = (img.format or "").upper()
            format_map = {
                "JPEG": "jpeg",
                "JPG": "jpeg",
                "PNG": "png",
                "WEBP": "webp",
                "JPEG2000": "jp2",
                "J2K": "jp2",
                "TIFF": "tiff",
                "PPM": "ppm",
                "PBM": "ppm",
                "PGM": "ppm",
            }
            return format_map.get(fmt, "unknown")
    except Exception as e:
        logger.debug(f"Could not detect image format: {e}")
    return "unknown"


def save_jpeg2000(img: np.ndarray, output_path: str, quality: int = 85) -> None:
    """Save image as JPEG 2000 using PIL.

    Args:
        img: OpenCV image (BGR format)
        output_path: Output file path
        quality: Quality setting (1-100, higher = better quality, larger file)
    """
    try:
        # Convert BGR to RGB for PIL
        if len(img.shape) == 3 and img.shape[2] == 3:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = img

        pil_img = Image.fromarray(rgb_img)

        # JPEG 2000 quality is controlled by 'quality_mode' and 'quality_layers'
        if quality >= 95:
            # Near lossless
            pil_img.save(output_path, "JPEG2000", quality_mode="lossless")
        else:
            # Lossy compression - map quality to compression ratio
            ratio = max(5, int(100 - quality))
            pil_img.save(output_path, "JPEG2000", quality_mode="rates", quality_layers=[ratio])

        logger.debug(f"Saved JPEG 2000: {output_path} (quality={quality})")

    except Exception as e:
        logger.warning(f"JPEG 2000 save failed, falling back to PNG: {e}")
        # Fallback to PNG (lossless, universally readable)
        fallback_path = str(Path(output_path).with_suffix(".png"))
        cv2.imwrite(fallback_path, img)
        # Rename to expected path so downstream code finds it
        if fallback_path != output_path:
            import shutil

            shutil.move(fallback_path, output_path)


def _apply_pdf_rotation(
    original_img: np.ndarray,
    img_path: str,
    page_num: int,
    pdf_rotation: int,
) -> tuple[np.ndarray, int, int, bool]:
    """Apply PDF /Rotate metadata, accounting for EXIF orientation.

    Returns:
        Tuple of (rotated_image, orig_h, orig_w, image_prerotated).
    """
    orig_h, orig_w = original_img.shape[:2]
    if pdf_rotation == 0:
        return original_img, orig_h, orig_w, False

    exif_degrees = 0
    try:
        with Image.open(img_path) as pil_check:
            exif_data = pil_check.getexif()
            exif_orient = exif_data.get(274, 1)
            _exif_to_deg = {1: 0, 2: 0, 3: 180, 4: 180, 5: 90, 6: 90, 7: 270, 8: 270}
            exif_degrees = _exif_to_deg.get(exif_orient, 0)
    except Exception as e:
        logger.debug("EXIF orientation read failed for page %d: %s", page_num, e)

    effective_rotation = (pdf_rotation - exif_degrees) % 360

    if exif_degrees:
        logger.info(
            f"Page {page_num}: EXIF orientation={exif_degrees}°, "
            f"PDF /Rotate={pdf_rotation}° → effective={effective_rotation}°"
        )

    if effective_rotation == 90:
        original_img = cv2.rotate(original_img, cv2.ROTATE_90_CLOCKWISE)
    elif effective_rotation == 180:
        original_img = cv2.rotate(original_img, cv2.ROTATE_180)
    elif effective_rotation == 270:
        original_img = cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if effective_rotation != 0:
        orig_h, orig_w = original_img.shape[:2]
        logger.info(
            f"Page {page_num}: Applied effective rotation {effective_rotation}° → {orig_w}×{orig_h}"
        )

    logger.info(
        f"Page {page_num}: Image is display-oriented "
        f"(pdf_rotation={pdf_rotation}°, exif={exif_degrees}°)"
    )
    return original_img, orig_h, orig_w, True


def _determine_output_format(img_path: str, config: Any) -> tuple[str, int, int]:
    """Determine output image format, quality, and detected quality.

    Returns:
        Tuple of (img_format, img_quality, detected_quality).
    """
    img_format = getattr(config, "image_export_format", "original")
    img_quality = getattr(config, "image_export_quality", 85)
    auto_detect = getattr(config, "auto_detect_quality", True)

    if img_format == "jpg":
        img_format = "jpeg"

    detected_quality = 0
    if auto_detect and img_format == "original":
        detected_quality = detect_image_quality(img_path)
        if detected_quality > 0:
            img_quality = detected_quality

    if img_format == "original":
        detected_fmt = detect_original_format(img_path)
        if detected_fmt in ("jpeg", "png", "jp2", "tiff"):
            img_format = detected_fmt
        elif detected_fmt == "ppm":
            img_format = "jpeg"
            if img_quality == 0:
                img_quality = 85
        elif detected_fmt == "webp":
            img_format = "jpeg"
            if img_quality == 0:
                img_quality = 85
        else:
            img_ext = Path(img_path).suffix.lower()
            if img_ext in (".jpg", ".jpeg"):
                img_format = "jpeg"
            elif img_ext == ".jp2":
                img_format = "jp2"
            elif img_ext == ".png":
                img_format = "png"
            else:
                img_format = "jpeg"
                if img_quality == 0:
                    img_quality = 85

    if img_format in ("webp", "avif"):
        img_format = "jpeg"

    if img_format == "tiff":
        img_format = "png"

    return img_format, img_quality, detected_quality


def _save_processed_image(processed_img: np.ndarray, img_format: str, img_quality: int) -> str:
    """Save processed image to temp file in the appropriate format.

    Returns:
        Path to the saved temporary file.
    """
    if img_format == "jp2":
        suffix = ".jp2"
        write_params = None
    elif img_format == "jpeg":
        suffix = ".jpg"
        write_params = [cv2.IMWRITE_JPEG_QUALITY, img_quality]
    elif img_format == "png":
        suffix = ".png"
        write_params = None
    else:
        suffix = ".jpg"
        write_params = [cv2.IMWRITE_JPEG_QUALITY, img_quality or 85]

    fd, temp_out = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    if img_format == "jp2":
        save_jpeg2000(processed_img, temp_out, img_quality)
    elif write_params:
        cv2.imwrite(temp_out, processed_img, write_params)
    else:
        cv2.imwrite(temp_out, processed_img)

    return temp_out


def process_page(args: dict[str, Any]) -> dict[str, Any]:
    """Worker function for parallel page processing.

    Must be at module level for pickling with ProcessPoolExecutor.

    Args:
        args: Dictionary with page_num, img_path, config, pdf_rotation

    Returns:
        Dictionary with processing results or error info
    """
    try:
        page_num = args["page_num"]
        img_path = args["img_path"]
        config = args["config"]

        if img_path is None:
            return {
                "page_num": page_num,
                "success": False,
                "skipped": True,
                "error": "No image path provided",
            }

        preprocessor = ImagePreprocessor(config)

        probmap_max_side = args.get("probmap_max_side", 0)
        if probmap_max_side > 0:
            preprocessor.probmap_max_side = probmap_max_side

        logger.info(
            f"Page {page_num} config: "
            f"scanner={config.enable_scanner_effect}, "
            f"perspective={config.enable_perspective_correction}, "
            f"deskew={config.enable_deskew}, orientation={config.enable_orientation_detection}, "
            f"preprocessing={config.enable_preprocessing}"
        )

        original_img = load_image_with_exif_rotation(Path(img_path))
        if original_img is None:
            return {"page_num": page_num, "error": f"Could not read image: {img_path}"}

        pdf_rotation = args.get("pdf_rotation", 0)
        original_img, orig_h, orig_w, image_prerotated = _apply_pdf_rotation(
            original_img, img_path, page_num, pdf_rotation
        )

        if pdf_rotation == 0:
            orientation_angle = preprocessor.detect_orientation(original_img)
            if orientation_angle != 0:
                original_img = preprocessor.correct_orientation(original_img, orientation_angle)
                orig_h, orig_w = original_img.shape[:2]
        else:
            orientation_angle = 0

        processed_img = preprocessor.process(original_img)

        img_format, img_quality, detected_quality = _determine_output_format(img_path, config)

        temp_out = _save_processed_image(processed_img, img_format, img_quality)

        # Save a lossless PNG for OCR when the PDF image is lossy-compressed.
        # JPEG artifacts (even at q=85) can degrade OCR accuracy — e.g.
        # "CAIXA" misread as "CAIZA".  The OCR subprocess must always
        # receive an uncompressed image for reliable text recognition.
        temp_ocr = None
        if img_format not in ("png",):
            fd_ocr, temp_ocr = tempfile.mkstemp(suffix=".png")
            os.close(fd_ocr)
            cv2.imwrite(temp_ocr, processed_img)

        result = {
            "page_num": page_num,
            "temp_out_path": temp_out,
            "temp_ocr_path": temp_ocr,
            "orientation_angle": orientation_angle,
            "image_prerotated": image_prerotated,
            "original_pdf_rotation": pdf_rotation,
            "orig_h": orig_h,
            "orig_w": orig_w,
            "proc_h": processed_img.shape[0],
            "proc_w": processed_img.shape[1],
            "detected_quality": detected_quality,
            "image_format": img_format,
            "original_encoding": args.get("original_encoding", ""),
            "success": True,
        }

        del original_img, processed_img

        return result

    except Exception as e:
        import traceback

        return {
            "page_num": args.get("page_num", -1),
            "error": str(e),
            "traceback": traceback.format_exc(),
            "success": False,
        }
