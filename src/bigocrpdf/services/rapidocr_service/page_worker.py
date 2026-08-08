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
from bigocrpdf.services.rapidocr_service.pdf_page_geometry import render_pdf_page_to_ppm
from bigocrpdf.services.rapidocr_service.preprocessor import ImagePreprocessor
from bigocrpdf.services.rapidocr_service.resource_manager import select_pdf_page_render_dpi

logger = logging.getLogger(__name__)

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


def _write_cv_image(
    output_path: str,
    image: np.ndarray,
    parameters: list[int] | None = None,
) -> None:
    """Write an image and raise when OpenCV reports an encoder failure."""
    written = (
        cv2.imwrite(output_path, image, parameters)
        if parameters is not None
        else cv2.imwrite(output_path, image)
    )
    if not written:
        raise OSError(f"OpenCV could not write image: {output_path}")


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
    try:
        with Image.open(img_path) as img:
            if quality := _detect_jpeg_quality_from_tables(img):
                return quality
            return _lossless_or_default_quality(img.format)

    except Exception as e:
        logger.debug(f"Could not detect image quality: {e}")
    return 0  # Not detectable


def _detect_jpeg_quality_from_tables(img: Image.Image) -> int:
    qtables = getattr(img, "quantization", None)
    if not qtables:
        return 0

    first_table = list(qtables.values())[0] if isinstance(qtables, dict) else qtables[0]
    scaling_factors = [
        (actual * 100.0) / base
        for actual, base in zip(first_table, _IJG_LUMA_BASE, strict=False)
        if base > 0 and actual > 0
    ]
    if not scaling_factors:
        return 0

    avg_scaling = _median(scaling_factors)
    if avg_scaling < 100:
        quality = int(round((200 - avg_scaling) / 2))
    else:
        quality = int(round(5000.0 / avg_scaling))
    return max(1, min(100, quality))


def _median(values: list[float]) -> float:
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 0:
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2
    return sorted_values[mid]


def _lossless_or_default_quality(img_format: str | None) -> int:
    if img_format == "PNG":
        return 100
    if img_format == "WEBP":
        return 85
    if img_format in ("JPEG2000", "J2K"):
        return 95
    return 0


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
        try:
            _write_cv_image(fallback_path, img)
            # Preserve the expected path so downstream code remains format-agnostic.
            os.replace(fallback_path, output_path)
        finally:
            Path(fallback_path).unlink(missing_ok=True)


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

    img_format = _normalize_requested_format(img_format)

    detected_quality = 0
    if auto_detect and img_format == "original":
        detected_quality = detect_image_quality(img_path)
        if detected_quality > 0:
            # Use at least quality 75 to avoid generation loss when
            # re-encoding low-quality source JPEGs.  Re-encoding at the
            # original low quality (e.g. 32) adds new artifacts on top of
            # the existing ones, visibly degrading the image.
            img_quality = max(detected_quality, 75)

    if img_format == "original":
        img_format, img_quality = _format_from_original_image(img_path, img_quality)

    img_format = _normalize_output_format(img_format)

    return img_format, img_quality, detected_quality


def _normalize_requested_format(img_format: str) -> str:
    return "jpeg" if img_format == "jpg" else img_format


def _format_from_original_image(img_path: str, img_quality: int) -> tuple[str, int]:
    detected_fmt = detect_original_format(img_path)
    if detected_fmt in ("jpeg", "png", "jp2", "tiff"):
        return detected_fmt, img_quality
    if detected_fmt in ("ppm", "webp"):
        return "jpeg", img_quality or 85

    return _format_from_extension(img_path, img_quality)


def _format_from_extension(img_path: str, img_quality: int) -> tuple[str, int]:
    img_ext = Path(img_path).suffix.lower()
    if img_ext in (".jpg", ".jpeg"):
        return "jpeg", img_quality
    if img_ext == ".jp2":
        return "jp2", img_quality
    if img_ext == ".png":
        return "png", img_quality
    return "jpeg", img_quality or 85


def _normalize_output_format(img_format: str) -> str:
    if img_format in ("webp", "avif"):
        return "jpeg"
    if img_format == "tiff":
        return "png"
    return img_format


def _save_processed_image(
    processed_img: np.ndarray,
    img_format: str,
    img_quality: int,
    scratch_dir: Path | str | None = None,
) -> str:
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

    fd, temp_out = tempfile.mkstemp(suffix=suffix, dir=scratch_dir)
    os.close(fd)

    try:
        if img_format == "jp2":
            save_jpeg2000(processed_img, temp_out, img_quality)
        else:
            _write_cv_image(temp_out, processed_img, write_params)
        return temp_out
    except BaseException:
        Path(temp_out).unlink(missing_ok=True)
        raise


def _missing_image_path_result(page_num: int) -> dict[str, Any]:
    return {
        "page_num": page_num,
        "success": False,
        "skipped": True,
        "error": "No image path provided",
    }


def _create_page_preprocessor(
    config: Any, args: dict[str, Any], page_num: int
) -> ImagePreprocessor:
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
    return preprocessor


def _load_page_source_image(img_path: str) -> np.ndarray | None:
    return load_image_with_exif_rotation(Path(img_path))


def _replace_with_rendered_source(
    original_img: np.ndarray,
    args: dict[str, Any],
    page_num: int,
) -> tuple[np.ndarray, str | None]:
    if not args.get("use_rendered_source", False):
        return original_img, None

    input_pdf = args.get("input_pdf")
    if not input_pdf:
        return original_img, None

    config = args.get("config")
    preferred_dpi = int(getattr(config, "fallback_render_dpi", 300))
    render_dpi = select_pdf_page_render_dpi(
        input_pdf,
        page_num,
        preferred_dpi,
        float(getattr(config, "max_render_megapixels", 45)),
    )
    if render_dpi != preferred_dpi:
        logger.info(
            f"Page {page_num}: reducing rendered-source DPI {preferred_dpi} -> {render_dpi}"
        )
    rendered_source_path = render_pdf_page_to_ppm(
        input_pdf,
        page_num,
        render_dpi,
        output_dir=args.get("scratch_dir"),
    )
    if not rendered_source_path:
        return original_img, None

    rendered_img = cv2.imread(rendered_source_path, cv2.IMREAD_COLOR)
    if rendered_img is None:
        return original_img, rendered_source_path

    logger.info(
        f"Page {page_num}: using pdftoppm-rendered source "
        f"({rendered_img.shape[1]}x{rendered_img.shape[0]}) "
        f"instead of extracted BG layer"
    )
    return rendered_img, rendered_source_path


def _orient_page_image(
    original_img: np.ndarray,
    img_path: str,
    page_num: int,
    pdf_rotation: int,
    has_rendered_source: bool,
    args: dict[str, Any],
    preprocessor: ImagePreprocessor,
) -> tuple[np.ndarray, int, bool, int, int]:
    if args.get("skip_rotation", False) or has_rendered_source:
        orig_h, orig_w = original_img.shape[:2]
        image_prerotated = pdf_rotation != 0
    else:
        original_img, orig_h, orig_w, image_prerotated = _apply_pdf_rotation(
            original_img, img_path, page_num, pdf_rotation
        )

    if pdf_rotation != 0:
        return original_img, 0, image_prerotated, orig_h, orig_w

    orientation_angle = preprocessor.detect_orientation(original_img)
    if orientation_angle != 0:
        original_img = preprocessor.correct_orientation(original_img, orientation_angle)
        orig_h, orig_w = original_img.shape[:2]
    return original_img, orientation_angle, image_prerotated, orig_h, orig_w


def _preprocess_page_image(
    original_img: np.ndarray,
    args: dict[str, Any],
    preprocessor: ImagePreprocessor,
) -> np.ndarray:
    if args.get("skip_geometric", False):
        preprocessor.geometry_applied = False
        return original_img
    return preprocessor.process(original_img)


def _save_ocr_image_if_needed(
    processed_img: np.ndarray,
    img_format: str,
    scratch_dir: Path | str | None = None,
) -> str | None:
    if img_format == "png":
        return None

    fd_ocr, temp_ocr = tempfile.mkstemp(suffix=".png", dir=scratch_dir)
    os.close(fd_ocr)
    try:
        _write_cv_image(temp_ocr, processed_img)
        return temp_ocr
    except BaseException:
        Path(temp_ocr).unlink(missing_ok=True)
        raise


def _page_success_result(
    page_num: int,
    args: dict[str, Any],
    temp_out: str,
    temp_ocr: str | None,
    orientation_angle: int,
    image_prerotated: bool,
    pdf_rotation: int,
    orig_h: int,
    orig_w: int,
    processed_img: np.ndarray,
    detected_quality: int,
    img_format: str,
    geometry_applied: bool,
    crop_applied: bool,
    crop_offset_px: tuple[int, int],
    crop_original_size_px: tuple[int, int] | None,
) -> dict[str, Any]:
    return {
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
        "geometry_applied": geometry_applied,
        "crop_applied": crop_applied,
        "crop_offset_px": crop_offset_px,
        "crop_original_size_px": crop_original_size_px,
        "success": True,
    }


def _cleanup_rendered_source(rendered_source_path: str | None) -> None:
    if not rendered_source_path:
        return
    try:
        os.unlink(rendered_source_path)
    except OSError:
        pass


def _cleanup_owned_page_outputs(*paths: str | None) -> None:
    for path in paths:
        if not path:
            continue
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


def process_page(args: dict[str, Any]) -> dict[str, Any]:
    """Worker function for parallel page processing.

    Must be at module level for pickling with ProcessPoolExecutor.

    Args:
        args: Dictionary with page_num, img_path, config, pdf_rotation

    Returns:
        Dictionary with processing results or error info
    """
    rendered_source_path: str | None = None
    temp_out: str | None = None
    temp_ocr: str | None = None
    ownership_transferred = False
    try:
        page_num = args["page_num"]
        img_path = args["img_path"]
        config = args["config"]

        if img_path is None:
            return _missing_image_path_result(page_num)

        preprocessor = _create_page_preprocessor(config, args, page_num)
        original_img = _load_page_source_image(img_path)
        if original_img is None:
            return {"page_num": page_num, "error": f"Could not read image: {img_path}"}

        original_img, rendered_source_path = _replace_with_rendered_source(
            original_img,
            args,
            page_num,
        )

        pdf_rotation = args.get("pdf_rotation", 0)
        original_img, orientation_angle, image_prerotated, orig_h, orig_w = _orient_page_image(
            original_img,
            img_path,
            page_num,
            pdf_rotation,
            bool(rendered_source_path),
            args,
            preprocessor,
        )
        processed_img = _preprocess_page_image(original_img, args, preprocessor)

        img_format, img_quality, detected_quality = _determine_output_format(img_path, config)
        scratch_dir = args.get("scratch_dir")
        temp_out = _save_processed_image(
            processed_img,
            img_format,
            img_quality,
            scratch_dir,
        )
        temp_ocr = _save_ocr_image_if_needed(processed_img, img_format, scratch_dir)
        result = _page_success_result(
            page_num,
            args,
            temp_out,
            temp_ocr,
            orientation_angle,
            image_prerotated,
            pdf_rotation,
            orig_h,
            orig_w,
            processed_img,
            detected_quality,
            img_format,
            preprocessor.geometry_applied,
            preprocessor.crop_applied,
            preprocessor.crop_offset_px,
            preprocessor.crop_original_size_px,
        )

        del original_img, processed_img

        ownership_transferred = True
        return result

    except Exception as e:
        import traceback

        return {
            "page_num": args.get("page_num", -1),
            "error": str(e),
            "traceback": traceback.format_exc(),
            "success": False,
        }
    finally:
        _cleanup_rendered_source(rendered_source_path)
        if not ownership_transferred:
            _cleanup_owned_page_outputs(temp_out, temp_ocr)
