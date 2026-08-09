"""OCR command implementation for the BigOcrPdf CLI.

allow-noisy-log: OCR commands print progress and user-facing diagnostics.
"""

import argparse
import logging
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bigocrpdf.cli_parser import _parse_page_list

if TYPE_CHECKING:
    from bigocrpdf.services.rapidocr_service.config import OCRConfig


def _cmd_ocr(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle the 'ocr' command."""
    config = _build_ocr_config(args)
    page_range, page_set = _parse_ocr_page_selection(args.pages)
    if args.dewarp_only:
        return _run_dewarp_only(args, config, page_range, logger, page_set=page_set)
    return _run_full_ocr(args, config, page_range, logger, page_set=page_set)


def _build_ocr_config(args: argparse.Namespace) -> "OCRConfig":
    """Build OCRConfig from parsed CLI arguments."""
    from bigocrpdf.services.rapidocr_service.config import OCRConfig

    scanner_enabled = True  # default
    if args.no_scanner:
        scanner_enabled = False
    elif args.scanner:
        scanner_enabled = True

    # Enable preprocessing master switch if any enhancement is requested
    enable_preprocessing = any(
        [
            args.auto_contrast,
            args.auto_brightness,
            args.denoise,
        ]
    )

    return OCRConfig(
        language="latin",
        dpi=args.dpi,
        workers=args.workers,
        engine_type=args.engine_type,
        model_type=args.model_type,
        pdf_mode=args.pdf_mode,
        rec_batch_num=args.rec_batch_num,
        use_textline_cls=args.use_textline_cls,
        gpu_backend=args.gpu_backend,
        gpu_device_id=args.gpu_device_id,
        gpu_fp16=args.gpu_fp16,
        gpu_fallback_to_cpu=not args.no_gpu_fallback,
        # Geometric
        enable_baseline_dewarp=not args.no_dewarp,
        enable_deskew=not args.no_deskew,
        enable_perspective_correction=not args.no_perspective,
        enable_orientation_detection=not args.no_orientation,
        # Enhancements
        enable_scanner_effect=scanner_enabled,
        enable_preprocessing=enable_preprocessing,
        enable_auto_contrast=args.auto_contrast,
        enable_auto_brightness=args.auto_brightness,
        enable_denoise=args.denoise,
        enable_border_clean=args.border_clean,
        enable_vintage_look=args.vintage or args.vintage_bw,
        vintage_bw=args.vintage_bw,
        # Output
        convert_to_pdfa=args.pdfa,
        page_layout=args.page_layout,
        image_export_format=args.image_format,
        image_export_quality=args.image_quality,
        # Behavior
        replace_existing_ocr=args.replace_existing_ocr,
        force_full_ocr=args.force_full_ocr,
    )


def _parse_ocr_page_selection(
    pages: str | None,
) -> tuple[tuple[int, int] | None, set[int] | None]:
    """Parse OCR page selection into extraction range and optional sparse page set."""
    page_range = None
    page_set = None
    if pages:
        pages_list = _parse_page_list(pages)
        if len(pages_list) == 1:
            page_range = (pages_list[0], pages_list[0])
        elif pages_list:
            # Check if continuous range
            if pages_list == list(range(pages_list[0], pages_list[-1] + 1)):
                page_range = (pages_list[0], pages_list[-1])
            else:
                page_set = set(pages_list)
                page_range = (min(pages_list), max(pages_list))
    return page_range, page_set


# ---------------------------------------------------------------------------
# OCR helpers (moved from old cli.py)
# ---------------------------------------------------------------------------


def _load_pdf_page_rotations(input_path: Path) -> list[int]:
    """Return each page /Rotate value normalized to 0, 90, 180, or 270."""
    from bigocrpdf.services.rapidocr_service.rotation import extract_page_rotations

    return [page.original_pdf_rotation for page in extract_page_rotations(input_path)]


def _effective_dewarp_page_range(
    page_range: tuple[int, int] | None,
    page_set: set[int] | None,
) -> tuple[int, int] | None:
    """Return the extraction range needed for dewarp-only processing."""
    return (min(page_set), max(page_set)) if page_set else page_range


def _load_dewarp_image(img_path: Path, page_num: int, page_rotations: list[int], logger) -> Any:
    """Load an extracted page image and apply original PDF rotation."""
    import cv2
    import numpy as np
    from PIL import Image as PILImage

    with PILImage.open(img_path) as pil_img:
        max_pixels = 200_000_000  # ~200 MP ≈ ~600 MB at 3 bytes/px
        w, h = pil_img.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            logger.warning(
                f"Page {page_num}: image {w}×{h} too large, downsampling to {new_w}×{new_h}"
            )
            pil_img.thumbnail((new_w, new_h), PILImage.Resampling.LANCZOS)

        if pil_img.mode == "RGB":
            rgb = np.array(pil_img)
        else:
            with pil_img.convert("RGB") as converted:
                rgb = np.array(converted)

    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if page_num > len(page_rotations):
        return img
    rotation = page_rotations[page_num - 1]
    rotations = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if rotation in rotations:
        img = cv2.rotate(img, rotations[rotation])
        logger.info(f"Page {page_num}: applied /Rotate={rotation}°")
    return img


def _save_dewarp_images(
    img,
    page_num: int,
    save_dir: Path,
    preprocessor,
    logger: logging.Logger,
) -> None:
    """Save original and preprocessed debug images for one page."""
    import cv2

    logger.info(f"Page {page_num}: {img.shape[1]}×{img.shape[0]} px")

    orig_path = save_dir / f"page_{page_num:02d}_original.png"
    if not cv2.imwrite(str(orig_path), img):
        raise OSError(f"Failed to save image: {orig_path}")
    logger.info(f"  Saved original: {orig_path}")

    t0 = time.perf_counter()
    processed = preprocessor.process(img)
    elapsed = time.perf_counter() - t0

    proc_path = save_dir / f"page_{page_num:02d}_processed.png"
    if not cv2.imwrite(str(proc_path), processed):
        raise OSError(f"Failed to save image: {proc_path}")
    logger.info(f"  Saved processed: {proc_path} ({elapsed:.2f}s)")

    if img.shape != processed.shape:
        logger.info(
            f"  Geometry: {img.shape[1]}×{img.shape[0]} → {processed.shape[1]}×{processed.shape[0]}"
        )


def _run_dewarp_only(
    args: argparse.Namespace,
    config: "OCRConfig",
    page_range: tuple[int, int] | None,
    logger: logging.Logger,
    page_set: set[int] | None = None,
) -> int:
    """Run only preprocessing (dewarp/deskew) and save images."""
    save_dir = args.save_preprocessed or Path(tempfile.gettempdir()) / "bigocrpdf_debug"
    save_dir.mkdir(parents=True, exist_ok=True)

    from bigocrpdf.services.rapidocr_service.pdf_extractor import PDFImageExtractor
    from bigocrpdf.services.rapidocr_service.preprocessor import ImagePreprocessor

    extractor = PDFImageExtractor(config.dpi, config.max_render_megapixels)
    preprocessor = ImagePreprocessor(config)

    page_rotations = _load_pdf_page_rotations(args.input)
    effective_range = _effective_dewarp_page_range(page_range, page_set)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        logger.info(f"Extracting images from {args.input}...")
        images = extractor.extract(args.input, tmp_path, page_range=effective_range)

        for i, img_path in enumerate(images):
            if img_path is None:
                continue
            page_num = (effective_range[0] if effective_range else 1) + i

            if page_set and page_num not in page_set:
                continue

            img = _load_dewarp_image(img_path, page_num, page_rotations, logger)
            _save_dewarp_images(img, page_num, save_dir, preprocessor, logger)

    logger.info(f"All preprocessed images saved to {save_dir}")
    return 0


def _run_full_ocr(
    args: argparse.Namespace,
    config: "OCRConfig",
    page_range: tuple[int, int] | None,
    logger: logging.Logger,
    page_set: set[int] | None = None,
) -> int:
    """Run the full OCR pipeline."""
    from bigocrpdf.services.rapidocr_service.backend import ProfessionalPDFOCR
    from bigocrpdf.services.rapidocr_service.ocr_document_io import (
        OcrPdfPublication,
        complete_ocr_document,
        ocr_document_sidecar_path,
        publish_ocr_pdf_publications,
    )

    ocr = ProfessionalPDFOCR(config)

    def progress_cb(current: int, total: int, message: str) -> None:
        print(f"\r[{current}/{total}] {message}", end="", flush=True)

    logger.info(f"Processing {args.input} → {args.output}")
    logger.info(
        f"Config: lang={config.language}, dpi={config.dpi}, workers={config.workers}, "
        f"dewarp={config.enable_baseline_dewarp}, deskew={config.enable_deskew}, "
        f"perspective={config.enable_perspective_correction}, "
        f"scanner={config.enable_scanner_effect}, "
        f"replace_ocr={config.replace_existing_ocr}, "
        f"force_full={config.force_full_ocr}"
    )

    t0 = time.perf_counter()
    try:
        requested_output = Path(args.output)
        structured_document = None
        with tempfile.TemporaryDirectory(
            prefix=".bigocr_ocr_",
            dir=requested_output.parent,
        ) as staging_name:
            staging_dir = Path(staging_name)
            staged_output = staging_dir / requested_output.name
            input_pdf = args.input
            selected_pages = None
            if page_set:
                selected_pages = sorted(page_set)
            elif page_range:
                selected_pages = list(range(page_range[0], page_range[1] + 1))

            if selected_pages:
                input_pdf = _prepare_selected_pages(
                    args.input,
                    staging_dir / "selected-pages.pdf",
                    selected_pages,
                )
                logger.info("Selected pages: %s", args.pages)

            stats = ocr.process(
                input_pdf,
                staged_output,
                progress_callback=progress_cb,
            )
            if stats.split_output_files:
                published_outputs = publish_ocr_pdf_publications(
                    [
                        OcrPdfPublication(
                            staged_pdf=Path(part),
                            requested_pdf=requested_output.parent / Path(part).name,
                            unavailable_reason="split-page-mapping-unavailable",
                        )
                        for part in stats.split_output_files
                    ],
                    overwrite=True,
                    family_root=requested_output,
                )
                stats.split_output_files = [str(path) for path in published_outputs]
            else:
                structured_document = complete_ocr_document(
                    stats.ocr_document,
                    pages_total=stats.pages_total,
                    pages_processed=stats.pages_processed,
                )
                published_outputs = publish_ocr_pdf_publications(
                    [
                        OcrPdfPublication(
                            staged_pdf=staged_output,
                            requested_pdf=requested_output,
                            document=structured_document,
                        )
                    ],
                    overwrite=True,
                    family_root=requested_output,
                )
        elapsed = time.perf_counter() - t0
        print()  # newline after progress

        pages = stats.pages_processed
        confidence = stats.average_confidence
        logger.info(f"Done: {pages} pages, {confidence:.1%} avg confidence, {elapsed:.1f}s total")
        if len(published_outputs) == 1 and structured_document is not None:
            logger.info(
                "Saved OCR document sidecar: %s",
                ocr_document_sidecar_path(published_outputs[0]),
            )
        elif len(published_outputs) > 1 and stats.ocr_document.pages:
            logger.warning(
                "Split output uses per-part sidecar invalidation; "
                "structured OCR export is unavailable for the parts"
            )

        if args.save_preprocessed:
            logger.info("Note: --save-preprocessed requires --dewarp-only mode")

        return 0

    except Exception as e:
        elapsed = time.perf_counter() - t0
        print()
        logger.error(f"Fatal error after {elapsed:.1f}s: {e}")
        return 1


def _prepare_selected_pages(input_pdf: Path, output_pdf: Path, pages: list[int]) -> Path:
    """Return a PDF containing exactly the requested 1-indexed pages."""
    import pikepdf

    with pikepdf.open(input_pdf) as source:
        total_pages = len(source.pages)
        invalid_pages = [page for page in pages if page > total_pages]
        if invalid_pages:
            raise ValueError(
                f"Page {invalid_pages[0]} is out of range; document has {total_pages} pages"
            )
        if pages == list(range(1, total_pages + 1)):
            return input_pdf

        with pikepdf.Pdf.new() as selected:
            for page in pages:
                selected.pages.append(source.pages[page - 1])
            selected.docinfo.update(selected.copy_foreign(source.docinfo))
            selected.save(output_pdf)
    return output_pdf
