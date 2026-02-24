"""Text Layer Rendering Mixin for ProfessionalPDFOCR."""

import gc
import os
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pikepdf
from PIL import Image
from reportlab.pdfgen import canvas

from bigocrpdf.services.rapidocr_service.config import OCRResult, ProcessingStats
from bigocrpdf.services.rapidocr_service.page_worker import (
    process_page,
)
from bigocrpdf.services.rapidocr_service.pdf_assembly import (
    append_text_to_page,
    create_text_layer_commands,
    overlay_text_on_original,
)
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class BackendTextLayerMixin:
    """Mixin providing text layer creation and rendering methods."""

    def _create_text_layer_commands(
        self,
        ocr_results: list[OCRResult],
        img_x: float,
        img_y: float,
        img_width: float,
        img_height: float,
        scale_x: float,
        scale_y: float,
    ) -> list[str]:
        """Create PDF text layer commands for OCR results."""
        return create_text_layer_commands(
            ocr_results, img_x, img_y, img_width, img_height, scale_x, scale_y
        )

    def _append_text_to_page(
        self, pdf: pikepdf.Pdf, page: pikepdf.Page, text_commands: list[str]
    ) -> None:
        """Append text layer commands to a PDF page."""
        append_text_to_page(pdf, page, text_commands)

    def _determine_page_mode(
        self,
        result: dict,
        proc_w: int,
        proc_h: int,
    ) -> tuple[bool, bool]:
        """Determine standalone vs overlay mode for a page.

        Returns:
            Tuple of (use_processed_for_page, geometry_changed)
        """
        orig_h = result.get("orig_h", proc_h)
        orig_w = result.get("orig_w", proc_w)
        total_size = orig_h + orig_w
        dim_change = abs(orig_h - proc_h) + abs(orig_w - proc_w)
        change_ratio = dim_change / total_size if total_size > 0 else 0
        geometry_changed = change_ratio > 0.05 or (result.get("orientation_angle", 0) != 0)

        # When the image was pre-rotated (PDF /Rotate applied), the image
        # dimensions no longer match the original MediaBox and standalone
        # mode must be used so the /Rotate is not needed on the output page.
        image_prerotated = result.get("image_prerotated", False)
        original_pdf_rotation = result.get("original_pdf_rotation", 0)
        if image_prerotated and original_pdf_rotation != 0:
            geometry_changed = True

        # Force standalone mode when user explicitly chose a non-original format
        # (e.g. JPEG custom quality), otherwise the format/quality settings are ignored
        format_changed = self.config.image_export_format not in ("original", "")

        # Force standalone mode when appearance-altering effects are enabled,
        # otherwise the processed image is discarded and the original is kept
        appearance_changed = (
            self.config.enable_border_clean
            or self.config.enable_scanner_effect
            or (
                self.config.enable_preprocessing
                and (
                    self.config.enable_auto_contrast
                    or self.config.enable_auto_brightness
                    or self.config.enable_denoise
                    or self.config.enable_vintage_look
                )
            )
        )

        # Geometric preprocessing that may change pixel content without
        # changing dimensions (e.g. small deskew rotation)
        geometric_preprocessing = (
            self.config.enable_deskew or self.config.enable_perspective_correction
        )

        use_processed_for_page = (
            geometry_changed or format_changed or appearance_changed or geometric_preprocessing
        )

        if geometry_changed:
            logger.info(
                f"Page {result.get('page_num', '?')}: significant geometry change "
                f"({orig_w}x{orig_h} → {proc_w}x{proc_h}, "
                f"{change_ratio:.1%}), using processed image in PDF"
            )

        if format_changed and not geometry_changed:
            logger.debug(
                f"Page {result.get('page_num', '?')}: using processed image "
                f"(export format: {self.config.image_export_format})"
            )

        if appearance_changed and not geometry_changed and not format_changed:
            logger.debug(
                f"Page {result.get('page_num', '?')}: using processed image "
                f"(appearance effects enabled)"
            )

        return use_processed_for_page, geometry_changed

    def _render_ocr_to_page(
        self,
        c: canvas.Canvas,
        ocr_image: np.ndarray,
        page_num: int,
        pdf_width: float,
        pdf_height: float,
        pdf_rotation: int,
        ocr_img_size: tuple[int, int],
        use_processed_for_page: bool,
        draw_image_path: str | None,
        stats: ProcessingStats,
        precomputed_ocr: list[OCRResult] | None = None,
    ) -> float:
        """Run OCR on image and render results to a PDF page.

        Args:
            precomputed_ocr: If provided, skip OCR subprocess and use these
                results directly. This enables parallel OCR across pages.

        Returns:
            Total confidence contribution from this page
        """
        if precomputed_ocr is not None:
            ocr_results = precomputed_ocr
            logger.info(f"OCR page {page_num}: {len(ocr_results)} text regions (pre-computed)")
        else:
            ocr_results = self._run_ocr(ocr_image)
            logger.info(f"OCR page {page_num}: {len(ocr_results)} text regions")

        ocr_results = self._fix_vertical_overlaps(ocr_results)

        c.setPageSize((pdf_width, pdf_height))

        # If using processed image, draw it on the PDF
        if use_processed_for_page and draw_image_path:
            c.drawImage(draw_image_path, 0, 0, width=pdf_width, height=pdf_height)

        total_confidence = 0.0
        if ocr_results:
            # Use raw OCR results (pixel coordinates) throughout.
            # The renderer handles pixel→point conversion via DPI and
            # rotation via canvas transforms — no pre-transformation needed.
            # Previously, transform_ocr_coords_for_rotation() scaled coords
            # to PDF points, and then create_text_layer() divided by DPI
            # again, causing a double-conversion that pushed text to the
            # bottom-left corner.

            # Accumulate text for stats
            try:
                formatted_page_text = self._format_ocr_text(ocr_results, float(ocr_img_size[0]))
                stats.full_text += formatted_page_text + "\n\n"
            except Exception as e:
                logger.error(f"Error formatting text: {e}")
                stats.full_text += " ".join(r.text for r in ocr_results) + "\n\n"

            # Collect structured OCR data (pixel coords + pixel dimensions
            # so that percentage calculations and height→point conversions
            # are each applied exactly once)
            stats.ocr_boxes.extend(
                self._collect_ocr_boxes(
                    ocr_results,
                    page_num,
                    float(ocr_img_size[0]),
                    float(ocr_img_size[1]),
                )
            )

            regions_added = self.renderer.render(c, ocr_results, ocr_img_size, pdf_rotation)
            stats.total_text_regions += regions_added

            page_conf = sum(r.confidence for r in ocr_results) / len(ocr_results)
            total_confidence = page_conf * regions_added
        else:
            logger.warning(f"No text detected on page {page_num}")

        c.showPage()
        stats.pages_processed += 1
        return total_confidence

    @staticmethod
    def _rotate_image_for_overlay(image: np.ndarray, rotation: int) -> np.ndarray:
        """Rotate an image by the given PDF rotation angle for OCR in overlay mode."""
        if rotation == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif rotation == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def _process_page_result(
        self,
        c: canvas.Canvas,
        result: dict,
        work_item: dict,
        page_rotations: list[dict],
        page_num: int,
        stats: ProcessingStats,
        force_overlay: bool = False,
    ) -> tuple[float, bool]:
        """Process a single page result from the parallel worker.

        Args:
            force_overlay: When True, forces overlay mode even if
                _determine_page_mode would choose standalone. Used for
                pages with native text in editor-merged files.

        Returns:
            Tuple of (confidence_contribution, geometry_changed)
        """
        # Handle skipped pages (None input)
        if work_item["img_path"] is None:
            page_info = (
                page_rotations[page_num - 1]
                if page_num <= len(page_rotations)
                else {"rotation": 0, "mediabox": None}
            )
            mediabox = page_info["mediabox"]
            if mediabox:
                pdf_width = mediabox[2] - mediabox[0]
                pdf_height = mediabox[3] - mediabox[1]
            else:
                pdf_width, pdf_height = 595, 842
            c.setPageSize((pdf_width, pdf_height))
            c.showPage()
            stats.pages_processed += 1
            logger.info(f"Page {page_num}: Skipped (no image), added blank text page.")
            return 0.0, False

        if not result.get("success"):
            logger.warning(f"Failed to process page {page_num}: {result.get('error')}")
            stats.warnings.append(f"Page {page_num} failed: {result.get('error')}")
            c.setPageSize((595, 842))
            c.showPage()
            return 0.0, False

        # Load processed image from temp file
        temp_path = result["temp_out_path"]
        try:
            # Use PIL to read (supports JP2 and other formats that cv2 may not)
            try:
                pil_img = Image.open(temp_path)
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                processed_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as read_err:
                # Fallback to cv2 for standard formats
                processed_img = cv2.imread(temp_path)
                if processed_img is None:
                    raise ValueError(
                        f"Could not read temp image {temp_path}: {read_err}"
                    ) from read_err

            proc_h, proc_w = processed_img.shape[:2]
            ocr_image = processed_img

            page_info = (
                page_rotations[page_num - 1]
                if page_num <= len(page_rotations)
                else {"rotation": 0, "mediabox": None}
            )
            rotation = page_info.get("rotation", 0)

            use_processed_for_page, geometry_changed = self._determine_page_mode(
                result, proc_w, proc_h
            )

            # Force overlay mode for pages with native text (editor-merged files)
            # so the original page content (text + layout) is preserved intact.
            if force_overlay:
                use_processed_for_page = False

            if use_processed_for_page:
                # STANDALONE MODE: embed processed image in the PDF page.
                # Compute page size from image pixel dimensions and the
                # extraction DPI so the page has proper physical dimensions
                # (e.g. ~A4) and the image is not distorted.
                draw_image_path = temp_path
                dpi = self.config.dpi or 300
                pdf_width = proc_w * 72.0 / dpi
                pdf_height = proc_h * 72.0 / dpi
                logger.info(
                    f"Page {page_num}: page size {pdf_width:.1f}×{pdf_height:.1f} pt "
                    f"from {proc_w}×{proc_h} px @ {dpi} DPI"
                )

                pdf_rotation = 0
                ocr_img_size = (proc_w, proc_h)
            else:
                # OVERLAY MODE
                # When the image was pre-rotated (/Rotate applied in worker),
                # it is already in display orientation — skip the rotation.
                if result.get("image_prerotated"):
                    pass  # Already display-oriented
                elif rotation != 0:
                    ocr_image = self._rotate_image_for_overlay(ocr_image, rotation)
                    logger.info(
                        f"Rotated OCR image for page {page_num} by "
                        f"{rotation} degrees (overlay mode)"
                    )

                ocr_img_h, ocr_img_w = ocr_image.shape[:2]
                pdf_rotation = rotation
                mediabox = page_info["mediabox"]
                if mediabox:
                    pdf_width = mediabox[2] - mediabox[0]
                    pdf_height = mediabox[3] - mediabox[1]
                else:
                    pdf_width, pdf_height = float(ocr_img_w), float(ocr_img_h)
                ocr_img_size = (ocr_img_w, ocr_img_h)
                draw_image_path = None

            # Use pre-computed OCR results from pool worker when available.
            # This avoids a sequential subprocess call per page and is the
            # key optimisation that enables parallel OCR across pages.
            precomputed_ocr = None
            ocr_raw = result.get("ocr_raw")
            if ocr_raw and ocr_raw.get("boxes"):
                # Apply confidence filter matching _run_ocr post-processing
                min_score = self.config.text_score_threshold
                precomputed_ocr = [
                    OCRResult(text=t, box=b, confidence=s)
                    for t, b, s in zip(
                        ocr_raw["txts"], ocr_raw["boxes"], ocr_raw["scores"], strict=False
                    )
                    if s >= min_score
                ]

            confidence = self._render_ocr_to_page(
                c,
                ocr_image,
                page_num,
                pdf_width,
                pdf_height,
                pdf_rotation,
                ocr_img_size,
                use_processed_for_page,
                draw_image_path,
                stats,
                precomputed_ocr=precomputed_ocr,
            )

            # Explicitly release large arrays to prevent memory accumulation
            del processed_img, ocr_image
            if "pil_img" in locals():
                del pil_img

            return confidence, use_processed_for_page

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _create_text_layer_pdf(
        self,
        image_paths: list[Path],
        output_pdf: Path,
        page_rotations: list[dict],
        stats: ProcessingStats,
        progress_callback: Callable[[int, int, str], None] | None,
    ):
        """Create a PDF with only invisible text layer (no images).

        Uses sequential preprocessing with a single persistent OCR subprocess
        to minimize memory usage (one model instance, one page in memory at a time).
        """
        c = canvas.Canvas(str(output_pdf))
        total_pages = len(image_paths)
        total_confidence = 0.0
        page_standalone_flags: list[bool] = []

        # Start persistent OCR subprocess (model loaded once)
        ocr_proc = self._launch_ocr_subprocess()
        logger.info(
            f"Text layer: {total_pages} pages, sequential preprocessing, "
            f"1 persistent OCR subprocess"
        )

        try:
            for i, p in enumerate(image_paths, 1):
                if hasattr(self, "cancel_event") and self.cancel_event.is_set():
                    logger.info("Processing cancelled by user — stopping page loop")
                    raise InterruptedError("Processing cancelled by user")

                page_info = (
                    page_rotations[i - 1]
                    if i <= len(page_rotations)
                    else {"rotation": 0, "mediabox": None}
                )
                pdf_rotation = page_info.get("rotation", 0)

                work_item = {
                    "page_num": i,
                    "img_path": str(p) if p is not None else None,
                    "config": self.config,
                    "pdf_rotation": pdf_rotation,
                    "run_ocr": False,
                }

                # Step 1: Preprocess
                result = process_page(work_item)

                # Step 2: OCR via persistent subprocess
                if result.get("success") and result.get("temp_out_path"):
                    ocr_raw = self._ocr_image_via_subprocess(ocr_proc, result["temp_out_path"])
                    result["ocr_raw"] = ocr_raw

                # Step 3: Render text layer
                progress_pct = 10 + int((i / total_pages) * 70)
                if progress_callback:
                    progress_callback(
                        progress_pct,
                        100,
                        _("Processing page {0}/{1}...").format(i, total_pages),
                    )

                try:
                    confidence, needs_standalone = self._process_page_result(
                        c, result, work_item, page_rotations, i, stats
                    )
                    total_confidence += confidence
                    page_standalone_flags.append(needs_standalone)
                except Exception as page_err:
                    logger.error(f"Error processing page {i}: {page_err}")
                    stats.warnings.append(f"Page {i} failed: {page_err}")
                    c.setPageSize((595, 842))
                    c.showPage()
                    page_standalone_flags.append(False)

                # Free page data immediately
                del result
                gc.collect()
                # Force glibc to return freed pages to OS
                try:
                    import ctypes

                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass

                # Restart OCR subprocess every 3 pages to limit memory growth
                if (i + 1) % 3 == 0 and i < len(image_paths) - 1:
                    self._stop_ocr_subprocess(ocr_proc)
                    ocr_proc = self._launch_ocr_subprocess()

        finally:
            self._stop_ocr_subprocess(ocr_proc)

        c.save()
        stats.average_confidence = total_confidence
        self._page_standalone_flags = page_standalone_flags
        logger.debug(f"Text layer PDF created: {output_pdf}")

    def _overlay_text_on_original(
        self,
        original_pdf_path: Path,
        text_layer_pdf_path: Path,
        output_pdf_path: Path,
    ) -> None:
        """Overlay text layer PDF on original PDF, preserving everything."""
        overlay_text_on_original(original_pdf_path, text_layer_pdf_path, output_pdf_path)
