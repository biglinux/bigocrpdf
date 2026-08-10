"""Text Layer Rendering Mixin for ProfessionalPDFOCR."""
# Host attributes are supplied by ProfessionalPDFOCR's explicit mixin composition.
# pyright: reportAttributeAccessIssue=false

import gc
import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pikepdf

# Restart OCR subprocess every N pages to limit memory growth
_OCR_RESTART_INTERVAL = 3
from PIL import Image
from reportlab.pdfgen import canvas

from bigocrpdf.services.rapidocr_service.backend_text_layer_geometry import (
    _processed_page_dimensions,
)
from bigocrpdf.services.rapidocr_service.config import OcrLine, OcrPage, OCRResult, ProcessingStats
from bigocrpdf.services.rapidocr_service.native_text_verification import (
    extract_native_text_spans,
    verify_ocr_lines_with_native_spans,
)
from bigocrpdf.services.rapidocr_service.ocr_document_structure import build_ocr_lines_from_results
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

        if result.get("image_prerotated", False) and result.get("original_pdf_rotation", 0) != 0:
            geometry_changed = True

        # Coordinate-space changes from perspective/dewarp/deskew that
        # don't alter dimensions still require standalone mode so OCR
        # coordinates match the displayed image.
        if result.get("geometry_applied", False):
            geometry_changed = True

        format_changed = self.config.image_export_format not in ("original", "")

        # Standalone mode replaces the original image with the processed one.
        # Only use it when geometry or format actually changed:
        #   - geometry_changed: dimensions/orientation changed OR coordinate
        #     space changed by perspective/dewarp/deskew
        #   - format_changed: user explicitly requested a different format
        # Appearance effects (scanner, vintage, etc.) improve OCR accuracy
        # but should NOT trigger image replacement, because re-encoding
        # low-quality JPEGs causes generation loss and destroys FG/BG
        # layer separation in mixed-mode PDFs.
        use_processed_for_page = geometry_changed or format_changed

        page_label = result.get("page_num", "?")
        if geometry_changed:
            logger.info(
                f"Page {page_label}: geometry/coordinate change "
                f"({orig_w}x{orig_h} → {proc_w}x{proc_h}, "
                f"{change_ratio:.1%}), using processed image in PDF"
            )
        elif format_changed:
            logger.debug(
                f"Page {page_label}: using processed image "
                f"(export format: {self.config.image_export_format})"
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
        image_rect: tuple[float, float, float, float] | None = None,
        input_pdf: Path | None = None,
        retry_level: int = 0,
        preprocess_trace: dict | None = None,
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
            ocr_results = self._ocr.run(ocr_image)
            logger.info(f"OCR page {page_num}: {len(ocr_results)} text regions")

        ocr_results = self._ocr.fix_vertical_overlaps(ocr_results)

        c.setPageSize((pdf_width, pdf_height))

        # If using processed image, draw it on the PDF
        if use_processed_for_page and draw_image_path:
            # Convert to JPEG for embedding — reportlab uses DCTDecode
            # (JPEG passthrough) which is much faster and smaller than
            # ASCII85+FlateDecode used for PNG.
            jpg_path = draw_image_path + ".jpg"
            try:
                with Image.open(draw_image_path) as source:
                    with source.convert("RGB") as rgb_image:
                        rgb_image.save(jpg_path, "JPEG", quality=95)
                c.drawImage(jpg_path, 0, 0, width=pdf_width, height=pdf_height)
            finally:
                Path(jpg_path).unlink(missing_ok=True)

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
                formatted_page_text = self._text_formatting.format(
                    ocr_results, float(ocr_img_size[0])
                )
                stats.full_text += formatted_page_text + "\n\n"
            except Exception as e:
                logger.error(f"Error formatting text: {e}")
                stats.full_text += " ".join(r.text for r in ocr_results) + "\n\n"

            # Collect structured OCR data (pixel coords + pixel dimensions
            # so that percentage calculations and height→point conversions
            # are each applied exactly once)
            stats.ocr_boxes.extend(
                self._text_formatting.collect_boxes(
                    ocr_results,
                    page_num,
                    float(ocr_img_size[0]),
                    float(ocr_img_size[1]),
                )
            )

            # In overlay mode, image DPI may differ from config DPI, so
            # pass actual page dimensions for correct coordinate mapping.
            # Always map OCR pixel coords to the actual PDF page dimensions.
            # In standalone mode the processed image is drawn at pdf_width×pdf_height,
            # so text coordinates must match that space — not config DPI which
            # may differ from the rendering DPI (e.g. pdftoppm at 150 vs config 300).
            # In overlay mode the same logic applies to the original page.
            overlay_page_size = (pdf_width, pdf_height)
            image_offset = None
            if overlay_page_size and image_rect:
                ix, iy, iw, ih = image_rect
                if abs(iw - pdf_width) > 2 or abs(ih - pdf_height) > 2:
                    overlay_page_size = (iw, ih)
                    image_offset = (ix, iy)
                    logger.info(
                        f"Page {page_num}: image offset ({ix:.1f}, {iy:.1f}), "
                        f"display size {iw:.1f}×{ih:.1f} pt "
                        f"(page {pdf_width:.1f}×{pdf_height:.1f})"
                    )
            regions_added = self.renderer.render(
                c,
                ocr_results,
                ocr_img_size,
                pdf_rotation,
                page_size_pts=overlay_page_size,
                image_offset=image_offset,
            )
            stats.total_text_regions += regions_added

            page_conf = sum(r.confidence for r in ocr_results) / len(ocr_results)
            total_confidence = page_conf * regions_added
        else:
            logger.warning(f"No text detected on page {page_num}")

        ocr_lines = build_ocr_lines_from_results(list(ocr_results))
        diagnostics = {
            "pdf_width": pdf_width,
            "pdf_height": pdf_height,
            "pdf_rotation": pdf_rotation,
            "use_processed_for_page": use_processed_for_page,
        }
        if preprocess_trace:
            # Which geometric correction ran on this page, from the worker.
            diagnostics["preprocess"] = preprocess_trace
        text_layer_quality = "ocr" if ocr_results else "absent"
        ocr_lines, verified_quality = self._auto_verify_ocr_lines(
            ocr_lines,
            input_pdf,
            page_num,
            ocr_img_size,
            image_rect,
            use_processed_for_page,
            pdf_rotation,
            diagnostics,
        )
        if verified_quality:
            text_layer_quality = verified_quality

        stats.ocr_document.append_page(
            OcrPage(
                page_index=page_num,
                width_px=int(ocr_img_size[0]),
                height_px=int(ocr_img_size[1]),
                dpi=int(getattr(self.config, "dpi", 300) or 300),
                text_results=list(ocr_results),
                lines=ocr_lines,
                text_layer_quality=text_layer_quality,
                retry_level=retry_level,
                diagnostics=diagnostics,
            )
        )

        c.showPage()
        stats.pages_processed += 1
        return total_confidence

    def _auto_verify_ocr_lines(
        self,
        ocr_lines: list[OcrLine],
        input_pdf: Path | None,
        page_num: int,
        ocr_img_size: tuple[int, int],
        image_rect: tuple[float, float, float, float] | None,
        use_processed_for_page: bool,
        pdf_rotation: int,
        diagnostics: dict[str, object],
    ) -> tuple[list[OcrLine], str | None]:
        if not ocr_lines or input_pdf is None or self.config.pdf_mode != "auto_verified":
            return ocr_lines, None
        if use_processed_for_page or pdf_rotation != 0:
            diagnostics["auto_verified"] = {
                "status": "skipped",
                "reason": "changed_geometry_or_rotation",
            }
            return ocr_lines, None

        native_spans = extract_native_text_spans(
            input_pdf,
            page_num,
            ocr_img_size,
            source_rect_pts=image_rect,
        )
        verified_page = verify_ocr_lines_with_native_spans(ocr_lines, native_spans)
        diagnostics["auto_verified"] = {
            "status": "checked",
            "native_spans": verified_page.native_spans,
            "accepted_lines": verified_page.accepted_lines,
            "rejected_lines": verified_page.rejected_lines,
        }
        quality = "auto_verified" if verified_page.accepted_lines else None
        return verified_page.lines, quality

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

    def _handle_skipped_page(
        self,
        c,
        page_rotations,
        page_num,
        stats,
    ) -> tuple[float, bool]:
        """Add a blank page for a skipped (None input) page."""
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

    @staticmethod
    def _load_processed_image(temp_path: str) -> np.ndarray:
        """Load a processed image from a temp file (PIL with cv2 fallback)."""
        try:
            with Image.open(temp_path) as source:
                with source.convert("RGB") as rgb_image:
                    image = np.array(rgb_image)
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except Exception as read_err:
            img = cv2.imread(temp_path)
            if img is None:
                raise ValueError(f"Could not read temp image {temp_path}: {read_err}") from read_err
            return img

    def _setup_overlay_mode(
        self,
        result,
        page_info,
        ocr_image,
        page_num,
    ) -> tuple[np.ndarray, int, float, float, tuple[int, int]]:
        """Compute overlay-mode parameters.

        Returns (ocr_image, pdf_rotation, pdf_width, pdf_height, ocr_img_size).
        """
        rotation = page_info.get("rotation", 0)
        if result.get("image_prerotated"):
            pass
        elif rotation != 0:
            ocr_image = self._rotate_image_for_overlay(ocr_image, rotation)
            logger.info(
                f"Rotated OCR image for page {page_num} by {rotation} degrees (overlay mode)"
            )
        ocr_img_h, ocr_img_w = ocr_image.shape[:2]
        mediabox = page_info["mediabox"]
        if mediabox:
            pdf_width = mediabox[2] - mediabox[0]
            pdf_height = mediabox[3] - mediabox[1]
        else:
            pdf_width, pdf_height = float(ocr_img_w), float(ocr_img_h)
        return ocr_image, rotation, pdf_width, pdf_height, (ocr_img_w, ocr_img_h)

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
        """Process a single page result from the parallel worker."""
        if work_item["img_path"] is None:
            return self._handle_skipped_page(c, page_rotations, page_num, stats)

        if not result.get("success"):
            logger.warning(f"Failed to process page {page_num}: {result.get('error')}")
            stats.warnings.append(f"Page {page_num} failed: {result.get('error')}")
            c.setPageSize((595, 842))
            c.showPage()
            return 0.0, False

        temp_path = result["temp_out_path"]
        try:
            processed_img = self._load_processed_image(temp_path)
            proc_h, proc_w = processed_img.shape[:2]
            ocr_image = processed_img

            page_info = (
                page_rotations[page_num - 1]
                if page_num <= len(page_rotations)
                else {"rotation": 0, "mediabox": None}
            )

            use_processed_for_page, geometry_changed = self._determine_page_mode(
                result, proc_w, proc_h
            )
            if force_overlay and not result.get("geometry_applied", False):
                # force_overlay preserves the original composite of a masked
                # (JBIG2 foreground/background) page. It must not apply once a
                # geometric correction has moved the pixels: the OCR
                # coordinates were measured on the corrected image, and there
                # is no inverse transform to bring them back, so drawing them
                # over the original would displace every word on the page.
                use_processed_for_page = False
            elif force_overlay:
                logger.debug(
                    f"Page {page_num}: keeping standalone mode despite force_overlay, "
                    "because geometric correction changed the coordinate space"
                )

            if use_processed_for_page:
                (
                    draw_image_path,
                    pdf_rotation,
                    pdf_width,
                    pdf_height,
                    ocr_img_size,
                    page_image_rect,
                ) = self._processed_page_render_params(
                    result, page_info, temp_path, proc_w, proc_h, page_num
                )
            else:
                (
                    draw_image_path,
                    pdf_rotation,
                    pdf_width,
                    pdf_height,
                    ocr_img_size,
                    page_image_rect,
                ) = self._overlay_page_render_params(result, page_info, ocr_image, page_num)
                if "ocr_img_w" in result and "ocr_img_h" in result:
                    ocr_img_size = (result["ocr_img_w"], result["ocr_img_h"])

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
                precomputed_ocr=self._precomputed_ocr_results(result),
                image_rect=page_image_rect,
                input_pdf=Path(work_item["input_pdf"]) if work_item.get("input_pdf") else None,
                retry_level=int(result.get("retry_level", 0)),
                preprocess_trace=result.get("preprocess_trace"),
            )

            del processed_img, ocr_image
            return confidence, use_processed_for_page

        finally:
            Path(temp_path).unlink(missing_ok=True)
            ocr_path = result.get("temp_ocr_path")
            if ocr_path and ocr_path != temp_path:
                Path(ocr_path).unlink(missing_ok=True)

    def _processed_page_render_params(
        self,
        result: dict,
        page_info: dict,
        temp_path: str,
        proc_w: int,
        proc_h: int,
        page_num: int,
    ) -> tuple[str, int, float, float, tuple[int, int], None]:
        pdf_width, pdf_height = _processed_page_dimensions(result, page_info, proc_w, proc_h)
        logger.info(
            f"Page {page_num}: page size {pdf_width:.1f}×{pdf_height:.1f} pt "
            f"from {proc_w}×{proc_h} px"
        )
        return temp_path, 0, pdf_width, pdf_height, (proc_w, proc_h), None

    def _overlay_page_render_params(
        self,
        result: dict,
        page_info: dict,
        ocr_image: np.ndarray,
        page_num: int,
    ) -> tuple[None, int, float, float, tuple[int, int], tuple[float, float, float, float] | None]:
        ocr_image, pdf_rotation, pdf_width, pdf_height, ocr_img_size = self._setup_overlay_mode(
            result, page_info, ocr_image, page_num
        )
        return None, pdf_rotation, pdf_width, pdf_height, ocr_img_size, page_info.get("image_rect")

    def _precomputed_ocr_results(self, result: dict) -> list[OCRResult] | None:
        ocr_raw = result.get("ocr_raw")
        if not ocr_raw or not ocr_raw.get("boxes"):
            return None
        min_score = self.config.text_score_threshold
        return [
            OCRResult(text=t, box=b, confidence=s)
            for t, b, s in zip(ocr_raw["txts"], ocr_raw["boxes"], ocr_raw["scores"], strict=False)
            if s >= min_score
        ]

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

        scratch_temp_dir = tempfile.TemporaryDirectory(prefix="rapidocr_pages_")
        scratch_dir = Path(scratch_temp_dir.name)
        ocr_proc = None
        try:
            # Start persistent OCR subprocess (model loaded once)
            ocr_proc = self._ocr_subprocess.launch()
            logger.info(
                f"Text layer: {total_pages} pages, sequential preprocessing, "
                f"1 persistent OCR subprocess"
            )

            for i, p in enumerate(image_paths, 1):
                self._raise_if_text_layer_cancelled()
                work_item = self._text_layer_work_item(
                    i,
                    p,
                    page_rotations,
                    scratch_dir,
                )

                try:
                    confidence, needs_standalone = self._run_text_layer_page(
                        c,
                        work_item,
                        page_rotations,
                        i,
                        total_pages,
                        stats,
                        ocr_proc,
                        progress_callback,
                    )
                    total_confidence += confidence
                    page_standalone_flags.append(needs_standalone)
                except InterruptedError:
                    raise
                except Exception as page_err:
                    logger.error(f"Error processing page {i}: {page_err}")
                    stats.warnings.append(f"Page {i} failed: {page_err}")
                    c.setPageSize((595, 842))
                    c.showPage()
                    page_standalone_flags.append(False)

                self._release_text_layer_page_memory()
                ocr_proc = self._restart_text_layer_ocr_if_needed(ocr_proc, i, total_pages)

        finally:
            try:
                if ocr_proc is not None:
                    self._ocr_subprocess.stop(ocr_proc)
            finally:
                try:
                    scratch_temp_dir.cleanup()
                except OSError as error:
                    logger.warning("Could not fully clean page scratch directory: %s", error)

        c.save()
        stats.average_confidence = total_confidence
        self._page_standalone_flags = page_standalone_flags
        logger.debug(f"Text layer PDF created: {output_pdf}")

    def _raise_if_text_layer_cancelled(self) -> None:
        if hasattr(self, "cancel_event") and self.cancel_event.is_set():
            logger.info("Processing cancelled by user — stopping page loop")
            raise InterruptedError("Processing cancelled by user")

    def _text_layer_work_item(
        self,
        page_num: int,
        image_path: Path,
        page_rotations: list[dict],
        scratch_dir: Path,
    ) -> dict:
        page_info = (
            page_rotations[page_num - 1]
            if page_num <= len(page_rotations)
            else {"rotation": 0, "mediabox": None}
        )
        masked_pages = getattr(getattr(self, "extractor", None), "masked_pages", set())
        masked = page_num in masked_pages
        use_rendered_source = masked and self._geometry_or_format_changes_enabled()
        work_item = {
            "page_num": page_num,
            "img_path": str(image_path) if image_path is not None else None,
            "config": self.config,
            "pdf_rotation": page_info.get("rotation", 0),
            "skip_geometric": masked and not use_rendered_source,
            "run_ocr": False,
            "scratch_dir": str(scratch_dir),
        }
        input_pdf = getattr(self, "_input_pdf", None)
        if use_rendered_source and input_pdf:
            work_item["use_rendered_source"] = True
            work_item["input_pdf"] = str(input_pdf)
        return work_item

    def _geometry_or_format_changes_enabled(self) -> bool:
        return self.config.image_export_format not in ("original", "") or (
            self.config.enable_deskew
            or self.config.enable_perspective_correction
            or self.config.enable_baseline_dewarp
        )

    def _run_text_layer_page(
        self,
        c: canvas.Canvas,
        work_item: dict,
        page_rotations: list[dict],
        page_num: int,
        total_pages: int,
        stats: ProcessingStats,
        ocr_proc,
        progress_callback: Callable[[int, int, str], None] | None,
    ) -> tuple[float, bool]:
        result = process_page(work_item)
        if result.get("success") and result.get("temp_out_path"):
            result["ocr_raw"] = self._ocr_subprocess.recognize(ocr_proc, result["temp_out_path"])

        if progress_callback:
            progress_callback(
                10 + int((page_num / total_pages) * 70),
                100,
                _("Processing page {0}/{1}...").format(page_num, total_pages),
            )

        return self._process_page_result(c, result, work_item, page_rotations, page_num, stats)

    @staticmethod
    def _release_text_layer_page_memory() -> None:
        gc.collect()
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    def _restart_text_layer_ocr_if_needed(self, ocr_proc, page_num: int, total_pages: int):
        if (page_num + 1) % _OCR_RESTART_INTERVAL != 0 or page_num >= total_pages - 1:
            return ocr_proc
        self._ocr_subprocess.stop(ocr_proc)
        try:
            return self._ocr_subprocess.launch()
        except Exception:
            logger.warning("OCR subprocess restart failed, retrying once")
            try:
                return self._ocr_subprocess.launch()
            except Exception:
                logger.error("OCR subprocess restart failed twice, aborting")
                raise

    def _overlay_text_on_original(
        self,
        original_pdf_path: Path,
        text_layer_pdf_path: Path,
        output_pdf_path: Path,
    ) -> None:
        """Overlay text layer PDF on original PDF, preserving everything."""
        overlay_text_on_original(original_pdf_path, text_layer_pdf_path, output_pdf_path)
