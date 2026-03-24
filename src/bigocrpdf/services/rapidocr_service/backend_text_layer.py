"""Text Layer Rendering Mixin for ProfessionalPDFOCR."""

import gc
import os
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pikepdf

# Restart OCR subprocess every N pages to limit memory growth
_OCR_RESTART_INTERVAL = 3
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


def _extract_image_rect_from_page(
    pdf_path: Path, page_num: int
) -> tuple[float, float, float, float] | None:
    """Extract the display rectangle of the main image on a PDF page.

    Parses the page's content stream, tracks the CTM through q/Q/cm
    operators, and finds the bounding box where the largest image
    XObject is drawn.

    Returns (x, y, width, height) in PDF points with origin at
    the page's bottom-left, or None if no image is found.
    """
    try:
        with pikepdf.open(pdf_path) as pdf:
            if page_num < 1 or page_num > len(pdf.pages):
                return None
            page = pdf.pages[page_num - 1]

            # Find image XObjects and their pixel areas
            resources = page.get("/Resources")
            if not resources:
                return None
            xobjects = resources.get("/XObject")
            if not xobjects:
                return None

            image_areas: dict[str, int] = {}
            for name in xobjects.keys():
                try:
                    xobj = xobjects[name]
                    subtype = str(xobj.get("/Subtype", ""))
                    if subtype == "/Image":
                        w = int(xobj.get("/Width", 0))
                        h = int(xobj.get("/Height", 0))
                        image_areas[str(name)] = w * h
                except Exception:
                    continue

            if not image_areas:
                return None

            target_name = max(image_areas, key=image_areas.get)

            # Parse content stream and track CTM
            ctm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            ctm_stack: list[list[float]] = []

            def _mat_mul(m1: list[float], m2: list[float]) -> list[float]:
                a1, b1, c1, d1, e1, f1 = m1
                a2, b2, c2, d2, e2, f2 = m2
                return [
                    a1 * a2 + b1 * c2,
                    a1 * b2 + b1 * d2,
                    c1 * a2 + d1 * c2,
                    c1 * b2 + d1 * d2,
                    e1 * a2 + f1 * c2 + e2,
                    e1 * b2 + f1 * d2 + f2,
                ]

            ops = pikepdf.parse_content_stream(page)
            for operands, operator in ops:
                op = str(operator)
                if op == "q":
                    ctm_stack.append(ctm[:])
                elif op == "Q":
                    if ctm_stack:
                        ctm = ctm_stack.pop()
                elif op == "cm" and len(operands) >= 6:
                    m = [float(operands[i]) for i in range(6)]
                    ctm = _mat_mul(m, ctm)
                elif op == "Do" and operands:
                    if str(operands[0]) == target_name:
                        a, b, c, d, e, f = ctm
                        xs = [e, a + e, c + e, a + c + e]
                        ys = [f, b + f, d + f, b + d + f]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        iw, ih = x_max - x_min, y_max - y_min
                        # Validate that the image covers a significant
                        # portion of the page — small decorative images
                        # (barcodes, logos) should not be used as the
                        # image rect for text coordinate mapping.
                        mb = page.get("/MediaBox")
                        if mb:
                            pw = float(mb[2]) - float(mb[0])
                            ph = float(mb[3]) - float(mb[1])
                            if pw > 0 and ph > 0:
                                coverage = (iw * ih) / (pw * ph)
                                if coverage < 0.25:
                                    logger.debug(
                                        f"Page {page_num}: largest image "
                                        f"{iw:.1f}×{ih:.1f} covers only "
                                        f"{coverage:.1%} of page — ignoring"
                                    )
                                    return None
                        return (x_min, y_min, iw, ih)

    except Exception as exc:
        logger.debug(f"Failed to extract image rect for page {page_num}: {exc}")
    return None


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

    def _has_appearance_effects(self) -> bool:
        """Check if appearance-altering effects are enabled in config."""
        cfg = self.config
        return (
            cfg.enable_border_clean
            or cfg.enable_scanner_effect
            or (
                cfg.enable_preprocessing
                and (
                    cfg.enable_auto_contrast
                    or cfg.enable_auto_brightness
                    or cfg.enable_denoise
                    or cfg.enable_vintage_look
                )
            )
        )

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
            # Convert to JPEG for embedding — reportlab uses DCTDecode
            # (JPEG passthrough) which is much faster and smaller than
            # ASCII85+FlateDecode used for PNG.
            jpg_path = draw_image_path + ".jpg"
            try:
                Image.open(draw_image_path).convert("RGB").save(jpg_path, "JPEG", quality=95)
                c.drawImage(jpg_path, 0, 0, width=pdf_width, height=pdf_height)
            finally:
                if os.path.exists(jpg_path):
                    os.remove(jpg_path)

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
            pil_img = Image.open(temp_path)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
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
            if force_overlay:
                use_processed_for_page = False

            if use_processed_for_page:
                draw_image_path = temp_path
                # Use original PDF page dimensions when available so that
                # native images at non-standard DPI keep the correct page
                # size.  Swap width/height when the image has been rotated:
                #   - /Rotate pages whose images were pre-rotated
                #   - Orientation correction (90°/270° detected by the worker)
                mediabox = page_info.get("mediabox")
                page_rot = page_info.get("rotation", 0)
                prerotated = result.get("image_prerotated", False)
                orientation_angle = result.get("orientation_angle", 0)
                need_swap = (prerotated and page_rot in (90, 270)) or orientation_angle in (90, 270)
                if mediabox:
                    mb_w = float(mediabox[2]) - float(mediabox[0])
                    mb_h = float(mediabox[3]) - float(mediabox[1])
                    if need_swap:
                        pdf_width, pdf_height = mb_h, mb_w
                    else:
                        pdf_width, pdf_height = mb_w, mb_h
                else:
                    dpi = self.config.dpi or 300
                    pdf_width = proc_w * 72.0 / dpi
                    pdf_height = proc_h * 72.0 / dpi
                logger.info(
                    f"Page {page_num}: page size {pdf_width:.1f}×{pdf_height:.1f} pt "
                    f"from {proc_w}×{proc_h} px"
                )
                pdf_rotation = 0
                ocr_img_size = (proc_w, proc_h)
                page_image_rect = None
            else:
                ocr_image, pdf_rotation, pdf_width, pdf_height, ocr_img_size = (
                    self._setup_overlay_mode(result, page_info, ocr_image, page_num)
                )
                draw_image_path = None
                page_image_rect = page_info.get("image_rect")
                # When OCR ran on a higher-quality rendered image (pdftoppm
                # for DjVu-like pages), OCR coordinates are in the rendered
                # image space.  Override ocr_img_size so the renderer
                # scales coordinates correctly.
                if "ocr_img_w" in result and "ocr_img_h" in result:
                    ocr_img_size = (result["ocr_img_w"], result["ocr_img_h"])

            precomputed_ocr = None
            ocr_raw = result.get("ocr_raw")
            if ocr_raw and ocr_raw.get("boxes"):
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
                image_rect=page_image_rect,
            )

            del processed_img, ocr_image
            return confidence, use_processed_for_page

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            ocr_path = result.get("temp_ocr_path")
            if ocr_path and ocr_path != temp_path and os.path.exists(ocr_path):
                os.remove(ocr_path)

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

                masked_pages = getattr(getattr(self, "extractor", None), "masked_pages", set())
                masked = i in masked_pages
                format_changed = self.config.image_export_format not in ("original", "")
                geometry_enabled = (
                    self.config.enable_deskew
                    or self.config.enable_perspective_correction
                    or self.config.enable_baseline_dewarp
                )
                use_rendered_source = masked and (format_changed or geometry_enabled)
                work_item = {
                    "page_num": i,
                    "img_path": str(p) if p is not None else None,
                    "config": self.config,
                    "pdf_rotation": pdf_rotation,
                    "skip_geometric": masked and not use_rendered_source,
                    "run_ocr": False,
                }
                if use_rendered_source:
                    input_pdf = getattr(self, "_input_pdf", None)
                    if input_pdf:
                        work_item["use_rendered_source"] = True
                        work_item["input_pdf"] = str(input_pdf)

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

                # Restart OCR subprocess periodically to limit memory growth
                if (i + 1) % _OCR_RESTART_INTERVAL == 0 and i < len(image_paths) - 1:
                    self._stop_ocr_subprocess(ocr_proc)
                    try:
                        ocr_proc = self._launch_ocr_subprocess()
                    except Exception:
                        logger.warning("OCR subprocess restart failed, retrying once")
                        try:
                            ocr_proc = self._launch_ocr_subprocess()
                        except Exception:
                            logger.error("OCR subprocess restart failed twice, aborting")
                            raise

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
