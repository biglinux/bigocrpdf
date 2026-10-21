"""Text Layer Rendering Mixin for ProfessionalPDFOCR."""
# Host attributes are supplied by ProfessionalPDFOCR's explicit mixin composition.
# pyright: reportAttributeAccessIssue=false

from pathlib import Path

from PIL import Image
from reportlab.pdfgen import canvas

from bigocrpdf.services.ocr.backend_text_layer_geometry import (
    _processed_page_dimensions,
)
from bigocrpdf.services.ocr.config import OcrLine, OcrPage, OCRResult, ProcessingStats
from bigocrpdf.services.ocr.native_text_verification import (
    extract_native_text_spans,
    verify_ocr_lines_with_native_spans,
)
from bigocrpdf.services.ocr.ocr_document_structure import build_ocr_lines_from_results
from bigocrpdf.services.ocr.ocr_postprocess import fix_vertical_overlaps
from bigocrpdf.services.ocr.ppocr_native import ocr_results as page_lines
from bigocrpdf.utils.logger import logger


class BackendTextLayerMixin:
    """Mixin providing text layer creation and rendering methods."""

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
        # Appearance effects (scanner, vintage) do not trigger replacement:
        # re-encoding low-quality JPEGs causes generation loss and destroys
        # the FG/BG layer separation of mixed-mode PDFs.
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
        ocr_results: list[OCRResult],
        page_num: int,
        pdf_width: float,
        pdf_height: float,
        pdf_rotation: int,
        ocr_img_size: tuple[int, int],
        use_processed_for_page: bool,
        draw_image_path: str | None,
        stats: ProcessingStats,
        image_rect: tuple[float, float, float, float] | None = None,
        input_pdf: Path | None = None,
        retry_level: int = 0,
        preprocess_trace: dict | None = None,
    ) -> float:
        """Render a page's OCR results to a PDF page.

        Returns:
            Total confidence contribution from this page
        """
        logger.info(f"OCR page {page_num}: {len(ocr_results)} text regions")
        ocr_results = fix_vertical_overlaps(ocr_results)

        c.setPageSize((pdf_width, pdf_height))

        # reportlab embeds a JPEG file as it is (DCTDecode), which is much
        # faster and smaller than the ASCII85+FlateDecode it writes for
        # other formats, so those are converted to JPEG first.
        if use_processed_for_page and draw_image_path and draw_image_path.endswith(".jpg"):
            c.drawImage(draw_image_path, 0, 0, width=pdf_width, height=pdf_height)
        elif use_processed_for_page and draw_image_path:
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
            # rotation via canvas transforms; converting them here as well
            # would scale twice and push the text to the bottom-left corner.

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
            # so text coordinates must match that space, not the config DPI, which
            # may differ from the rendering DPI (e.g. pdftoppm at 150 vs config 300).
            # In overlay mode the same logic applies to the original page.
            overlay_page_size = (pdf_width, pdf_height)
            image_offset = None
            if image_rect:
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
                dpi=self.config.dpi,
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

    def _handle_skipped_page(self, c, page_num, stats) -> tuple[float, bool]:
        """Add a blank page, at the size the caller set, for a skipped (None input) page."""
        c.showPage()
        stats.pages_processed += 1
        logger.info(f"Page {page_num}: Skipped (no image), added blank text page.")
        return 0.0, False

    def _setup_overlay_mode(
        self,
        result,
        page_info,
        proc_size: tuple[int, int],
        page_num,
    ) -> tuple[int, float, float, tuple[int, int]]:
        """Compute overlay-mode parameters for an OCR image of ``proc_size`` (w, h),
        measured turned by /Rotate unless the worker already rotated it.

        Returns (pdf_rotation, pdf_width, pdf_height, ocr_img_size), the page
        size taken in the frame the OCR image was measured in.
        """
        rotation = page_info.get("rotation", 0)
        ocr_img_w, ocr_img_h = proc_size
        if not result.get("image_prerotated") and rotation in (90, 270):
            ocr_img_w, ocr_img_h = ocr_img_h, ocr_img_w
            logger.info(f"Page {page_num}: OCR image measured turned by {rotation} degrees")
        mediabox = page_info["mediabox"]
        if mediabox:
            pdf_width = mediabox[2] - mediabox[0]
            pdf_height = mediabox[3] - mediabox[1]
            # A prerotated image shows the page as displayed, turned from its MediaBox.
            if result.get("image_prerotated") and rotation in (90, 270):
                pdf_width, pdf_height = pdf_height, pdf_width
        else:
            pdf_width, pdf_height = float(ocr_img_w), float(ocr_img_h)
        return rotation, pdf_width, pdf_height, (ocr_img_w, ocr_img_h)

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

        The canvas page already has the PDF page's size; a page drawn from its
        image sets its own.
        """
        if work_item["img_path"] is None:
            return self._handle_skipped_page(c, page_num, stats)

        if not result.get("success"):
            logger.warning(f"Failed to process page {page_num}: {result.get('error')}")
            stats.failed_pages[page_num] = str(result.get("error"))
            c.showPage()
            return 0.0, False

        temp_path = result["temp_out_path"]
        try:
            proc_w, proc_h = result["proc_w"], result["proc_h"]
            page_info = (
                page_rotations[page_num - 1]
                if page_num <= len(page_rotations)
                else {"rotation": 0, "mediabox": None}
            )

            use_processed_for_page, _ = self._determine_page_mode(result, proc_w, proc_h)
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
                draw_image_path = temp_path
                pdf_rotation = 0
                pdf_width, pdf_height = _processed_page_dimensions(
                    result, page_info, proc_w, proc_h
                )
                ocr_img_size = (proc_w, proc_h)
                page_image_rect = None
                logger.info(
                    f"Page {page_num}: page size {pdf_width:.1f}×{pdf_height:.1f} pt "
                    f"from {proc_w}×{proc_h} px"
                )
            else:
                draw_image_path = None
                pdf_rotation, pdf_width, pdf_height, ocr_img_size = self._setup_overlay_mode(
                    result, page_info, (proc_w, proc_h), page_num
                )
                # A page render holds the whole page, not only its image.
                page_rendered = work_item.get("page_rendered") or work_item.get(
                    "use_rendered_source"
                )
                page_image_rect = None if page_rendered else page_info.get("image_rect")
                if "ocr_img_w" in result and "ocr_img_h" in result:
                    ocr_img_size = (result["ocr_img_w"], result["ocr_img_h"])

            confidence = self._render_ocr_to_page(
                c,
                self._page_ocr_results(result, page_num, stats),
                page_num,
                pdf_width,
                pdf_height,
                pdf_rotation,
                ocr_img_size,
                use_processed_for_page,
                draw_image_path,
                stats,
                image_rect=page_image_rect,
                input_pdf=Path(work_item["input_pdf"]) if work_item.get("input_pdf") else None,
                retry_level=int(result.get("retry_level", 0)),
                preprocess_trace=result.get("preprocess_trace"),
            )

            return confidence, use_processed_for_page

        finally:
            Path(temp_path).unlink(missing_ok=True)
            ocr_path = result.get("temp_ocr_path")
            if ocr_path and ocr_path != temp_path:
                Path(ocr_path).unlink(missing_ok=True)

    @staticmethod
    def _page_ocr_results(result: dict, page_num: int, stats: ProcessingStats) -> list[OCRResult]:
        """The page's text lines; a page the worker could not read counts as failed."""
        ocr_raw = result.get("ocr_raw")
        if ocr_raw is None:
            stats.failed_pages[page_num] = result.get("ocr_error", "no OCR reply")
            return []
        return page_lines(ocr_raw)
