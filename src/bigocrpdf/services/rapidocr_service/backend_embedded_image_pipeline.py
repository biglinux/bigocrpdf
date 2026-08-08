"""Embedded-image OCR helpers for the RapidOCR backend pipeline."""
# Host attributes are supplied by ProfessionalPDFOCR's explicit mixin composition.
# pyright: reportAttributeAccessIssue=false

import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pikepdf

from bigocrpdf.constants import MIN_IMAGE_BOX_SIZE_PX, MIN_TEXT_BOX_HEIGHT_PX, MIN_TEXT_BOX_WIDTH_PX
from bigocrpdf.services.rapidocr_service.config import OCRBoxData, OCRResult, ProcessingStats
from bigocrpdf.services.rapidocr_service.ocr_postprocess import refine_ocr_results
from bigocrpdf.services.rapidocr_service.ocr_runtime_diagnostics import (
    record_ocr_runtime_diagnostics,
)
from bigocrpdf.services.rapidocr_service.pdf_extractor import (
    extract_image_positions,
    load_image_with_exif_rotation,
)
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class BackendEmbeddedImagePipelineMixin:
    """OCR images embedded inside otherwise native-text PDF pages."""

    @staticmethod
    def _get_page_end_ctm(page) -> tuple[float, ...]:
        """Return the active CTM at the end of a page's content stream.

        Tracks q/Q (graphics state save/restore) and cm (concat matrix)
        operators to determine what transform is in effect when new
        content is appended to the page.
        """
        identity = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        ctm = identity
        stack: list[tuple[float, ...]] = []

        try:
            ops = pikepdf.parse_content_stream(page)
        except Exception:
            return identity

        for operands, operator in ops:
            op = str(operator)
            if op == "q":
                stack.append(ctm)
            elif op == "Q":
                if stack:
                    ctm = stack.pop()
            elif op == "cm" and len(operands) == 6:
                vals = [float(v) for v in operands]
                a2, b2, c2, d2, e2, f2 = vals
                a1, b1, c1, d1, e1, f1 = ctm
                # CTM' = M × CTM  (PDF pre-multiplication)
                ctm = (
                    a2 * a1 + b2 * c1,
                    a2 * b1 + b2 * d1,
                    c2 * a1 + d2 * c1,
                    c2 * b1 + d2 * d1,
                    e2 * a1 + f2 * c1 + e1,
                    e2 * b1 + f2 * d1 + f1,
                )

        return ctm

    @staticmethod
    def _invert_ctm(
        ctm: tuple[float, ...],
    ) -> tuple[float, ...] | None:
        """Return the inverse of a 2D affine CTM, or None if identity."""
        a, b, c, d, e, f = ctm
        # If already identity, no correction needed
        if (
            abs(a - 1) < 1e-6
            and abs(b) < 1e-6
            and abs(c) < 1e-6
            and abs(d - 1) < 1e-6
            and abs(e) < 1e-6
            and abs(f) < 1e-6
        ):
            return None

        det = a * d - b * c
        if abs(det) < 1e-12:
            return None

        return (
            d / det,
            -b / det,
            -c / det,
            a / det,
            (c * f - d * e) / det,
            (b * e - a * f) / det,
        )

    def _extract_xobj_image(self, xobj, xobj_name: str, PdfImage):
        """Extract a numpy RGB array from a PDF XObject, or None."""
        if not hasattr(xobj, "Width"):
            return None
        px_w, px_h = int(xobj.Width), int(xobj.Height)
        if px_w < MIN_TEXT_BOX_WIDTH_PX or px_h < MIN_TEXT_BOX_HEIGHT_PX:
            return None
        try:
            with PdfImage(xobj).as_pil_image() as source:
                with source.convert("RGB") as rgb_image:
                    return np.array(rgb_image)
        except Exception as e:
            logger.debug(f"Could not extract {xobj_name}: {e}")
            return None

    def _ocr_page_embedded_images(
        self,
        img_positions,
        xobjects,
        ocr_proc,
        stats,
        page_num,
        PdfImage,
    ) -> list[str]:
        """OCR all embedded images in one page; return text overlay commands."""
        text_commands: list[str] = []
        for img_pos in img_positions:
            if img_pos.width < 15 or img_pos.height < 15:
                continue
            if img_pos.name not in xobjects:
                continue
            img_array = self._extract_xobj_image(
                xobjects[img_pos.name],
                img_pos.name,
                PdfImage,
            )
            if img_array is None:
                continue
            ocr_results = self._ocr_via_persistent(img_array, ocr_proc)
            if not ocr_results:
                continue
            img_h, img_w = img_array.shape[:2]
            scale_x = img_pos.width / img_w if img_w else 1
            scale_y = img_pos.height / img_h if img_h else 1
            cmds = self._create_text_layer_commands(
                ocr_results,
                img_pos.x,
                img_pos.y,
                img_pos.width,
                img_pos.height,
                scale_x,
                scale_y,
            )
            for cmd in cmds:
                if cmd not in ("q", "Q"):
                    text_commands.append(cmd)
            stats.total_text_regions += len(ocr_results)
            logger.debug(f"Page {page_num}: OCR'd {img_pos.name} ({len(ocr_results)} text regions)")
        return text_commands

    def _ocr_native_text_page_images(
        self,
        merged_pdf_path: Path,
        native_text_pages: set[int],
        stats: ProcessingStats,
        progress_callback=None,
    ) -> None:
        from pikepdf import PdfImage

        image_positions = extract_image_positions(merged_pdf_path)

        # Filter to only native text pages that have images
        pages_to_process = {
            p: imgs for p, imgs in image_positions.items() if p in native_text_pages and imgs
        }

        if not pages_to_process:
            logger.info("No images found in native text pages, skipping")
            return

        total_images = sum(len(imgs) for imgs in pages_to_process.values())
        logger.info(
            f"OCR'ing {total_images} images in {len(pages_to_process)} "
            f"native text pages: {sorted(pages_to_process.keys())}"
        )

        ocr_proc = self._ocr_subprocess.launch()
        try:
            worker_runtime = self._ocr_subprocess.wait_until_ready(ocr_proc)
            record_ocr_runtime_diagnostics(
                stats,
                self.config,
                self._check_openvino_available,
                int(ocr_proc._bigocr_threads),
                1,
                worker_runtime,
            )
            with pikepdf.open(merged_pdf_path, allow_overwriting_input=True) as pdf:
                self._ocr_native_text_page_image_overlays(
                    pdf, pages_to_process, ocr_proc, stats, progress_callback, PdfImage
                )
                pdf.save(merged_pdf_path)
        finally:
            self._ocr_subprocess.stop(ocr_proc)

        logger.info(f"Native text page image OCR complete ({total_images} images)")

    def _ocr_native_text_page_image_overlays(
        self,
        pdf,
        pages_to_process: dict,
        ocr_proc,
        stats: ProcessingStats,
        progress_callback,
        PdfImage,
    ) -> None:
        pages_done = 0
        total_native = len(pages_to_process)
        for page_num, img_positions in sorted(pages_to_process.items()):
            if page_num > len(pdf.pages):
                continue
            if progress_callback:
                pct = 87 + int(8 * pages_done / total_native)
                progress_callback(
                    pct,
                    100,
                    _("Processing page {0}/{1}...").format(pages_done + 1, total_native),
                )
            pages_done += 1
            page = pdf.pages[page_num - 1]
            xobjects = (
                page.Resources.XObject
                if "/Resources" in page and "/XObject" in page.Resources
                else {}
            )
            page_text_commands = self._ocr_page_embedded_images(
                img_positions, xobjects, ocr_proc, stats, page_num, PdfImage
            )
            if page_text_commands:
                self._append_text_to_page(
                    pdf, page, self._wrap_page_text_commands(page, page_text_commands)
                )

    def _wrap_page_text_commands(self, page, page_text_commands: list[str]) -> list[str]:
        end_ctm = self._get_page_end_ctm(page)
        wrapped = ["q"]
        inv = self._invert_ctm(end_ctm)
        if inv:
            a, b, c, d, e, f = inv
            wrapped.append(f"{a:.6f} {b:.6f} {c:.6f} {d:.6f} {e:.6f} {f:.6f} cm")
        wrapped.extend(page_text_commands)
        wrapped.append("Q")
        return wrapped

    def _ocr_via_persistent(
        self,
        image: np.ndarray,
        ocr_proc: subprocess.Popen,
    ) -> list[OCRResult]:
        """Run OCR on a numpy image using the persistent subprocess.

        Writes the image to a temp file, sends the path to the persistent
        subprocess, and converts the raw dict result to OCRResult objects.
        """
        fd, temp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            if not cv2.imwrite(temp_path, image):
                logger.warning("Could not write temporary image for embedded OCR")
                return []
            raw = self._ocr_subprocess.recognize(ocr_proc, temp_path)
            if not raw or raw.get("error") or not raw.get("boxes"):
                return []

            # Refine via persistent subprocess
            def _persistent_ocr(crop_path: str) -> dict | None:
                return self._ocr_subprocess.recognize(ocr_proc, crop_path)

            raw = refine_ocr_results(raw, temp_path, _persistent_ocr)

            results = []
            for i in range(len(raw["boxes"])):
                results.append(OCRResult(raw["txts"][i], raw["boxes"][i], raw["scores"][i]))

            min_score = self.config.text_score_threshold
            results = [r for r in results if r.confidence >= min_score]
            return results
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def _ocr_image_in_page(
        self,
        img_path: Path,
        img_pos,
        pdf: "pikepdf.Pdf",
        page: "pikepdf.Page",
        page_num: int,
        page_width: float,
        page_height: float,
        stats: ProcessingStats,
        ocr_proc: subprocess.Popen | None = None,
        skip_preprocessing: bool = False,
    ) -> list[str]:
        # Skip very small images (likely icons or decorations)
        if img_pos.width < MIN_IMAGE_BOX_SIZE_PX or img_pos.height < MIN_IMAGE_BOX_SIZE_PX:
            logger.debug(f"Skipping small image: {img_pos.width}x{img_pos.height}")
            return []

        img = load_image_with_exif_rotation(img_path)
        if img is None:
            logger.warning(f"Could not load image: {img_path}")
            return []

        if skip_preprocessing:
            processed_img = img
        else:
            processed_img = self.preprocessor.process(img)
            self._replace_pdf_image(page, img_pos.name, processed_img)

        if ocr_proc is not None:
            ocr_results = self._ocr_via_persistent(processed_img, ocr_proc)
        else:
            ocr_results = self._ocr.run(processed_img)

        if not ocr_results:
            logger.debug(f"No text found in image at ({img_pos.x}, {img_pos.y})")
            return []

        img_h, img_w = processed_img.shape[:2]
        scale_x = img_pos.width / img_w
        scale_y = img_pos.height / img_h

        text_commands = self._create_text_layer_commands(
            ocr_results,
            img_pos.x,
            img_pos.y,
            img_pos.width,
            img_pos.height,
            scale_x,
            scale_y,
        )
        self._append_text_to_page(pdf, page, text_commands)

        stats.total_text_regions += len(ocr_results)

        formatted_text = self._text_formatting.format(ocr_results, float(img_w))

        self._record_positioned_ocr_boxes(
            ocr_results, img_pos, scale_x, scale_y, page_width, page_height, page_num, stats
        )

        logger.debug(
            f"Added {len(ocr_results)} text regions for image at ({img_pos.x:.1f}, {img_pos.y:.1f})"
        )

        return [formatted_text] if formatted_text else []

    @staticmethod
    def _record_positioned_ocr_boxes(
        ocr_results: list[OCRResult],
        img_pos,
        scale_x: float,
        scale_y: float,
        page_width: float,
        page_height: float,
        page_num: int,
        stats: ProcessingStats,
    ) -> None:
        for result in ocr_results:
            xs = [p[0] for p in result.box]
            ys = [p[1] for p in result.box]
            box_x = min(xs) * scale_x + img_pos.x
            box_y = min(ys) * scale_y + img_pos.y
            box_w = (max(xs) - min(xs)) * scale_x
            box_h = (max(ys) - min(ys)) * scale_y
            stats.ocr_boxes.append(
                OCRBoxData(
                    text=result.text,
                    x=(box_x / page_width) * 100 if page_width > 0 else 0,
                    y=(box_y / page_height) * 100 if page_height > 0 else 0,
                    width=(box_w / page_width) * 100 if page_width > 0 else 0,
                    height=box_h,
                    confidence=result.confidence,
                    page_num=page_num,
                )
            )

    @staticmethod
    def _replace_pdf_image(page, img_name: str, img_array: np.ndarray) -> None:
        """Replace a PDF image XObject with a preprocessed image."""
        import io

        from PIL import Image

        try:
            xobj = page.Resources.XObject[img_name]
        except (AttributeError, KeyError):
            logger.warning(f"Could not find image {img_name} to replace")
            return

        h, w = img_array.shape[:2]
        is_gray = len(img_array.shape) == 2

        if is_gray:
            pil_img = Image.fromarray(img_array, mode="L")
            colorspace = pikepdf.Name.DeviceGray
        else:
            pil_img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
            colorspace = pikepdf.Name.DeviceRGB

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95)

        xobj.write(buf.getvalue(), filter=pikepdf.Name.DCTDecode)
        xobj[pikepdf.Name.Width] = w
        xobj[pikepdf.Name.Height] = h
        xobj[pikepdf.Name.ColorSpace] = colorspace
        xobj[pikepdf.Name.BitsPerComponent] = 8

        # Remove transparency mask — JPEG does not support alpha
        if pikepdf.Name.SMask in xobj:
            del xobj[pikepdf.Name.SMask]
