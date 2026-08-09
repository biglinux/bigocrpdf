"""Mixed Content PDF Processing Mixin — PDFs with both text and images."""
# Host attributes are supplied by ProfessionalPDFOCR's explicit mixin composition.
# pyright: reportAttributeAccessIssue=false

import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import pikepdf

from bigocrpdf.constants import PDF_TOOL_TIMEOUT_SECS
from bigocrpdf.services.rapidocr_service.config import ProcessingStats
from bigocrpdf.services.rapidocr_service.ocr_runtime_diagnostics import (
    record_ocr_runtime_diagnostics,
)
from bigocrpdf.services.rapidocr_service.pdf_assembly import strip_invisible_text
from bigocrpdf.services.rapidocr_service.pdf_extractor import (
    ImagePosition,
    PdfImageInfo,
    extract_image_positions,
    page_has_ocr_text,
    parse_pdfimages_list,
)
from bigocrpdf.services.rapidocr_service.pipeline_mixed_content_pages import (
    _index_extracted_images,
    _mixed_excluded_pages,
    _mixed_progress_bands,
    _mixed_render_candidates,
    _pdf_page_size,
    _position_image_pairs,
    _reflow_text,
    _render_mixed_page_image,
)
from bigocrpdf.services.rapidocr_service.resource_manager import (
    enforce_image_resource_limits,
    select_pdf_page_render_dpi,
)
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class MixedContentMixin:
    """Mixin providing mixed-content PDF processing (text + image pages)."""

    def _run_mixed_ocr_pass(
        self,
        input_pdf: Path,
        pdf,
        images_dir: Path,
        temp_dir: str,
        image_positions: dict,
        pdfimages_map: dict[int, list[PdfImageInfo]],
        render_candidates: set[int],
        excluded_pages: set[int],
        stats: ProcessingStats,
        ocr_texts: list[str],
        ocr_proc: subprocess.Popen,
        progress_callback: Callable[[int, int, str], None] | None,
    ) -> None:
        positioned_image_positions = {
            page_num: imgs
            for page_num, imgs in image_positions.items()
            if page_num not in render_candidates
        }
        render_pages = render_candidates - excluded_pages
        render_band, pos_band = _mixed_progress_bands(render_pages, positioned_image_positions)

        if render_pages:
            self._ocr_rendered_pages(
                input_pdf,
                pdf,
                sorted(render_pages),
                stats,
                ocr_texts,
                ocr_proc,
                progress_callback,
                temp_dir=temp_dir,
                progress_start=5,
                progress_band=render_band,
            )

        extracted_images = []
        if positioned_image_positions:
            if progress_callback:
                progress_callback(5 + render_band, 100, _("Extracting images..."))
            extracted_images = self._extract_and_filter_images(input_pdf, images_dir)

        self._ocr_image_pages(
            pdf,
            positioned_image_positions,
            extracted_images,
            sum(len(imgs) for imgs in positioned_image_positions.values()),
            stats,
            ocr_texts,
            ocr_proc,
            progress_callback,
            excluded_pages=excluded_pages,
            pdfimages_map=pdfimages_map,
            progress_start=5 + render_band,
            progress_band=pos_band,
        )

    def _raise_if_mixed_cancelled(self) -> None:
        if hasattr(self, "cancel_event") and self.cancel_event.is_set():
            logger.info("Processing cancelled by user in mixed content mode")
            raise InterruptedError("Processing cancelled by user")

    def _process_mixed_content_pdf(
        self,
        input_pdf: Path,
        output_pdf: Path,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> ProcessingStats:
        """Process mixed content PDF (text + images)."""
        import shutil

        start_time = time.time()
        stats = ProcessingStats()

        if progress_callback:
            progress_callback(0, 100, _("Analyzing PDF structure..."))

        image_positions = extract_image_positions(input_pdf)
        pdfimages_map, masked_pages = parse_pdfimages_list(input_pdf)
        enforce_image_resource_limits(
            (
                (page_num, image.width, image.height)
                for page_num, images in pdfimages_map.items()
                for image in images
            ),
            self.config,
        )

        if not image_positions and not pdfimages_map:
            logger.info("No images found in PDF. Copying original.")
            with pikepdf.open(input_pdf) as source_pdf:
                stats.pages_total = len(source_pdf.pages)
            stats.pages_processed = stats.pages_total
            shutil.copy2(input_pdf, output_pdf)
            stats.warnings.append("No images found to OCR in mixed content PDF")
            self._calculate_final_stats(stats, start_time)
            return stats

        missed_pages = set(pdfimages_map.keys()) - set(image_positions.keys())
        with pikepdf.open(input_pdf) as pdf_scan:
            total_pages = len(pdf_scan.pages)
            render_candidates = _mixed_render_candidates(
                pdf_scan, image_positions, pdfimages_map, masked_pages
            )
        stats.pages_total = total_pages

        total_images = sum(len(imgs) for imgs in image_positions.values())
        logger.info(
            f"Found {total_images} positioned image(s) across "
            f"{len(image_positions)} page(s), "
            f"{len(missed_pages)} unpositioned, "
            f"{len(masked_pages)} masked, "
            f"{len(render_candidates)} page(s) to render"
        )

        all_image_pages = set(image_positions.keys()) | set(pdfimages_map.keys())
        text_only_pages = set(range(1, total_pages + 1)) - all_image_pages
        if text_only_pages:
            native_text = self._extract_native_text(input_pdf, text_only_pages)
            logger.info(f"Extracted native text for {len(text_only_pages)} text-only page(s)")
        else:
            native_text = ""
            logger.info("All pages have images; skipping native text extraction")
        ocr_texts: list[str] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            images_dir = Path(temp_dir) / "images"
            images_dir.mkdir()

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
                with pikepdf.open(input_pdf, allow_overwriting_input=True) as pdf:
                    excluded_pages = _mixed_excluded_pages(self.config.page_modifications)
                    self._run_mixed_ocr_pass(
                        input_pdf,
                        pdf,
                        images_dir,
                        temp_dir,
                        image_positions,
                        pdfimages_map,
                        render_candidates,
                        excluded_pages,
                        stats,
                        ocr_texts,
                        ocr_proc,
                        progress_callback,
                    )

                    # Remove excluded pages before saving
                    if excluded_pages:
                        for idx in sorted(excluded_pages, reverse=True):
                            if 0 < idx <= len(pdf.pages):
                                del pdf.pages[idx - 1]
                                logger.info(f"Removed excluded page {idx} from output")

                    if progress_callback:
                        progress_callback(90, 100, _("Saving PDF..."))
                    positioned_pages = {
                        page_num
                        for page_num in image_positions
                        if page_num not in render_candidates
                    }
                    render_pages = render_candidates - excluded_pages
                    stats.pages_processed = (
                        len(positioned_pages)
                        + len(render_pages)
                        - len(excluded_pages & all_image_pages)
                    )
                    pdf.save(output_pdf)
            finally:
                self._ocr_subprocess.stop(ocr_proc)

        self._post_process_mixed(output_pdf, stats, native_text, ocr_texts, progress_callback)
        self._calculate_final_stats(stats, start_time)

        if progress_callback:
            progress_callback(100, 100, _("Done!"))

        logger.info(f"Mixed content processing complete in {stats.processing_time_seconds:.1f}s")
        logger.info(f"Pages: {stats.pages_processed}, Text regions: {stats.total_text_regions}")
        return stats

    def _ocr_image_pages(
        self,
        pdf,
        image_positions: dict,
        extracted_images: list,
        total_images: int,
        stats: ProcessingStats,
        ocr_texts: list[str],
        ocr_proc: subprocess.Popen,
        progress_callback: Callable[[int, int, str], None] | None,
        excluded_pages: set[int] | None = None,
        pdfimages_map: dict[int, list[PdfImageInfo]] | None = None,
        progress_start: int = 10,
        progress_band: int = 80,
    ) -> None:
        """OCR all image-bearing pages, modifying the PDF in place.

        Uses *pdfimages_map* (from ``pdfimages -list``) for correct
        alignment between extracted image files and page positions.
        """
        enhance = getattr(self.config, "enhance_embedded_images", False)
        logger.info(f"Mixed content: enhance_embedded_images={enhance}")
        processed_images = 0
        _excluded = excluded_pages or set()
        _pdfmap = pdfimages_map or {}

        for page_num in sorted(image_positions.keys()):
            processed_images += MixedContentMixin._ocr_positioned_image_page(
                self,
                pdf,
                page_num,
                image_positions[page_num],
                _pdfmap.get(page_num, []),
                extracted_images,
                processed_images,
                total_images,
                stats,
                ocr_texts,
                ocr_proc,
                progress_callback,
                _excluded,
                progress_start,
                progress_band,
            )

    def _ocr_positioned_image_page(
        self,
        pdf,
        page_num: int,
        page_imgs: list[ImagePosition],
        page_img_infos: list[PdfImageInfo],
        extracted_images: list[Path],
        processed_images: int,
        total_images: int,
        stats: ProcessingStats,
        ocr_texts: list[str],
        ocr_proc: subprocess.Popen,
        progress_callback: Callable[[int, int, str], None] | None,
        excluded_pages: set[int],
        progress_start: int,
        progress_band: int,
    ) -> int:
        MixedContentMixin._raise_if_mixed_cancelled(self)
        if page_num in excluded_pages:
            logger.info(f"Page {page_num}: excluded from OCR, skipping ({len(page_imgs)} image(s))")
            return 0

        page = pdf.pages[page_num - 1]
        if not MixedContentMixin._prepare_page_for_reocr(self, pdf, page, page_num, len(page_imgs)):
            return 0
        if progress_callback:
            progress_callback(
                progress_start + int(progress_band * processed_images / max(total_images, 1)),
                100,
                _("OCR page {0}...").format(page_num),
            )

        page_width, page_height = _pdf_page_size(page)
        return MixedContentMixin._ocr_positioned_image_pairs(
            self,
            pdf,
            page,
            page_num,
            page_width,
            page_height,
            _position_image_pairs(page, page_imgs, page_img_infos),
            _index_extracted_images(extracted_images),
            stats,
            ocr_texts,
            ocr_proc,
        )

    def _prepare_page_for_reocr(self, pdf, page, page_num: int, image_count: int) -> bool:
        if not page_has_ocr_text(page):
            return True
        if not self.config.replace_existing_ocr:
            logger.info(
                f"Page {page_num}: already has OCR text layer, skipping ({image_count} image(s))"
            )
            return False
        stripped = strip_invisible_text(page, pdf)
        if stripped:
            logger.info(f"Page {page_num}: stripped {stripped} old OCR text block(s) before re-OCR")
        return True

    def _ocr_positioned_image_pairs(
        self,
        pdf,
        page,
        page_num: int,
        page_width: float,
        page_height: float,
        pairs: list[tuple[ImagePosition, PdfImageInfo | None]],
        idx_to_path: dict[int, Path],
        stats: ProcessingStats,
        ocr_texts: list[str],
        ocr_proc: subprocess.Popen,
    ) -> int:
        processed = 0
        for img_pos, info in pairs:
            img_path = idx_to_path.get(info.idx) if info is not None else None
            if img_path is None:
                logger.warning(f"Page {page_num}: no extracted image for position {img_pos.name}")
                continue
            try:
                texts = self._ocr_image_in_page(
                    img_path,
                    img_pos,
                    pdf,
                    page,
                    page_num,
                    page_width,
                    page_height,
                    stats,
                    ocr_proc=ocr_proc,
                    skip_preprocessing=not getattr(self.config, "enhance_embedded_images", False),
                )
                ocr_texts.extend(texts)
                processed += 1 if texts else 0
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {e}")
                stats.warnings.append(f"Failed to OCR image: {e}")
        return processed

    def _ocr_rendered_pages(
        self,
        input_pdf: Path,
        pdf: "pikepdf.Pdf",
        page_nums: list[int],
        stats: ProcessingStats,
        ocr_texts: list[str],
        ocr_proc: subprocess.Popen,
        progress_callback: Callable[[int, int, str], None] | None,
        temp_dir: str,
        progress_start: int = 5,
        progress_band: int = 80,
    ) -> None:
        """Render missed pages with pdftoppm and OCR the result.

        For pages whose images are inside nested Form XObjects (invisible
        to ``extract_image_positions``), we render the composited page at
        configured fallback DPI and add OCR text as an overlay.
        """

        render_dir = Path(temp_dir) / "rendered"
        render_dir.mkdir(exist_ok=True)

        for render_idx, page_num in enumerate(page_nums):
            MixedContentMixin._ocr_rendered_page(
                self,
                input_pdf,
                pdf,
                page_num,
                render_idx,
                len(page_nums),
                render_dir,
                stats,
                ocr_texts,
                ocr_proc,
                progress_callback,
                progress_start,
                progress_band,
            )

    def _ocr_rendered_page(
        self,
        input_pdf: Path,
        pdf,
        page_num: int,
        render_idx: int,
        total_render_pages: int,
        render_dir: Path,
        stats: ProcessingStats,
        ocr_texts: list[str],
        ocr_proc: subprocess.Popen,
        progress_callback: Callable[[int, int, str], None] | None,
        progress_start: int,
        progress_band: int,
    ) -> None:
        import cv2

        MixedContentMixin._raise_if_mixed_cancelled(self)
        if page_num < 1 or page_num > len(pdf.pages):
            return
        page = pdf.pages[page_num - 1]
        if not MixedContentMixin._prepare_rendered_page_for_reocr(self, pdf, page, page_num):
            return
        if progress_callback:
            progress_callback(
                progress_start + int(progress_band * render_idx / max(total_render_pages, 1)),
                100,
                _("OCR page {0} (rendered)...").format(page_num),
            )

        preferred_dpi = int(getattr(self.config, "fallback_render_dpi", 300))
        render_dpi = select_pdf_page_render_dpi(
            input_pdf,
            page_num,
            preferred_dpi,
            float(getattr(self.config, "max_render_megapixels", 45)),
        )
        if render_dpi != preferred_dpi:
            logger.info(
                f"Page {page_num}: reducing mixed-page render DPI {preferred_dpi} -> {render_dpi}"
            )
        rendered_path = _render_mixed_page_image(input_pdf, render_dir, page_num, render_dpi)
        if rendered_path is None:
            return
        img = cv2.imread(str(rendered_path))
        rendered_path.unlink(missing_ok=True)
        if img is None:
            logger.warning(f"Could not read rendered image: {rendered_path}")
            return

        ocr_results = self._ocr_via_persistent(img, ocr_proc)
        if not ocr_results:
            logger.debug(f"Page {page_num}: rendered — no OCR text found")
            return

        MixedContentMixin._append_rendered_page_ocr(
            self, pdf, page, page_num, img, ocr_results, stats, ocr_texts
        )

    def _prepare_rendered_page_for_reocr(self, pdf, page, page_num: int) -> bool:
        if not page_has_ocr_text(page):
            return True
        if not self.config.replace_existing_ocr:
            logger.info(f"Page {page_num}: rendered — already has OCR, skipping")
            return False
        stripped = strip_invisible_text(page, pdf)
        if stripped:
            logger.info(f"Page {page_num}: stripped {stripped} old OCR block(s) before re-OCR")
        return True

    def _append_rendered_page_ocr(
        self,
        pdf,
        page,
        page_num: int,
        img,
        ocr_results,
        stats: ProcessingStats,
        ocr_texts: list[str],
    ) -> None:
        page_width, page_height = _pdf_page_size(page)
        img_h, img_w = img.shape[:2]
        text_commands = self._create_text_layer_commands(
            ocr_results,
            0.0,
            0.0,
            page_width,
            page_height,
            page_width / img_w,
            page_height / img_h,
        )
        self._append_text_to_page(pdf, page, text_commands)
        stats.total_text_regions += len(ocr_results)
        formatted = self._text_formatting.format(ocr_results, float(img_w))
        if formatted:
            ocr_texts.append(formatted)
        logger.info(f"Page {page_num}: rendered OCR — {len(ocr_results)} text regions")

    def _post_process_mixed(
        self,
        output_pdf: Path,
        stats: ProcessingStats,
        native_text: str,
        ocr_texts: list[str],
        progress_callback: Callable[[int, int, str], None] | None,
    ) -> None:
        """Split output if needed and combine text results."""
        max_mb = self.config.max_file_size_mb
        if max_mb > 0:
            file_size_mb = output_pdf.stat().st_size / (1024 * 1024)
            if file_size_mb > max_mb:
                if progress_callback:
                    progress_callback(92, 100, _("Splitting PDF by size limit..."))
                split_parts = self._split_pdf_by_size(output_pdf, max_mb)
                if split_parts:
                    stats.split_output_files = [str(p) for p in split_parts]

        parts = []
        if native_text and native_text.strip():
            parts.append(native_text.strip())
        if ocr_texts:
            parts.append("\n".join(ocr_texts))
        stats.full_text = "\n\n".join(parts)

    @staticmethod
    def _extract_native_text(input_pdf: Path, pages: set[int] | None = None) -> str:
        """Extract existing text from PDF using pdftotext.

        Uses plain mode (no -layout) for cleaner text flow, then
        post-processes with conservative reflow for mid-sentence joins.
        """
        try:
            result = subprocess.run(
                ["pdftotext", str(input_pdf), "-"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                return ""
        except Exception as e:
            logger.warning(f"Could not extract native text: {e}")
            return ""

        text = result.stdout

        if pages is not None:
            selected = [
                pt for i, pt in enumerate(text.split("\f")) if (i + 1) in pages and pt.strip()
            ]
            if not selected:
                return ""
            text = "\f".join(selected)

        text = text.replace("\f", "\n\n")
        text = _reflow_text(text)
        return text

    @staticmethod
    def _extract_and_filter_images(input_pdf: Path, images_dir: Path) -> list[Path]:
        """Extract images from PDF and filter out masks/small icons.

        Uses pdfimages -all for native extraction. If any images are in
        formats that OpenCV/PIL cannot read (JBIG2, CCITT), re-extracts
        without -all to get universally readable PBM/PPM files instead.
        """
        UNSUPPORTED_EXTS = frozenset({".jb2e", ".jb2g", ".ccitt"})

        cmd = ["pdfimages", "-all", str(input_pdf), str(images_dir / "img")]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=PDF_TOOL_TIMEOUT_SECS,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"pdfimages failed: {e.stderr}")
            raise RuntimeError(f"Failed to extract images: {e}") from e

        extracted = sorted(images_dir.glob("img-*"))

        # Check for unsupported formats (JBIG2, CCITT fax)
        has_unsupported = any(f.suffix.lower() in UNSUPPORTED_EXTS for f in extracted)

        if has_unsupported:
            logger.info(
                "Detected JBIG2/CCITT images, re-extracting as PBM/PPM for OCR compatibility"
            )
            # Clean all extracted files
            for f in extracted:
                try:
                    f.unlink()
                except OSError:
                    pass

            # Re-extract without -all: produces PBM/PPM/PGM (universally readable)
            cmd_pbm = ["pdfimages", str(input_pdf), str(images_dir / "img")]
            try:
                subprocess.run(
                    cmd_pbm,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=PDF_TOOL_TIMEOUT_SECS,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                logger.error(f"pdfimages fallback failed: {e.stderr}")
                raise RuntimeError(f"Failed to extract images: {e}") from e

            extracted = sorted(images_dir.glob("img-*"))

        filtered = [
            img
            for img in extracted
            if not (img.stat().st_size < 5000 and img.suffix.lower() == ".png")
        ]
        logger.info(f"Extracted {len(filtered)} images (after filtering masks)")
        return filtered
