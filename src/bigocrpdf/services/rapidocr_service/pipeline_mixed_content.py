"""Mixed Content PDF Processing Mixin — PDFs with both text and images."""

import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import pikepdf

from bigocrpdf.services.rapidocr_service.config import ProcessingStats
from bigocrpdf.services.rapidocr_service.pdf_extractor import (
    extract_image_positions,
    page_has_ocr_text,
)
from bigocrpdf.services.rapidocr_service.pdf_assembly import strip_invisible_text
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger

import re

_MULTI_BLANK_RE = re.compile(r"\n{4,}")


def _try_join_line(para: str, stripped_next: str) -> str | None:
    """Try joining next line to current paragraph via hyphen or mid-sentence.

    Returns the joined string, or None if lines should not be joined.
    """
    para_end = para.rstrip()
    if not para_end or not stripped_next:
        return None

    # Hyphenated word break
    if (
        para_end.endswith("-")
        and len(para_end) > 1
        and para_end[-2].isalpha()
        and stripped_next[0].islower()
    ):
        return para_end[:-1] + stripped_next

    # Mid-sentence continuation
    last_ch = para_end[-1]
    if (last_ch.isalpha() or last_ch == ",") and stripped_next[0].islower():
        return para_end + " " + stripped_next

    return None


def _reflow_text(text: str) -> str:
    """Conservative reflow: join only mid-sentence continuations."""
    lines = text.split("\n")
    reflowed: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            reflowed.append("")
            i += 1
            continue

        para = line.rstrip()
        i += 1
        while i < len(lines):
            if not lines[i].strip():
                break
            joined = _try_join_line(para, lines[i].strip())
            if joined is None:
                break
            para = joined
            i += 1

        reflowed.append(para)

    return _MULTI_BLANK_RE.sub("\n\n\n", "\n".join(reflowed))


class MixedContentMixin:
    """Mixin providing mixed-content PDF processing (text + image pages)."""

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

        if not image_positions:
            logger.info("No images found in PDF. Copying original.")
            shutil.copy2(input_pdf, output_pdf)
            stats.warnings.append("No images found to OCR in mixed content PDF")
            self._calculate_final_stats(stats, start_time)
            return stats

        total_images = sum(len(imgs) for imgs in image_positions.values())
        logger.info(f"Found {total_images} image(s) across {len(image_positions)} page(s)")

        with pikepdf.open(input_pdf) as pdf_count:
            total_pages = len(pdf_count.pages)
        stats.pages_total = total_pages

        pages_with_images = set(image_positions.keys())
        text_only_pages = set(range(1, total_pages + 1)) - pages_with_images
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

            if progress_callback:
                progress_callback(5, 100, _("Extracting images..."))

            extracted_images = self._extract_and_filter_images(input_pdf, images_dir)

            ocr_proc = self._launch_ocr_subprocess()
            try:
                self._wait_for_ocr_ready(ocr_proc)
                with pikepdf.open(input_pdf, allow_overwriting_input=True) as pdf:
                    # Build set of excluded pages from editor modifications
                    excluded_pages: set[int] = set()
                    if self.config.page_modifications:
                        for mod in self.config.page_modifications:
                            pn = mod.get("page_number")
                            if pn and (mod.get("deleted") or not mod.get("included_for_ocr", True)):
                                excluded_pages.add(pn)

                    self._ocr_image_pages(
                        pdf,
                        image_positions,
                        extracted_images,
                        total_images,
                        stats,
                        ocr_texts,
                        ocr_proc,
                        progress_callback,
                        excluded_pages=excluded_pages,
                    )

                    # Remove excluded pages before saving
                    if excluded_pages:
                        for idx in sorted(excluded_pages, reverse=True):
                            if 0 < idx <= len(pdf.pages):
                                del pdf.pages[idx - 1]
                                logger.info(f"Removed excluded page {idx} from output")

                    if progress_callback:
                        progress_callback(90, 100, _("Saving PDF..."))
                    stats.pages_processed = len(image_positions) - len(
                        excluded_pages & set(image_positions.keys())
                    )
                    pdf.save(output_pdf)
            finally:
                self._stop_ocr_subprocess(ocr_proc)

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
    ) -> None:
        """OCR all image-bearing pages, modifying the PDF in place."""
        enhance = getattr(self.config, "enhance_embedded_images", False)
        logger.info(f"Mixed content: enhance_embedded_images={enhance}")
        processed_images = 0
        current_img_idx = 0
        _excluded = excluded_pages or set()

        for page_num in sorted(image_positions.keys()):
            if hasattr(self, "cancel_event") and self.cancel_event.is_set():
                logger.info("Processing cancelled by user in mixed content mode")
                raise InterruptedError("Processing cancelled by user")

            page_imgs = image_positions[page_num]

            # Skip excluded pages entirely (no preprocessing, no OCR)
            if page_num in _excluded:
                logger.info(f"Page {page_num}: excluded from OCR, skipping ({len(page_imgs)} image(s))")
                current_img_idx += len(page_imgs)
                continue

            page = pdf.pages[page_num - 1]
            mediabox = page.mediabox
            page_height = float(mediabox[3]) - float(mediabox[1])
            page_width = float(mediabox[2]) - float(mediabox[0])

            if page_has_ocr_text(page):
                if not self.config.replace_existing_ocr:
                    logger.info(
                        f"Page {page_num}: already has OCR text layer, skipping "
                        f"({len(page_imgs)} image(s))"
                    )
                    current_img_idx += len(page_imgs)
                    continue
                stripped = strip_invisible_text(page, pdf)
                if stripped:
                    logger.info(
                        f"Page {page_num}: stripped {stripped} old OCR text block(s) before re-OCR"
                    )

            if progress_callback:
                pct = 10 + int(80 * processed_images / total_images)
                progress_callback(pct, 100, _("OCR page {0}...").format(page_num))

            for img_pos in page_imgs:
                if current_img_idx >= len(extracted_images):
                    logger.warning(f"Image index {current_img_idx} exceeds extracted images")
                    break
                img_path = extracted_images[current_img_idx]
                current_img_idx += 1
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
                        skip_preprocessing=not getattr(
                            self.config, "enhance_embedded_images", False
                        ),
                    )
                    ocr_texts.extend(texts)
                    if texts:
                        processed_images += 1
                except Exception as e:
                    logger.error(f"Error processing image {img_path}: {e}")
                    stats.warnings.append(f"Failed to OCR image: {e}")

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
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
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
                subprocess.run(cmd_pbm, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
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
