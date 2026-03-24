"""Mixed Content PDF Processing Mixin — PDFs with both text and images."""

import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import pikepdf

from bigocrpdf.services.rapidocr_service.config import ProcessingStats
from bigocrpdf.services.rapidocr_service.pdf_assembly import strip_invisible_text
from bigocrpdf.services.rapidocr_service.pdf_extractor import (
    PdfImageInfo,
    extract_image_positions,
    page_has_ocr_text,
    parse_pdfimages_list,
)
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger

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
        pdfimages_map, masked_pages = parse_pdfimages_list(input_pdf)

        if not image_positions and not pdfimages_map:
            logger.info("No images found in PDF. Copying original.")
            shutil.copy2(input_pdf, output_pdf)
            stats.warnings.append("No images found to OCR in mixed content PDF")
            self._calculate_final_stats(stats, start_time)
            return stats

        # Pages that pdfimages finds but extract_image_positions misses
        # (e.g. images inside nested Form XObjects).
        missed_pages = set(pdfimages_map.keys()) - set(image_positions.keys())

        # Pages with JBIG2 masks (DjVu-like) are better handled via
        # pdftoppm rendering (composites BG+FG+mask into readable image)
        # than by OCR'ing the degraded BG component separately.
        render_candidates = missed_pages | (masked_pages & set(image_positions.keys()))

        # Detect pages with multiple overlapping full-page images
        # (e.g. DjVu-like BG+FG layers without masks).  Rendering via
        # pdftoppm composites them correctly, whereas trying to OCR each
        # layer separately produces duplicate / misaligned text.
        with pikepdf.open(input_pdf) as pdf_scan:
            total_pages = len(pdf_scan.pages)
            for pg_num, imgs in image_positions.items():
                if len(imgs) < 2 or pg_num in render_candidates:
                    continue
                page_obj = pdf_scan.pages[pg_num - 1]
                mb = page_obj.mediabox
                page_area = (float(mb[2]) - float(mb[0])) * (float(mb[3]) - float(mb[1]))
                if page_area <= 0:
                    continue
                full_count = sum(1 for p in imgs if (p.width * p.height) / page_area > 0.5)
                if full_count >= 2:
                    render_candidates.add(pg_num)
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

                    # Only send non-masked positioned pages to _ocr_image_pages.
                    # Masked pages go to _ocr_rendered_pages instead.
                    positioned_image_positions = {
                        p: imgs for p, imgs in image_positions.items() if p not in render_candidates
                    }
                    positioned_total = sum(
                        len(imgs) for imgs in positioned_image_positions.values()
                    )

                    # Compute progress proportions.  Both render and
                    # positioned paths share a 5–85 % progress band.
                    render_pages = render_candidates - excluded_pages
                    n_render = len(render_pages) if render_pages else 0
                    n_positioned = len(positioned_image_positions)
                    n_total_ocr = n_render + n_positioned or 1
                    render_band = int(80 * n_render / n_total_ocr)  # share of 80 pp
                    pos_band = 80 - render_band

                    # OCR render_candidates via pdftoppm rendering (page-by-page,
                    # no bulk image extraction needed).
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

                    # Extract images only for pages that need position-based OCR.
                    extracted_images: list[Path] = []
                    if positioned_image_positions:
                        ext_pct = 5 + render_band
                        if progress_callback:
                            progress_callback(ext_pct, 100, _("Extracting images..."))
                        extracted_images = self._extract_and_filter_images(input_pdf, images_dir)

                    self._ocr_image_pages(
                        pdf,
                        positioned_image_positions,
                        extracted_images,
                        positioned_total,
                        stats,
                        ocr_texts,
                        ocr_proc,
                        progress_callback,
                        excluded_pages=excluded_pages,
                        pdfimages_map=pdfimages_map,
                        progress_start=5 + render_band,
                        progress_band=pos_band,
                    )

                    # Remove excluded pages before saving
                    if excluded_pages:
                        for idx in sorted(excluded_pages, reverse=True):
                            if 0 < idx <= len(pdf.pages):
                                del pdf.pages[idx - 1]
                                logger.info(f"Removed excluded page {idx} from output")

                    if progress_callback:
                        progress_callback(90, 100, _("Saving PDF..."))
                    stats.pages_processed = (
                        len(positioned_image_positions)
                        + len(render_pages)
                        - len(excluded_pages & all_image_pages)
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
            if hasattr(self, "cancel_event") and self.cancel_event.is_set():
                logger.info("Processing cancelled by user in mixed content mode")
                raise InterruptedError("Processing cancelled by user")

            page_imgs = image_positions[page_num]
            page_img_infos: list[PdfImageInfo] = _pdfmap.get(page_num, [])

            # Skip excluded pages entirely (no preprocessing, no OCR)
            if page_num in _excluded:
                logger.info(
                    f"Page {page_num}: excluded from OCR, skipping ({len(page_imgs)} image(s))"
                )
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
                    continue
                stripped = strip_invisible_text(page, pdf)
                if stripped:
                    logger.info(
                        f"Page {page_num}: stripped {stripped} old OCR text block(s) before re-OCR"
                    )

            if progress_callback:
                pct = progress_start + int(progress_band * processed_images / max(total_images, 1))
                progress_callback(pct, 100, _("OCR page {0}...").format(page_num))

            # Match each ImagePosition to its extracted file using
            # pdfimages_map indices.  Select the images with the largest
            # compressed size (best quality indicator for DjVu-like PDFs
            # where BG layers are tiny but have large pixel dimensions).
            if len(page_img_infos) >= len(page_imgs):
                # Sort by compressed size descending — highest quality first
                sorted_infos = sorted(page_img_infos, key=lambda info: info.comp_size, reverse=True)
                selected_files: list[tuple[Path, int]] = []
                for info in sorted_infos[: len(page_imgs)]:
                    if info.idx < len(extracted_images):
                        selected_files.append((extracted_images[info.idx], info.idx))
            else:
                # Fallback: no pdfimages_map or mismatch
                selected_files = []

            for pos_i, img_pos in enumerate(page_imgs):
                if pos_i < len(selected_files):
                    img_path = selected_files[pos_i][0]
                else:
                    logger.warning(f"Page {page_num}: no extracted image for position {pos_i}")
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
        150 DPI and add OCR text as an overlay.
        """
        import cv2

        render_dir = Path(temp_dir) / "rendered"
        render_dir.mkdir(exist_ok=True)

        for render_idx, page_num in enumerate(page_nums):
            if hasattr(self, "cancel_event") and self.cancel_event.is_set():
                raise InterruptedError("Processing cancelled by user")

            if page_num < 1 or page_num > len(pdf.pages):
                continue

            page = pdf.pages[page_num - 1]

            if page_has_ocr_text(page):
                if not self.config.replace_existing_ocr:
                    logger.info(f"Page {page_num}: rendered — already has OCR, skipping")
                    continue
                stripped = strip_invisible_text(page, pdf)
                if stripped:
                    logger.info(
                        f"Page {page_num}: stripped {stripped} old OCR block(s) before re-OCR"
                    )

            if progress_callback:
                pct = progress_start + int(progress_band * render_idx / max(len(page_nums), 1))
                progress_callback(pct, 100, _("OCR page {0} (rendered)...").format(page_num))

            # Render single page via pdftoppm (PPM, 150 DPI)
            out_prefix = str(render_dir / f"p{page_num}")
            try:
                subprocess.run(
                    [
                        "pdftoppm",
                        "-r",
                        "150",
                        "-f",
                        str(page_num),
                        "-l",
                        str(page_num),
                        str(input_pdf),
                        out_prefix,
                    ],
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                logger.warning(f"pdftoppm failed for page {page_num}: {exc}")
                continue

            # Find the rendered file (pdftoppm names: prefix-NNNNNN.ppm)
            rendered_files = sorted(render_dir.glob(f"p{page_num}-*.ppm"))
            if not rendered_files:
                logger.warning(f"No rendered image for page {page_num}")
                continue

            rendered_path = rendered_files[0]
            img = cv2.imread(str(rendered_path))
            if img is None:
                logger.warning(f"Could not read rendered image: {rendered_path}")
                rendered_path.unlink(missing_ok=True)
                continue

            # OCR the rendered image
            ocr_results = self._ocr_via_persistent(img, ocr_proc)

            # Clean up rendered file
            rendered_path.unlink(missing_ok=True)

            if not ocr_results:
                logger.debug(f"Page {page_num}: rendered — no OCR text found")
                continue

            # Position text over the entire page
            mediabox = page.mediabox
            page_width = float(mediabox[2]) - float(mediabox[0])
            page_height = float(mediabox[3]) - float(mediabox[1])
            img_h, img_w = img.shape[:2]
            scale_x = page_width / img_w
            scale_y = page_height / img_h

            text_commands = self._create_text_layer_commands(
                ocr_results,
                0.0,  # x offset (full page)
                0.0,  # y offset (full page)
                page_width,
                page_height,
                scale_x,
                scale_y,
            )
            self._append_text_to_page(pdf, page, text_commands)

            stats.total_text_regions += len(ocr_results)
            formatted = self._format_ocr_text(ocr_results, float(img_w))
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
