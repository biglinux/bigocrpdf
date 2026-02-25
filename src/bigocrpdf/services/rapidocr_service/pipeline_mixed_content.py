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
    extract_image_positions,
    page_has_ocr_text,
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
        ocr_texts: list[str] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            images_dir = Path(temp_dir) / "images"
            images_dir.mkdir()

            if progress_callback:
                progress_callback(5, 100, _("Extracting images..."))

            images_by_page = self._extract_images_by_page(input_pdf, images_dir)

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

                    # Detect pages with custom font encoding BEFORE image OCR
                    # so we can skip per-image OCR on those pages (full-page
                    # OCR will cover everything including images).
                    custom_font_pages = self._detect_custom_font_pages(
                        pdf, pages_with_images, excluded_pages
                    )

                    self._ocr_image_pages(
                        pdf,
                        image_positions,
                        images_by_page,
                        total_images,
                        stats,
                        ocr_texts,
                        ocr_proc,
                        progress_callback,
                        excluded_pages=excluded_pages | custom_font_pages,
                    )

                    # Full-page OCR for pages with custom/subset fonts that
                    # pdftotext cannot decode.  Strips original text operators
                    # first (safe because _ocr_image_pages skipped these pages)
                    # then adds a complete OCR text layer.
                    full_ocr_pages = self._ocr_custom_font_regions(
                        pdf,
                        input_pdf,
                        custom_font_pages,
                        ocr_texts,
                        stats,
                        ocr_proc,
                        excluded_pages,
                        temp_dir,
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

        # Extract native text from pages that were NOT fully OCR'd
        # to avoid duplicating text already captured by supplementary OCR.
        all_pages = set(range(1, total_pages + 1))
        native_text_pages = all_pages - (full_ocr_pages or set())
        if native_text_pages:
            native_text = self._extract_native_text(input_pdf, native_text_pages)
            if native_text.strip():
                logger.info(
                    f"Extracted native text from {len(native_text_pages)} page(s) "
                    f"(excluded {len(full_ocr_pages or set())} fully-OCR'd page(s))"
                )
            else:
                native_text = ""
        else:
            native_text = ""

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
        images_by_page: dict[int, list[Path]],
        total_images: int,
        stats: ProcessingStats,
        ocr_texts: list[str],
        ocr_proc: subprocess.Popen,
        progress_callback: Callable[[int, int, str], None] | None,
        excluded_pages: set[int] | None = None,
    ) -> None:
        """OCR all image-bearing pages, modifying the PDF in place.

        Uses per-page image mapping to ensure correct correspondence
        between extracted image files and their positions in the PDF.
        """
        enhance = getattr(self.config, "enhance_embedded_images", False)
        logger.info(f"Mixed content: enhance_embedded_images={enhance}")
        processed_images = 0
        _excluded = excluded_pages or set()

        for page_num in sorted(image_positions.keys()):
            if hasattr(self, "cancel_event") and self.cancel_event.is_set():
                logger.info("Processing cancelled by user in mixed content mode")
                raise InterruptedError("Processing cancelled by user")

            page_imgs = image_positions[page_num]

            # Skip excluded pages entirely (no preprocessing, no OCR)
            if page_num in _excluded:
                logger.info(f"Page {page_num}: excluded from OCR, skipping ({len(page_imgs)} image(s))")
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
                pct = 10 + int(80 * processed_images / total_images)
                progress_callback(pct, 100, _("OCR page {0}...").format(page_num))

            # Get image files for this specific page
            page_image_files = images_by_page.get(page_num, [])

            for pos_idx, img_pos in enumerate(page_imgs):
                if pos_idx >= len(page_image_files):
                    logger.warning(
                        f"Page {page_num}: image position {pos_idx} ({img_pos.name}) "
                        f"has no corresponding extracted file "
                        f"({len(page_image_files)} files for {len(page_imgs)} positions)"
                    )
                    break
                img_path = page_image_files[pos_idx]
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

    @staticmethod
    def _detect_custom_font_pages(
        pdf, pages_with_images: set[int], excluded_pages: set[int]
    ) -> set[int]:
        """Return page numbers that have text rendered with custom/subset fonts.

        Scans BT/ET blocks in the content stream for Tj operators whose
        text payload is mostly non-ASCII — a strong indicator of a custom
        CMap font that pdftotext cannot decode.
        """
        import re

        _excluded = excluded_pages or set()
        result: set[int] = set()

        for page_num in sorted(pages_with_images):
            if page_num in _excluded:
                continue

            page = pdf.pages[page_num - 1]
            contents = page.get("/Contents")
            if not contents:
                continue

            raw_parts: list[bytes] = []
            if isinstance(contents, pikepdf.Array):
                for s in contents:
                    try:
                        raw_parts.append(s.read_bytes())
                    except Exception:
                        pass
            else:
                try:
                    raw_parts.append(contents.read_bytes())
                except Exception:
                    continue

            full_stream = b"".join(raw_parts).decode("latin-1", errors="ignore")
            for m in re.finditer(r"BT\b(.*?)ET\b", full_stream, re.DOTALL):
                block = m.group(1)
                text_ops = re.findall(r"\(([^)]*)\)\s*Tj", block)
                for text_content in text_ops:
                    if text_content and len(text_content) >= 2:
                        printable_letters = sum(
                            1 for c in text_content if c.isalpha() and ord(c) < 128
                        )
                        total = len(text_content)
                        if total > 0 and printable_letters / total < 0.3:
                            result.add(page_num)
                            break
                if page_num in result:
                    break

        if result:
            logger.info(
                f"Detected custom font encoding on page(s): "
                f"{', '.join(str(p) for p in sorted(result))}"
            )
        return result

    @staticmethod
    def _strip_text_operators(pdf, page) -> None:
        """Remove BT..ET text blocks from a page's content stream.

        This is used before adding a full-page OCR text layer so that
        pdftotext does not return duplicate text (original + OCR).
        Image drawing (Do), graphics state, and other non-text operators
        are preserved.
        """
        import re

        contents = page.get("/Contents")
        if not contents:
            return

        raw_parts: list[bytes] = []
        if isinstance(contents, pikepdf.Array):
            for s in contents:
                try:
                    raw_parts.append(s.read_bytes())
                except Exception:
                    pass
        else:
            try:
                raw_parts.append(contents.read_bytes())
            except Exception:
                return

        full_stream = b"".join(raw_parts)
        # Remove BT..ET blocks (text drawing).  PDF spec guarantees
        # BT/ET are not nested so a non-greedy match is safe.
        # Replace with a newline to avoid gluing adjacent operators
        # (e.g. Q BT...ET q becoming Qq which is an invalid operator).
        cleaned = re.sub(rb"BT\b.*?ET\b", b"\n", full_stream, flags=re.DOTALL)
        if cleaned == full_stream:
            return  # nothing to strip

        page["/Contents"] = pdf.make_stream(cleaned)

    @staticmethod
    def _box_to_rect(box: list[list[float]]) -> tuple[float, float, float, float]:
        """Convert 4-point box to (x1, y1, x2, y2) rectangle."""
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return (min(xs), min(ys), max(xs), max(ys))

    @staticmethod
    def _rect_iou(a: tuple, b: tuple) -> float:
        """Intersection-over-Union of two axis-aligned rectangles."""
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _ocr_text_in_native(ocr_text: str, native_words: set[str]) -> bool:
        """Check if an OCR result is redundant given native text words.

        Returns True when ALL significant words in the OCR text already
        appear in the native word set.  Handles accent differences
        (OCR "Transito" vs native "Trânsito") via ASCII folding.
        Also catches partial fragments like "PA" from "PARTICULAR".
        """
        import unicodedata

        def _fold(s: str) -> str:
            nfkd = unicodedata.normalize("NFKD", s.lower())
            return "".join(
                c for c in nfkd
                if not unicodedata.combining(c) and c.isalnum()
            )

        words = ocr_text.strip().split()
        if not words:
            return True
        matched = 0
        for w in words:
            folded = _fold(w)
            if not folded:
                matched += 1
                continue
            if folded in native_words:
                matched += 1
                continue
            # Check if this is a prefix of a native word (catches OCR
            # fragments like "PA" from "PARTICULAR")
            if len(folded) <= 5 and any(
                nw.startswith(folded) for nw in native_words
            ):
                matched += 1
        return matched / len(words) >= 0.8

    def _ocr_with_tiles(
        self,
        img,
        ocr_proc,
        tile_rows: int = 2,
        tile_cols: int = 2,
        overlap_ratio: float = 0.1,
    ) -> list:
        """OCR an image, then re-OCR as tiles for small text detection.

        The PP-OCRv5 detection model can miss small text in large images.
        After the full-image pass we split into overlapping tiles, OCR each,
        and merge back only genuinely new detections.
        """
        from bigocrpdf.services.rapidocr_service.config import OCRResult

        # Full-image pass
        if ocr_proc is not None:
            full_results = self._ocr_via_persistent(img, ocr_proc)
        else:
            full_results = self._run_ocr(img)

        all_results: list[OCRResult] = list(full_results) if full_results else []

        h, w = img.shape[:2]
        if max(h, w) < 2000:
            return all_results  # image small enough, tiling unnecessary

        tile_h = h // tile_rows
        tile_w = w // tile_cols
        overlap_h = int(tile_h * overlap_ratio)
        overlap_w = int(tile_w * overlap_ratio)

        for row in range(tile_rows):
            for col in range(tile_cols):
                y1 = max(0, row * tile_h - overlap_h)
                y2 = min(h, (row + 1) * tile_h + overlap_h)
                x1 = max(0, col * tile_w - overlap_w)
                x2 = min(w, (col + 1) * tile_w + overlap_w)

                tile = img[y1:y2, x1:x2]

                if ocr_proc is not None:
                    tile_results = self._ocr_via_persistent(tile, ocr_proc)
                else:
                    tile_results = self._run_ocr(tile)

                if not tile_results:
                    continue

                for tr in tile_results:
                    # Map tile coordinates back to full image
                    tr.box = [[p[0] + x1, p[1] + y1] for p in tr.box]

                    # Dedup via bounding-box IoU.  Two detections with
                    # overlapping boxes (IoU > 0.3) are considered the same
                    # text region — keep the one with higher confidence.
                    tr_rect = self._box_to_rect(tr.box)
                    is_dup = False
                    for idx, er in enumerate(all_results):
                        er_rect = self._box_to_rect(er.box)
                        if self._rect_iou(tr_rect, er_rect) > 0.3:
                            # Replace if tile version has higher confidence
                            if tr.confidence > er.confidence:
                                all_results[idx] = tr
                            is_dup = True
                            break
                    if not is_dup:
                        all_results.append(tr)

        return all_results

    def _ocr_custom_font_regions(
        self,
        pdf,
        input_pdf: Path,
        custom_font_pages: set[int],
        ocr_texts: list[str],
        stats: ProcessingStats,
        ocr_proc: subprocess.Popen,
        excluded_pages: set[int],
        temp_dir: str,
    ) -> set[int]:
        """Render and OCR pages that have text with custom/subset fonts.

        Pages to process are provided by ``_detect_custom_font_pages``.
        For each page, the original text operators are stripped and replaced
        with a full-page OCR invisible text layer.

        Returns the set of page numbers that received full-page OCR,
        so the caller can exclude them from native text extraction
        (avoiding text duplication).
        """
        _excluded = excluded_pages or set()
        render_dir = Path(temp_dir) / "page_renders"
        fully_ocrd_pages: set[int] = set()

        for page_num in sorted(custom_font_pages):
            if page_num in _excluded:
                continue

            page = pdf.pages[page_num - 1]

            logger.info(
                f"Page {page_num}: rendering full page for custom-font OCR"
            )

            # Render the full page
            render_dir.mkdir(parents=True, exist_ok=True)
            prefix = str(render_dir / f"page_{page_num}")
            cmd = [
                "pdftoppm",
                "-f", str(page_num),
                "-l", str(page_num),
                "-r", str(self.config.dpi),
                "-png",
                "-singlefile",
                str(input_pdf),
                prefix,
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                logger.warning(f"Page {page_num}: pdftoppm render failed: {e.stderr}")
                continue

            rendered_path = Path(f"{prefix}.png")
            if not rendered_path.exists():
                continue

            try:
                from bigocrpdf.services.rapidocr_service.pdf_extractor import (
                    load_image_with_exif_rotation,
                )

                img = load_image_with_exif_rotation(rendered_path)
                if img is None:
                    continue

                page_ocr_results = self._ocr_with_tiles(img, ocr_proc)

                if not page_ocr_results:
                    continue

                # Extract native text from this page to avoid duplication.
                # Only add OCR invisible text for regions pdftotext cannot
                # already decode from the original fonts.
                native_text = ""
                try:
                    native_proc = subprocess.run(
                        ["pdftotext", "-f", str(page_num), "-l", str(page_num),
                         str(input_pdf), "-"],
                        capture_output=True, text=True, timeout=10,
                    )
                    native_text = native_proc.stdout or ""
                except (subprocess.SubprocessError, OSError):
                    pass

                # Filter OCR noise: CID glyph artifacts and very short
                # non-word results the model misreads from icons/symbols.
                import re as _re
                _CID_RE = _re.compile(
                    r"^\s*[\(\[]?\s*c?i?d?\s*:\s*[\)\]]?\s*$", _re.I
                )
                page_ocr_results = [
                    r for r in page_ocr_results
                    if not _CID_RE.match(r.text)
                ]

                if native_text.strip():
                    # Build set of ASCII-folded native words for robust
                    # dedup even when OCR groups words differently or
                    # misses accents.
                    import unicodedata

                    def _fold_word(s: str) -> str:
                        nfkd = unicodedata.normalize("NFKD", s.lower())
                        return "".join(
                            c for c in nfkd
                            if not unicodedata.combining(c) and c.isalnum()
                        )

                    native_words = {
                        _fold_word(w) for w in native_text.split()
                    }
                    native_words.discard("")
                    page_ocr_results = [
                        r for r in page_ocr_results
                        if not self._ocr_text_in_native(r.text, native_words)
                    ]

                if not page_ocr_results:
                    # Native text covers everything, nothing new from OCR
                    fully_ocrd_pages.add(page_num)
                    continue

                # Get page dimensions for coordinate mapping
                mediabox = page.mediabox
                page_height_pts = float(mediabox[3]) - float(mediabox[1])
                page_width_pts = float(mediabox[2]) - float(mediabox[0])
                img_h, img_w = img.shape[:2]
                scale_x = page_width_pts / img_w
                scale_y = page_height_pts / img_h

                # Add invisible text layer for the full page.
                # Original text operators are preserved so the visual
                # appearance stays unchanged.  The OCR layer provides
                # selectable text for regions pdftotext cannot decode.
                text_commands = self._create_text_layer_commands(
                    page_ocr_results,
                    0.0,  # img_x = 0 (full page)
                    0.0,  # img_y = 0 (full page)
                    page_width_pts,
                    page_height_pts,
                    scale_x,
                    scale_y,
                )
                self._append_text_to_page(pdf, page, text_commands)

                stats.total_text_regions += len(page_ocr_results)
                formatted = self._format_ocr_text(page_ocr_results, float(img_w))
                if formatted:
                    ocr_texts.append(formatted)

                logger.info(
                    f"Page {page_num}: supplementary OCR found "
                    f"{len(page_ocr_results)} text region(s)"
                )
                fully_ocrd_pages.add(page_num)
            except Exception as e:
                logger.warning(f"Page {page_num}: supplementary OCR failed: {e}")
            finally:
                try:
                    rendered_path.unlink()
                except OSError:
                    pass

        return fully_ocrd_pages

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

    @staticmethod
    def _extract_images_by_page(input_pdf: Path, images_dir: Path) -> dict[int, list[Path]]:
        """Extract images from PDF with proper per-page mapping.

        Uses pdfimages -list to create an accurate mapping between
        extracted files and their source pages, excluding soft masks
        and other non-content images. This ensures correct correspondence
        between extracted image files and image positions from
        extract_image_positions().

        Returns:
            dict mapping page number (1-based) to list of image file paths
            in content-stream order.
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
            for f in extracted:
                try:
                    f.unlink()
                except OSError:
                    pass

            cmd_pbm = ["pdfimages", str(input_pdf), str(images_dir / "img")]
            try:
                subprocess.run(cmd_pbm, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"pdfimages fallback failed: {e.stderr}")
                raise RuntimeError(f"Failed to extract images: {e}") from e

        # Parse pdfimages -list to get page-to-image mapping (excluding smasks)
        cmd_list = ["pdfimages", "-list", str(input_pdf)]
        try:
            list_result = subprocess.run(cmd_list, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            logger.warning("pdfimages -list failed, falling back to flat extraction")
            filtered = sorted(
                f for f in images_dir.glob("img-*")
                if not (f.stat().st_size < 5000 and f.suffix.lower() == ".png")
            )
            return {1: filtered} if filtered else {}

        # Build page -> [image_index] mapping, skipping smasks/masks
        page_image_indices: dict[int, list[int]] = {}
        lines = list_result.stdout.splitlines()
        start_parsing = False
        for line in lines:
            if line.startswith("---"):
                start_parsing = True
                continue
            if not start_parsing:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    page_num = int(parts[0])
                    img_idx = int(parts[1])
                    img_type = parts[2]
                    # Only include actual images, not soft masks or masks
                    if img_type == "image":
                        page_image_indices.setdefault(page_num, []).append(img_idx)
                except ValueError:
                    continue

        # Map indices to actual files
        result_map: dict[int, list[Path]] = {}
        for page_num, indices in sorted(page_image_indices.items()):
            page_files: list[Path] = []
            for idx in indices:
                found = None
                for pattern in (f"img-{idx:03d}.*", f"img-{idx:04d}.*"):
                    matches = list(images_dir.glob(pattern))
                    if matches:
                        found = matches[0]
                        break
                if found:
                    page_files.append(found)
            if page_files:
                result_map[page_num] = page_files

        total = sum(len(v) for v in result_map.values())
        logger.info(f"Extracted {total} images across {len(result_map)} page(s) (per-page mapping)")
        return result_map
