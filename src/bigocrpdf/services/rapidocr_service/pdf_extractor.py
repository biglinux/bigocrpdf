import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pikepdf

from bigocrpdf.constants import PDF_TOOL_TIMEOUT_SECS
from bigocrpdf.services.rapidocr_service.pdf_image_analysis import (
    ImagePosition,
    PdfImageInfo,
    _get_page_xobjects,
    extract_image_positions,
    match_positions_to_images,
    parse_pdfimages_list,
)
from bigocrpdf.services.rapidocr_service.pdf_page_geometry import (
    extract_content_streams,
    get_page_image_encodings,
    load_image_with_exif_rotation,
    merge_page_fonts,
    transform_ocr_coords_for_rotation,
)
from bigocrpdf.services.rapidocr_service.pdf_validation import classify_text_layer
from bigocrpdf.services.rapidocr_service.resource_manager import select_render_dpi_for_page
from bigocrpdf.utils.logger import logger

__all__ = [
    "ImagePosition",
    "PDFImageExtractor",
    "PdfImageInfo",
    "_get_page_xobjects",
    "extract_content_streams",
    "extract_image_positions",
    "get_page_image_encodings",
    "get_pages_with_native_text",
    "has_native_text",
    "has_trusted_native_text",
    "load_image_with_exif_rotation",
    "match_positions_to_images",
    "merge_page_fonts",
    "page_has_ocr_text",
    "parse_pdfimages_list",
    "transform_ocr_coords_for_rotation",
]


def _page_has_visible_text(page: pikepdf.Page) -> bool:
    """Check if a page has visible non-OCR text."""
    text_state = _page_text_state(page)
    return text_state[0]


def _page_has_invisible_ocr_text(page: pikepdf.Page) -> bool:
    text_state = _page_text_state(page)
    return text_state[1]


def _page_text_state(page: pikepdf.Page) -> tuple[bool, bool]:
    """Return ``(has_visible_text, has_invisible_ocr_text)`` for one page."""
    try:
        ops = list(pikepdf.parse_content_stream(page))
    except Exception:
        return _page_text_state_from_raw(page)
    return _scan_page_text_ops(ops)


@dataclass
class _PageTextScanState:
    has_visible_text: bool = False
    has_invisible_ocr_text: bool = False
    graphics_state_is_invisible: bool = False
    in_text: bool = False
    text_block_is_invisible: bool = False
    invisible_stack: list[bool] = field(default_factory=list)


def _scan_page_text_ops(ops: list[Any]) -> tuple[bool, bool]:
    state = _PageTextScanState()
    for operands, operator in ops:
        _update_page_text_scan_state(state, operands, str(operator))
    return state.has_visible_text, state.has_invisible_ocr_text


def _update_page_text_scan_state(
    state: _PageTextScanState,
    operands: list,
    op: str,
) -> None:
    if op == "q":
        state.invisible_stack.append(state.graphics_state_is_invisible)
        return
    if op == "Q":
        state.graphics_state_is_invisible = (
            state.invisible_stack.pop() if state.invisible_stack else False
        )
        return
    if op == "gs" and _is_ocr_invisible_graphics_state(operands):
        state.graphics_state_is_invisible = True
        return
    if op == "BT":
        state.in_text = True
        state.text_block_is_invisible = state.graphics_state_is_invisible
        return
    if op == "ET":
        state.in_text = False
        state.text_block_is_invisible = False
        return
    if op == "Tr" and state.in_text and operands:
        state.text_block_is_invisible = state.graphics_state_is_invisible or int(operands[0]) == 3
        return
    if op in {"Tj", "TJ"} and state.in_text:
        _record_page_text_operator(state)


def _record_page_text_operator(state: _PageTextScanState) -> None:
    if state.text_block_is_invisible:
        state.has_invisible_ocr_text = True
    else:
        state.has_visible_text = True


def _is_ocr_invisible_graphics_state(operands: list) -> bool:
    return bool(operands) and str(operands[0]) == "/GSOcrInvisible"


def _page_text_state_from_raw(page: pikepdf.Page) -> tuple[bool, bool]:
    import re

    has_visible_text = False
    has_invisible_ocr_text = False
    for raw in _page_raw_content_streams(page):
        text = raw.decode("latin-1", errors="ignore")
        has_invisible_ocr_text = has_invisible_ocr_text or bool(
            re.search(r"\b3\s+Tr\b", text) or "/GSOcrInvisible" in text
        )
        for match in re.finditer(r"BT\b(.*?)ET\b", text, re.DOTALL):
            block = match.group(1)
            if re.search(r"\bTj\b|\bTJ\b", block) and not re.search(r"\b3\s+Tr\b", block):
                has_visible_text = has_visible_text or "/GSOcrInvisible" not in text
    return has_visible_text, has_invisible_ocr_text


def _page_raw_content_streams(page: pikepdf.Page) -> list[bytes]:
    contents = page.get("/Contents")
    if not contents:
        return []

    raw_parts: list[bytes] = []
    streams = list(contents) if isinstance(contents, pikepdf.Array) else [contents]
    for stream in streams:
        try:
            raw_parts.append(stream.read_bytes())
        except Exception:
            pass
    return raw_parts


def _bounded_page_range(
    total_pages: int,
    page_range: tuple[int, int] | None,
) -> tuple[int, int]:
    if not page_range:
        return 1, total_pages
    start_page, end_page = page_range
    return max(1, start_page), min(total_pages, end_page)


def _clean_pdfimages_output(output_dir: Path) -> None:
    for file_path in output_dir.glob("*"):
        try:
            file_path.unlink()
        except OSError:
            pass


def _cleanup_pdfimages_objects(output_dir: Path) -> None:
    for file_path in output_dir.glob("obj-*"):
        try:
            file_path.unlink()
        except OSError:
            pass


def _page_dimensions_and_text_pages(
    pdf_path: Path,
    start_page: int,
    end_page: int,
) -> tuple[dict[int, tuple[float, float]], set[int]]:
    page_dimensions: dict[int, tuple[float, float]] = {}
    text_rich_pages: set[int] = set()
    with pikepdf.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            if not start_page <= page_num <= end_page:
                continue
            if hasattr(page, "mediabox") and page.mediabox:
                page_dimensions[page_num] = _page_dimensions(page)
            if _page_has_substantial_vector_text(page):
                text_rich_pages.add(page_num)
    return page_dimensions, text_rich_pages


def _page_dimensions(page: pikepdf.Page) -> tuple[float, float]:
    user_unit = float(page.get("/UserUnit", 1))
    if not math.isfinite(user_unit) or user_unit <= 0:
        raise ValueError("PDF page has an invalid UserUnit")
    return (
        abs(float(page.mediabox[2]) - float(page.mediabox[0])) * user_unit,
        abs(float(page.mediabox[3]) - float(page.mediabox[1])) * user_unit,
    )


def _page_has_substantial_vector_text(page: pikepdf.Page) -> bool:
    fonts = page.get("/Resources", {}).get("/Font", {})
    if len(fonts) < 2:
        return False
    contents = page.get("/Contents")
    if contents is None:
        return False
    try:
        raw = (
            b"".join(bytes(ref.read_bytes()) for ref in contents)
            if isinstance(contents, pikepdf.Array)
            else bytes(contents.read_bytes())
        )
    except Exception:
        return False
    return raw.count(b"Tj") + raw.count(b"TJ") >= 20


def _append_fallback_or_skip_text_page(
    current_page: int,
    text_rich_pages: set[int],
    fallback_pages: list[int],
    reason: str,
) -> None:
    if current_page in text_rich_pages:
        logger.info(f"Page {current_page}: {reason} but has vector text, skipping OCR")
        return
    fallback_pages.append(current_page)
    logger.info(f"Page {current_page}: {reason} found, will render with pdftoppm")


def _small_page_image_reason(
    page_dim: tuple[float, float] | None,
    img_width: int,
    img_height: int,
) -> str | None:
    if not page_dim:
        return None
    expected_dpi = 300
    expected_w = page_dim[0] / 72.0 * expected_dpi
    expected_h = page_dim[1] / 72.0 * expected_dpi
    expected_area = expected_w * expected_h
    coverage = (img_width * img_height) / expected_area if expected_area > 0 else 1.0
    width_ratio = img_width / expected_w if expected_w > 0 else 1.0
    height_ratio = img_height / expected_h if expected_h > 0 else 1.0
    if coverage >= 0.15 and width_ratio >= 0.45 and height_ratio >= 0.45:
        return None
    if coverage < 0.15:
        return f"area {coverage:.0%}"
    return f"dimensions {img_width}x{img_height} ({width_ratio:.0%}w, {height_ratio:.0%}h)"


def _add_pdfimages_mapping_line(
    line: str,
    mapping: dict[int, list[tuple[int, int, int]]],
    masked_pages: set[int],
) -> None:
    parts = line.split()
    if len(parts) < 5:
        return
    try:
        page_num = int(parts[0])
        image_index = int(parts[1])
        image_width = int(parts[3])
        image_height = int(parts[4])
    except ValueError:
        return

    if parts[2] == "image":
        mapping.setdefault(page_num, []).append((image_index, image_width, image_height))
    elif parts[2] == "mask":
        masked_pages.add(page_num)


def has_native_text(pdf_path: Path) -> bool:
    """
    Check if a PDF has native (non-OCR) visible text content.

    Opens the PDF with pikepdf and checks whether any page contains
    visible text blocks (not invisible OCR with render mode 3).
    This correctly distinguishes:
    - Image-only PDFs (scanned documents, or images + invisible OCR) → False
    - Mixed content PDFs (real typeset text + images) → True

    Args:
        pdf_path: Path to the PDF file

    Returns:
        True if the PDF has visible native text content
    """
    try:
        with pikepdf.open(pdf_path) as pdf:
            for page in pdf.pages:
                if _page_has_visible_text(page):
                    return True
        return False
    except Exception as e:
        logger.warning(f"Could not check for native text: {e}")
        return False


def extract_native_text_for_quality(pdf_path: Path) -> str:
    """Extract native PDF text for trust classification."""
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
        logger.warning(f"Could not extract native text for quality check: {exc}")
        return ""
    if result.returncode != 0:
        logger.warning(f"Native text extraction failed: {result.stderr.strip()}")
        return ""
    return result.stdout


def has_trusted_native_text(pdf_path: Path) -> bool:
    """Return True when the native text layer is extractable and not lossy."""
    extracted = extract_native_text_for_quality(pdf_path)
    quality = classify_text_layer(extracted)
    if quality.status != "trusted":
        logger.info(f"Native text layer rejected: {quality.reason}")
        return False
    return True


def get_pages_with_native_text(pdf_path: Path, total_pages: int) -> set[int]:
    """Detect which pages have native (non-OCR) visible text content.

    Uses pikepdf to identify pages that contain visible text blocks
    (not invisible OCR with render mode 3), so they can be preserved
    as-is during OCR.

    Args:
        pdf_path: Path to the PDF file
        total_pages: Total number of pages

    Returns:
        Set of 1-based page numbers that have visible native text
    """
    pages_with_text: set[int] = set()
    try:
        with pikepdf.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                if page_num > total_pages:
                    break
                if _page_has_visible_text(page):
                    pages_with_text.add(page_num)
    except Exception as e:
        logger.warning(f"Could not detect native text pages: {e}")
    return pages_with_text


def page_has_ocr_text(page: pikepdf.Page) -> bool:
    """Check if a PDF page has invisible OCR text."""
    return _page_has_invisible_ocr_text(page)


class PDFImageExtractor:
    """Native PDF image extraction without re-encoding.

    Uses pdfimages -all to extract images directly from PDFs.
    This is more efficient than rendering pages with pdftoppm because:
    - No re-encoding of images (preserves original quality)
    - No upscaling of low-DPI content
    - Much faster and uses less memory

    Falls back to pdftoppm for formats that OpenCV/PIL cannot read
    (JBIG2, CCITT fax), which are common in scanned document PDFs.
    """

    # Extensions that OpenCV and PIL cannot read natively
    _UNSUPPORTED_EXTENSIONS = frozenset({".jb2e", ".jb2g", ".ccitt"})

    def __init__(self, dpi: int | None = None, max_render_megapixels: float = 45):
        # DPI parameter kept for API compatibility but not used for extraction
        # (pdfimages extracts at native resolution)
        self.dpi = dpi
        self.max_render_megapixels = max_render_megapixels
        # Track which 1-indexed pages were rendered via pdftoppm
        # (rotation already baked into the image)
        self.rendered_pages: set[int] = set()
        # Pages with image masks (DjVu-like FG/BG layers)
        self.masked_pages: set[int] = set()

    def extract(
        self,
        pdf_path: Path,
        output_dir: Path,
        page_range: tuple[int, int] | None = None,
        skip_pages: set[int] | None = None,
    ) -> list[Path | None]:
        """Extract native images from PDF ensuring correct page mapping.

        When page_range is provided, only images from those pages are
        extracted (using pdfimages -f/-l flags), significantly reducing
        disk usage for large documents.

        For images stored in formats that OpenCV/PIL cannot decode
        (JBIG2, CCITT), falls back to pdftoppm page rendering.

        Args:
            pdf_path: Path to the PDF file.
            output_dir: Directory to extract images to.
            page_range: Optional (start, end) 1-indexed page range.
            skip_pages: Optional set of 1-indexed page numbers to skip entirely.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        self.rendered_pages = set()

        with pikepdf.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

        start_page, end_page = _bounded_page_range(total_pages, page_range)
        num_pages_to_process = end_page - start_page + 1
        results: list[Path | None] = [None] * num_pages_to_process
        image_mapping = self._get_image_mapping(pdf_path, page_range)

        _clean_pdfimages_output(output_dir)
        self._run_pdfimages_extract(pdf_path, output_dir, start_page, end_page, page_range)
        _skip = skip_pages or set()
        fallback_pages: list[int] = []
        page_dimensions, text_rich_pages = _page_dimensions_and_text_pages(
            pdf_path, start_page, end_page
        )

        for i in range(num_pages_to_process):
            current_page = start_page + i
            if current_page in _skip:
                continue
            self._process_extracted_page(
                current_page,
                i,
                output_dir,
                image_mapping,
                page_dimensions,
                text_rich_pages,
                fallback_pages,
                results,
            )

        _cleanup_pdfimages_objects(output_dir)
        if fallback_pages:
            self._render_fallback_pages(
                pdf_path,
                output_dir,
                fallback_pages,
                results,
                start_page,
                page_dimensions,
            )

        return results

    def _run_pdfimages_extract(
        self,
        pdf_path: Path,
        output_dir: Path,
        start_page: int,
        end_page: int,
        page_range: tuple[int, int] | None,
    ) -> None:
        logger.info(f"Extracting images from PDF pages {start_page}-{end_page} using pdfimages...")
        cmd = ["pdfimages"]
        if page_range:
            cmd.extend(["-f", str(start_page), "-l", str(end_page)])
        cmd.extend(["-all", str(pdf_path), str(output_dir / "obj")])

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

    def _process_extracted_page(
        self,
        current_page: int,
        result_index: int,
        output_dir: Path,
        image_mapping: dict[int, list[tuple[int, int, int]]],
        page_dimensions: dict[int, tuple[float, float]],
        text_rich_pages: set[int],
        fallback_pages: list[int],
        results: list[Path | None],
    ) -> None:
        img_entries = image_mapping.get(current_page, [])
        if not img_entries:
            _append_fallback_or_skip_text_page(
                current_page, text_rich_pages, fallback_pages, "no images"
            )
            return

        image_choice = self._best_readable_image(output_dir, img_entries)
        if image_choice is None:
            _append_fallback_or_skip_text_page(
                current_page, text_rich_pages, fallback_pages, "no readable images"
            )
            return

        valid_img_path, best_w, best_h = image_choice
        reason = _small_page_image_reason(page_dimensions.get(current_page), best_w, best_h)
        if reason is not None:
            if current_page in text_rich_pages:
                logger.info(
                    f"Page {current_page}: small image {best_w}x{best_h} "
                    f"({reason}) but has vector text, skipping OCR"
                )
            else:
                fallback_pages.append(current_page)
                logger.info(
                    f"Page {current_page}: largest image {best_w}x{best_h} "
                    f"insufficient ({reason}), will render with pdftoppm"
                )
            return

        dest = output_dir / f"page_{current_page}{valid_img_path.suffix}"
        if not dest.exists():
            valid_img_path.rename(dest)
            results[result_index] = dest

    def _best_readable_image(
        self,
        output_dir: Path,
        img_entries: list[tuple[int, int, int]],
    ) -> tuple[Path, int, int] | None:
        for idx, img_w, img_h in sorted(img_entries, key=lambda e: e[1] * e[2], reverse=True):
            found = self._find_file_for_index(output_dir, idx)
            if found and found.suffix.lower() not in self._UNSUPPORTED_EXTENSIONS:
                return found, img_w, img_h
        return None

    def _render_fallback_pages(
        self,
        pdf_path: Path,
        output_dir: Path,
        pages: list[int],
        results: list[Path | None],
        start_page: int,
        page_dimensions: dict[int, tuple[float, float]],
    ) -> None:
        """Render specific pages via pdftoppm when pdfimages produces unreadable formats.

        Uses pdftoppm to render each page to PNG at the configured DPI.
        Multiple pages are rendered in parallel via threads.
        """
        import os as _os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        preferred_dpi = int(self.dpi or 300)
        max_render_workers = min(len(pages), _os.cpu_count() or 4)

        def _render_one(page_num: int) -> tuple[int, Path | None]:
            width_pts, height_pts = page_dimensions.get(page_num, (0.0, 0.0))
            render_dpi = select_render_dpi_for_page(
                width_pts,
                height_pts,
                preferred_dpi,
                self.max_render_megapixels,
            )
            if render_dpi != preferred_dpi:
                logger.info(
                    f"Page {page_num}: reducing pdftoppm render DPI "
                    f"{preferred_dpi} -> {render_dpi} to stay under "
                    f"{self.max_render_megapixels:.1f} MP"
                )
            prefix = str(output_dir / f"render_{page_num}")
            cmd = [
                "pdftoppm",
                "-f",
                str(page_num),
                "-l",
                str(page_num),
                "-r",
                str(render_dpi),
                "-png",
                "-singlefile",
                str(pdf_path),
                prefix,
            ]
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=PDF_TOOL_TIMEOUT_SECS,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                logger.error(f"pdftoppm fallback failed for page {page_num}: {e.stderr}")
                return page_num, None

            rendered = Path(f"{prefix}.png")
            if rendered.exists():
                dest = output_dir / f"page_{page_num}.png"
                rendered.rename(dest)
                logger.info(f"Page {page_num}: rendered via pdftoppm ({dest.name})")
                return page_num, dest
            logger.warning(f"pdftoppm produced no output for page {page_num}")
            return page_num, None

        with ThreadPoolExecutor(max_workers=max_render_workers) as tp:
            for page_num, dest in (
                fut.result() for fut in as_completed({tp.submit(_render_one, p): p for p in pages})
            ):
                if dest is not None:
                    idx = page_num - start_page
                    if 0 <= idx < len(results):
                        results[idx] = dest
                        self.rendered_pages.add(page_num)

    def _get_image_mapping(
        self,
        pdf_path: Path,
        page_range: tuple[int, int] | None = None,
    ) -> dict[int, list[tuple[int, int, int]]]:
        """Map page numbers to image info using pdfimages -list.

        When page_range is provided, uses -f/-l flags so that image
        indices match the files produced by a corresponding pdfimages -all
        call with the same range.

        Also populates ``self.masked_pages`` — a set of page numbers that
        contain image masks (SMask / soft-mask).  Pages with masks typically
        use DjVu-like foreground/background layer separation where the
        extracted background layer is heavily compressed (~Q10) and
        unusable for OCR on its own.  Such pages should be rendered via
        pdftoppm to get the composited image.

        Returns:
            Dict mapping page_num -> list of (image_index, width, height).
        """
        cmd = ["pdfimages", "-list"]
        if page_range:
            cmd.extend(["-f", str(page_range[0]), "-l", str(page_range[1])])
        cmd.append(str(pdf_path))
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=PDF_TOOL_TIMEOUT_SECS,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return {}

        mapping: dict[int, list[tuple[int, int, int]]] = {}
        self.masked_pages: set[int] = set()
        lines = result.stdout.splitlines()
        start_parsing = False
        for line in lines:
            if line.startswith("---"):
                start_parsing = True
                continue
            if not start_parsing:
                continue

            _add_pdfimages_mapping_line(line, mapping, self.masked_pages)
        return mapping

    def _find_file_for_index(self, output_dir: Path, idx: int) -> Path | None:
        pattern = f"obj-{idx:03d}.*"
        matches = list(output_dir.glob(pattern))
        if not matches:
            pattern = f"obj-{idx:04d}.*"
            matches = list(output_dir.glob(pattern))
        return matches[0] if matches else None
