import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pikepdf
from PIL import Image

from bigocrpdf.services.rapidocr_service.config import OCRResult
from bigocrpdf.utils.logger import logger


@dataclass
class ImagePosition:
    """Position and metadata of an image within a PDF page."""

    name: str  # Resource name (e.g. /Im1)
    page_num: int  # 1-based page number
    x: float  # X coordinate (PDF points)
    y: float  # Y coordinate (PDF points)
    width: float  # Display width (PDF points)
    height: float  # Display height (PDF points)


def _page_has_visible_text(page: pikepdf.Page) -> bool:
    """Check if a page has visible (non-invisible-OCR) text.

    Returns True if the page contains any BT/ET text block that does NOT
    use render mode 3 (invisible).  Pages with only ``3 Tr`` text are
    treated as image-only since the text is just a previous OCR layer.
    """
    import re

    contents = page.get("/Contents")
    if not contents:
        return False

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
            return False

    for raw in raw_parts:
        text = raw.decode("latin-1", errors="ignore")
        # Find all BT ... ET blocks
        for m in re.finditer(r"BT\b(.*?)ET\b", text, re.DOTALL):
            block = m.group(1)
            # If it has a Tj/TJ operator but does NOT set invisible mode (3 Tr)
            if re.search(r"\bTj\b|\bTJ\b", block) and not re.search(r"\b3\s+Tr\b", block):
                return True
    return False


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
    """Check if a PDF page has invisible OCR text (render mode 3).

    Invisible text with render mode 3 (Tr=3) is the standard way OCR engines
    add searchable text layers over images. This detects such layers to avoid
    duplicate OCR when replace_existing_ocr is False.
    """
    import re

    contents = page.get("/Contents")
    if not contents:
        return False

    streams = []
    if isinstance(contents, pikepdf.Array):
        for s in contents:
            try:
                streams.append(s.read_bytes())
            except Exception:
                pass
    else:
        try:
            streams.append(contents.read_bytes())
        except Exception:
            return False

    for raw in streams:
        text = raw.decode("latin-1", errors="ignore")
        if re.search(r"\b3\s+Tr\b", text):
            return True
    return False


def _multiply_ctm(current: list[float], m: list[float]) -> list[float]:
    """Multiply two 3x3 affine matrices (stored as 6 elements)."""
    a1, b1, c1, d1, e1, f1 = current
    a2, b2, c2, d2, e2, f2 = m
    return [
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2,
        e1 * b2 + f1 * d2 + f2,
    ]


def _get_page_xobjects(page) -> dict[str, dict]:
    """Return {name: {width, height}} for image XObjects on a page."""
    xobjects = {}
    if "/Resources" in page and "/XObject" in page.Resources:
        for name, xobj in page.Resources.XObject.items():
            if xobj.get("/Subtype") == "/Image":
                xobjects[str(name)] = {
                    "width": int(xobj.get("/Width", 0)),
                    "height": int(xobj.get("/Height", 0)),
                }
    return xobjects


def _parse_page_images(commands, xobjects, page_num) -> list[ImagePosition]:
    """Parse content stream commands and return ImagePosition list."""
    identity = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    ctm_stack: list[list[float]] = []
    ctm = list(identity)
    positions: list[ImagePosition] = []

    for operands, operator in commands:
        op = str(operator)
        if op == "q":
            ctm_stack.append(list(ctm))
        elif op == "Q":
            if ctm_stack:
                ctm = ctm_stack.pop()
        elif op == "cm" and len(operands) == 6:
            try:
                m = [float(x) for x in operands]
                ctm = _multiply_ctm(m, ctm)
            except (ValueError, TypeError):
                pass
        elif op == "Do" and len(operands) == 1:
            img_name = str(operands[0])
            if img_name in xobjects:
                a, b, c, d, e, f = ctm
                width = (a * a + b * b) ** 0.5
                height = (c * c + d * d) ** 0.5
                x = e
                y = f - height if d < 0 else f
                positions.append(
                    ImagePosition(
                        name=img_name,
                        page_num=page_num,
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                    )
                )
                logger.debug(
                    f"Found image {img_name} on page {page_num}: "
                    f"pos=({x:.1f}, {y:.1f}), size={width:.1f}x{height:.1f}"
                )

    return positions


def extract_image_positions(pdf_path: Path) -> dict[int, list[ImagePosition]]:
    """Extract positions and metadata of all images in a PDF."""
    positions: dict[int, list[ImagePosition]] = {}

    with pikepdf.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            xobjects = _get_page_xobjects(page)
            if not xobjects:
                continue

            try:
                commands = pikepdf.parse_content_stream(page)
            except Exception as e:
                logger.debug(f"Failed to parse content stream for page {page_num}: {e}")
                continue

            page_positions = _parse_page_images(commands, xobjects, page_num)
            if page_positions:
                positions[page_num] = page_positions

    return positions


@dataclass
class PdfImageInfo:
    """Metadata for a single image entry from ``pdfimages -list``."""

    idx: int
    img_type: str
    width: int
    height: int
    comp_size: int  # compressed size in bytes (from pdfimages -list "size" column)


def _parse_size_field(s: str) -> int:
    """Parse a size field like '249K', '5411B', '1.2M' into bytes."""
    s = s.strip()
    if s.endswith("K"):
        return int(float(s[:-1]) * 1024)
    if s.endswith("M"):
        return int(float(s[:-1]) * 1024 * 1024)
    if s.endswith("G"):
        return int(float(s[:-1]) * 1024 * 1024 * 1024)
    if s.endswith("B"):
        return int(float(s[:-1]))
    try:
        return int(float(s))
    except ValueError:
        return 0


def parse_pdfimages_list(
    pdf_path: Path,
) -> tuple[dict[int, list[PdfImageInfo]], set[int]]:
    """Parse ``pdfimages -list`` and return per-page image info.

    Returns:
        A tuple of (mapping, masked_pages) where *mapping* is
        ``{page_num: [PdfImageInfo, …]}`` (masks/smasks excluded)
        and *masked_pages* is the set of pages that have JBIG2 mask
        entries (DjVu-like FG/BG layers).
    """
    try:
        result = subprocess.run(
            ["pdfimages", "-list", str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return {}, set()
    except Exception as exc:
        logger.warning(f"pdfimages -list failed: {exc}")
        return {}, set()

    mapping: dict[int, list[PdfImageInfo]] = {}
    masked_pages: set[int] = set()
    for line in result.stdout.splitlines()[2:]:  # skip header lines
        parts = line.split()
        if len(parts) < 14:
            continue
        try:
            page_num = int(parts[0])
            img_idx = int(parts[1])
            img_type = parts[2]
            width = int(parts[3])
            height = int(parts[4])
            # "size" is at position 14 (0-indexed) in the standard layout:
            # page num type width height color comp bpc enc interp object ID x-ppi y-ppi size ratio
            comp_size = _parse_size_field(parts[14])
        except (ValueError, IndexError):
            continue
        if img_type in ("mask", "smask"):
            masked_pages.add(page_num)
            continue
        info = PdfImageInfo(
            idx=img_idx,
            img_type=img_type,
            width=width,
            height=height,
            comp_size=comp_size,
        )
        mapping.setdefault(page_num, []).append(info)
    return mapping, masked_pages


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

    def __init__(self, dpi: int | None = None):
        # DPI parameter kept for API compatibility but not used for extraction
        # (pdfimages extracts at native resolution)
        self.dpi = dpi
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

        # 1. Determine Total Pages and Range
        with pikepdf.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

        start_page = 1
        end_page = total_pages
        if page_range:
            start_page, end_page = page_range
            start_page = max(1, start_page)
            end_page = min(total_pages, end_page)

        num_pages_to_process = end_page - start_page + 1
        results: list[Path | None] = [None] * num_pages_to_process

        # 2. Get Image Mapping (Page -> [ImageIndex, ...])
        # Use same page range so indices match extracted files
        image_mapping = self._get_image_mapping(pdf_path, page_range)

        # 3. Clean and Extract
        for f in output_dir.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass

        logger.info(f"Extracting images from PDF pages {start_page}-{end_page} using pdfimages...")
        cmd = ["pdfimages"]
        if page_range:
            cmd.extend(["-f", str(start_page), "-l", str(end_page)])
        cmd.extend(["-all", str(pdf_path), str(output_dir / "obj")])

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"pdfimages failed: {e.stderr}")
            raise RuntimeError(f"Failed to extract images: {e}") from e

        # 4. Process extracted files based on mapping
        _skip = skip_pages or set()
        fallback_pages: list[int] = []

        # Get page dimensions and text content info for decision making
        page_dimensions: dict[int, tuple[float, float]] = {}
        text_rich_pages: set[int] = set()
        with pikepdf.open(pdf_path) as pdf:
            for pn, page in enumerate(pdf.pages, 1):
                if start_page <= pn <= end_page:
                    if hasattr(page, "mediabox") and page.mediabox:
                        pw = float(page.mediabox[2]) - float(page.mediabox[0])
                        ph = float(page.mediabox[3]) - float(page.mediabox[1])
                        page_dimensions[pn] = (pw, ph)
                    # Detect pages with substantial vector text content.
                    # Such pages should NOT be converted to bitmap via pdftoppm
                    # because that destroys text quality and searchability.
                    fonts = page.get("/Resources", {}).get("/Font", {})
                    if len(fonts) >= 2:
                        cs = page.get("/Contents")
                        if cs is not None:
                            try:
                                if isinstance(cs, pikepdf.Array):
                                    raw = b"".join(
                                        bytes(pdf.get_object(ref).read_bytes()) for ref in cs
                                    )
                                else:
                                    raw = bytes(cs.read_bytes())
                                text_ops = raw.count(b"Tj") + raw.count(b"TJ")
                                if text_ops >= 20:
                                    text_rich_pages.add(pn)
                            except Exception:
                                pass  # content parsing failed; treat as non-text

        for i in range(num_pages_to_process):
            current_page = start_page + i

            # Skip excluded pages entirely (no extraction, no rendering)
            if current_page in _skip:
                continue

            img_entries = image_mapping.get(current_page, [])

            if not img_entries:
                # No images at all — vector-only page.
                # If it has substantial text content, skip it entirely
                # (don't render as bitmap; preserves vector text quality).
                if current_page in text_rich_pages:
                    logger.info(f"Page {current_page}: no images but has vector text, skipping OCR")
                    continue
                fallback_pages.append(current_page)
                logger.info(f"Page {current_page}: no images found, will render with pdftoppm")
                continue

            # Sort by pixel area (largest first) to pick the best image
            img_entries_sorted = sorted(img_entries, key=lambda e: e[1] * e[2], reverse=True)

            valid_img_path: Path | None = None
            best_w, best_h = 0, 0
            for idx, img_w, img_h in img_entries_sorted:
                found = self._find_file_for_index(output_dir, idx)
                if found:
                    ext = found.suffix.lower()
                    if ext in self._UNSUPPORTED_EXTENSIONS:
                        continue  # skip unreadable, try next
                    valid_img_path = found
                    best_w, best_h = img_w, img_h
                    break

            if valid_img_path is None:
                # All images were unsupported formats or not found
                if current_page in text_rich_pages:
                    logger.info(
                        f"Page {current_page}: no readable images but has vector text, skipping OCR"
                    )
                    continue
                fallback_pages.append(current_page)
                logger.info(
                    f"Page {current_page}: no readable images found, will render with pdftoppm"
                )
                continue

            # Check if the best image is too small relative to the page.
            # PDF page dimensions are in points (72 pt/inch). At 300 DPI,
            # a full A4 page is ~2480x3508 px.  We compare the image's
            # pixel area and dimensions to the expected at 150 DPI (generous
            # minimum).  Fallback to rendering if:
            #   - area covers less than 15% of expected page area, OR
            #   - either dimension is less than 45% of expected (catches
            #     narrow strips like barcodes, tiled page fragments, headers)
            page_dim = page_dimensions.get(current_page)
            if page_dim:
                expected_dpi = 150  # generous minimum
                expected_w = page_dim[0] / 72.0 * expected_dpi
                expected_h = page_dim[1] / 72.0 * expected_dpi
                expected_area = expected_w * expected_h
                img_area = best_w * best_h
                coverage = img_area / expected_area if expected_area > 0 else 1.0
                w_ratio = best_w / expected_w if expected_w > 0 else 1.0
                h_ratio = best_h / expected_h if expected_h > 0 else 1.0
                if coverage < 0.15 or w_ratio < 0.45 or h_ratio < 0.45:
                    reason = (
                        f"area {coverage:.0%}"
                        if coverage < 0.15
                        else f"dimensions {best_w}x{best_h} ({w_ratio:.0%}w, {h_ratio:.0%}h)"
                    )
                    if current_page in text_rich_pages:
                        logger.info(
                            f"Page {current_page}: small image {best_w}x{best_h} "
                            f"({reason}) but has vector text, skipping OCR"
                        )
                        continue
                    fallback_pages.append(current_page)
                    logger.info(
                        f"Page {current_page}: largest image {best_w}x{best_h} "
                        f"insufficient ({reason}), will render with pdftoppm"
                    )
                    continue

            dest = output_dir / f"page_{current_page}{valid_img_path.suffix}"
            if not dest.exists():
                valid_img_path.rename(dest)
                results[i] = dest

        # Cleanup unused extracted files (including unreadable ones)
        for f in output_dir.glob("obj-*"):
            try:
                f.unlink()
            except OSError:
                pass

        # 5. Fallback: render pages with unsupported image formats via pdftoppm
        if fallback_pages:
            self._render_fallback_pages(pdf_path, output_dir, fallback_pages, results, start_page)

        return results

    def _render_fallback_pages(
        self,
        pdf_path: Path,
        output_dir: Path,
        pages: list[int],
        results: list[Path | None],
        start_page: int,
    ) -> None:
        """Render specific pages via pdftoppm when pdfimages produces unreadable formats.

        Uses pdftoppm to render each page to PNG at the configured DPI.
        Multiple pages are rendered in parallel via threads.
        """
        import os as _os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        dpi = str(self.dpi or 300)
        max_render_workers = min(len(pages), _os.cpu_count() or 4)

        def _render_one(page_num: int) -> tuple[int, Path | None]:
            prefix = str(output_dir / f"render_{page_num}")
            cmd = [
                "pdftoppm",
                "-f",
                str(page_num),
                "-l",
                str(page_num),
                "-r",
                dpi,
                "-png",
                "-singlefile",
                str(pdf_path),
                prefix,
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
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
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
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

            parts = line.split()
            if len(parts) >= 5:
                try:
                    p_num = int(parts[0])
                    i_idx = int(parts[1])
                    type_str = parts[2]
                    img_w = int(parts[3])
                    img_h = int(parts[4])
                    if type_str == "image":
                        if p_num not in mapping:
                            mapping[p_num] = []
                        mapping[p_num].append((i_idx, img_w, img_h))
                    elif type_str == "mask":
                        self.masked_pages.add(p_num)
                except ValueError:
                    continue
        return mapping

    def _find_file_for_index(self, output_dir: Path, idx: int) -> Path | None:
        pattern = f"obj-{idx:03d}.*"
        matches = list(output_dir.glob(pattern))
        if not matches:
            pattern = f"obj-{idx:04d}.*"
            matches = list(output_dir.glob(pattern))
        return matches[0] if matches else None


def transform_ocr_coords_for_rotation(
    ocr_results: list[OCRResult],
    ocr_img_size: tuple[int, int],
    pdf_page_size: tuple[float, float],
    rotation: int,
) -> list[OCRResult]:
    """
    Transform OCR coordinates to match PDF page with rotation.

    OCR is performed on the corrected (upright) image.
    PDF pages may have a /Rotate attribute that rotates the display.

    Args:
        ocr_results: OCR results with coordinates from upright image
        ocr_img_size: (width, height) of image used for OCR
        pdf_page_size: (width, height) from PDF MediaBox
        rotation: PDF page /Rotate value (0, 90, 180, 270)

    Returns:
        Transformed OCR results matching PDF coordinate system
    """
    # 2026-02-05: Fix bug in rotation - was swapping W/H incorrectly
    if rotation == 0:
        # No rotation - just scale coordinates to match PDF dimensions
        ocr_w, ocr_h = ocr_img_size
        pdf_w, pdf_h = pdf_page_size
        scale_x = pdf_w / ocr_w
        scale_y = pdf_h / ocr_h

        transformed = []
        for result in ocr_results:
            new_box = [[p[0] * scale_x, p[1] * scale_y] for p in result.box]
            transformed.append(OCRResult(result.text, new_box, result.confidence))
        return transformed

    elif rotation == 90:
        # PDF page is rotated 90° clockwise
        ocr_w, ocr_h = ocr_img_size
        pdf_w, pdf_h = pdf_page_size

        # Scale factors:
        # OCR Width maps to PDF Height (physical)
        # OCR Height maps to PDF Width (physical)
        scale_x = pdf_w / ocr_h
        scale_y = pdf_h / ocr_w

        transformed = []
        for result in ocr_results:
            new_box = []
            for p in result.box:
                # Transform: portrait (x, y) -> landscape with 90° rotation
                # New X (Visual) comes from Old Y
                # New Y (Visual) comes from Old Width - Old X
                new_x = p[1] * scale_x
                new_y = (ocr_w - p[0]) * scale_y
                new_box.append([new_x, new_y])
            transformed.append(OCRResult(result.text, new_box, result.confidence))
        return transformed

    elif rotation == 180:
        # PDF page is rotated 180°
        ocr_w, ocr_h = ocr_img_size
        pdf_w, pdf_h = pdf_page_size
        scale_x = pdf_w / ocr_w
        scale_y = pdf_h / ocr_h

        transformed = []
        for result in ocr_results:
            new_box = []
            for p in result.box:
                new_x = (ocr_w - p[0]) * scale_x
                new_y = (ocr_h - p[1]) * scale_y
                new_box.append([new_x, new_y])
            transformed.append(OCRResult(result.text, new_box, result.confidence))
        return transformed

    elif rotation == 270:
        # PDF page is rotated 270° clockwise (or 90° counter-clockwise)
        ocr_w, ocr_h = ocr_img_size
        pdf_w, pdf_h = pdf_page_size

        # For 270° rotation (90° counter-clockwise):
        # OCR Height maps to PDF Width, OCR Width maps to PDF Height
        scale_x = pdf_w / ocr_h
        scale_y = pdf_h / ocr_w

        transformed = []
        for result in ocr_results:
            new_box = []
            for p in result.box:
                new_x = (ocr_h - p[1]) * scale_x
                new_y = p[0] * scale_y
                new_box.append([new_x, new_y])
            transformed.append(OCRResult(result.text, new_box, result.confidence))
        return transformed

    else:
        logger.warning(f"Unsupported rotation: {rotation}°, using no transformation")
        return ocr_results


def extract_content_streams(
    contents: Any,
    target_pdf: pikepdf.Pdf,
    copy_foreign: bool = True,
) -> list:
    """Extract content streams from PDF contents object."""
    streams = []
    if isinstance(contents, pikepdf.Array):
        for stream in contents:
            if copy_foreign:
                streams.append(target_pdf.copy_foreign(stream))
            else:
                streams.append(stream)
    else:
        if copy_foreign:
            streams.append(target_pdf.copy_foreign(contents))
        else:
            streams.append(contents)
    return streams


def merge_page_fonts(
    orig_page: pikepdf.Page,
    text_resources: pikepdf.Dictionary,
    original_pdf: pikepdf.Pdf,
) -> None:
    """Merge fonts from text layer resources into original page."""
    if "/Font" not in text_resources:
        return

    # Ensure original page has resources
    if "/Resources" not in orig_page:
        orig_page["/Resources"] = pikepdf.Dictionary()

    if "/Font" not in orig_page["/Resources"]:
        orig_page["/Resources"]["/Font"] = pikepdf.Dictionary()

    # Copy each font if not already present
    for font_name, font_obj in text_resources["/Font"].items():
        try:
            if font_name not in orig_page["/Resources"]["/Font"]:
                orig_page["/Resources"]["/Font"][font_name] = original_pdf.copy_foreign(font_obj)
        except Exception as e:
            logger.debug(f"Could not copy font {font_name}: {e}")


def load_image_with_exif_rotation(img_path: Path) -> np.ndarray:
    """Load image and apply EXIF orientation correction."""
    from PIL import ImageOps

    try:
        pil_img = Image.open(img_path)
        pil_img = ImageOps.exif_transpose(pil_img)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        img_rgb = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr
    except Exception:
        return cv2.imread(str(img_path))


def get_page_image_encodings(
    pdf_path: Path,
    page_range: tuple[int, int] | None = None,
) -> dict[int, str]:
    """Detect the image encoding used on each page of a PDF.

    Parses ``pdfimages -list`` output to determine what compression
    each page uses (jbig2, ccitt, jpeg, flate, jpx, etc.).
    When a page has multiple images, the first image's encoding is used.

    Args:
        pdf_path: Path to the PDF file.
        page_range: Optional (first, last) 1-based page range.

    Returns:
        Mapping of 1-based page number to encoding string
        (e.g. ``{1: "jbig2", 2: "jbig2", 3: "jpeg"}``).
    """
    cmd = ["pdfimages", "-list"]
    if page_range:
        cmd.extend(["-f", str(page_range[0]), "-l", str(page_range[1])])
    cmd.append(str(pdf_path))

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}

    encodings: dict[int, str] = {}
    parsing = False
    for line in result.stdout.splitlines():
        if line.startswith("---"):
            parsing = True
            continue
        if not parsing:
            continue
        parts = line.split()
        # Columns: page num type width height color comp bpc enc ...
        if len(parts) >= 9:
            try:
                page_num = int(parts[0])
                img_type = parts[2]
                enc = parts[8].lower()
                if img_type == "image" and page_num not in encodings:
                    encodings[page_num] = enc
            except (ValueError, IndexError):
                continue
    return encodings
