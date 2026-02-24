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


def has_native_text(pdf_path: Path) -> bool:
    """
    Check if a PDF has native (non-OCR) text content.

    Uses pdftotext to extract text and checks if there's meaningful content.
    This helps distinguish between:
    - Image-only PDFs (scanned documents) - no native text
    - Mixed content PDFs (text + images) - has native text

    Args:
        pdf_path: Path to the PDF file

    Returns:
        True if the PDF has native text content
    """
    try:
        result = subprocess.run(
            ["pdftotext", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return False

        # Check if there's meaningful text (more than just whitespace)
        text = result.stdout.strip()
        # Filter out form feed characters and check for actual content
        text = text.replace("\f", "").strip()
        return len(text) > 10  # Arbitrary threshold for "meaningful" text

    except Exception as e:
        logger.warning(f"Could not check for native text: {e}")
        return False


def get_pages_with_native_text(pdf_path: Path, total_pages: int) -> set[int]:
    """Detect which pages have native (non-OCR) text content.

    Uses pdftotext per-page to identify pages that already contain
    meaningful text, so they can be preserved as-is during OCR.

    Args:
        pdf_path: Path to the PDF file
        total_pages: Total number of pages

    Returns:
        Set of 1-based page numbers that have native text
    """
    pages_with_text: set[int] = set()
    for page_num in range(1, total_pages + 1):
        try:
            result = subprocess.run(
                ["pdftotext", "-f", str(page_num), "-l", str(page_num), str(pdf_path), "-"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                text = result.stdout.replace("\f", "").strip()
                if len(text) > 10:
                    pages_with_text.add(page_num)
        except Exception:
            pass
    return pages_with_text


def extract_image_positions(pdf_path: Path) -> dict[int, list[ImagePosition]]:
    """Extract positions and metadata of all images in a PDF.

    Uses pikepdf's content stream parser to track the current transformation
    matrix (CTM) and find image draw commands (Do). Handles all PDF generators
    including Chrome/Skia, which use complex multi-step CTM operations.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary mapping page number (1-indexed) to list of ImagePosition
    """
    positions: dict[int, list[ImagePosition]] = {}

    with pikepdf.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_positions = []

            # Get XObjects (images) metadata
            xobjects = {}
            if "/Resources" in page and "/XObject" in page.Resources:
                for name, xobj in page.Resources.XObject.items():
                    if xobj.get("/Subtype") == "/Image":
                        xobjects[str(name)] = {
                            "width": int(xobj.get("/Width", 0)),
                            "height": int(xobj.get("/Height", 0)),
                        }

            if not xobjects:
                continue

            # Parse content stream with pikepdf (handles all PDF variants)
            try:
                commands = pikepdf.parse_content_stream(page)
            except Exception as e:
                logger.debug(f"Failed to parse content stream for page {page_num}: {e}")
                continue

            # Track graphics state with a CTM stack
            # CTM = [a, b, c, d, e, f] (2D affine transform)
            identity = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            ctm_stack: list[list[float]] = []
            ctm = list(identity)

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
                        # CTM maps the unit square [0,0]-[1,1] to the image area.
                        # Width = length of vector (a, b), Height = length of (c, d)
                        a, b, c, d, e, f = ctm
                        width = (a * a + b * b) ** 0.5
                        height = (c * c + d * d) ** 0.5
                        x = e
                        # When d < 0 (y-axis flip, common in Chrome/Skia PDFs),
                        # f is the TOP of the image; adjust to get the BOTTOM
                        # (PDF convention: y=0 at bottom of page).
                        y = f - height if d < 0 else f

                        pos = ImagePosition(
                            name=img_name,
                            page_num=page_num,
                            x=x,
                            y=y,
                            width=width,
                            height=height,
                        )
                        page_positions.append(pos)
                        logger.debug(
                            f"Found image {img_name} on page {page_num}: "
                            f"pos=({x:.1f}, {y:.1f}), size={width:.1f}x{height:.1f}"
                        )

            if page_positions:
                positions[page_num] = page_positions

    return positions


class PDFImageExtractor:
    """Native PDF image extraction without re-encoding.

    Uses pdfimages -all to extract images directly from PDFs.
    This is more efficient than rendering pages with pdftoppm because:
    - No re-encoding of images (preserves original quality)
    - No upscaling of low-DPI content
    - Much faster and uses less memory
    """

    def __init__(self, dpi: int | None = None):
        # DPI parameter kept for API compatibility but not used for extraction
        # (pdfimages extracts at native resolution)
        self.dpi = dpi

    def extract(
        self,
        pdf_path: Path,
        output_dir: Path,
        page_range: tuple[int, int] | None = None,
    ) -> list[Path | None]:
        """Extract native images from PDF ensuring correct page mapping.

        When page_range is provided, only images from those pages are
        extracted (using pdfimages -f/-l flags), significantly reducing
        disk usage for large documents.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

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
        for i in range(num_pages_to_process):
            current_page = start_page + i
            img_indices = image_mapping.get(current_page, [])

            if not img_indices:
                continue

            valid_img_path: Path | None = None
            for idx in img_indices:
                found = self._find_file_for_index(output_dir, idx)
                if found:
                    valid_img_path = found
                    break

            if valid_img_path:
                ext = valid_img_path.suffix
                dest = output_dir / f"page_{current_page}{ext}"
                if not dest.exists():
                    valid_img_path.rename(dest)
                    results[i] = dest

        # Cleanup unused extracted files
        for f in output_dir.glob("obj-*"):
            try:
                f.unlink()
            except OSError:
                pass

        return results

    def _get_image_mapping(
        self,
        pdf_path: Path,
        page_range: tuple[int, int] | None = None,
    ) -> dict[int, list[int]]:
        """Map page numbers to image indices using pdfimages -list.

        When page_range is provided, uses -f/-l flags so that image
        indices match the files produced by a corresponding pdfimages -all
        call with the same range.
        """
        cmd = ["pdfimages", "-list"]
        if page_range:
            cmd.extend(["-f", str(page_range[0]), "-l", str(page_range[1])])
        cmd.append(str(pdf_path))
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            return {}

        mapping: dict[int, list[int]] = {}
        lines = result.stdout.splitlines()
        start_parsing = False
        for line in lines:
            if line.startswith("---"):
                start_parsing = True
                continue
            if not start_parsing:
                continue

            parts = line.split()
            if len(parts) >= 4:
                try:
                    p_num = int(parts[0])
                    i_idx = int(parts[1])
                    type_str = parts[2]
                    if type_str == "image":
                        if p_num not in mapping:
                            mapping[p_num] = []
                        mapping[p_num].append(i_idx)
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
