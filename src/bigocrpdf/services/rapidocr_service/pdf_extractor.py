import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pikepdf
from PIL import Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from bigocrpdf.constants import FONT_SIZE_SCALE_FACTOR
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


def extract_image_positions(pdf_path: Path) -> dict[int, list[ImagePosition]]:
    """
    Extract positions and metadata of all images in a PDF.

    Parses the PDF content stream to find image placement commands (cm/Do).
    Returns a dictionary mapping page numbers to lists of ImagePosition.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary mapping page number (1-indexed) to list of ImagePosition
    """
    import re

    import pikepdf

    positions: dict[int, list[ImagePosition]] = {}

    with pikepdf.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_positions = []

            # 2026-01-20 fix: Handle rotated pages by checking /Rotate
            # But here we are extracting raw placement coordinates.
            # The coordinates in 'cm' are relative to the page's User Space.
            # We store them as is, and adjust later if needed.

            mediabox = page.mediabox
            _ = float(mediabox[3]) - float(mediabox[1])  # page_height (reserved)

            # Get XObjects (images) metadata
            xobjects = {}
            if "/Resources" in page and "/XObject" in page.Resources:
                for name, xobj in page.Resources.XObject.items():
                    if xobj.get("/Subtype") == "/Image":
                        xobjects[str(name)] = {
                            "width": int(xobj.get("/Width", 0)),
                            "height": int(xobj.get("/Height", 0)),
                        }

            # Parse content stream to find image placements
            contents = page.get("/Contents")
            if not contents:
                continue

            # Handle array of content streams
            streams = []
            if isinstance(contents, pikepdf.Array):
                for item in contents:
                    streams.append(bytes(item.get_stream_buffer()))
            else:
                streams.append(bytes(contents.get_stream_buffer()))

            content_text = b"".join(streams).decode("latin-1", errors="replace")

            # Pattern to find transformation matrix + image draw commands
            # Format: [a 0 0 d tx ty] cm ... /ImageName Do
            # where a=width, d=height, tx=x, ty=y
            pattern = (
                r"q\s+([0-9.]+)\s+0\s+0\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+cm\s*/(\w+)\s+Do"
            )
            for match in re.finditer(pattern, content_text):
                width = float(match.group(1))
                height = float(match.group(2))
                x = float(match.group(3))
                y = float(match.group(4))
                img_name = "/" + match.group(5)

                # Get native dimensions from XObject
                native_w = native_h = 0
                if img_name in xobjects:
                    native_w = xobjects[img_name]["width"]
                    native_h = xobjects[img_name]["height"]

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
                    f"pos=({x:.1f}, {y:.1f}), size={width:.1f}x{height:.1f}, "
                    f"native={native_w}x{native_h}"
                )

            if page_positions:
                positions[page_num] = page_positions

    return positions


class PDFImageExtractor:
    """Native PDF image extraction without re-encoding."""

    def __init__(self, dpi: int = None):
        # DPI parameter kept for API compatibility but not used
        self.dpi = dpi

    def render_pages(
        self,
        pdf_path: Path,
        output_dir: Path,
        page_range: tuple[int, int] | None = None,
    ) -> list[Path | None]:
        """Render full PDF pages as images using pdftoppm.

        Unlike extract(), which only pulls embedded image objects,
        this renders the complete page (text + graphics + images)
        at the configured DPI.  Use this when replacing existing
        OCR on mixed-content PDFs.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Clean output directory
        for f in output_dir.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass

        with pikepdf.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

        start_page = 1
        end_page = total_pages
        if page_range:
            start_page, end_page = page_range
            start_page = max(1, start_page)
            end_page = min(total_pages, end_page)

        num_pages = end_page - start_page + 1
        dpi = self.dpi or 300

        prefix = str(output_dir / "page")
        cmd = [
            "pdftoppm",
            "-r",
            str(dpi),
            "-jpeg",
            "-f",
            str(start_page),
            "-l",
            str(end_page),
            str(pdf_path),
            prefix,
        ]

        logger.info(f"Rendering pages {start_page}-{end_page} at {dpi} DPI using pdftoppm...")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"pdftoppm failed: {e.stderr}")
            raise RuntimeError(f"Failed to render pages: {e}") from e

        # Collect rendered files and rename to page_N.jpg
        results: list[Path | None] = [None] * num_pages
        for i in range(num_pages):
            page_num = start_page + i
            # pdftoppm names files like: page-1.jpg, page-01.jpg, page-001.jpg
            candidates = list(output_dir.glob(f"page-{page_num}.*"))
            if not candidates:
                # Try zero-padded variants
                for width in (2, 3, 4, 6):
                    candidates = list(output_dir.glob(f"page-{page_num:0>{width}}.*"))
                    if candidates:
                        break
            if candidates:
                src = candidates[0]
                dest = output_dir / f"page_{page_num}{src.suffix}"
                if src != dest:
                    src.rename(dest)
                results[i] = dest

        return results

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
        results = [None] * num_pages_to_process

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

            valid_img_path = None
            for idx in img_indices:
                f = self._find_file_for_index(output_dir, idx)
                if f:
                    valid_img_path = f
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

        mapping = {}
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


class TextLayerRenderer:
    """Renders invisible text layer on PDF for searchability."""

    def __init__(self, dpi: int = 300):
        self.dpi = dpi

    def render(
        self,
        canvas_obj: canvas.Canvas,
        ocr_results: list[OCRResult],
        img_size: tuple[int, int],
    ) -> int:
        """Render invisible text layer. Returns number of text regions added."""
        img_w, img_h = img_size
        count = 0

        # Sort results (top-to-bottom, left-to-right) simple approach
        # A full reading-order sort is better but complex.
        # Ideally import `_sort_for_reading_order` from backend if it wasn't tightly coupled relative to self
        # For now, simplistic sort:
        sorted_results = sorted(
            ocr_results, key=lambda r: (min(p[1] for p in r.box), min(p[0] for p in r.box))
        )

        for result in sorted_results:
            if not result.text.strip():
                continue
            try:
                self._render_text_region(canvas_obj, result, img_h)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to render text region: {e}")
        return count

    def _sort_for_reading_order(self, results: list[OCRResult], *args) -> list[OCRResult]:
        """Sort OCR results in reading order (top-to-bottom, left-to-right)."""
        # Sort by vertical position (Top edge) then horizontal (Left edge)
        # Using min(y) for top edge and min(x) for left edge
        # Note: box is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        return sorted(results, key=lambda r: (min(p[1] for p in r.box), min(p[0] for p in r.box)))

    def _render_text_region(
        self,
        canvas_obj: canvas.Canvas,
        result: OCRResult,
        img_h: int,
    ):
        box = result.box
        text = result.text

        width = np.linalg.norm(np.array(box[1]) - np.array(box[0]))
        height = np.linalg.norm(np.array(box[3]) - np.array(box[0]))

        dx = box[1][0] - box[0][0]
        dy = box[1][1] - box[0][1]
        angle = np.degrees(np.arctan2(dy, dx))

        if width <= 0 or height <= 0:
            return

        font_name = "Helvetica"
        font_size = height * FONT_SIZE_SCALE_FACTOR

        try:
            text_width = pdfmetrics.stringWidth(text, font_name, font_size)
        except Exception:
            text_width = len(text) * font_size * 0.5

        if text_width > 0:
            h_scale = (width / text_width) * 100
        else:
            h_scale = 100

        h_scale = max(50, min(200, h_scale))

        left_x = min(p[0] for p in box)
        top_y_img = min(p[1] for p in box)  # Image coords (0,0 top-left)

        # PDF Coords (0,0 bottom-left)
        pdf_x = left_x
        pdf_y = img_h - top_y_img - height

        canvas_obj.saveState()

        if abs(angle) > 1:
            center_x = np.mean([p[0] for p in box])
            center_y = img_h - np.mean([p[1] for p in box])
            canvas_obj.translate(center_x, center_y)
            canvas_obj.rotate(-angle)
            text_x = -width / 2
            text_y = -height / 2 + (height * 0.15)
        else:
            text_x = pdf_x
            text_y = pdf_y + (height * 0.15)

        text_obj = canvas_obj.beginText()
        text_obj.setTextRenderMode(3)  # Invisible
        text_obj.setFont(font_name, font_size)
        text_obj.setHorizScale(h_scale)
        text_obj.setTextOrigin(text_x, text_y)
        text_obj.textOut(text)
        canvas_obj.drawText(text_obj)
        canvas_obj.restoreState()


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
    source_pdf: pikepdf.Pdf,
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
