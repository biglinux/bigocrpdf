"""PDF page geometry, stream and image-loading helpers."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pikepdf
from PIL import Image

from bigocrpdf.constants import PDF_TOOL_TIMEOUT_SECS
from bigocrpdf.utils.logger import logger


def render_pdf_page_to_ppm(
    pdf_path: Path | str,
    page_num: int,
    dpi: int = 300,
    *,
    output_dir: Path | str | None = None,
) -> str | None:
    """Render one composited PDF page to an owned temporary PPM."""
    fd, prefix = tempfile.mkstemp(
        suffix="",
        prefix=f"bigocr_render_{page_num}_",
        dir=output_dir,
    )
    os.close(fd)
    os.unlink(prefix)
    rendered = Path(f"{prefix}.ppm")
    command = [
        "pdftoppm",
        "-f",
        str(page_num),
        "-l",
        str(page_num),
        "-r",
        str(dpi),
        "-singlefile",
        str(pdf_path),
        prefix,
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=PDF_TOOL_TIMEOUT_SECS,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        rendered.unlink(missing_ok=True)
        logger.warning("pdftoppm render failed for page %d: %s", page_num, error)
        return None
    return str(rendered) if rendered.is_file() else None


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
) -> dict[str, str]:
    """Merge fonts from text layer resources into original page.

    Returns the names that had to change, old name to new.

    A name that already exists on the page belongs to the page, not to us.
    Keeping it used to mean our font was silently dropped and our text read
    through a stranger's encoding: our layer writes subset code 1 for whatever
    character it happened to meet first, so a page carrying an unrelated
    ``/F2+0`` decoded every accent as some other accent -- ``Jürgen`` came out
    ``Jçrgen``. Names like ``/F1`` are as generic as they look, and re-running
    OCR on a file we produced hits this every time.
    """
    if "/Font" not in text_resources:
        return {}

    # Ensure original page has resources
    if "/Resources" not in orig_page:
        orig_page["/Resources"] = pikepdf.Dictionary()

    if "/Font" not in orig_page["/Resources"]:
        orig_page["/Resources"]["/Font"] = pikepdf.Dictionary()

    page_fonts = orig_page["/Resources"]["/Font"]
    renames: dict[str, str] = {}
    for font_name, font_obj in text_resources["/Font"].items():
        target = (
            font_name if font_name not in page_fonts else _free_font_name(page_fonts, font_name)
        )
        try:
            page_fonts[target] = original_pdf.copy_foreign(font_obj)
        except Exception as e:
            # Recorded only once the font is installed. A rename pointing at a
            # font that failed to copy would send the stream to a resource
            # that is not there, which is worse than the wrong font.
            logger.debug(f"Could not copy font {font_name}: {e}")
            continue
        if target != font_name:
            renames[font_name] = target
    return renames


def _free_font_name(page_fonts: pikepdf.Object, wanted: str) -> str:
    """Return a font resource name the page is not already using."""
    stem = wanted.lstrip("/")
    suffix = 1
    while f"/BigOcr{stem}_{suffix}" in page_fonts:
        suffix += 1
    return f"/BigOcr{stem}_{suffix}"


def load_image_with_exif_rotation(img_path: Path) -> np.ndarray | None:
    """Load image and apply EXIF orientation correction."""
    from PIL import ImageOps

    try:
        with Image.open(img_path) as source:
            with ImageOps.exif_transpose(source) as oriented:
                with oriented.convert("RGB") as rgb_image:
                    img_rgb = np.array(rgb_image)
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
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
