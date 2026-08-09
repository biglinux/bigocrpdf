"""
BigOcrPdf - PDF Utilities Module

This module provides shared utility functions for PDF file operations.
Centralizes PDF-related functionality to avoid code duplication.
"""

import glob
import os
import subprocess
from contextlib import ExitStack
from functools import lru_cache
from typing import Any

# Note: pdftoppm is no longer used. PDF rendering uses Poppler GI library directly.
# Only pdfinfo and pdfimages are still used as subprocesses.
from bigocrpdf.utils.logger import logger


def get_pdf_page_count(file_path: str, use_cache: bool = True) -> int:
    """Get the number of pages in a PDF file using pdfinfo.

    Args:
        file_path: Path to the PDF file
        use_cache: Whether to use cached values (default: True)

    Returns:
        Number of pages, or 0 if unable to determine
    """
    if not file_path or not os.path.exists(file_path):
        return 0

    if not use_cache:
        return _get_page_count_uncached(file_path)
    try:
        stat = os.stat(file_path)
    except OSError:
        return 0
    return _get_page_count_cached(file_path, stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=256)
def _get_page_count_cached(file_path: str, _mtime_ns: int, _size: int) -> int:
    return _get_page_count_uncached(file_path)


def _get_page_count_uncached(file_path: str) -> int:
    """Get page count without caching.

    Args:
        file_path: Path to the PDF file

    Returns:
        Number of pages, or 0 if unable to determine
    """
    try:
        result = subprocess.run(
            ["pdfinfo", file_path],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,  # Prevent hanging on corrupted files
        )

        for line in result.stdout.split("\n"):
            if line.startswith("Pages:"):
                return int(line.split(":")[1].strip())

    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout getting page count for {os.path.basename(file_path)}")
    except ValueError as e:
        logger.warning(f"Invalid page count value in {os.path.basename(file_path)}: {e}")
    except Exception as e:
        logger.error(f"Error getting page count for {os.path.basename(file_path)}: {e}")

    return 0


def get_pdf_thumbnail(file_path: str, max_size: int = 200) -> bytes | None:
    """Get a thumbnail of the first page of a PDF.

    Uses Poppler library directly for rendering, avoiding subprocess overhead.

    Args:
        file_path: Path to the PDF file
        max_size: Maximum width/height in pixels (default: 200)

    Returns:
        PNG image bytes or None if extraction failed
    """
    if not file_path or not os.path.exists(file_path):
        return None

    return render_pdf_page_to_png(file_path, page_num=0, size=max_size)


def get_pdf_info(file_path: str) -> dict[str, Any]:
    """Get metadata about a PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        Dictionary with PDF info (pages, title, author, file_size, etc.)
    """
    info: dict[str, Any] = {
        "pages": 0,
        "title": "",
        "author": "",
        "file_size": 0,
        "file_size_mb": 0.0,
    }

    if not file_path or not os.path.exists(file_path):
        return info

    # Get file size
    try:
        size = os.path.getsize(file_path)
        info["file_size"] = size
        info["file_size_mb"] = round(size / (1024 * 1024), 2)
    except OSError:
        pass

    # Get page count and metadata from one pdfinfo snapshot.
    try:
        result = subprocess.run(
            ["pdfinfo", file_path],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )

        for line in result.stdout.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                if key == "pages":
                    try:
                        info["pages"] = int(value)
                    except ValueError:
                        logger.debug("Invalid PDF page count: %s", value)
                elif key == "title" and value:
                    info["title"] = value
                elif key == "author" and value:
                    info["author"] = value

    except (OSError, subprocess.SubprocessError) as e:
        logger.debug(f"Could not get PDF metadata: {e}")

    return info


def extract_images_for_odf(
    pdf_path: str,
    ocr_text: str = "",
) -> tuple[list[str], list[str]]:
    """Extract images from PDF for ODF export using pdfimages.

    Args:
        pdf_path: Path to the PDF file
        ocr_text: Optional OCR text to associate with the first image

    Returns:
        Tuple of (image_paths, ocr_texts) where ocr_texts[0] contains
        the provided text and the rest are empty strings.
    """
    images: list[str] = []
    ocr_texts: list[str] = []

    if not pdf_path or not os.path.exists(pdf_path):
        return images, ocr_texts

    try:
        from bigocrpdf.utils.temp_manager import mkdtemp

        temp_dir = mkdtemp(prefix="odf_images_")
        image_prefix = os.path.join(temp_dir, "image")

        result = subprocess.run(
            ["pdfimages", "-all", pdf_path, image_prefix],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            patterns = ["*.png", "*.jpg", "*.jpeg", "*.ppm", "*.pbm"]
            for pattern in patterns:
                images.extend(glob.glob(os.path.join(temp_dir, pattern)))
            images.sort()

            if images and ocr_text:
                ocr_texts = [ocr_text] + [""] * (len(images) - 1)

            logger.info(f"Extracted {len(images)} images for ODF export")
        else:
            logger.warning(f"pdfimages failed: {result.stderr}")

    except Exception as e:
        logger.error(f"Error extracting images for ODF: {e}")

    return images, ocr_texts


def open_file_with_default_app(file_path: str) -> bool:
    """Open a file using the system's default application.

    Uses Gio.AppInfo to directly launch the default handler without
    showing an application chooser dialog.

    Args:
        file_path: Path to the file to open

    Returns:
        True if the launch was requested successfully
    """
    if not file_path or not os.path.exists(file_path):
        logger.warning(f"Cannot open file: path does not exist: {file_path}")
        return False

    try:
        import gi

        gi.require_version("Gio", "2.0")
        from gi.repository import Gio

        gfile = Gio.File.new_for_path(file_path)
        uri = gfile.get_uri()
        Gio.AppInfo.launch_default_for_uri(uri, None)
        return True
    except Exception as e:
        logger.warning(f"Gio launch failed for {file_path}: {e}, trying xdg-open")
        try:
            import subprocess

            subprocess.Popen(
                ["xdg-open", file_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception as e2:
            logger.error(f"Failed to open file {file_path}: {e2}")
            return False


def render_pdf_page_to_png(
    pdf_path: str,
    page_num: int,
    size: int = 200,
) -> bytes | None:
    """Render a single PDF page to PNG bytes using Poppler.

    Uses the Poppler GObject Introspection library directly instead of
    spawning a subprocess, which is faster and avoids pdftoppm dependency.

    Args:
        pdf_path: Path to the PDF file
        page_num: Page number (0-indexed)
        size: Maximum dimension in pixels

    Returns:
        PNG image bytes, or None if rendering failed
    """
    try:
        import gi

        gi.require_version("Poppler", "0.18")
        from gi.repository import GLib, Poppler

        # Poppler requires absolute paths for filename_to_uri
        abs_path = os.path.abspath(pdf_path)
        uri = GLib.filename_to_uri(abs_path, None)
        doc = Poppler.Document.new_from_file(uri, None)
        if not doc or page_num >= doc.get_n_pages():
            return None

        page = doc.get_page(page_num)
        if not page:
            return None

        pw, ph = page.get_size()
        scale = size / max(pw, ph)
        render_w = int(pw * scale)
        render_h = int(ph * scale)

        import cairo

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, render_w, render_h)
        ctx = cairo.Context(surface)
        ctx.scale(scale, scale)
        # White background
        ctx.set_source_rgb(1.0, 1.0, 1.0)
        ctx.paint()
        page.render(ctx)

        # Convert to PNG bytes
        import io

        buf = io.BytesIO()
        surface.write_to_png(buf)
        return buf.getvalue()

    except Exception as e:
        logger.error(f"Error rendering PDF page: {e}")
        return None


# --- Image extensions recognized as valid inputs ---
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".avif")


def is_image_file(file_path: str) -> bool:
    """Check if a file path has a supported image extension.

    Args:
        file_path: Path to check

    Returns:
        True if the file is a supported image type
    """
    return file_path.lower().endswith(IMAGE_EXTENSIONS)


def images_to_pdf(image_paths: list[str], output_path: str | None = None) -> str:
    """Convert one or more images to a single PDF file.

    Uses Pillow to convert images, applying EXIF rotation and converting
    RGBA/LA modes to RGB for PDF compatibility.

    Args:
        image_paths: List of image file paths to convert
        output_path: Optional output path. If None, a temp file in /tmp is created.

    Returns:
        Path to the created PDF file

    Raises:
        ValueError: If no valid images provided
        RuntimeError: If conversion fails
    """
    from PIL import Image, ImageOps

    if not image_paths:
        raise ValueError("No image paths provided")

    with ExitStack() as resources:
        images: list[Image.Image] = []
        for path in image_paths:
            try:
                source = resources.enter_context(Image.open(path))
                image = ImageOps.exif_transpose(source)
                if image is not source:
                    resources.callback(image.close)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                    resources.callback(image.close)
                images.append(image)
            except Exception as e:
                logger.error(f"Failed to open image {path}: {e}")

        if not images:
            raise ValueError("No valid images could be opened")

        owns_output = output_path is None
        if owns_output:
            from bigocrpdf.utils.temp_manager import mkstemp

            base = os.path.splitext(os.path.basename(image_paths[0]))[0]
            if len(image_paths) > 1:
                base = f"{base}_+{len(image_paths) - 1}"
            fd, output_path = mkstemp(prefix=f"bigocr_{base}_", suffix=".pdf")
            os.close(fd)

        try:
            first_img = images[0]
            rest = images[1:]
            first_img.save(output_path, format="PDF", save_all=True, append_images=rest)
        except Exception as e:
            if owns_output:
                from bigocrpdf.utils.temp_manager import remove_file

                remove_file(output_path)
            raise RuntimeError(f"Failed to create PDF from images: {e}") from e

        logger.info(f"Created PDF from {len(images)} image(s): {output_path}")
        return output_path


# Map the user-facing page-layout setting to a PDF catalog ``/PageLayout``
# name (ISO 32000 §7.7.2). ``"default"`` omits the key so the viewer chooses.
PAGE_LAYOUT_NAMES: dict[str, str] = {
    "single": "/SinglePage",
    "continuous": "/OneColumn",
    "two_page": "/TwoColumnLeft",
}


def set_root_page_layout(pdf: Any, page_layout: str) -> bool:
    """Set the document catalog ``/PageLayout`` from the page-layout setting.

    Controls how viewers (notably mobile readers) arrange pages on open:
    one page at a time, continuous vertical scroll, or two-up. ``"default"``
    (or any unknown value) leaves the catalog untouched so the viewer decides.

    ``pdf`` is a ``pikepdf.Pdf``. Returns True when a name was written.
    """
    import pikepdf

    name = PAGE_LAYOUT_NAMES.get(page_layout)
    if name is None:
        return False
    pdf.Root.PageLayout = pikepdf.Name(name)
    return True
