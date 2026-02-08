"""
BigOcrPdf - PDF Utilities Module

This module provides shared utility functions for PDF file operations.
Centralizes PDF-related functionality to avoid code duplication.
"""

import glob
import os
import shutil
import subprocess
import tempfile
from typing import Any

from bigocrpdf.utils.logger import logger

# Cache for PDF page counts to avoid repeated subprocess calls
_page_count_cache: dict[str, int] = {}


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

    # Check cache first
    if use_cache and file_path in _page_count_cache:
        return _page_count_cache[file_path]

    page_count = _get_page_count_uncached(file_path)

    # Cache the result
    if use_cache and page_count > 0:
        _page_count_cache[file_path] = page_count

    return page_count


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


def extract_pdf_page_images(
    file_path: str,
    page_range: tuple[int, int] | None = None,
) -> tuple[str | None, dict[int, str] | None]:
    """Extract page images from PDF for visual analysis.

    Args:
        file_path: Path to PDF file
        page_range: Optional (start, end) tuple for page range (1-indexed, inclusive)

    Returns:
        Tuple of (temp_dir_path, dict mapping page_num to image_path).
        Caller is responsible for cleaning up temp_dir_path.
    """
    try:
        temp_dir = tempfile.mkdtemp(prefix="bigocr_visual_")

        # Use pdftoppm to extract images
        # We assume pdftoppm is installed (poppler-utils)
        cmd = ["pdftoppm", "-png"]

        # Add page range options if specified
        if page_range is not None:
            start_page, end_page = page_range
            if start_page > 0:
                cmd.extend(["-f", str(start_page)])
            if end_page > 0:
                cmd.extend(["-l", str(end_page)])

        cmd.extend([file_path, os.path.join(temp_dir, "page")])
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Map images to page numbers
        page_images = {}
        for img_path in glob.glob(os.path.join(temp_dir, "page-*.png")):
            # Extract page number from filename (pdftoppm uses 1-based indexing in filenames usually)
            # Format: page-1.png or page-01.png
            try:
                basename = os.path.basename(img_path)
                num_part = basename.replace("page-", "").replace(".png", "")
                page_num = int(num_part)
                page_images[page_num] = img_path
            except ValueError:
                continue

        if not page_images:
            logger.warning("No images extracted from PDF")
            # Clean up empty temp dir
            shutil.rmtree(temp_dir)
            return None, None

        return temp_dir, page_images

    except Exception as e:
        logger.error(f"Failed to extract page images: {e}")
        # Use local temp_dir variable if it exists
        if "temp_dir" in locals() and temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return None, None


def get_pdf_thumbnail(file_path: str, max_size: int = 200) -> bytes | None:
    """Get a thumbnail of the first page of a PDF.

    Args:
        file_path: Path to the PDF file
        max_size: Maximum width/height in pixels (default: 200)

    Returns:
        PNG image bytes or None if extraction failed
    """
    if not file_path or not os.path.exists(file_path):
        return None

    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="bigocr_thumb_")
        temp_img = os.path.join(temp_dir, "thumb")

        # Use pdftoppm to extract first page only as JPEG for smaller size
        # -f 1 -l 1 = first page only
        # -scale-to limits the larger dimension
        cmd = [
            "pdftoppm",
            "-png",
            "-f",
            "1",
            "-l",
            "1",
            "-scale-to",
            str(max_size),
            "-singlefile",
            file_path,
            temp_img,
        ]

        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            timeout=30,
        )

        # Read the generated image
        img_path = temp_img + ".png"
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                return f.read()

        return None

    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout generating thumbnail for {os.path.basename(file_path)}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to generate thumbnail: {e.stderr.decode() if e.stderr else e}")
    except Exception as e:
        logger.error(f"Error generating thumbnail: {e}")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    return None


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

    # Get page count
    info["pages"] = get_pdf_page_count(file_path)

    # Get PDF metadata using pdfinfo
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

                if key == "title" and value:
                    info["title"] = value
                elif key == "author" and value:
                    info["author"] = value

    except Exception as e:
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
        temp_dir = tempfile.mkdtemp(prefix="odf_images_")
        image_prefix = os.path.join(temp_dir, "image")

        result = subprocess.run(
            ["pdfimages", "-all", pdf_path, image_prefix],
            capture_output=True,
            text=True,
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
    """Open a file using the system's default application (xdg-open).

    Args:
        file_path: Path to the file to open

    Returns:
        True if the command was launched successfully
    """
    if not file_path or not os.path.exists(file_path):
        logger.warning(f"Cannot open file: path does not exist: {file_path}")
        return False

    try:
        subprocess.Popen(
            ["xdg-open", file_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except FileNotFoundError:
        logger.error("xdg-open not found. Cannot open file with default application.")
        return False
    except Exception as e:
        logger.error(f"Failed to open file {file_path}: {e}")
        return False


def render_pdf_page_to_png(
    pdf_path: str,
    page_num: int,
    size: int = 200,
) -> bytes | None:
    """Render a single PDF page to PNG bytes using pdftoppm.

    Args:
        pdf_path: Path to the PDF file
        page_num: Page number (0-indexed)
        size: Maximum dimension in pixels

    Returns:
        PNG image bytes, or None if rendering failed
    """
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="bigocr_render_")
        temp_prefix = os.path.join(temp_dir, "page")

        # pdftoppm uses 1-based page numbers
        page_1based = page_num + 1

        cmd = [
            "pdftoppm",
            "-png",
            "-singlefile",
            "-f",
            str(page_1based),
            "-l",
            str(page_1based),
            "-scale-to",
            str(size),
            pdf_path,
            temp_prefix,
        ]

        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=30,
        )

        output_file = f"{temp_prefix}.png"
        if os.path.exists(output_file):
            with open(output_file, "rb") as f:
                return f.read()

        return None

    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout rendering page {page_num} of {os.path.basename(pdf_path)}")
    except subprocess.CalledProcessError as e:
        logger.debug(f"pdftoppm failed for page {page_num}: {e}")
    except Exception as e:
        logger.error(f"Error rendering PDF page: {e}")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

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

    images: list[Image.Image] = []
    for path in image_paths:
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
        except Exception as e:
            logger.error(f"Failed to open image {path}: {e}")

    if not images:
        raise ValueError("No valid images could be opened")

    if output_path is None:
        # Create a temp file with a descriptive name
        if len(image_paths) == 1:
            base = os.path.splitext(os.path.basename(image_paths[0]))[0]
        else:
            base = "merged_images"
        fd, output_path = tempfile.mkstemp(prefix=f"bigocr_{base}_", suffix=".pdf", dir="/tmp")
        os.close(fd)

    try:
        first_img = images[0]
        rest = images[1:] if len(images) > 1 else []
        first_img.save(output_path, format="PDF", save_all=True, append_images=rest)
        logger.info(f"Created PDF from {len(images)} image(s): {output_path}")
        return output_path
    except Exception as e:
        raise RuntimeError(f"Failed to create PDF from images: {e}") from e
