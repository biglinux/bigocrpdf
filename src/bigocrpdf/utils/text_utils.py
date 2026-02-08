"""
BigOcrPdf - Text Utilities Module

This module provides shared utility functions for text extraction and handling.
Centralizes text-related functionality to avoid code duplication.
"""

import os

from bigocrpdf.utils.logger import logger


def read_text_from_sidecar(sidecar_path: str) -> str | None:
    """Read text from a sidecar .txt file.

    Args:
        sidecar_path: Path to the sidecar text file

    Returns:
        Text content, or None if file doesn't exist or can't be read
    """
    if not sidecar_path or not os.path.exists(sidecar_path):
        return None

    try:
        with open(sidecar_path, encoding="utf-8") as f:
            text = f.read()

        if text:
            logger.info(f"Read {len(text)} characters from sidecar file")
            return text

    except UnicodeDecodeError:
        # Try with different encoding
        try:
            with open(sidecar_path, encoding="latin-1") as f:
                text = f.read()
            if text:
                logger.info(f"Read {len(text)} characters from sidecar (latin-1)")
                return text
        except Exception as e:
            logger.error(f"Error reading sidecar with fallback encoding: {e}")
    except Exception as e:
        logger.error(f"Error reading sidecar file: {e}")

    return None


def group_ocr_text_by_page(ocr_boxes: list, num_pages: int) -> list[str]:
    """Group OCR text by page number.

    Args:
        ocr_boxes: List of OCR boxes with text and page_num attributes
        num_pages: Number of pages

    Returns:
        List of text strings per page
    """
    pages: dict[int, list[str]] = {i + 1: [] for i in range(num_pages)}

    for box in ocr_boxes:
        page_num = getattr(box, "page_num", 1)
        if page_num in pages:
            pages[page_num].append(box.text)

    return ["\n".join(pages.get(i + 1, [])) for i in range(num_pages)]
