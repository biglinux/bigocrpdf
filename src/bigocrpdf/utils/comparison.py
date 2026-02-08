"""
BigOcrPdf - PDF Comparison Utilities

This module provides functionality to compare PDFs before and after OCR processing.
It generates comparison data including file sizes, thumbnails, and text content.
"""

import os
from dataclasses import dataclass
from typing import Any

from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.pdf_utils import get_pdf_info, get_pdf_thumbnail


@dataclass
class PDFComparisonResult:
    """Result of comparing two PDFs (before and after OCR)."""

    # File paths
    input_path: str
    output_path: str

    # File sizes
    input_size_bytes: int = 0
    output_size_bytes: int = 0

    # Page counts
    input_pages: int = 0
    output_pages: int = 0

    # Thumbnails (PNG bytes)
    input_thumbnail: bytes | None = None
    output_thumbnail: bytes | None = None

    # Extracted text (output only, as input has no OCR)
    extracted_text: str = ""

    @property
    def input_size_mb(self) -> float:
        """Get input size in megabytes."""
        return round(self.input_size_bytes / (1024 * 1024), 2)

    @property
    def output_size_mb(self) -> float:
        """Get output size in megabytes."""
        return round(self.output_size_bytes / (1024 * 1024), 2)

    @property
    def size_change_mb(self) -> float:
        """Get size change in megabytes."""
        return round((self.output_size_bytes - self.input_size_bytes) / (1024 * 1024), 2)

    @property
    def size_change_percent(self) -> float:
        """Get size change as percentage (positive = larger, negative = smaller)."""
        if self.input_size_bytes <= 0:
            return 0.0
        return round(
            ((self.output_size_bytes - self.input_size_bytes) / self.input_size_bytes) * 100, 1
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (without thumbnails for JSON serialization)."""
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "input_size_bytes": self.input_size_bytes,
            "output_size_bytes": self.output_size_bytes,
            "input_size_mb": self.input_size_mb,
            "output_size_mb": self.output_size_mb,
            "size_change_bytes": self.output_size_bytes - self.input_size_bytes,
            "size_change_mb": self.size_change_mb,
            "size_change_percent": self.size_change_percent,
            "input_pages": self.input_pages,
            "output_pages": self.output_pages,
            "has_input_thumbnail": self.input_thumbnail is not None,
            "has_output_thumbnail": self.output_thumbnail is not None,
            "text_length": len(self.extracted_text),
        }


def compare_pdfs(
    input_path: str,
    output_path: str,
    extracted_text: str = "",
    include_thumbnails: bool = True,
    thumbnail_size: int = 200,
) -> PDFComparisonResult:
    """Compare a PDF before and after OCR processing.

    Args:
        input_path: Path to the original PDF (before OCR)
        output_path: Path to the processed PDF (after OCR)
        extracted_text: The OCR-extracted text content
        include_thumbnails: Whether to generate thumbnails
        thumbnail_size: Maximum dimension for thumbnails

    Returns:
        PDFComparisonResult with all comparison data
    """
    result = PDFComparisonResult(
        input_path=input_path,
        output_path=output_path,
        extracted_text=extracted_text,
    )

    # Get input file info
    if os.path.exists(input_path):
        input_info = get_pdf_info(input_path)
        result.input_size_bytes = input_info.get("file_size", 0)
        result.input_pages = input_info.get("pages", 0)

        if include_thumbnails:
            result.input_thumbnail = get_pdf_thumbnail(input_path, max_size=thumbnail_size)
    else:
        logger.warning(f"Input file not found for comparison: {input_path}")

    # Get output file info
    if os.path.exists(output_path):
        output_info = get_pdf_info(output_path)
        result.output_size_bytes = output_info.get("file_size", 0)
        result.output_pages = output_info.get("pages", 0)

        if include_thumbnails:
            result.output_thumbnail = get_pdf_thumbnail(output_path, max_size=thumbnail_size)
    else:
        logger.warning(f"Output file not found for comparison: {output_path}")

    return result


def get_batch_statistics(results: list[PDFComparisonResult]) -> dict[str, Any]:
    """Calculate aggregate statistics for a batch of comparisons.

    Args:
        results: List of comparison results

    Returns:
        Dictionary with aggregate statistics
    """
    if not results:
        return {
            "total_files": 0,
            "total_input_size_bytes": 0,
            "total_output_size_bytes": 0,
            "total_size_change_bytes": 0,
            "total_pages": 0,
            "total_words": 0,
        }

    total_input = sum(r.input_size_bytes for r in results)
    total_output = sum(r.output_size_bytes for r in results)
    total_pages = sum(r.output_pages for r in results)
    total_words = sum(len(r.extracted_text.split()) for r in results)
    files_larger = sum(1 for r in results if r.output_size_bytes > r.input_size_bytes)
    files_smaller = sum(1 for r in results if r.output_size_bytes < r.input_size_bytes)

    return {
        "total_files": len(results),
        "total_input_size_bytes": total_input,
        "total_output_size_bytes": total_output,
        "total_input_size_mb": round(total_input / (1024 * 1024), 2),
        "total_output_size_mb": round(total_output / (1024 * 1024), 2),
        "total_size_change_bytes": total_output - total_input,
        "total_size_change_mb": round((total_output - total_input) / (1024 * 1024), 2),
        "average_size_change_percent": (
            round(((total_output - total_input) / total_input) * 100, 1) if total_input > 0 else 0
        ),
        "total_pages": total_pages,
        "total_words": total_words,
        "files_larger": files_larger,
        "files_smaller": files_smaller,
        "files_same_size": len(results) - files_larger - files_smaller,
    }
