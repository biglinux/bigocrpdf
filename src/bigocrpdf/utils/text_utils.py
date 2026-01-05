"""
BigOcrPdf - Text Utilities Module

This module provides shared utility functions for text extraction and handling.
Centralizes text-related functionality to avoid code duplication.
"""

import os
import subprocess

from bigocrpdf.utils.logger import logger


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text directly from a PDF file using pdftotext.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text content, or empty string if extraction fails
    """
    if not pdf_path or not os.path.exists(pdf_path):
        return ""

    try:
        result = subprocess.run(
            ["pdftotext", pdf_path, "-"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,  # Prevent hanging on large files
        )

        if result.stdout and len(result.stdout.strip()) > 0:
            logger.info(f"Extracted {len(result.stdout)} characters from PDF")
            return result.stdout

    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout extracting text from {os.path.basename(pdf_path)}")
    except Exception as e:
        logger.warning(f"Failed to extract text from PDF: {e}")

    return ""


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


def get_sidecar_path(pdf_path: str) -> str:
    """Get the standard sidecar file path for a PDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Path to the corresponding .txt sidecar file
    """
    return os.path.splitext(pdf_path)[0] + ".txt"


def get_temp_sidecar_path(pdf_path: str, temp_dir: str | None = None) -> str:
    """Get the temporary sidecar file path for a PDF.

    Args:
        pdf_path: Path to the PDF file
        temp_dir: Optional custom temp directory. If None, uses .temp subdirectory.

    Returns:
        Path to the temporary sidecar file
    """
    output_dir = os.path.dirname(pdf_path)

    if temp_dir is None:
        temp_dir = os.path.join(output_dir, ".temp")

    base_name = os.path.basename(os.path.splitext(pdf_path)[0])
    return os.path.join(temp_dir, f"temp_{base_name}.txt")


def find_extracted_text(pdf_path: str, temp_dir: str | None = None) -> str:
    """Find and return extracted text from various sources.

    Searches in order:
    1. Standard sidecar file (.txt next to PDF)
    2. Temporary sidecar file (in .temp directory)
    3. Direct PDF text extraction

    Args:
        pdf_path: Path to the PDF file
        temp_dir: Optional custom temp directory

    Returns:
        Extracted text, or empty string if not found
    """
    # Try standard sidecar
    sidecar_path = get_sidecar_path(pdf_path)
    text = read_text_from_sidecar(sidecar_path)
    if text:
        return text

    # Try temp sidecar
    temp_sidecar = get_temp_sidecar_path(pdf_path, temp_dir)
    text = read_text_from_sidecar(temp_sidecar)
    if text:
        return text

    # Try direct extraction
    return extract_text_from_pdf(pdf_path)


def cleanup_temp_sidecar(pdf_path: str, temp_dir: str | None = None) -> bool:
    """Clean up temporary sidecar file for a PDF.

    Args:
        pdf_path: Path to the PDF file
        temp_dir: Optional custom temp directory

    Returns:
        True if file was deleted, False otherwise
    """
    temp_sidecar = get_temp_sidecar_path(pdf_path, temp_dir)

    if os.path.exists(temp_sidecar):
        try:
            os.remove(temp_sidecar)
            logger.info(f"Deleted temp sidecar: {os.path.basename(temp_sidecar)}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete temp sidecar: {e}")

    return False


def cleanup_temp_directory(output_dir: str) -> bool:
    """Clean up empty .temp directory.

    Args:
        output_dir: Directory containing the .temp subdirectory

    Returns:
        True if directory was removed, False otherwise
    """
    temp_dir = os.path.join(output_dir, ".temp")

    if os.path.exists(temp_dir):
        try:
            # Only remove if empty
            if not os.listdir(temp_dir):
                os.rmdir(temp_dir)
                logger.info("Removed empty temp directory")
                return True
        except Exception as e:
            logger.debug(f"Could not remove temp directory: {e}")

    return False
