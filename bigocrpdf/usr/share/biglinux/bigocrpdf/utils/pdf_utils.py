"""
BigOcrPdf - PDF Utilities Module

This module provides shared utility functions for PDF file operations.
Centralizes PDF-related functionality to avoid code duplication.
"""

import os
import subprocess
from typing import Dict, Optional

from utils.logger import logger

# Cache for PDF page counts to avoid repeated subprocess calls
_page_count_cache: Dict[str, int] = {}


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
            timeout=10  # Prevent hanging on corrupted files
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


def clear_page_count_cache(file_path: Optional[str] = None) -> None:
    """Clear the page count cache.
    
    Args:
        file_path: If provided, clear only this file's cache entry.
                   If None, clear the entire cache.
    """
    global _page_count_cache
    
    if file_path is not None:
        _page_count_cache.pop(file_path, None)
    else:
        _page_count_cache.clear()


def validate_pdf_file(file_path: str) -> tuple[bool, str]:
    """Validate a PDF file for processing.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        Tuple of (is_valid, error_message).
        If valid, error_message is empty string.
    """
    if not file_path:
        return False, "Empty file path"
    
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    if not os.path.isfile(file_path):
        return False, "Path is not a file"
    
    if not file_path.lower().endswith('.pdf'):
        return False, "Not a PDF file"
    
    if not os.access(file_path, os.R_OK):
        return False, "File is not readable"
    
    # Check file size (empty files are invalid)
    try:
        if os.path.getsize(file_path) == 0:
            return False, "File is empty"
    except OSError:
        return False, "Cannot read file size"
    
    return True, ""


def get_pdf_file_info(file_path: str) -> Dict[str, any]:
    """Get comprehensive information about a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dictionary with file information:
        - pages: int
        - size_bytes: int
        - filename: str
        - directory: str
        - is_valid: bool
    """
    is_valid, error = validate_pdf_file(file_path)
    
    if not is_valid:
        return {
            "pages": 0,
            "size_bytes": 0,
            "filename": os.path.basename(file_path) if file_path else "",
            "directory": os.path.dirname(file_path) if file_path else "",
            "is_valid": False,
            "error": error
        }
    
    try:
        size_bytes = os.path.getsize(file_path)
    except OSError:
        size_bytes = 0
    
    return {
        "pages": get_pdf_page_count(file_path),
        "size_bytes": size_bytes,
        "filename": os.path.basename(file_path),
        "directory": os.path.dirname(file_path),
        "is_valid": True,
        "error": ""
    }
