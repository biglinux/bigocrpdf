"""
BigOcrPdf - Export Service Module

Handles exporting OCR results to TXT and ODF files.
Encapsulates business logic for text/document export that was previously in window.py.
"""

import os
import subprocess
from pathlib import Path

from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


def save_text_file(
    output_file: str,
    extracted_text: str,
    separate_folder: str | None = None,
    ocr_boxes: list | None = None,
) -> str | None:
    """Save extracted text to a .txt file.

    When the OCR'd PDF is available, uses structured formatting
    (proper paragraph separation, aligned tables, heading detection)
    via the TSV converter pipeline. Falls back to raw text otherwise.

    Args:
        output_file: Path to the related output PDF file (used to derive name)
        extracted_text: The extracted text content (fallback)
        separate_folder: Optional separate folder for txt output
        ocr_boxes: Optional OCR boxes for spread detection

    Returns:
        Path to the saved txt file, or None on failure
    """
    try:
        # Try structured export using pdftotext TSV pipeline
        text_to_save = extracted_text
        if os.path.isfile(output_file) and output_file.lower().endswith(".pdf"):
            try:
                from bigocrpdf.utils.tsv_odf_converter import convert_pdf_to_text

                structured = convert_pdf_to_text(output_file)
                if structured.strip():
                    text_to_save = structured
            except Exception as e:
                logger.warning(f"Structured text extraction failed, using raw text: {e}")

        # Detect and split book spreads if OCR data is available
        if text_to_save is extracted_text and ocr_boxes:
            from bigocrpdf.utils.spread_detector import (
                detect_and_split_spreads,
                split_text_by_spreads,
            )

            _split_boxes, split_map = detect_and_split_spreads(ocr_boxes)
            if split_map:
                text_to_save = split_text_by_spreads(extracted_text, ocr_boxes, split_map)

        if separate_folder:
            txt_dir = str(Path(separate_folder).resolve())
            os.makedirs(txt_dir, exist_ok=True)
        else:
            txt_dir = os.path.dirname(output_file)

        base_name = os.path.splitext(os.path.basename(output_file))[0]
        txt_path = os.path.join(txt_dir, f"{base_name}.txt")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text_to_save)

        logger.info(_("Saved extracted text to: {0}").format(txt_path))
        return txt_path

    except OSError as e:
        logger.error(_("Failed to save text file: {0}").format(e))
        return None


def save_odf_file(
    output_file: str,
    extracted_text: str,
    ocr_boxes: list,
    source_pdf: str,
    include_images: bool = True,
    use_formatting: bool = True,
) -> str | None:
    """Save OCR'd PDF as a structured ODF document using pdftotext TSV.

    Uses pdftotext -tsv on the OCR'd PDF (output_file) to extract
    word-level positions, then assembles headings, paragraphs, tables
    and key-value pairs into a styled ODF file.

    Args:
        output_file: Path to the OCR'd PDF (has text layer). Also used to derive ODF name.
        extracted_text: The extracted text content (unused, kept for API compat).
        ocr_boxes: List of OCR boxes (unused, kept for API compat).
        source_pdf: Path to the original source PDF (unused).
        include_images: Whether to embed page images in the ODF document.
        use_formatting: Whether to use coordinate-based formatting (unused).

    Returns:
        Path to the saved ODF file, or None on failure.
    """
    try:
        odf_dir = os.path.dirname(output_file)
        base_name = os.path.splitext(os.path.basename(output_file))[0]
        odf_path = os.path.join(odf_dir, f"{base_name}.odt")

        from bigocrpdf.utils.tsv_odf_converter import convert_pdf_to_odf

        convert_pdf_to_odf(output_file, odf_path, include_images=include_images)

        logger.info(f"Saved ODF file to: {odf_path}")
        return odf_path

    except (OSError, ImportError, subprocess.SubprocessError) as e:
        logger.error(f"Failed to save ODF file: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error saving ODF file: {e}")
        return None


def _boxes_to_ocr_data(ocr_boxes: list) -> list:
    """Convert OCR boxes to OCRTextData objects.

    Args:
        ocr_boxes: List of raw OCR box objects

    Returns:
        List of OCRTextData objects
    """
    from bigocrpdf.utils.odf_exporter import OCRTextData

    return [
        OCRTextData(
            text=box.text,
            x=box.x,
            y=box.y,
            width=box.width,
            height=box.height,
            confidence=getattr(box, "confidence", 1.0),
            page_num=getattr(box, "page_num", 1),
            is_bold=getattr(box, "is_bold", False),
            is_underlined=getattr(box, "is_underlined", False),
        )
        for box in ocr_boxes
    ]
