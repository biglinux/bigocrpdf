"""
BigOcrPdf - Export Service Module

Handles exporting OCR results to TXT and ODF files.
Encapsulates business logic for text/document export that was previously in window.py.
"""

import os

from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


def save_text_file(
    output_file: str,
    extracted_text: str,
    separate_folder: str | None = None,
) -> str | None:
    """Save extracted text to a .txt file.

    Args:
        output_file: Path to the related output PDF file (used to derive name)
        extracted_text: The extracted text content to save
        separate_folder: Optional separate folder for txt output

    Returns:
        Path to the saved txt file, or None on failure
    """
    try:
        if separate_folder:
            txt_dir = separate_folder
            os.makedirs(txt_dir, exist_ok=True)
        else:
            txt_dir = os.path.dirname(output_file)

        base_name = os.path.splitext(os.path.basename(output_file))[0]
        txt_path = os.path.join(txt_dir, f"{base_name}.txt")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)

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
    """Save extracted text to an ODF file.

    Args:
        output_file: Path to the related output PDF file (used to derive name)
        extracted_text: The extracted text content to save
        ocr_boxes: List of OCR boxes with position information
        source_pdf: Path to the original source PDF
        include_images: Whether to include images in the ODF
        use_formatting: Whether to use OCR coordinate-based formatting

    Returns:
        Path to the saved ODF file, or None on failure
    """
    from bigocrpdf.utils.odf_exporter import ODFExporter
    from bigocrpdf.utils.text_utils import group_ocr_text_by_page

    try:
        odf_dir = os.path.dirname(output_file)
        base_name = os.path.splitext(os.path.basename(output_file))[0]
        odf_path = os.path.join(odf_dir, f"{base_name}.odt")

        exporter = ODFExporter()

        if include_images and source_pdf and os.path.exists(source_pdf):
            from bigocrpdf.utils.pdf_utils import extract_images_for_odf

            images, _ = extract_images_for_odf(source_pdf)

            if images:
                if use_formatting and ocr_boxes:
                    ocr_data = _boxes_to_ocr_data(ocr_boxes)
                    exporter.export_formatted_with_images(ocr_data, images, odf_path)
                else:
                    page_texts = group_ocr_text_by_page(ocr_boxes, len(images))
                    exporter.export_paged_images(images, page_texts, odf_path)
            else:
                # Fallback: no images extracted
                if use_formatting and ocr_boxes:
                    ocr_data = _boxes_to_ocr_data(ocr_boxes)
                    exporter.export_structured_data(ocr_data, odf_path)
                else:
                    exporter.export_text(extracted_text, odf_path)
        elif use_formatting and ocr_boxes:
            ocr_data = _boxes_to_ocr_data(ocr_boxes)
            exporter.export_structured_data(ocr_data, odf_path)
        else:
            exporter.export_text(extracted_text, odf_path)

        logger.info(f"Saved ODF file to: {odf_path}")
        return odf_path

    except Exception as e:
        logger.error(f"Failed to save ODF file: {e}")
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
        )
        for box in ocr_boxes
    ]
