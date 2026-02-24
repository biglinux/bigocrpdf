"""
BigOcrPdf - PDF Page Operations

Functions for manipulating PDF pages: rotation, deletion, reordering, and OCR selection.
Uses pikepdf for PDF manipulation operations.
"""

from bigocrpdf.ui.pdf_editor.page_model import PDFDocument
from bigocrpdf.utils.logger import logger


def rotate_pages(doc: PDFDocument, page_indices: list[int], degrees: int) -> bool:
    """Rotate pages by the specified degrees.

    Args:
        doc: The PDFDocument to modify
        page_indices: List of page indices (0-indexed positions) to rotate
        degrees: Rotation angle (90, 180, 270, or -90)

    Returns:
        True if successful, False otherwise
    """
    if not page_indices:
        return False

    # Normalize degrees
    degrees = degrees % 360
    if degrees not in (0, 90, 180, 270):
        degrees = round(degrees / 90) * 90 % 360

    if degrees == 0:
        return True  # No rotation needed

    try:
        for idx in page_indices:
            page = doc.get_page_by_position(idx)
            if page:
                if degrees == 90:
                    page.rotate_right()
                elif degrees == -90 or degrees == 270:
                    page.rotate_left()
                elif degrees == 180:
                    page.rotate_right()
                    page.rotate_right()

        doc.mark_modified()
        logger.info(f"Rotated {len(page_indices)} page(s) by {degrees}°")
        return True

    except Exception as e:
        logger.error(f"Failed to rotate pages: {e}")
        return False


def delete_pages(
    doc: PDFDocument | None, page_indices: list[int], hard_delete: bool = False
) -> bool:
    """Delete pages (soft delete by default).

    Soft delete marks pages as deleted but doesn't remove them.
    Hard delete removes pages from the document.

    Args:
        doc: The PDFDocument to modify
        page_indices: List of page indices to delete
        hard_delete: If True, permanently remove pages

    Returns:
        True if successful
    """
    if not page_indices or doc is None:
        return False

    try:
        if hard_delete:
            # Remove pages from the list (in reverse order to maintain indices)
            pages_to_remove = []
            for idx in sorted(page_indices, reverse=True):
                page = doc.get_page_by_position(idx)
                if page:
                    pages_to_remove.append(page)

            for page in pages_to_remove:
                doc.pages.remove(page)

            # Update positions
            doc.update_positions()
        else:
            # Soft delete: mark pages as deleted
            for idx in page_indices:
                page = doc.get_page_by_position(idx)
                if page:
                    page.deleted = True

        doc.mark_modified()
        logger.info(f"Deleted {len(page_indices)} page(s) (hard={hard_delete})")
        return True

    except Exception as e:
        logger.error(f"Failed to delete pages: {e}")
        return False


def set_ocr_selection(doc: PDFDocument, page_indices: list[int], selected: bool) -> bool:
    """Set OCR selection state for pages.

    Args:
        doc: The PDFDocument to modify
        page_indices: List of page indices to modify
        selected: Whether pages should be included for OCR

    Returns:
        True if successful
    """
    if not page_indices:
        return False

    try:
        for idx in page_indices:
            page = doc.get_page_by_position(idx)
            if page:
                page.included_for_ocr = selected

        doc.mark_modified()
        logger.info(f"Set OCR selection to {selected} for {len(page_indices)} page(s)")
        return True

    except Exception as e:
        logger.error(f"Failed to set OCR selection: {e}")
        return False


def select_all_for_ocr(doc: PDFDocument) -> bool:
    """Select all pages for OCR.

    Args:
        doc: The PDFDocument to modify

    Returns:
        True if successful
    """
    try:
        for page in doc.pages:
            if not page.deleted:
                page.included_for_ocr = True

        doc.mark_modified()
        logger.info("Selected all pages for OCR")
        return True

    except Exception as e:
        logger.error(f"Failed to select all for OCR: {e}")
        return False


def deselect_all_for_ocr(doc: PDFDocument) -> bool:
    """Deselect all pages from OCR.

    Args:
        doc: The PDFDocument to modify

    Returns:
        True if successful
    """
    try:
        for page in doc.pages:
            page.included_for_ocr = False

        doc.mark_modified()
        logger.info("Deselected all pages from OCR")
        return True

    except Exception as e:
        logger.error(f"Failed to deselect all from OCR: {e}")
        return False


def _add_image_page(source_file, page_state, new_pdf, opened_pdfs, opened_streams):
    """Convert an image file to a PDF page and append it to new_pdf."""
    import io

    import pikepdf
    from PIL import Image, ImageOps

    img = Image.open(source_file)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA"):
        img = img.convert("RGB")

    pdf_bytes = io.BytesIO()
    img.save(pdf_bytes, format="PDF")
    pdf_bytes.seek(0)
    opened_streams.append(pdf_bytes)

    temp_pdf = pikepdf.Pdf.open(pdf_bytes)
    page = temp_pdf.pages[0]
    if page_state.rotation != 0:
        page.Rotate = page_state.rotation
    new_pdf.pages.append(page)

    if "temp_images" not in opened_pdfs:
        opened_pdfs["temp_images"] = []  # type: ignore
    opened_pdfs["temp_images"].append(temp_pdf)  # type: ignore


def _resolve_source_rotation(src_page) -> int:
    """Resolve the effective /Rotate from a page, including inherited values."""
    try:
        if "/Rotate" in src_page:
            return int(src_page["/Rotate"])
        node = src_page.obj if hasattr(src_page, "obj") else src_page
        while node is not None:
            if "/Rotate" in node:
                return int(node["/Rotate"])
            parent = node.get("/Parent")
            node = parent if parent is not None else None
    except Exception:
        pass
    return 0


def _add_pdf_page(source_file, page_state, new_pdf, opened_pdfs):
    """Copy a PDF page to new_pdf with rotation handling."""
    import pikepdf

    if source_file not in opened_pdfs:
        opened_pdfs[source_file] = pikepdf.open(source_file)

    src_pdf = opened_pdfs[source_file]
    original_idx = page_state.page_number - 1

    if original_idx < 0 or original_idx >= len(src_pdf.pages):
        logger.warning(f"Invalid page index {original_idx} for {source_file}")
        return

    src_page = src_pdf.pages[original_idx]
    source_rotation = _resolve_source_rotation(src_page)

    new_pdf.pages.append(src_page)
    new_page = new_pdf.pages[-1]

    final_rotation = (source_rotation + page_state.rotation) % 360
    if final_rotation != 0:
        new_page.Rotate = final_rotation
    elif "/Rotate" in new_page:
        del new_page["/Rotate"]

    if page_state.rotation != 0 or source_rotation != 0:
        logger.info(
            f"Page {page_state.source_file}:{page_state.page_number} "
            f"rotation: source={source_rotation} + editor={page_state.rotation} "
            f"= {final_rotation}"
        )


_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp")


def apply_changes_to_pdf(doc: PDFDocument | None, output_path: str) -> bool:
    """Apply all changes and save to a new PDF file.

    Merges pages from multiple source files (PDFs and Images).
    Uses pikepdf for PDF manipulation and PIL for image conversion.

    Args:
        doc: The PDFDocument with changes
        output_path: Path for the output PDF

    Returns:
        True if successful
    """
    if doc is None:
        return False
    try:
        import io

        import pikepdf

        opened_pdfs: dict[str, pikepdf.Pdf] = {}
        opened_streams: list[io.BytesIO] = []
        new_pdf = pikepdf.Pdf.new()
        active_pages = doc.get_active_pages()

        for page_state in active_pages:
            source_file = page_state.source_file or doc.path
            if not source_file:
                logger.warning(f"Skipping page with no source file: {page_state}")
                continue

            is_image = source_file.lower().endswith(_IMAGE_EXTENSIONS)

            try:
                if is_image:
                    _add_image_page(source_file, page_state, new_pdf, opened_pdfs, opened_streams)
                else:
                    _add_pdf_page(source_file, page_state, new_pdf, opened_pdfs)
            except Exception as e:
                logger.error(f"Failed to process {source_file}: {e}")
                continue

        new_pdf.save(output_path)
        logger.info(f"Saved modified PDF to {output_path}")
        return True

    except ImportError:
        logger.error("pikepdf or PIL is not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to save PDF: {e}")
        return False
    finally:
        # Always close all opened PDF handles
        for key, handle in opened_pdfs.items():
            try:
                if key == "temp_images":
                    for tmp in handle:  # type: ignore
                        tmp.close()
                else:
                    handle.close()
            except Exception:
                pass
