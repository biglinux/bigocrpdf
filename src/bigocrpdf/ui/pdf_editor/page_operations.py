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


def rotate_pages_left(doc: PDFDocument, page_indices: list[int]) -> bool:
    """Rotate pages 90° counter-clockwise.

    Args:
        doc: The PDFDocument to modify
        page_indices: List of page indices to rotate

    Returns:
        True if successful
    """
    return rotate_pages(doc, page_indices, 270)


def rotate_pages_right(doc: PDFDocument, page_indices: list[int]) -> bool:
    """Rotate pages 90° clockwise.

    Args:
        doc: The PDFDocument to modify
        page_indices: List of page indices to rotate

    Returns:
        True if successful
    """
    return rotate_pages(doc, page_indices, 90)


def delete_pages(doc: PDFDocument, page_indices: list[int], hard_delete: bool = False) -> bool:
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
    if not page_indices:
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


def reorder_pages(doc: PDFDocument, source_indices: list[int], target_position: int) -> bool:
    """Reorder pages by moving source pages to target position.

    Args:
        doc: The PDFDocument to modify
        source_indices: List of page indices to move
        target_position: Target position (0-indexed)

    Returns:
        True if successful
    """
    if not source_indices:
        return False

    try:
        active_pages = doc.get_active_pages()
        if not active_pages:
            return False

        # Validate target position
        target_position = max(0, min(target_position, len(active_pages)))

        # Get pages to move
        pages_to_move = []
        for idx in sorted(source_indices):
            if 0 <= idx < len(active_pages):
                pages_to_move.append(active_pages[idx])

        if not pages_to_move:
            return False

        # Remove pages from their current positions
        remaining_pages = [p for p in active_pages if p not in pages_to_move]

        # Adjust target position if needed (after removing pages)
        adjusted_target = target_position
        for idx in sorted(source_indices):
            if idx < target_position:
                adjusted_target -= 1

        adjusted_target = max(0, min(adjusted_target, len(remaining_pages)))

        # Insert pages at target position
        new_order = (
            remaining_pages[:adjusted_target] + pages_to_move + remaining_pages[adjusted_target:]
        )

        # Update positions
        for i, page in enumerate(new_order):
            page.position = i

        doc.mark_modified()
        logger.info(f"Reordered {len(pages_to_move)} page(s) to position {target_position}")
        return True

    except Exception as e:
        logger.error(f"Failed to reorder pages: {e}")
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


def apply_changes_to_pdf(doc: PDFDocument, output_path: str) -> bool:
    """Apply all changes and save to a new PDF file.

    Merges pages from multiple source files (PDFs and Images).
    Uses pikepdf for PDF manipulation and PIL for image conversion.

    Args:
        doc: The PDFDocument with changes
        output_path: Path for the output PDF

    Returns:
        True if successful
    """
    try:
        import io

        import pikepdf
        from PIL import Image, ImageOps

        # Cache for opened PDF documents to avoid re-opening
        # Map: filename -> pikepdf.Pdf
        opened_pdfs: dict[str, pikepdf.Pdf] = {}
        # Keep references to streams to prevent GC while pikepdf needs them
        opened_streams: list[io.BytesIO] = []

        # Create destination PDF
        new_pdf = pikepdf.Pdf.new()

        # Get pages in their new order
        active_pages = doc.get_active_pages()

        for page_state in active_pages:
            source_file = page_state.source_file
            if not source_file:
                # Fallback to main doc path if source not set
                source_file = doc.path

            if not source_file:
                logger.warning(f"Skipping page with no source file: {page_state}")
                continue

            # Check if source is an image
            lower_path = source_file.lower()
            is_image = lower_path.endswith(
                (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp")
            )

            if is_image:
                try:
                    # Convert image to a single-page PDF in memory
                    img = Image.open(source_file)

                    # Apply EXIF rotation so the image is stored upright
                    # Without this, images from cameras may appear sideways
                    img = ImageOps.exif_transpose(img)

                    # Convert to RGB (remove alpha) to avoid issues
                    if img.mode in ("RGBA", "LA"):
                        img = img.convert("RGB")

                    pdf_bytes = io.BytesIO()
                    img.save(pdf_bytes, format="PDF")
                    pdf_bytes.seek(0)

                    # Store stream reference
                    opened_streams.append(pdf_bytes)

                    # Open this temp PDF with pikepdf
                    temp_pdf = pikepdf.Pdf.open(pdf_bytes)

                    # Take the first (only) page
                    page = temp_pdf.pages[0]

                    # Apply editor rotation directly on the page
                    if page_state.rotation != 0:
                        page.Rotate = page_state.rotation

                    new_pdf.pages.append(page)

                    # Keep reference alive until save
                    if "temp_images" not in opened_pdfs:
                        opened_pdfs["temp_images"] = []  # type: ignore
                    opened_pdfs["temp_images"].append(temp_pdf)  # type: ignore

                except Exception as e:
                    logger.error(f"Failed to process image {source_file}: {e}")
                    continue

            else:
                # It's a PDF
                if source_file not in opened_pdfs:
                    try:
                        opened_pdfs[source_file] = pikepdf.open(source_file)
                    except Exception as e:
                        logger.error(f"Failed to open source PDF {source_file}: {e}")
                        continue

                src_pdf = opened_pdfs[source_file]

                # PageState.page_number is 1-indexed relative to the source file
                original_idx = page_state.page_number - 1

                if original_idx < 0 or original_idx >= len(src_pdf.pages):
                    logger.warning(f"Invalid page index {original_idx} for {source_file}")
                    continue

                src_page = src_pdf.pages[original_idx]

                # Resolve the EFFECTIVE rotation from the source page.
                # /Rotate can be inherited from parent Pages nodes and would
                # be lost when copying to a new PDF. We must read it BEFORE
                # appending and explicitly set it on the destination page.
                source_rotation = 0
                try:
                    # Try direct key first (most common case)
                    if "/Rotate" in src_page:
                        source_rotation = int(src_page["/Rotate"])
                    else:
                        # Check inherited rotation via pikepdf property
                        # Walk up the page tree to find inherited /Rotate
                        node = src_page.obj if hasattr(src_page, "obj") else src_page
                        while node is not None:
                            if "/Rotate" in node:
                                source_rotation = int(node["/Rotate"])
                                break
                            parent = node.get("/Parent")
                            node = parent if parent is not None else None
                except Exception:
                    source_rotation = 0

                # Copy page to new PDF
                new_pdf.pages.append(src_page)
                new_page = new_pdf.pages[-1]

                # Compute final rotation: source + editor
                final_rotation = (source_rotation + page_state.rotation) % 360

                # Always set /Rotate explicitly (ensures inherited values are
                # materialized and editor rotation is applied)
                if final_rotation != 0:
                    new_page.Rotate = final_rotation
                elif "/Rotate" in new_page:
                    # Remove stale direct /Rotate if final rotation is 0
                    del new_page["/Rotate"]

                if page_state.rotation != 0 or source_rotation != 0:
                    logger.info(
                        f"Page {page_state.source_file}:{page_state.page_number} "
                        f"rotation: source={source_rotation} + editor={page_state.rotation} "
                        f"= {final_rotation}"
                    )

        # Save the new PDF
        new_pdf.save(output_path)
        logger.info(f"Saved modified PDF to {output_path}")

        # Close all handles
        for key, handle in opened_pdfs.items():
            if key == "temp_images":
                for tmp in handle:  # type: ignore
                    tmp.close()
            else:
                handle.close()

        return True

    except ImportError:
        logger.error("pikepdf or PIL is not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to save PDF: {e}")
        return False
