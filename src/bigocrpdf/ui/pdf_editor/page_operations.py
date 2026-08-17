"""
BigOcrPdf - PDF Page Operations

Functions for manipulating PDF pages: rotation, deletion, reordering, and OCR selection.
Uses pikepdf for PDF manipulation operations.
"""

import math
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from bigocrpdf.ui.pdf_editor.page_model import PageState, PDFDocument
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
        return False

    try:
        changed_pages = 0
        for idx in dict.fromkeys(page_indices):
            page = doc.get_page_by_position(idx)
            if page:
                page.rotate(degrees)
                changed_pages += 1

        if not changed_pages:
            return False
        doc.mark_modified()
        logger.info(f"Rotated {changed_pages} page(s) by {degrees}°")
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
            for idx in sorted(set(page_indices), reverse=True):
                page = doc.get_page_by_position(idx)
                if page:
                    pages_to_remove.append(page)

            if not pages_to_remove:
                return False
            changed_pages = len(pages_to_remove)
            for page in pages_to_remove:
                doc.pages.remove(page)

            # Update positions
            doc.update_positions()
        else:
            # Soft delete: mark pages as deleted
            changed_pages = 0
            for idx in dict.fromkeys(page_indices):
                page = doc.get_page_by_position(idx)
                if page and not page.deleted:
                    page.deleted = True
                    changed_pages += 1
            if not changed_pages:
                return False

        doc.mark_modified()
        logger.info(f"Deleted {changed_pages} page(s) (hard={hard_delete})")
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
        changed_pages = 0
        for idx in dict.fromkeys(page_indices):
            page = doc.get_page_by_position(idx)
            if page and page.included_for_ocr != selected:
                page.included_for_ocr = selected
                changed_pages += 1

        if not changed_pages:
            return False
        doc.mark_modified()
        logger.info(f"Set OCR selection to {selected} for {changed_pages} page(s)")
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
        changed_pages = 0
        for page in doc.pages:
            if not page.deleted and not page.included_for_ocr:
                page.included_for_ocr = True
                changed_pages += 1

        if not changed_pages:
            return False
        doc.mark_modified()
        logger.info("Marked every page for OCR")
        return True

    except Exception as e:
        logger.error(f"Failed to mark every page for OCR: {e}")
        return False


def deselect_all_for_ocr(doc: PDFDocument) -> bool:
    """Deselect all pages from OCR.

    Args:
        doc: The PDFDocument to modify

    Returns:
        True if successful
    """
    try:
        changed_pages = 0
        for page in doc.pages:
            if page.included_for_ocr:
                page.included_for_ocr = False
                changed_pages += 1

        if not changed_pages:
            return False
        doc.mark_modified()
        logger.info("Cleared OCR selection on every page")
        return True

    except Exception as e:
        logger.error(f"Failed to clear OCR selection: {e}")
        return False


def _add_image_page(source_file, page_state, new_pdf, resources: ExitStack):
    """Convert an image file to a PDF page and append it to new_pdf."""
    import io

    import pikepdf
    from PIL import Image, ImageOps

    source_image = resources.enter_context(Image.open(source_file))
    img = ImageOps.exif_transpose(source_image)
    if img is not source_image:
        resources.callback(img.close)
    if img.mode in ("RGBA", "LA"):
        img = img.convert("RGB")
        resources.callback(img.close)

    pdf_bytes = resources.enter_context(io.BytesIO())
    img.save(pdf_bytes, format="PDF")
    pdf_bytes.seek(0)

    temp_pdf = resources.enter_context(pikepdf.Pdf.open(pdf_bytes))
    page = temp_pdf.pages[0]
    # Force the page box to the image's native pixel size (1 px = 1 pt).
    # PIL sizes the page from the image DPI, so images of the same resolution
    # but differing embedded DPI would otherwise yield inconsistent physical
    # page sizes — which mobile viewers (honoring per-page MediaBox) render at
    # different scales. Pinning to native pixels keeps every image page uniform.
    page.MediaBox = [0, 0, img.width, img.height]
    final_rotation = page_state.rotation
    if final_rotation != 0:
        page.Rotate = final_rotation
    new_pdf.pages.append(page)
    _apply_page_flips(new_pdf, new_pdf.pages[-1], page_state, final_rotation)


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


def _page_flip_matrix(
    page_box: list[float],
    rotation: int,
    *,
    horizontal: bool,
    vertical: bool,
) -> tuple[float, float, float, float, float, float]:
    """Return a native-space matrix for flips expressed on display axes."""
    if len(page_box) != 4:
        raise ValueError("PDF page box must contain four coordinates")
    x0, y0, x1, y1 = (float(value) for value in page_box)
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        raise ValueError("PDF page box coordinates must be finite")
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        raise ValueError("PDF page has an invalid page box")
    rotation %= 360
    if rotation not in (0, 90, 180, 270):
        raise ValueError("PDF page rotation must be a multiple of 90 degrees")
    quarter_turn = rotation in (90, 270)
    native_horizontal = vertical if quarter_turn else horizontal
    native_vertical = horizontal if quarter_turn else vertical
    return (
        -1.0 if native_horizontal else 1.0,
        0.0,
        0.0,
        -1.0 if native_vertical else 1.0,
        x0 + x1 if native_horizontal else 0.0,
        y0 + y1 if native_vertical else 0.0,
    )


def _matrix_point(
    matrix: tuple[float, ...],
    x: float,
    y: float,
) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    return a * x + c * y + e, b * x + d * y + f


def _matrix_bounds(
    matrix: tuple[float, ...],
    box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    corners = (
        _matrix_point(matrix, x0, y0),
        _matrix_point(matrix, x0, y1),
        _matrix_point(matrix, x1, y0),
        _matrix_point(matrix, x1, y1),
    )
    return (
        min(x for x, _y in corners),
        min(y for _x, y in corners),
        max(x for x, _y in corners),
        max(y for _x, y in corners),
    )


def _compose_matrices(
    outer: tuple[float, ...],
    inner: tuple[float, ...],
) -> tuple[float, float, float, float, float, float]:
    """Return the affine transform ``outer ∘ inner``."""
    oa, ob, oc, od, oe, of = outer
    ia, ib, ic, id_, ie, if_ = inner
    return (
        oa * ia + oc * ib,
        ob * ia + od * ib,
        oa * ic + oc * id_,
        ob * ic + od * id_,
        oa * ie + oc * if_ + oe,
        ob * ie + od * if_ + of,
    )


def _normalized_box(values) -> tuple[float, float, float, float] | None:
    if values is None or len(values) != 4:
        return None
    try:
        raw_x0, raw_y0, raw_x1, raw_y1 = (float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (raw_x0, raw_y0, raw_x1, raw_y1)):
        return None
    x0, x1 = sorted((raw_x0, raw_x1))
    y0, y1 = sorted((raw_y0, raw_y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _appearance_matrix(stream) -> tuple[float, float, float, float, float, float] | None:
    raw_matrix = stream.get("/Matrix", [1, 0, 0, 1, 0, 0])
    if len(raw_matrix) != 6:
        return None
    try:
        matrix = tuple(float(value) for value in raw_matrix)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in matrix):
        return None
    return matrix  # type: ignore[return-value]


def _wrap_appearance_stream(
    new_pdf,
    stream,
    old_rect: tuple[float, float, float, float],
    new_rect: tuple[float, float, float, float],
    page_matrix: tuple[float, ...],
):
    """Create an independent Form that mirrors one annotation appearance."""
    import pikepdf

    bbox = _normalized_box(stream.get("/BBox"))
    appearance_matrix = _appearance_matrix(stream)
    if bbox is None or appearance_matrix is None:
        return stream
    transformed_bbox = _matrix_bounds(appearance_matrix, bbox)
    bbox_width = transformed_bbox[2] - transformed_bbox[0]
    bbox_height = transformed_bbox[3] - transformed_bbox[1]
    rect_width = old_rect[2] - old_rect[0]
    rect_height = old_rect[3] - old_rect[1]
    if bbox_width <= 0 or bbox_height <= 0:
        return stream
    scale_x = rect_width / bbox_width
    scale_y = rect_height / bbox_height
    fit_matrix = (
        scale_x,
        0.0,
        0.0,
        scale_y,
        old_rect[0] - scale_x * transformed_bbox[0],
        old_rect[1] - scale_y * transformed_bbox[1],
    )
    wrapper_matrix = _compose_matrices(page_matrix, fit_matrix)
    content = pikepdf.unparse_content_stream(
        [
            pikepdf.ContentStreamInstruction([], pikepdf.Operator("q")),
            pikepdf.ContentStreamInstruction(
                [pikepdf.Real(value) for value in wrapper_matrix],
                pikepdf.Operator("cm"),
            ),
            pikepdf.ContentStreamInstruction(
                [pikepdf.Name("/Original")],
                pikepdf.Operator("Do"),
            ),
            pikepdf.ContentStreamInstruction([], pikepdf.Operator("Q")),
        ]
    )
    wrapper = new_pdf.make_stream(content)
    wrapper.Type = pikepdf.Name("/XObject")
    wrapper.Subtype = pikepdf.Name("/Form")
    wrapper.FormType = 1
    wrapper.BBox = pikepdf.Array(new_rect)
    wrapper.Matrix = pikepdf.Array([1, 0, 0, 1, 0, 0])
    wrapper.Resources = pikepdf.Dictionary(
        {
            "/XObject": pikepdf.Dictionary(
                {
                    "/Original": stream,
                }
            )
        }
    )
    return wrapper


def _wrap_appearance_value(
    new_pdf,
    value,
    old_rect: tuple[float, float, float, float],
    new_rect: tuple[float, float, float, float],
    page_matrix: tuple[float, ...],
):
    import pikepdf

    if isinstance(value, pikepdf.Stream):
        return _wrap_appearance_stream(
            new_pdf,
            value,
            old_rect,
            new_rect,
            page_matrix,
        )
    if isinstance(value, pikepdf.Dictionary):
        wrapped = pikepdf.Dictionary()
        for key, child in value.items():
            wrapped[key] = _wrap_appearance_value(
                new_pdf,
                child,
                old_rect,
                new_rect,
                page_matrix,
            )
        return wrapped
    return value


def _transform_annotation_appearances(
    new_pdf,
    annotation,
    old_rect: tuple[float, float, float, float],
    new_rect: tuple[float, float, float, float],
    matrix: tuple[float, ...],
) -> None:
    import pikepdf

    appearances = annotation.get("/AP")
    if not isinstance(appearances, pikepdf.Dictionary):
        return
    wrapped_appearances = pikepdf.Dictionary()
    for key, value in appearances.items():
        if str(key) in {"/N", "/R", "/D"}:
            value = _wrap_appearance_value(
                new_pdf,
                value,
                old_rect,
                new_rect,
                matrix,
            )
        wrapped_appearances[key] = value
    annotation.AP = wrapped_appearances


def _transform_annotation_geometry(new_pdf, page, matrix: tuple[float, ...]) -> None:
    """Keep common annotation hit regions aligned with transformed content."""
    import pikepdf

    def point(x: float, y: float) -> tuple[float, float]:
        return _matrix_point(matrix, x, y)

    def transform_pairs(values) -> pikepdf.Array:
        transformed: list[float] = []
        for index in range(0, len(values), 2):
            x, y = point(float(values[index]), float(values[index + 1]))
            transformed.extend((x, y))
        return pikepdf.Array(transformed)

    for annotation in page.get("/Annots", []):
        rect = annotation.get("/Rect")
        old_rect = _normalized_box(rect)
        if old_rect is not None:
            new_rect = _matrix_bounds(matrix, old_rect)
            _transform_annotation_appearances(
                new_pdf,
                annotation,
                old_rect,
                new_rect,
                matrix,
            )
            annotation.Rect = pikepdf.Array(new_rect)
        for key in ("/QuadPoints", "/Vertices", "/L"):
            values = annotation.get(key)
            if values is not None and len(values) % 2 == 0:
                annotation[key] = transform_pairs(values)
        ink_lists = annotation.get("/InkList")
        if ink_lists is not None:
            annotation.InkList = pikepdf.Array(
                [
                    transform_pairs(stroke) if len(stroke) % 2 == 0 else pikepdf.Array(stroke)
                    for stroke in ink_lists
                ]
            )


def _apply_page_flips(new_pdf, page, page_state: PageState, final_rotation: int) -> None:
    """Mirror page contents without rasterizing text, vectors, or images."""
    if not page_state.flip_horizontal and not page_state.flip_vertical:
        return

    import pikepdf

    matrix = _page_flip_matrix(
        list(page.cropbox),
        final_rotation,
        horizontal=page_state.flip_horizontal,
        vertical=page_state.flip_vertical,
    )
    prefix = pikepdf.unparse_content_stream(
        [
            pikepdf.ContentStreamInstruction([], pikepdf.Operator("q")),
            pikepdf.ContentStreamInstruction(
                [pikepdf.Real(value) for value in matrix],
                pikepdf.Operator("cm"),
            ),
        ]
    )
    suffix = pikepdf.unparse_content_stream(
        [pikepdf.ContentStreamInstruction([], pikepdf.Operator("Q"))]
    )
    page.contents_add(prefix, prepend=True)
    page.contents_add(suffix)
    _transform_annotation_geometry(new_pdf, page, matrix)


def _add_pdf_page(source_file, page_state, new_pdf, opened_pdfs, resources: ExitStack):
    """Copy a PDF page to new_pdf with rotation handling."""
    import pikepdf

    if source_file not in opened_pdfs:
        opened_pdfs[source_file] = resources.enter_context(pikepdf.open(source_file))

    src_pdf = opened_pdfs[source_file]
    original_idx = page_state.page_number - 1

    if original_idx < 0 or original_idx >= len(src_pdf.pages):
        raise IndexError(f"Invalid page index {original_idx} for {source_file}")

    src_page = src_pdf.pages[original_idx]
    source_rotation = _resolve_source_rotation(src_page)

    new_pdf.pages.append(src_page)
    new_page = new_pdf.pages[-1]

    final_rotation = (source_rotation + page_state.rotation) % 360
    if final_rotation != 0:
        new_page.Rotate = final_rotation
    elif "/Rotate" in new_page:
        del new_page["/Rotate"]

    _apply_page_flips(new_pdf, new_page, page_state, final_rotation)

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
        import pikepdf

        with ExitStack() as resources:
            new_pdf = pikepdf.Pdf.new()
            resources.callback(new_pdf.close)
            opened_pdfs: dict[str, pikepdf.Pdf] = {}
            active_pages = [page for page in doc.get_active_pages() if page.included_for_ocr]

            _append_active_pages_to_pdf(
                active_pages,
                doc.path,
                new_pdf,
                opened_pdfs,
                resources,
            )

            # Apply the viewer page layout (/PageLayout) from the shared setting.
            from bigocrpdf.utils.config_manager import get_config_manager
            from bigocrpdf.utils.pdf_utils import set_root_page_layout

            layout = get_config_manager().get("output.page_layout", "default")
            set_root_page_layout(new_pdf, layout)

            new_pdf.save(output_path)
        logger.info(f"Saved modified PDF to {output_path}")
        return True

    except ImportError:
        logger.error("pikepdf or PIL is not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to save PDF: {e}")
        return False


def apply_changes_to_pdf_atomically(
    doc: PDFDocument | None,
    output_path: str | Path,
) -> bool:
    """Materialize editor changes privately, then publish the complete PDF."""
    if doc is None:
        return False
    destination = Path(output_path)
    try:
        with TemporaryDirectory(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        ) as staging_name:
            staged_path = Path(staging_name) / destination.name
            if not apply_changes_to_pdf(doc, str(staged_path)):
                return False
            from bigocrpdf.utils.durable_writes import publish_file_atomically

            publish_file_atomically(
                staged_path,
                destination,
                overwrite=True,
            )
        return True
    except (OSError, ValueError) as error:
        logger.error("Failed to publish edited PDF: %s", error)
        return False


def _append_active_pages_to_pdf(
    active_pages: list[PageState],
    doc_path: str | None,
    new_pdf,
    opened_pdfs: dict,
    resources: ExitStack,
) -> None:
    for page_state in active_pages:
        source_file = page_state.source_file or doc_path
        if not source_file:
            raise ValueError(f"Page {page_state.page_number} has no source file")
        _append_source_page_to_pdf(source_file, page_state, new_pdf, opened_pdfs, resources)


def _append_source_page_to_pdf(
    source_file: str,
    page_state: PageState,
    new_pdf,
    opened_pdfs: dict,
    resources: ExitStack,
) -> None:
    if source_file.lower().endswith(_IMAGE_EXTENSIONS):
        _add_image_page(source_file, page_state, new_pdf, resources)
    else:
        _add_pdf_page(source_file, page_state, new_pdf, opened_pdfs, resources)
