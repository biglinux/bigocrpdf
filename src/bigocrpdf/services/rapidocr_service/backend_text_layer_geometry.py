"""Geometry helpers for rendering OCR text layers."""

from pathlib import Path

import pikepdf

from bigocrpdf.utils.logger import logger

ImageRect = tuple[float, float, float, float]


def _extract_image_rect_from_page(pdf_path: Path, page_num: int) -> ImageRect | None:
    """Extract the display rectangle of the main image on a PDF page.

    Parses the page's content stream, tracks the CTM through q/Q/cm
    operators, and finds the bounding box where the largest image
    XObject is drawn.

    Returns (x, y, width, height) in PDF points with origin at
    the page's bottom-left, or None if no image is found.
    """
    try:
        with pikepdf.open(pdf_path) as pdf:
            if page_num < 1 or page_num > len(pdf.pages):
                return None
            return _extract_image_rect(pdf.pages[page_num - 1], page_num)

    except Exception as exc:
        logger.debug(f"Failed to extract image rect for page {page_num}: {exc}")
    return None


def _extract_image_rects_from_pdf(
    pdf_path: Path,
    page_count: int | None = None,
) -> list[ImageRect | None]:
    """Extract main-image rectangles in page order with one PDF open.

    ``page_count`` preserves the caller's expected output length. Missing pages
    and per-page parse failures produce ``None``, matching the unit wrapper.
    """
    expected_pages = max(0, page_count) if page_count is not None else None
    if expected_pages == 0:
        return []

    try:
        with pikepdf.open(pdf_path) as pdf:
            result_count = len(pdf.pages) if expected_pages is None else expected_pages
            rectangles: list[ImageRect | None] = []
            for page_index in range(result_count):
                page_num = page_index + 1
                if page_index >= len(pdf.pages):
                    rectangles.append(None)
                    continue
                try:
                    rectangles.append(_extract_image_rect(pdf.pages[page_index], page_num))
                except Exception as exc:
                    logger.debug(f"Failed to extract image rect for page {page_num}: {exc}")
                    rectangles.append(None)
            return rectangles
    except Exception as exc:
        logger.debug(f"Failed to open PDF for image-rect extraction: {exc}")
        return [None] * (expected_pages or 0)


def _extract_image_rect(page: pikepdf.Page, page_num: int) -> ImageRect | None:
    resources = page.get("/Resources")
    if not resources:
        return None
    xobjects = resources.get("/XObject")
    if not xobjects:
        return None

    image_areas = _image_xobject_areas(xobjects)
    if not image_areas:
        return None

    target_name = max(image_areas, key=lambda name: image_areas[name])
    return _tracked_image_rect(page, target_name, page_num)


def _image_xobject_areas(xobjects) -> dict[str, int]:
    image_areas: dict[str, int] = {}
    for name in xobjects.keys():
        try:
            xobj = xobjects[name]
            if str(xobj.get("/Subtype", "")) == "/Image":
                image_areas[str(name)] = int(xobj.get("/Width", 0)) * int(xobj.get("/Height", 0))
        except Exception:
            continue
    return image_areas


def _tracked_image_rect(
    page: pikepdf.Page,
    target_name: str,
    page_num: int,
) -> ImageRect | None:
    ctm = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    ctm_stack: list[list[float]] = []
    largest_rect: ImageRect | None = None

    for operands, operator in pikepdf.parse_content_stream(page):
        op = str(operator)
        if op == "q":
            ctm_stack.append(ctm[:])
        elif op == "Q" and ctm_stack:
            ctm = ctm_stack.pop()
        elif op == "cm" and len(operands) >= 6:
            ctm = _matrix_multiply([float(operands[i]) for i in range(6)], ctm)
        elif op == "Do" and operands and str(operands[0]) == target_name:
            rect = _image_rect_from_ctm(page, ctm, page_num)
            if rect is not None and (
                largest_rect is None or rect[2] * rect[3] > largest_rect[2] * largest_rect[3]
            ):
                largest_rect = rect
    return largest_rect


def _matrix_multiply(m1: list[float], m2: list[float]) -> list[float]:
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return [
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2,
        e1 * b2 + f1 * d2 + f2,
    ]


def _image_rect_from_ctm(
    page: pikepdf.Page,
    ctm: list[float],
    page_num: int,
) -> ImageRect | None:
    a, b, c, d, e, f = ctm
    xs = [e, a + e, c + e, a + c + e]
    ys = [f, b + f, d + f, b + d + f]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width, height = x_max - x_min, y_max - y_min
    if _image_rect_is_decorative(page, width, height, page_num):
        return None
    return x_min, y_min, width, height


def _image_rect_is_decorative(
    page: pikepdf.Page, width: float, height: float, page_num: int
) -> bool:
    mediabox = page.get("/MediaBox")
    if not mediabox:
        return False
    page_width = float(mediabox[2]) - float(mediabox[0])
    page_height = float(mediabox[3]) - float(mediabox[1])
    if page_width <= 0 or page_height <= 0:
        return False

    coverage = (width * height) / (page_width * page_height)
    if coverage >= 0.25:
        return False
    logger.debug(
        f"Page {page_num}: largest image {width:.1f}×{height:.1f} "
        f"covers only {coverage:.1%} of page — ignoring"
    )
    return True


def _processed_page_dimensions(
    result: dict,
    page_info: dict,
    proc_w: int,
    proc_h: int,
) -> tuple[float, float]:
    mediabox = page_info.get("mediabox")
    if not mediabox:
        return float(proc_w), float(proc_h)

    mediabox_width = float(mediabox[2]) - float(mediabox[0])
    mediabox_height = float(mediabox[3]) - float(mediabox[1])
    page_rotation = page_info.get("rotation", 0)
    prerotated = result.get("image_prerotated", False)
    orientation_angle = result.get("orientation_angle", 0)
    should_swap = (prerotated and page_rotation in (90, 270)) or orientation_angle in (90, 270)
    if should_swap:
        return mediabox_height, mediabox_width
    return mediabox_width, mediabox_height
