"""Geometry helpers for rendering OCR text layers."""

from pathlib import Path

import pikepdf

from bigocrpdf.services.ocr.pdf_image_analysis import _multiply_ctm
from bigocrpdf.utils.logger import logger

ImageRect = tuple[float, float, float, float]
# (mirrored, degrees): mirror left-right, then turn clockwise.
ImageTurn = tuple[bool, int]
ImagePlacement = tuple[ImageRect, ImageTurn]
# (pixel count, image object id, matrix the image is drawn with)
_ImageDraw = tuple[int, tuple[int, int], list[float]]

_IDENTITY = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def _extract_image_placements_from_pdf(
    pdf_path: Path,
    page_count: int | None = None,
) -> list[ImagePlacement | None]:
    """Extract where and how each page draws its main image, in page order,
    with one PDF open.

    The rectangle is (x, y, width, height) in PDF points from the page's
    bottom-left corner, where the largest image XObject is drawn. The turn
    maps the image's stored pixels to how that drawing shows them, before
    /Rotate.
    ``page_count`` fixes the output length; missing pages, pages without an
    image and per-page parse failures produce ``None``.
    """
    expected_pages = max(0, page_count) if page_count is not None else None
    if expected_pages == 0:
        return []

    try:
        with pikepdf.open(pdf_path) as pdf:
            result_count = len(pdf.pages) if expected_pages is None else expected_pages
            placements: list[ImagePlacement | None] = []
            for page_index in range(result_count):
                page_num = page_index + 1
                if page_index >= len(pdf.pages):
                    placements.append(None)
                    continue
                try:
                    placements.append(_extract_image_placement(pdf.pages[page_index], page_num))
                except Exception as exc:
                    logger.debug(f"Failed to extract image placement for page {page_num}: {exc}")
                    placements.append(None)
            return placements
    except Exception as exc:
        logger.debug(f"Failed to open PDF for image placement extraction: {exc}")
        return [None] * (expected_pages or 0)


def _extract_image_placement(page: pikepdf.Page, page_num: int) -> ImagePlacement | None:
    """Use the image with the most pixels, as the extractor does, at the
    largest rectangle it is drawn into."""
    draws: list[_ImageDraw] = []
    _collect_image_draws(page, page.get("/Resources"), _IDENTITY, draws, set())
    if not draws:
        return None

    target = max(draws, key=lambda draw: draw[0])[1]
    largest: ImagePlacement | None = None
    for _, image_id, ctm in draws:
        if image_id != target:
            continue
        rect = _image_rect_from_ctm(page, ctm, page_num)
        if rect is not None and (
            largest is None or rect[2] * rect[3] > largest[0][2] * largest[0][3]
        ):
            largest = rect, _image_turn_from_ctm(ctm)
    return largest


def _collect_image_draws(
    content: pikepdf.Page | pikepdf.Object,
    resources: pikepdf.Object | None,
    ctm: list[float],
    draws: list[_ImageDraw],
    read_forms: set[tuple[int, int]],
) -> None:
    """Record every image drawn by ``content`` and the forms it draws.

    Each form is read once per page: that ends self-drawing forms and bounds
    the work a crafted PDF can demand.
    """
    xobjects = resources.get("/XObject") if resources is not None else None
    ctm_stack: list[list[float]] = []
    for operands, operator in pikepdf.parse_content_stream(content):
        op = str(operator)
        if op == "q":
            ctm_stack.append(ctm)
        elif op == "Q" and ctm_stack:
            ctm = ctm_stack.pop()
        elif op == "cm" and len(operands) >= 6:
            ctm = _multiply_ctm([float(operands[i]) for i in range(6)], ctm)
        elif op == "Do" and operands and xobjects is not None:
            xobject = xobjects.get(str(operands[0]))
            if not isinstance(xobject, pikepdf.Stream):
                continue
            subtype = str(xobject.get("/Subtype", ""))
            if subtype == "/Image":
                pixels = int(xobject.get("/Width", 0)) * int(xobject.get("/Height", 0))
                draws.append((pixels, xobject.objgen, ctm))
            elif subtype == "/Form" and xobject.objgen not in read_forms:
                read_forms.add(xobject.objgen)
                matrix = [float(value) for value in xobject.get("/Matrix", _IDENTITY)]
                form_resources = xobject.get("/Resources", resources)
                _collect_image_draws(
                    xobject, form_resources, _multiply_ctm(matrix, ctm), draws, read_forms
                )


def _image_turn_from_ctm(ctm: list[float]) -> ImageTurn:
    """The unit square puts the first stored row at its top, so a negative
    scale or a quarter-turn in the matrix shows the pixels turned or mirrored
    while pdfimages extracts them as stored."""
    a, b, c, d = ctm[:4]
    mirrored = a * d - b * c < 0
    # Where the image's top-to-bottom axis points on the page, y growing down.
    down_x, down_y = -c, d
    if abs(down_y) >= abs(down_x):
        return mirrored, 0 if down_y >= 0 else 180
    return mirrored, 90 if down_x < 0 else 270


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
