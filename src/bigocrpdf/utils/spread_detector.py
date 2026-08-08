"""Spread Page Detector

Detects book spreads (two pages scanned as one image) and splits
OCR data into separate logical pages for export.

A spread is detected when OCR boxes cluster into two distinct groups
separated by a clear vertical gap near the horizontal center.
"""

from __future__ import annotations

from bigocrpdf.utils.logger import logger

# Minimum gap (as percentage of page width) between left and right columns
# to consider a page as a spread.  Typical book gutters are 5-15% of width.
_MIN_GAP_PCT = 4.0

# The split point must be within this range of the page center (percentage).
# E.g. 15 means the gap must be between 35% and 65% of page width.
_CENTER_TOLERANCE_PCT = 15.0


def detect_and_split_spreads(ocr_boxes: list) -> tuple[list, dict[int, tuple[int, int]]]:
    """Detect spread pages and split OCR boxes into logical pages.

    For each physical page, analyzes the X-coordinate distribution of boxes.
    If a clear vertical gap is found near the center, the page is split into
    two logical pages (left and right).

    Args:
        ocr_boxes: List of OCRBoxData objects with x (0-100%), page_num, etc.

    Returns:
        Tuple of:
        - New list of OCRBoxData with remapped page_num and adjusted x coordinates
        - Mapping dict: {original_page_num: (left_logical_num, right_logical_num)}
          Only contains entries for pages that were actually split.
    """
    if not ocr_boxes:
        return ocr_boxes, {}

    pages = _group_boxes_by_page(ocr_boxes)
    split_map = _build_spread_split_map(pages)

    if not split_map:
        return ocr_boxes, {}

    logger.info(
        f"Spread detection: {len(split_map)} spread pages found (pages {sorted(split_map.keys())})"
    )

    new_boxes = []
    logical_num = 1

    for orig_pn in sorted(pages.keys()):
        page_boxes = pages[orig_pn]

        if orig_pn in split_map:
            left_num, right_num = split_map[orig_pn]
            new_boxes.extend(_remap_split_page_boxes(page_boxes, left_num, right_num))
            logical_num = right_num + 1
        else:
            new_boxes.extend(_remap_unsplit_page_boxes(page_boxes, logical_num))
            logical_num += 1

    return new_boxes, split_map


def split_text_by_spreads(
    extracted_text: str,
    ocr_boxes: list,
    split_map: dict[int, tuple[int, int]],
) -> str:
    """Re-arrange extracted text splitting spreads into separate logical pages.

    Uses the OCR box positions to determine which text belongs to the
    left or right half of each spread page.

    Args:
        extracted_text: Original extracted text (page chunks separated by \\n\\n)
        ocr_boxes: Original (unsplit) OCR boxes
        split_map: Mapping from detect_and_split_spreads

    Returns:
        Reformatted text with spread pages split
    """
    if not split_map or not ocr_boxes:
        return extracted_text

    pages = _group_boxes_by_page(ocr_boxes)
    logical_pages: dict[int, list[str]] = {}
    logical_num = 1

    for orig_pn in sorted(pages.keys()):
        page_boxes = pages[orig_pn]

        if orig_pn in split_map:
            left_num, right_num = split_map[orig_pn]
            left_texts, right_texts = _split_page_texts(page_boxes)
            logical_pages[left_num] = left_texts
            logical_pages[right_num] = right_texts
            logical_num = right_num + 1
        else:
            logical_pages[logical_num] = _sorted_page_texts(page_boxes)
            logical_num += 1

    # Assemble final text
    parts = []
    for ln in sorted(logical_pages.keys()):
        texts = logical_pages[ln]
        if texts:
            parts.append("\n".join(texts))

    return "\n\n".join(parts)


def _group_boxes_by_page(ocr_boxes: list) -> dict[int, list]:
    pages: dict[int, list] = {}
    for box in ocr_boxes:
        page_num = getattr(box, "page_num", 1)
        pages.setdefault(page_num, []).append(box)
    return pages


def _build_spread_split_map(pages: dict[int, list]) -> dict[int, tuple[int, int]]:
    split_map: dict[int, tuple[int, int]] = {}
    logical_num = 1

    for orig_pn in sorted(pages.keys()):
        if _find_center_gap(pages[orig_pn]) is not None:
            split_map[orig_pn] = (logical_num, logical_num + 1)
            logical_num += 2
        else:
            logical_num += 1

    return split_map


def _remap_split_page_boxes(page_boxes: list, left_num: int, right_num: int) -> list:
    gap = _find_center_gap(page_boxes)
    if gap is None:
        return _remap_unsplit_page_boxes(page_boxes, left_num)

    return [_remap_box_for_split_page(box, gap, left_num, right_num) for box in page_boxes]


def _remap_box_for_split_page(box, gap: float, left_num: int, right_num: int):
    new_box = _clone_box(box)
    center_x = box.x + box.width / 2

    if center_x < gap:
        scale = 100.0 / gap if gap > 0 else 1.0
        new_box.page_num = left_num
        new_box.x = box.x * scale
        new_box.width = box.width * scale
        return new_box

    right_width = 100.0 - gap
    scale = 100.0 / right_width if right_width > 0 else 1.0
    new_box.page_num = right_num
    new_box.x = (box.x - gap) * scale
    new_box.width = box.width * scale
    return new_box


def _remap_unsplit_page_boxes(page_boxes: list, logical_num: int) -> list:
    new_boxes = []
    for box in page_boxes:
        new_box = _clone_box(box)
        new_box.page_num = logical_num
        new_boxes.append(new_box)
    return new_boxes


def _split_page_texts(page_boxes: list) -> tuple[list[str], list[str]]:
    gap = _find_center_gap(page_boxes)
    if gap is None:
        return _sorted_page_texts(page_boxes), []

    left_items = []
    right_items = []
    for box in page_boxes:
        target = left_items if box.x + box.width / 2 < gap else right_items
        target.append((float(getattr(box, "y", 0)), str(getattr(box, "text", ""))))

    return _texts_sorted_by_y(left_items), _texts_sorted_by_y(right_items)


def _sorted_page_texts(page_boxes: list) -> list[str]:
    items = [(float(getattr(box, "y", 0)), str(getattr(box, "text", ""))) for box in page_boxes]
    return _texts_sorted_by_y(items)


def _texts_sorted_by_y(items: list[tuple[float, str]]) -> list[str]:
    items.sort(key=lambda item: item[0])
    return [text for _, text in items]


def _find_center_gap(boxes: list) -> float | None:
    """Find a clear vertical gap near the center of the page.

    Analyzes the X-coordinate distribution of OCR boxes to find a gap
    that separates left and right halves.

    Args:
        boxes: OCR boxes for a single page (x in 0-100%)

    Returns:
        Gap center position (percentage), or None if no spread detected
    """
    if len(boxes) < 4:
        return None

    intervals = sorted((box.x, box.x + box.width) for box in boxes)
    centers = sorted((start + end) / 2 for start, end in intervals)

    occupied = []
    for start, end in intervals:
        if occupied and start <= occupied[-1][1]:
            occupied[-1] = (occupied[-1][0], max(occupied[-1][1], end))
        else:
            occupied.append((start, end))

    best_gap = 0.0
    best_pos = 0.0
    center_lo = 50.0 - _CENTER_TOLERANCE_PCT
    center_hi = 50.0 + _CENTER_TOLERANCE_PCT

    for left, right in zip(occupied, occupied[1:], strict=False):
        gap_size = right[0] - left[1]
        gap_mid = (left[1] + right[0]) / 2

        if center_lo <= gap_mid <= center_hi and gap_size > best_gap:
            best_gap = gap_size
            best_pos = gap_mid

    if best_gap < _MIN_GAP_PCT:
        return None

    left_count = sum(1 for center in centers if center < best_pos)
    right_count = len(centers) - left_count
    if left_count < 2 or right_count < 2:
        return None

    return best_pos


def crop_spread_images(
    image_paths: list[str],
    ocr_boxes: list,
    split_map: dict[int, tuple[int, int]],
) -> list[str]:
    """Crop spread images into left/right halves matching logical pages.

    For each physical page in split_map, finds the gap position from OCR
    boxes and crops the corresponding image into two halves.

    Args:
        image_paths: Physical page image paths (sorted by page order)
        ocr_boxes: Original (unsplit) OCR boxes
        split_map: {physical_page_num: (left_logical, right_logical)}

    Returns:
        New list of image paths, one per logical page, in order.
        Non-spread pages keep their original image path.
    """
    import os

    from PIL import Image as PILImage

    if not split_map or not image_paths:
        return image_paths

    # Group original boxes by page to find gap positions
    pages = _group_boxes_by_page(ocr_boxes)

    logical_images: dict[int, str] = {}
    logical_num = 1

    for phys_idx, img_path in enumerate(image_paths):
        phys_page = phys_idx + 1

        if phys_page in split_map:
            left_num, right_num = split_map[phys_page]
            page_boxes = pages.get(phys_page, [])
            gap_pct = _find_center_gap(page_boxes) if page_boxes else None

            if gap_pct is None:
                gap_pct = 50.0

            try:
                with PILImage.open(img_path) as img:
                    w, h = img.size
                    gap_px = int(w * gap_pct / 100.0)

                    img_dir = os.path.dirname(img_path)
                    base = os.path.splitext(os.path.basename(img_path))[0]

                    left_path = os.path.join(img_dir, f"{base}_L{left_num}.png")
                    img.crop((0, 0, gap_px, h)).save(left_path)
                    logical_images[left_num] = left_path

                    right_path = os.path.join(img_dir, f"{base}_R{right_num}.png")
                    img.crop((gap_px, 0, w, h)).save(right_path)
                    logical_images[right_num] = right_path

            except Exception as e:
                logger.warning(f"Failed to crop spread image page {phys_page}: {e}")
                logical_images[left_num] = img_path
                logical_images[right_num] = img_path

            logical_num = right_num + 1
        else:
            logical_images[logical_num] = img_path
            logical_num += 1

    return [logical_images[k] for k in sorted(logical_images.keys())]


def _clone_box(box):
    """Create a shallow copy of an OCRBoxData preserving all fields."""
    from dataclasses import fields

    kwargs = {f.name: getattr(box, f.name) for f in fields(box)}
    return type(box)(**kwargs)
