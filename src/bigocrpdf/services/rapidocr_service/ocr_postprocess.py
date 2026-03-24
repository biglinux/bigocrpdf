"""
OCR post-processing utilities.

Functions for fixing overlapping text regions, merging nearby text segments,
refining low-confidence detections, and combining OCR results into coherent
reading order.
"""

import logging
import os
import tempfile
from collections.abc import Callable

import cv2
import numpy as np

from bigocrpdf.services.rapidocr_service.config import OCRResult

logger = logging.getLogger(__name__)

# --- Refinement constants ---
# Regions below this confidence are candidates for re-OCR refinement.
_REFINE_CONFIDENCE_THRESHOLD = 0.7
# A detection box must be taller than this multiple of the median box height
# to be considered "oversized" (likely spanning multiple text lines).
_REFINE_HEIGHT_RATIO = 1.5
# Minimum box width in pixels to consider for refinement.  Narrow boxes
# (icons, margin artefacts, etc.) are skipped.
_REFINE_MIN_BOX_WIDTH = 200
# Pixel padding around the detected box when cropping for re-OCR.
_REFINE_CROP_PADDING = 30
# Minimum average confidence for re-OCR results to replace the original.
_REFINE_MIN_REPLACEMENT_SCORE = 0.5


def fix_vertical_overlaps(results: list[OCRResult]) -> list[OCRResult]:
    """Fix vertical overlaps between text lines by adjusting their bounding boxes.

    Robustly handles stacking while ignoring crossing (vertical vs horizontal) text.

    Args:
        results: List of OCR results to fix

    Returns:
        List of OCR results with adjusted bounding boxes
    """
    if not results:
        return results

    # Sort by vertical position (Top edge)
    sorted_res = sorted(results, key=lambda r: min(p[1] for p in r.box))

    for i in range(1, len(sorted_res)):
        curr = sorted_res[i]
        prev = sorted_res[i - 1]

        # Get dimensions
        prev_ys = [p[1] for p in prev.box]
        curr_ys = [p[1] for p in curr.box]
        prev_xs = [p[0] for p in prev.box]
        curr_xs = [p[0] for p in curr.box]

        prev_top, prev_bottom = min(prev_ys), max(prev_ys)
        curr_top, curr_bottom = min(curr_ys), max(curr_ys)
        prev_left, prev_right = min(prev_xs), max(prev_xs)
        curr_left, curr_right = min(curr_xs), max(curr_xs)

        prev_h = prev_bottom - prev_top
        prev_w = prev_right - prev_left
        curr_h = curr_bottom - curr_top
        curr_w = curr_right - curr_left

        # Aspect Ratio check (Horizontal lines usually have W > H)
        prev_is_vert = prev_h > prev_w
        curr_is_vert = curr_h > curr_w

        if prev_is_vert != curr_is_vert:
            # One vertical, one horizontal -> They are crossing, not stacking.
            continue

        # Check vertical overlap
        if curr_top < prev_bottom:
            # Check horizontal overlap
            x_overlap_start = max(prev_left, curr_left)
            x_overlap_end = min(prev_right, curr_right)

            if x_overlap_end > x_overlap_start:
                overlap_amount = prev_bottom - curr_top

                # Safety check: Don't consume the entire line.
                min_h = min(prev_h, curr_h)
                if min_h > 0 and (overlap_amount / min_h) > 0.8:
                    continue

                # Calculate new split point
                mid_y = (prev_bottom + curr_top) / 2

                # Clamp mid_y to ensure we don't invert the box
                mid_y = max(prev_top + 1, min(mid_y, curr_bottom - 1))

                # Apply adjustment
                new_prev_box = [[p[0], min(p[1], mid_y)] for p in prev.box]
                new_curr_box = [[p[0], max(p[1], mid_y)] for p in curr.box]

                sorted_res[i - 1] = OCRResult(prev.text, new_prev_box, prev.confidence)
                sorted_res[i] = OCRResult(curr.text, new_curr_box, curr.confidence)

    return sorted_res


def _find_refine_candidates(
    boxes: list, txts: list[str], scores: list[float]
) -> tuple[list[int], float] | None:
    """Return (problem_indices, height_threshold) or None if nothing to refine."""
    n = len(boxes)
    if n == 0:
        return None

    heights = np.empty(n, dtype=np.float64)
    widths = np.empty(n, dtype=np.float64)
    for i, box in enumerate(boxes):
        pts = np.asarray(box)
        heights[i] = pts[:, 1].max() - pts[:, 1].min()
        widths[i] = pts[:, 0].max() - pts[:, 0].min()

    median_h = float(np.median(heights))
    if median_h <= 0:
        return None

    height_threshold = median_h * _REFINE_HEIGHT_RATIO
    problem_indices = []
    for i in range(n):
        if scores[i] >= _REFINE_CONFIDENCE_THRESHOLD:
            continue
        if widths[i] < _REFINE_MIN_BOX_WIDTH:
            continue
        if heights[i] > height_threshold or len(txts[i]) < widths[i] / 30:
            problem_indices.append(i)

    return (problem_indices, height_threshold) if problem_indices else None


def _reocr_region(
    img: np.ndarray,
    box: list,
    orig_txt: str,
    orig_score: float,
    ocr_fn: Callable[[str], dict | None],
) -> dict | None:
    """Re-OCR a single problem region; return replacement dict or None."""
    img_h, img_w = img.shape[:2]
    pts = np.asarray(box)
    x_min, y_min = pts.min(axis=0).astype(int)
    x_max, y_max = pts.max(axis=0).astype(int)

    pad = _REFINE_CROP_PADDING
    cy1, cy2 = max(0, y_min - pad), min(img_h, y_max + pad)
    cx1, cx2 = max(0, x_min - pad), min(img_w, x_max + pad)

    crop = img[cy1:cy2, cx1:cx2]
    fd, crop_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        cv2.imwrite(crop_path, crop)
        crop_result = ocr_fn(crop_path)
    finally:
        try:
            os.unlink(crop_path)
        except OSError:
            pass

    if not crop_result or not crop_result.get("boxes"):
        return None

    new_chars = sum(len(t) for t in crop_result["txts"])
    orig_chars = len(orig_txt)
    new_avg = float(np.mean(crop_result["scores"]))

    if new_chars <= orig_chars or new_avg < _REFINE_MIN_REPLACEMENT_SCORE:
        logger.debug(f"  Region kept original (re-OCR: {new_chars} chars, {new_avg:.3f})")
        return None

    adjusted_boxes = [
        [[p[0] + cx1, p[1] + cy1] for p in crop_box] for crop_box in crop_result["boxes"]
    ]
    logger.debug(
        f"  Region replaced ({orig_chars} chars, {orig_score:.3f}) "
        f"→ ({new_chars} chars, {new_avg:.3f})"
    )
    return {
        "boxes": adjusted_boxes,
        "txts": list(crop_result["txts"]),
        "scores": list(crop_result["scores"]),
    }


def refine_ocr_results(
    ocr_raw: dict,
    image_path: str,
    ocr_fn: Callable[[str], dict | None],
) -> dict:
    """Refine OCR results by re-processing low-confidence oversized detections.

    PP-OCR's text detection (DBNet) sometimes creates bounding boxes that span
    multiple text lines, particularly on photographed/scanned pages with
    curvature near the book spine.  When these merged boxes are sent to text
    recognition, the output is garbled with very low confidence.

    This refinement pass:
     1. Identifies detection boxes with low confidence AND excessive height
        (likely merging several text lines).
     2. Crops those regions from the preprocessed image.
     3. Re-runs OCR on each crop — PP-OCR correctly segments lines in the
        smaller image, producing accurate per-line detections.
     4. Replaces the original garbled results with the cleaner re-OCR output,
        adjusting coordinates back to the full-page coordinate system.

    Complexity is O(k) additional OCR calls where *k* is the number of
    low-confidence oversized regions (typically 0–6 per page).  Pages with
    no such regions are untouched (zero overhead).

    Args:
        ocr_raw:    Initial OCR result dict with ``boxes``, ``txts``, ``scores``.
        image_path: Path to the preprocessed image on disk.
        ocr_fn:     Callable that takes an image path and returns a raw OCR
                    result dict (same schema as *ocr_raw*), or ``None``
                    on failure.  This is typically a bound method that sends
                    the path to the persistent OCR subprocess.

    Returns:
        Refined OCR result dict (same schema).  If no regions qualify for
        refinement the original dict is returned unmodified.
    """
    boxes = ocr_raw.get("boxes")
    txts = ocr_raw.get("txts")
    scores = ocr_raw.get("scores")

    if not boxes or not txts or not scores:
        return ocr_raw

    candidates = _find_refine_candidates(boxes, txts, scores)
    if candidates is None:
        return ocr_raw

    problem_indices, height_threshold = candidates
    logger.info(
        f"OCR refinement: {len(problem_indices)} candidate region(s) "
        f"(conf<{_REFINE_CONFIDENCE_THRESHOLD}, height>{height_threshold:.0f}px)"
    )

    img = cv2.imread(image_path)
    if img is None:
        logger.warning(f"OCR refinement: could not read image {image_path}")
        return ocr_raw

    replacements: dict[int, dict] = {}
    for idx in problem_indices:
        repl = _reocr_region(img, boxes[idx], txts[idx], scores[idx], ocr_fn)
        if repl is not None:
            replacements[idx] = repl

    if not replacements:
        logger.info("OCR refinement: no regions improved, keeping original results")
        return ocr_raw

    n = len(boxes)
    final_boxes: list = []
    final_txts: list[str] = []
    final_scores: list[float] = []

    for i in range(n):
        if i in replacements:
            r = replacements[i]
            final_boxes.extend(r["boxes"])
            final_txts.extend(r["txts"])
            final_scores.extend(r["scores"])
        else:
            final_boxes.append(boxes[i])
            final_txts.append(txts[i])
            final_scores.append(scores[i])

    logger.info(
        f"OCR refinement: replaced {len(replacements)} region(s), "
        f"total {n} → {len(final_boxes)} detection(s)"
    )

    return {"boxes": final_boxes, "txts": final_txts, "scores": final_scores}
