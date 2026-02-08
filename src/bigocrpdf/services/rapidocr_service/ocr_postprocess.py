"""
OCR post-processing utilities.

Functions for fixing overlapping text regions, merging nearby text segments,
and combining OCR results into coherent reading order.
"""

import logging

from bigocrpdf.services.rapidocr_service.config import OCRResult

logger = logging.getLogger(__name__)


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
