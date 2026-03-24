"""
Visual Analyzer Module

OpenCV-based visual analysis of document page images to detect
structural elements: table borders, column separators, decorative
lines, and region boundaries.

Also provides text style detection (bold, underline) via ink density
analysis and morphological line detection.

Provides visual metadata to supplement OCR-based layout analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from bigocrpdf.utils.logger import logger

if TYPE_CHECKING:
    from bigocrpdf.utils.odf_types import OCRTextData

# Minimum line length as fraction of page dimension
MIN_LINE_FRACTION = 0.05
# Maximum gap between collinear segments to merge
MERGE_GAP_PX = 15
# Tolerance for line angle classification (degrees)
ANGLE_TOLERANCE = 2.0


@dataclass
class VisualLine:
    """A detected line segment in the page image."""

    x1: int
    y1: int
    x2: int
    y2: int
    orientation: str  # "horizontal" or "vertical"
    length: int


@dataclass
class TableRegion:
    """A detected table region defined by intersecting lines."""

    x: int
    y: int
    width: int
    height: int
    rows: int
    cols: int
    h_lines: list[int] = field(default_factory=list)  # Y positions of horizontal lines
    v_lines: list[int] = field(default_factory=list)  # X positions of vertical lines


@dataclass
class ColumnLayout:
    """Detected multi-column layout."""

    num_columns: int
    separators: list[float]  # X positions as percentage (0-100)
    column_ranges: list[tuple[float, float]]  # (left%, right%) for each column


@dataclass
class VisualAnalysisResult:
    """Complete visual analysis result for one page."""

    tables: list[TableRegion] = field(default_factory=list)
    columns: ColumnLayout | None = None
    h_separators: list[float] = field(default_factory=list)  # Y positions as percentage
    v_separators: list[float] = field(default_factory=list)  # X positions as percentage
    page_width: int = 0
    page_height: int = 0


def analyze_page_image(image_path: str) -> VisualAnalysisResult | None:
    """Analyze a page image for structural visual elements.

    Uses OpenCV to detect:
    - Horizontal and vertical lines (table borders, separators)
    - Table regions from line intersections
    - Multi-column layouts from vertical whitespace gaps

    Args:
        image_path: Path to the page image file

    Returns:
        VisualAnalysisResult or None if analysis fails
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"Visual analyzer: Could not read image {image_path}")
            return None

        h, w = img.shape[:2]
        result = VisualAnalysisResult(page_width=w, page_height=h)

        # Step 1: Detect lines
        h_lines, v_lines = _detect_lines(img, w, h)

        # Step 2: Detect tables from line intersections
        result.tables = _detect_tables(h_lines, v_lines, w, h)

        # Step 3: Detect decorative horizontal separators
        result.h_separators = _detect_separators(h_lines, v_lines, w, h)

        # Step 4: Detect multi-column layout from whitespace
        result.columns = _detect_columns(img, w, h)

        logger.debug(
            f"Visual analysis: {len(result.tables)} tables, "
            f"{len(result.h_separators)} separators, "
            f"columns={result.columns.num_columns if result.columns else 1}"
        )

        return result

    except Exception as e:
        logger.warning(f"Visual analysis failed for {image_path}: {e}")
        return None


def _detect_lines(img: np.ndarray, w: int, h: int) -> tuple[list[VisualLine], list[VisualLine]]:
    """Detect horizontal and vertical lines using morphological operations + HoughLinesP.

    Args:
        img: Grayscale image
        w: Image width
        h: Image height

    Returns:
        Tuple of (horizontal_lines, vertical_lines)
    """
    # Binarize
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    min_h_len = max(int(w * MIN_LINE_FRACTION), 30)
    min_v_len = max(int(h * MIN_LINE_FRACTION), 30)

    # Detect horizontal lines with morphological opening
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_h_len, 1))
    h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)

    # Detect vertical lines with morphological opening
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_v_len))
    v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)

    # Extract line segments with HoughLinesP
    h_lines = _extract_line_segments(h_mask, min_h_len, "horizontal")
    v_lines = _extract_line_segments(v_mask, min_v_len, "vertical")

    # Merge collinear segments
    h_lines = _merge_collinear(h_lines, "horizontal")
    v_lines = _merge_collinear(v_lines, "vertical")

    return h_lines, v_lines


def _extract_line_segments(mask: np.ndarray, min_length: int, orientation: str) -> list[VisualLine]:
    """Extract line segments from a binary mask using HoughLinesP.

    Args:
        mask: Binary mask with line regions
        min_length: Minimum line length in pixels
        orientation: Expected orientation

    Returns:
        List of VisualLine objects
    """
    lines_raw = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_length,
        maxLineGap=MERGE_GAP_PX,
    )

    if lines_raw is None:
        return []

    result = []
    for segment in lines_raw:
        x1, y1, x2, y2 = segment[0]
        length = int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

        # Classify orientation by angle
        if x2 != x1:
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        else:
            angle = 90.0

        if orientation == "horizontal" and angle <= ANGLE_TOLERANCE:
            result.append(
                VisualLine(x1=x1, y1=y1, x2=x2, y2=y2, orientation="horizontal", length=length)
            )
        elif orientation == "vertical" and abs(angle - 90) <= ANGLE_TOLERANCE:
            result.append(
                VisualLine(x1=x1, y1=y1, x2=x2, y2=y2, orientation="vertical", length=length)
            )

    return result


def _merge_collinear(lines: list[VisualLine], orientation: str) -> list[VisualLine]:
    """Merge collinear line segments that are close together.

    Args:
        lines: Line segments to merge
        orientation: "horizontal" or "vertical"

    Returns:
        Merged line segments
    """
    if not lines:
        return []

    if orientation == "horizontal":
        # Group by Y position (within tolerance)
        lines.sort(key=lambda ln: ln.y1)
        groups: list[list[VisualLine]] = []
        for ln in lines:
            placed = False
            for group in groups:
                if abs(ln.y1 - group[0].y1) <= 3:
                    group.append(ln)
                    placed = True
                    break
            if not placed:
                groups.append([ln])

        merged = []
        for group in groups:
            group.sort(key=lambda ln: min(ln.x1, ln.x2))
            # Merge overlapping/close segments
            current = group[0]
            for ln in group[1:]:
                seg_start = min(ln.x1, ln.x2)
                cur_end = max(current.x1, current.x2)
                if seg_start - cur_end <= MERGE_GAP_PX:
                    # Extend segment
                    new_x1 = min(current.x1, ln.x1)
                    new_x2 = max(current.x2, ln.x2)
                    avg_y = (current.y1 + ln.y1) // 2
                    length = new_x2 - new_x1
                    current = VisualLine(
                        x1=new_x1,
                        y1=avg_y,
                        x2=new_x2,
                        y2=avg_y,
                        orientation="horizontal",
                        length=length,
                    )
                else:
                    merged.append(current)
                    current = ln
            merged.append(current)
        return merged

    else:  # vertical
        lines.sort(key=lambda ln: ln.x1)
        groups = []
        for ln in lines:
            placed = False
            for group in groups:
                if abs(ln.x1 - group[0].x1) <= 3:
                    group.append(ln)
                    placed = True
                    break
            if not placed:
                groups.append([ln])

        merged = []
        for group in groups:
            group.sort(key=lambda ln: min(ln.y1, ln.y2))
            current = group[0]
            for ln in group[1:]:
                seg_start = min(ln.y1, ln.y2)
                cur_end = max(current.y1, current.y2)
                if seg_start - cur_end <= MERGE_GAP_PX:
                    new_y1 = min(current.y1, ln.y1)
                    new_y2 = max(current.y2, ln.y2)
                    avg_x = (current.x1 + ln.x1) // 2
                    length = new_y2 - new_y1
                    current = VisualLine(
                        x1=avg_x,
                        y1=new_y1,
                        x2=avg_x,
                        y2=new_y2,
                        orientation="vertical",
                        length=length,
                    )
                else:
                    merged.append(current)
                    current = ln
            merged.append(current)
        return merged


def _detect_tables(
    h_lines: list[VisualLine],
    v_lines: list[VisualLine],
    w: int,
    h: int,
) -> list[TableRegion]:
    """Detect table regions from intersecting horizontal and vertical lines.

    A table requires at least 3 horizontal and 3 vertical lines forming
    a grid pattern with consistent intersections.

    Args:
        h_lines: Horizontal line segments
        v_lines: Vertical line segments
        w: Image width
        h: Image height

    Returns:
        List of detected TableRegion objects
    """
    if len(h_lines) < 2 or len(v_lines) < 2:
        return []

    # Find intersection points
    intersections = []
    for hl in h_lines:
        for vl in v_lines:
            # Check if horizontal line's X range overlaps with vertical line's X
            h_min_x = min(hl.x1, hl.x2)
            h_max_x = max(hl.x1, hl.x2)
            v_x = (vl.x1 + vl.x2) // 2

            # Check if vertical line's Y range overlaps with horizontal line's Y
            v_min_y = min(vl.y1, vl.y2)
            v_max_y = max(vl.y1, vl.y2)
            h_y = (hl.y1 + hl.y2) // 2

            tolerance = 5
            if (
                h_min_x - tolerance <= v_x <= h_max_x + tolerance
                and v_min_y - tolerance <= h_y <= v_max_y + tolerance
            ):
                intersections.append((v_x, h_y))

    if len(intersections) < 4:
        return []

    # Cluster intersections into table regions
    # Use connected components approach: find groups of lines forming rectangles
    tables = []

    # Get unique Y positions of horizontal lines and X positions of vertical lines
    h_y_positions = sorted({(hl.y1 + hl.y2) // 2 for hl in h_lines})
    v_x_positions = sorted({(vl.x1 + vl.x2) // 2 for vl in v_lines})

    # Find contiguous groups of lines that form a table
    h_groups = _group_positions(h_y_positions, max_gap=h * 0.3)
    v_groups = _group_positions(v_x_positions, max_gap=w * 0.3)

    for h_group in h_groups:
        for v_group in v_groups:
            if len(h_group) >= 2 and len(v_group) >= 2:
                # Verify intersections exist for this grid
                grid_intersections = 0
                for hy in h_group:
                    for vx in v_group:
                        for ix, iy in intersections:
                            if abs(ix - vx) <= 10 and abs(iy - hy) <= 10:
                                grid_intersections += 1
                                break

                # Need at least 60% of expected intersections
                expected = len(h_group) * len(v_group)
                if grid_intersections >= expected * 0.6:
                    table = TableRegion(
                        x=min(v_group),
                        y=min(h_group),
                        width=max(v_group) - min(v_group),
                        height=max(h_group) - min(h_group),
                        rows=len(h_group) - 1,
                        cols=len(v_group) - 1,
                        h_lines=list(h_group),
                        v_lines=list(v_group),
                    )
                    tables.append(table)

    return tables


def _group_positions(positions: list[int], max_gap: float) -> list[list[int]]:
    """Group positions into clusters where consecutive positions are within max_gap.

    Args:
        positions: Sorted list of positions
        max_gap: Maximum gap between consecutive positions in the same group

    Returns:
        List of position groups
    """
    if not positions:
        return []

    groups: list[list[int]] = [[positions[0]]]
    for pos in positions[1:]:
        if pos - groups[-1][-1] <= max_gap:
            groups[-1].append(pos)
        else:
            groups.append([pos])

    return groups


def _detect_separators(
    h_lines: list[VisualLine],
    v_lines: list[VisualLine],
    w: int,
    h: int,
) -> list[float]:
    """Detect decorative horizontal separators (not part of tables).

    A separator is a long horizontal line that spans a significant portion
    of the page width and is not part of a table grid.

    Args:
        h_lines: Horizontal line segments
        v_lines: Vertical line segments
        w: Image width
        h: Image height

    Returns:
        List of separator Y positions as percentage (0-100)
    """
    # Lines that span >40% of page width and are isolated (no nearby vertical lines)
    min_span = w * 0.4
    separators = []

    # Build set of Y ranges covered by table regions (approximate)
    table_y_ranges = set()
    for vl in v_lines:
        y_min = min(vl.y1, vl.y2)
        y_max = max(vl.y1, vl.y2)
        for y in range(y_min, y_max + 1):
            table_y_ranges.add(y)

    for hl in h_lines:
        line_span = abs(hl.x2 - hl.x1)
        if line_span < min_span:
            continue

        h_y = (hl.y1 + hl.y2) // 2

        # Check if this line is part of a table (near vertical lines at same Y)
        near_vertical = False
        for vl in v_lines:
            v_min_y = min(vl.y1, vl.y2)
            v_max_y = max(vl.y1, vl.y2)
            if v_min_y - 10 <= h_y <= v_max_y + 10:
                near_vertical = True
                break

        if not near_vertical:
            separators.append((h_y / h) * 100)

    return sorted(separators)


def _detect_columns(img: np.ndarray, w: int, h: int) -> ColumnLayout | None:
    """Detect multi-column layout by analyzing vertical whitespace distribution.

    Projects the binary image horizontally and looks for wide vertical
    gaps that indicate column boundaries.

    Args:
        img: Grayscale image
        w: Image width
        h: Image height

    Returns:
        ColumnLayout if multi-column detected, None for single column
    """
    # Binarize
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Focus on the middle 60% of the page (skip header/footer)
    y_start = int(h * 0.2)
    y_end = int(h * 0.8)
    content_region = binary[y_start:y_end, :]

    # Horizontal projection: sum of ink pixels per column
    projection = np.sum(content_region, axis=0) / 255
    content_height = y_end - y_start

    # Normalize: fraction of content height that has ink
    projection_norm = projection / content_height

    # Find wide gaps (regions with very low ink density)
    gap_threshold = 0.02  # Less than 2% ink density
    min_gap_width = int(w * 0.02)  # At least 2% of page width

    in_gap = False
    gap_start = 0
    gaps: list[tuple[int, int]] = []

    # Skip margins: only look between 15% and 85% of page width
    margin_left = int(w * 0.15)
    margin_right = int(w * 0.85)

    for x in range(margin_left, margin_right):
        if projection_norm[x] < gap_threshold:
            if not in_gap:
                gap_start = x
                in_gap = True
        else:
            if in_gap:
                gap_width = x - gap_start
                if gap_width >= min_gap_width:
                    gaps.append((gap_start, x))
                in_gap = False

    if in_gap:
        gap_width = margin_right - gap_start
        if gap_width >= min_gap_width:
            gaps.append((gap_start, margin_right))

    if not gaps:
        return None

    # Convert gaps to column separators
    separators = [(g[0] + g[1]) / 2 / w * 100 for g in gaps]

    # Build column ranges
    num_columns = len(separators) + 1
    if num_columns > 5:
        # Unrealistic — probably not columns
        return None

    ranges: list[tuple[float, float]] = []
    prev_right = 0.0
    for sep in separators:
        ranges.append((prev_right, sep))
        prev_right = sep
    ranges.append((prev_right, 100.0))

    return ColumnLayout(
        num_columns=num_columns,
        separators=separators,
        column_ranges=ranges,
    )


# ---------------------------------------------------------------------------
# Text style detection (bold / underline) via distance-transform stroke width
# ---------------------------------------------------------------------------

# Stroke-width ratio threshold: box_stroke / page_median > this → bold
_BOLD_STROKE_RATIO = 1.35
# Minimum box area in pixels to attempt analysis (skip tiny artifacts)
_MIN_BOX_AREA_PX = 100
# Underline kernel width as fraction of box width
_UNDERLINE_KERNEL_FRACTION = 0.6
# Minimum underline length in pixels
_MIN_UNDERLINE_LEN = 10
# How far below the text baseline (as fraction of box height) to look
_UNDERLINE_SEARCH_BELOW = 0.25


def _compute_box_rects_and_strokes(
    ocr_items: list[OCRTextData],
    dist: np.ndarray,
    binary: np.ndarray,
    w: int,
    h: int,
) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    """Compute bounding box pixel rects and stroke widths for each OCR item."""
    stroke_widths: list[float] = []
    box_rects: list[tuple[int, int, int, int]] = []

    for item in ocr_items:
        x1 = max(0, int((item.x / 100.0) * w))
        y1 = max(0, int((item.y / 100.0) * h))
        x2 = min(w, int(((item.x + item.width) / 100.0) * w))
        box_h_pct = max(0.5, item.height * 0.12) if item.height > 0 else 2.0
        y2 = min(h, y1 + max(1, int((box_h_pct / 100.0) * h)))

        box_rects.append((x1, y1, x2, y2))
        box_area = (x2 - x1) * (y2 - y1)

        if box_area < _MIN_BOX_AREA_PX:
            stroke_widths.append(0.0)
            continue

        roi_dist = dist[y1:y2, x1:x2]
        roi_bin = binary[y1:y2, x1:x2]
        fg_distances = roi_dist[roi_bin > 0]
        if len(fg_distances) < 5:
            stroke_widths.append(0.0)
            continue

        median_half = float(np.median(fg_distances))
        stroke_widths.append(median_half * 2.0)

    return box_rects, stroke_widths


def _detect_bold(ocr_items: list[OCRTextData], stroke_widths: list[float]) -> None:
    """Mark items as bold based on stroke width relative to page median."""
    valid_strokes = sorted(s for s in stroke_widths if s > 0.5)
    if not valid_strokes:
        return
    page_median_stroke = valid_strokes[len(valid_strokes) // 2]
    if page_median_stroke > 0:
        for i, item in enumerate(ocr_items):
            if stroke_widths[i] > page_median_stroke * _BOLD_STROKE_RATIO:
                item.is_bold = True


def _detect_underlines(
    ocr_items: list[OCRTextData],
    box_rects: list[tuple[int, int, int, int]],
    h_lines_mask: np.ndarray,
    h: int,
) -> None:
    """Mark items as underlined based on horizontal line detection below the text."""
    for i, item in enumerate(ocr_items):
        x1, y1, x2, y2 = box_rects[i]
        box_w = x2 - x1
        box_h = y2 - y1
        if box_h < 3 or box_w < 5:
            continue

        search_top = y2
        search_bottom = min(h, y2 + max(3, int(box_h * _UNDERLINE_SEARCH_BELOW)))
        if search_top >= search_bottom:
            continue

        underline_roi = h_lines_mask[search_top:search_bottom, x1:x2]
        if underline_roi.size < 1:
            continue

        max_run = 0
        for row_idx in range(underline_roi.shape[0]):
            row = underline_roi[row_idx]
            run = 0
            for px in row:
                if px > 0:
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 0

        if max_run >= box_w * 0.5:
            item.is_underlined = True


def analyze_text_styles(
    image_path: str,
    ocr_items: list[OCRTextData],
) -> None:
    """Detect bold and underline styles for OCR items via OpenCV analysis.

    Modifies each ``OCRTextData`` in-place, setting ``is_bold`` and
    ``is_underlined`` flags based on visual analysis of the page image.

    Args:
        image_path: Path to the page image (grayscale or colour).
        ocr_items: OCR data items to annotate.  Modified in-place.
    """
    if not ocr_items:
        return

    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.debug(f"Style analysis: could not read {image_path}")
            return

        h, w = img.shape[:2]
        if h < 10 or w < 10:
            return

        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        box_rects, stroke_widths = _compute_box_rects_and_strokes(ocr_items, dist, binary, w, h)
        _detect_bold(ocr_items, stroke_widths)

        kernel_w = max(_MIN_UNDERLINE_LEN, int(w * 0.03))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
        h_lines_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
        _detect_underlines(ocr_items, box_rects, h_lines_mask, h)

        bold_count = sum(1 for it in ocr_items if it.is_bold)
        underline_count = sum(1 for it in ocr_items if it.is_underlined)
        logger.debug(
            f"Style analysis: {bold_count} bold, {underline_count} underlined "
            f"out of {len(ocr_items)} items"
        )

    except Exception as e:
        logger.warning(f"Text style analysis failed: {e}")
