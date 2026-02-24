"""Column and table detection for TSV-parsed page data.

Geometric analysis of word positions to detect multi-column layouts
and table regions in pdftotext TSV output.
"""

import re

from bigocrpdf.utils.tsv_parser import (
    BOUNDARY_SPLIT_MIN_GAP,
    CENTER_CLUSTER_TOLERANCE,
    COLUMN_CENTER_RANGE,
    COLUMN_GAP_THRESHOLD,
    COLUMN_MIN_VALLEY_WIDTH,
    COLUMN_VALLEY_BIN,
    HEADER_MISALIGN_TOLERANCE,
    HEADER_WORD_GAP,
    MAX_TABLE_CELL_LEN,
    MAX_TABLE_GAP,
    MIN_TABLE_ROWS,
    TABLE_TWO_COL_GAP,
    TextLine,
    Word,
)


# ── Column Detection ──


def _build_coverage_histogram(
    words: list[Word],
    page_left: float,
    page_width: float,
) -> tuple[list[int], int]:
    """Build a histogram of word coverage by x-position bin."""
    n_bins = max(1, int(page_width / COLUMN_VALLEY_BIN))
    coverage = [0] * n_bins
    for w in words:
        start_bin = max(0, int((w.left - page_left) / COLUMN_VALLEY_BIN))
        end_bin = min(n_bins - 1, int((w.right - page_left) / COLUMN_VALLEY_BIN))
        for b in range(start_bin, end_bin + 1):
            coverage[b] += 1
    return coverage, n_bins


def _find_valley(
    coverage: list[int],
    n_bins: int,
) -> tuple[int, int] | None:
    """Find a central valley in the coverage histogram.

    Returns (valley_start_bin, valley_end_bin) or None.
    """
    center_start = int(n_bins * COLUMN_CENTER_RANGE[0])
    center_end = int(n_bins * COLUMN_CENTER_RANGE[1])
    if center_start >= center_end:
        return None

    min_val = min(coverage[center_start:center_end])
    left_peak = max(coverage[:center_start]) if center_start > 0 else 0
    right_peak = max(coverage[center_end:]) if center_end < n_bins else 0
    if left_peak == 0 or right_peak == 0:
        return None

    avg_peak = (left_peak + right_peak) / 2
    if min_val > avg_peak * 0.3:
        return None

    valley_threshold = avg_peak * 0.3
    valley_start = None
    valley_end = None
    for b in range(center_start, center_end):
        if coverage[b] <= valley_threshold:
            if valley_start is None:
                valley_start = b
            valley_end = b

    if valley_start is None or valley_end is None:
        return None

    valley_width_pt = (valley_end - valley_start + 1) * COLUMN_VALLEY_BIN
    if valley_width_pt < COLUMN_MIN_VALLEY_WIDTH:
        return None

    return valley_start, valley_end


def detect_page_columns(words: list[Word]) -> list[tuple[float, float]] | None:
    """Detect if the page has a multi-column layout."""
    if len(words) < 20:
        return None

    page_left = min(w.left for w in words)
    page_right = max(w.right for w in words)
    page_width = page_right - page_left
    if page_width < 200:
        return None

    coverage, n_bins = _build_coverage_histogram(words, page_left, page_width)
    valley = _find_valley(coverage, n_bins)
    if valley is None:
        return None

    valley_start, valley_end = valley
    split_x = page_left + (valley_start + valley_end + 1) / 2 * COLUMN_VALLEY_BIN

    left_count = sum(1 for w in words if (w.left + w.right) / 2 < split_x)
    right_count = len(words) - left_count
    total = left_count + right_count
    if total > 0 and min(left_count, right_count) / total < 0.30:
        return None

    return [(page_left, split_x), (split_x, page_right)]


def split_words_by_columns(
    words: list[Word], columns: list[tuple[float, float]]
) -> list[list[Word]]:
    """Split words into columns based on their left x-position."""
    col_words: list[list[Word]] = [[] for _ in columns]
    for w in words:
        ref_x = w.left
        for i, (col_start, col_end) in enumerate(columns):
            if col_start <= ref_x <= col_end:
                col_words[i].append(w)
                break
        else:
            # Assign to nearest column
            min_dist = float("inf")
            best_col = 0
            for i, (col_start, col_end) in enumerate(columns):
                dist = min(abs(ref_x - col_start), abs(ref_x - col_end))
                if dist < min_dist:
                    min_dist = dist
                    best_col = i
            col_words[best_col].append(w)
    return col_words


# ── Table Detection ──


def _get_column_groups(words: list[Word], gap: float = COLUMN_GAP_THRESHOLD) -> list[list[Word]]:
    """Group words by x-gap."""
    if not words:
        return []
    sorted_w = sorted(words, key=lambda w: w.left)
    groups: list[list[Word]] = [[sorted_w[0]]]
    for w in sorted_w[1:]:
        if w.left - groups[-1][-1].right > gap:
            groups.append([w])
        else:
            groups[-1].append(w)
    return groups


def is_table_line(line: TextLine) -> bool:
    """Quick check if a line looks like a table row."""
    groups = _get_column_groups(line.words, gap=COLUMN_GAP_THRESHOLD)
    if len(groups) < 2:
        return False
    for i in range(1, len(groups)):
        gap = groups[i][0].left - groups[i - 1][-1].right
        if gap > MAX_TABLE_GAP:
            return False
    if len(groups) >= 3:
        return True
    # For 2-group lines, require a very clear gap
    if len(groups) == 2:
        actual_gap = groups[1][0].left - groups[0][-1].right
        return actual_gap >= TABLE_TWO_COL_GAP
    return False


def detect_table_region(lines: list[TextLine], start_idx: int) -> tuple[list[list[str]], int, int]:
    """Detect a table region starting at start_idx.

    Returns:
        (rows, end_idx, n_pre_headers) where n_pre_headers is the number
        of lines before start_idx consumed as table headers.
    """
    if start_idx >= len(lines):
        return [], start_idx, 0

    candidate_lines = _collect_table_candidates(lines, start_idx)
    if len(candidate_lines) < MIN_TABLE_ROWS:
        return [], start_idx, 0

    # Filter header-fragment lines misaligned to the right
    candidate_lines = _filter_misaligned_headers(candidate_lines)
    if len(candidate_lines) < MIN_TABLE_ROWS:
        return [], start_idx, 0

    col_centers = _determine_column_centers(candidate_lines)
    if len(col_centers) < 2:
        return [], start_idx, 0

    col_boundaries = [
        (col_centers[i] + col_centers[i + 1]) / 2 for i in range(len(col_centers) - 1)
    ]

    rows = _build_table_rows(candidate_lines, col_centers, col_boundaries)

    # Validate: reject if cells are too long (paragraph text, not table data)
    if not _validate_table_cell_lengths(rows):
        return [], start_idx, 0

    # Backward header scan
    header_rows, n_pre_headers = _scan_backward_headers(
        lines, start_idx, candidate_lines, col_centers
    )
    rows = header_rows + rows

    # Merge pre-header row with first data row when they complement each other
    if len(rows) >= 2 and n_pre_headers > 0:
        rows = _merge_complementary_headers(rows, col_centers)

    end_idx = candidate_lines[-1][0] + 1
    return rows, end_idx, n_pre_headers


def _collect_table_candidates(
    lines: list[TextLine], start_idx: int
) -> list[tuple[int, TextLine, list[list[Word]]]]:
    """Collect consecutive lines that satisfy table-line criteria."""
    candidate_lines = []
    idx = start_idx
    while idx < len(lines):
        groups = _get_column_groups(lines[idx].words, gap=COLUMN_GAP_THRESHOLD)
        is_valid = len(groups) >= 3
        if not is_valid and len(groups) == 2:
            gap = groups[1][0].left - groups[0][-1].right
            is_valid = gap >= TABLE_TWO_COL_GAP
        if is_valid:
            candidate_lines.append((idx, lines[idx], groups))
            idx += 1
        else:
            break
    return candidate_lines


def _filter_misaligned_headers(
    candidate_lines: list[tuple[int, TextLine, list[list[Word]]]],
) -> list[tuple[int, TextLine, list[list[Word]]]]:
    """Remove lines whose min_x is too far right of the median."""
    min_xs = [line.min_x for _, line, _ in candidate_lines]
    median_min_x = sorted(min_xs)[len(min_xs) // 2]
    return [
        (ci, ln, gps)
        for ci, ln, gps in candidate_lines
        if ln.min_x <= median_min_x + HEADER_MISALIGN_TOLERANCE
    ]


def _build_table_rows(
    candidate_lines: list[tuple[int, TextLine, list[list[Word]]]],
    col_centers: list[float],
    col_boundaries: list[float],
) -> list[list[str]]:
    """Build cell-text rows by distributing word groups into columns."""
    rows = []
    for _, line, _ in candidate_lines:
        groups = _get_column_groups(line.words)
        split_groups = _split_groups_at_boundaries(groups, col_boundaries)
        row = [""] * len(col_centers)
        for grp in split_groups:
            col_idx = min(
                range(len(col_centers)),
                key=lambda i: abs(grp[0].left - col_centers[i]),
            )
            grp_text = " ".join(w.text for w in grp)
            if row[col_idx]:
                row[col_idx] += " " + grp_text
            else:
                row[col_idx] = grp_text
        rows.append(row)
    return rows


def _validate_table_cell_lengths(rows: list[list[str]]) -> bool:
    """Reject tables whose median cell length exceeds the threshold."""
    all_cells = [c for r in rows for c in r if c]
    if all_cells:
        sorted_lens = sorted(len(c) for c in all_cells)
        median_len = sorted_lens[len(sorted_lens) // 2]
        if median_len > MAX_TABLE_CELL_LEN:
            return False
    return True


def _scan_backward_headers(
    lines: list[TextLine],
    start_idx: int,
    candidate_lines: list[tuple[int, TextLine, list[list[Word]]]],
    col_centers: list[float],
) -> tuple[list[list[str]], int]:
    """Scan lines above the table for header rows.

    Returns:
        (header_rows, n_pre_headers)
    """
    header_rows: list[list[str]] = []
    n_pre_headers = 0
    table_min_x = min(ln.min_x for _, ln, _ in candidate_lines)
    header_x_threshold = table_min_x + 50

    for j in range(start_idx - 1, max(start_idx - 4, -1), -1):
        prev_line = lines[j]
        y_gap = lines[j + 1].y - prev_line.y
        if y_gap > 25:
            break
        if prev_line.min_x < header_x_threshold:
            break
        # Header words must show column-like gaps (not continuous text)
        word_groups = _get_column_groups(prev_line.words, gap=HEADER_WORD_GAP)
        if len(word_groups) < 2:
            break
        row = _distribute_words_to_columns(prev_line.words, col_centers)
        populated = sum(1 for c in row if c)
        if populated < 2:
            break
        header_rows.insert(0, row)
        n_pre_headers += 1

    # Merge multiple pre-header rows into a single row
    if len(header_rows) > 1:
        merged = [""] * len(col_centers)
        for row in header_rows:
            for ci, cell in enumerate(row):
                if cell and ci < len(merged):
                    if merged[ci]:
                        merged[ci] += " " + cell
                    else:
                        merged[ci] = cell
        header_rows = [merged]

    return header_rows, n_pre_headers


def _merge_complementary_headers(
    rows: list[list[str]], col_centers: list[float]
) -> list[list[str]]:
    """Merge pre-header with first data row when they fill each other's gaps."""
    row0, row1 = rows[0], rows[1]
    n = len(col_centers)
    empty_in_0 = sum(1 for i in range(n) if not (row0[i] if i < len(row0) else ""))
    r1_cells = [row1[i] for i in range(n) if i < len(row1) and row1[i]]
    has_numeric = any(re.match(r"^\d+[.,]?\d*$", c.strip()) for c in r1_cells)
    if empty_in_0 > 0 and r1_cells and not has_numeric:
        merged = [""] * n
        for i in range(n):
            c0 = row0[i] if i < len(row0) else ""
            c1 = row1[i] if i < len(row1) else ""
            if c0 and c1:
                merged[i] = c0 + " " + c1
            else:
                merged[i] = c0 or c1
        return [merged] + rows[2:]
    return rows


def _determine_column_centers(candidate_lines: list) -> list[float]:
    """Find column centers from the most granular rows."""
    max_groups = max(len(gps) for _, _, gps in candidate_lines)
    if max_groups < 2:
        return []

    centers = []
    for _, _, groups in candidate_lines:
        if len(groups) >= max_groups:
            for grp in groups:
                centers.append((grp[0].left + grp[-1].right) / 2)

    if not centers:
        for _, _, groups in candidate_lines:
            for grp in groups:
                centers.append((grp[0].left + grp[-1].right) / 2)

    centers.sort()
    clusters: list[list[float]] = [[centers[0]]]
    for c in centers[1:]:
        if c - clusters[-1][-1] <= CENTER_CLUSTER_TOLERANCE:
            clusters[-1].append(c)
        else:
            clusters.append([c])

    return [sum(cl) / len(cl) for cl in clusters]


def _split_groups_at_boundaries(
    groups: list[list[Word]], boundaries: list[float]
) -> list[list[Word]]:
    """Split groups crossing column boundaries at the nearest internal gap."""
    result = []
    for grp in groups:
        result.extend(_recursive_boundary_split(grp, boundaries))
    return result


def _recursive_boundary_split(words: list[Word], boundaries: list[float]) -> list[list[Word]]:
    """Recursively split a word group at column boundaries."""
    if len(words) <= 1:
        return [words]
    grp_left = words[0].left
    grp_right = words[-1].right
    crossing = [b for b in boundaries if grp_left < b < grp_right]
    if not crossing:
        return [words]
    # Split at the gap closest to the first crossing boundary
    b = crossing[0]
    best_split = -1
    best_distance = float("inf")
    for i in range(len(words) - 1):
        gap_center = (words[i].right + words[i + 1].left) / 2
        dist = abs(gap_center - b)
        gap_size = words[i + 1].left - words[i].right
        if dist < best_distance and gap_size > BOUNDARY_SPLIT_MIN_GAP:
            best_distance = dist
            best_split = i + 1
    if best_split > 0:
        left_part = words[:best_split]
        right_part = words[best_split:]
        return _recursive_boundary_split(left_part, boundaries) + _recursive_boundary_split(
            right_part, boundaries
        )
    return [words]


def _distribute_words_to_columns(words: list[Word], col_centers: list[float]) -> list[str]:
    """Assign header words to table columns, grouping nearby words first."""
    n_cols = len(col_centers)
    row = [""] * n_cols
    # Group adjacent words (header text within a column is close together)
    groups = _get_column_groups(words, gap=8.0)
    for grp in groups:
        grp_text = " ".join(w.text for w in grp)
        # Use group's left edge for column assignment (more reliable than center)
        col_idx = min(range(n_cols), key=lambda ci: abs(grp[0].left - col_centers[ci]))
        if row[col_idx]:
            row[col_idx] += " " + grp_text
        else:
            row[col_idx] = grp_text
    return row
