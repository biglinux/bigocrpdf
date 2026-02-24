"""
Layout Analyzer Module

Advanced document layout analysis using statistical methods and efficient
binning algorithms. Detects text lines, table columns, paragraphs, and
headings from OCR data.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from bigocrpdf.utils.odf_types import OCRTextData

if TYPE_CHECKING:
    from bigocrpdf.utils.visual_analyzer import VisualAnalysisResult


class LayoutAnalyzer:
    """Advanced layout analysis using statistical methods and efficient binning.

    Uses efficient O(n log n) binning algorithms to detect:
    - Text lines (via Y-coordinate binning)
    - Table columns (via X-coordinate binning)
    - Paragraphs (via vertical spacing analysis)
    - Headings (via font size and text pattern analysis)

    Optionally enhanced with OpenCV visual analysis data for:
    - More accurate table detection via line detection
    - Multi-column layout detection
    - Decorative separator detection
    """

    def __init__(
        self,
        page_data: list[OCRTextData],
        visual_data: VisualAnalysisResult | None = None,
    ):
        """Initialize with OCR data for one page.

        Args:
            page_data: List of OCR text items for the page (coordinates as percentages 0-100)
            visual_data: Optional OpenCV visual analysis result for this page
        """
        self.items = page_data
        self.lines: list[dict] = []
        self.blocks: list[dict] = []  # Final blocks: 'paragraph', 'table', 'heading'
        self.visual_data = visual_data

        # Extract visual hints from OpenCV analysis
        if visual_data:
            self.footer_y_threshold = None
            # Convert table vertical line positions to percentage
            self.vertical_lines = visual_data.v_separators
            self.h_separators = visual_data.h_separators
            self.visual_tables = visual_data.tables
            self.column_layout = visual_data.columns
        else:
            self.footer_y_threshold = None
            self.vertical_lines = []
            self.h_separators = []
            self.visual_tables = []
            self.column_layout = None

        # Estimate page dimensions from data (all coordinates are 0-100%)
        if self.items:
            self.page_width = max(item.x + item.width for item in self.items) * 1.05
            self.page_height = max(item.y + item.height for item in self.items) * 1.05
        else:
            self.page_width = 100.0
            self.page_height = 100.0

    def analyze(self) -> list[dict]:
        """Perform full layout analysis.

        Returns:
            List of blocks, each with:
            - type: 'paragraph', 'table', or 'heading'
            - lines: list of line dicts (for paragraph/heading)
            - rows: list of row dicts (for table)
        """
        if not self.items:
            return []

        # Step 1: Group items into lines using Y-coordinate clustering
        self._detect_lines()

        # Step 2: Detect tables using X-coordinate clustering
        # and identify remaining content as paragraphs
        self._detect_blocks()

        return self.blocks

    def _detect_lines(self) -> None:
        """Group OCR items into lines using efficient Y-coordinate binning.

        Uses a simple O(n log n) sorting approach instead of expensive
        hierarchical clustering for better performance with large documents.
        """
        if len(self.items) < 2:
            if self.items:
                self.lines = [self._create_line([self.items[0]])]
            return

        # Determine adaptive threshold based on data distribution
        y_sorted = sorted(item.y for item in self.items)
        y_diffs = [y_sorted[i + 1] - y_sorted[i] for i in range(len(y_sorted) - 1)]
        y_diffs_filtered = [d for d in y_diffs if 0.1 < d < 5.0]

        if y_diffs_filtered:
            threshold = np.percentile(y_diffs_filtered, 50) * 0.7
            threshold = max(0.5, min(2.0, threshold))
        else:
            threshold = 1.0

        # Use efficient simple grouping (O(n log n) vs O(n²) for clustering)
        line_labels = self._simple_y_grouping(threshold)

        # Group items by line label
        line_groups: dict[int, list] = {}
        for i, item in enumerate(self.items):
            label = line_labels[i]
            if label not in line_groups:
                line_groups[label] = []
            line_groups[label].append(item)

        # Create line objects sorted by Y position
        line_data = []
        for items in line_groups.values():
            avg_y = np.mean([item.y for item in items])
            line_data.append((avg_y, items))

        line_data.sort(key=lambda x: x[0])
        self.lines = [self._create_line(items) for _, items in line_data]

    def _simple_y_grouping(self, threshold: float) -> list[int]:
        """Simple fallback Y grouping when clustering fails."""
        sorted_items = sorted(enumerate(self.items), key=lambda x: x[1].y)
        labels = [0] * len(self.items)
        current_label = 1
        current_y = sorted_items[0][1].y

        for idx, item in sorted_items:
            if abs(item.y - current_y) > threshold:
                current_label += 1
                current_y = item.y
            else:
                current_y = (current_y + item.y) / 2
            labels[idx] = current_label

        return labels

    def _create_line(self, items: list[OCRTextData]) -> dict:
        """Create a line dictionary from OCR items."""
        items.sort(key=lambda d: d.x)

        y_position = np.mean([d.y for d in items])
        avg_height = np.mean([d.height for d in items])
        min_x = min(d.x for d in items)
        max_x = max(d.x + d.width for d in items)

        columns = [
            {"x": d.x + d.width / 2, "text": d.text.strip(), "width": d.width} for d in items
        ]

        combined_text = self._combine_line_text(items)
        is_centered, is_right_aligned = self._detect_alignment(min_x, max_x)
        is_table_row = self._detect_table_row(items, y_position)
        is_after_separator = self._detect_after_separator(y_position)

        is_footer = y_position > (self.page_height * 0.88)
        if self.footer_y_threshold is not None:
            threshold_px = (self.footer_y_threshold / 100.0) * self.page_height
            if y_position > threshold_px:
                is_footer = True

        return {
            "text": combined_text,
            "y": y_position,
            "height": avg_height,
            "min_x": min_x,
            "max_x": max_x,
            "columns": columns,
            "is_heading": self._is_heading(combined_text, avg_height, min_x, max_x, y_position),
            "is_centered": is_centered,
            "is_right_aligned": is_right_aligned,
            "is_header": y_position < 10,
            "is_footer": is_footer,
            "is_table_row": is_table_row,
            "is_after_separator": is_after_separator,
            "items": items,
        }

    @staticmethod
    def _combine_line_text(items: list[OCRTextData]) -> str:
        """Combine OCR item texts with smart gap-based spacing."""
        parts: list[str] = []
        for i, item in enumerate(items):
            parts.append(item.text)
            if i < len(items) - 1:
                gap = items[i + 1].x - (item.x + item.width)
                parts.append("  " if gap > 1.5 else " ")
        return "".join(parts).strip()

    @staticmethod
    def _detect_alignment(min_x: float, max_x: float) -> tuple[bool, bool]:
        """Return (is_centered, is_right_aligned) based on margin analysis."""
        is_right_aligned = max_x > 80 and min_x > 30
        left_margin = min_x
        right_margin = 100 - max_x
        width = max_x - min_x
        is_centered = (
            not is_right_aligned
            and width < 60
            and left_margin > 20
            and right_margin > 20
            and abs(left_margin - right_margin) < 10
        )
        return is_centered, is_right_aligned

    def _detect_table_row(self, items: list[OCRTextData], y_position: float) -> bool:
        """Determine if a line is a table row using visual + statistical detection."""
        if self.visual_tables and len(items) >= 2 and self._in_visual_table_region(y_position):
            return True
        if self.vertical_lines and len(items) >= 2 and self._matches_visual_columns(items):
            return True
        return len(items) >= 3 and self._has_table_pattern(items)

    def _detect_after_separator(self, y_position: float) -> bool:
        """Check if line is just below a visual horizontal separator."""
        if not self.h_separators:
            return False
        return any(0 < y_position - sep_y < 3 for sep_y in self.h_separators)

    def _in_visual_table_region(self, y_position: float) -> bool:
        """Check if a Y position falls within any OpenCV-detected table region.

        Args:
            y_position: Y position as percentage (0-100)

        Returns:
            True if the position is inside a detected table
        """
        if not self.visual_tables or not self.visual_data:
            return False

        page_h = self.visual_data.page_height
        if page_h <= 0:
            return False

        for table in self.visual_tables:
            table_y_pct = (table.y / page_h) * 100
            table_end_pct = ((table.y + table.height) / page_h) * 100
            # Add small tolerance
            if table_y_pct - 2 <= y_position <= table_end_pct + 2:
                return True

        return False

    def _matches_visual_columns(self, items: list[OCRTextData]) -> bool:
        """Check if items align with detected visual vertical lines."""
        if not self.vertical_lines:
            return False

        page_width = self.page_width

        if not page_width and self.items:
            page_width = max(item.x + item.width for item in self.items) * 1.05

        if not page_width:
            return False

        item_centers_pct = [((item.x + item.width / 2) / page_width) * 100 for item in items]

        sorted_lines = sorted(self.vertical_lines)
        bins_occupied: set[int] = set()

        for center_pct in item_centers_pct:
            bin_idx = 0
            for line_pct in sorted_lines:
                if center_pct < line_pct:
                    break
                bin_idx += 1
            bins_occupied.add(bin_idx)

        return len(bins_occupied) >= 2

    def _is_heading(
        self,
        text: str,
        height: float,
        min_x: float,
        max_x: float,
        y_position: float = 50.0,
    ) -> bool:
        """Detect if text is a heading based on multiple features."""
        if not text:
            return False

        # Feature 1: Numbered section pattern
        if re.match(r"^\d+\.?\s*[A-ZÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÃÕÇ]", text) and len(text) < 80:
            return True

        # Feature 2: Short uppercase text
        words = text.split()
        if words and len(text) < 60:
            upper_words = sum(1 for w in words if len(w) >= 2 and w.isupper())
            if upper_words >= len(words) * 0.7:
                return True

        # Feature 3: Subsection patterns like "A3 - Busca"
        if re.match(r"^[A-Z]\d*\s*[-–—]\s*[A-Z]", text) and len(text) < 60:
            return True

        # Feature 4: Percentile/classification labels
        if re.match(r"^[A-Z][a-záéíóúàèìòùâêîôûãõç]+\s*[<>=≥≤]", text) and len(text) < 50:
            return True

        # Feature 5: Short centered text (likely title/subtitle)
        if min_x > 30 and max_x < 70 and len(text) < 50:
            return True

        # Feature 6: Title patterns - all caps
        if re.match(r"^[A-ZÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÃÕÇ\s]+$", text) and len(text) > 10 and len(text) < 60:
            return True

        return False

    def _has_table_pattern(self, items: list[OCRTextData]) -> bool:
        """Check if items show a table-like pattern (regular spacing)."""
        if len(items) < 3:
            return False

        gaps = []
        for i in range(1, len(items)):
            gap = items[i].x - (items[i - 1].x + items[i - 1].width)
            gaps.append(gap)

        large_gaps = sum(1 for g in gaps if g > 2.0)
        return large_gaps >= 2

    def _detect_blocks(self) -> None:
        """Detect content blocks (tables, paragraphs, headings, headers, footers)."""
        if not self.lines:
            return

        header_lines = []
        content_lines = []
        footer_lines = []

        for line in self.lines:
            if line.get("is_header"):
                header_lines.append(line)
            elif line.get("is_footer"):
                footer_lines.append(line)
            else:
                content_lines.append(line)

        if header_lines:
            self.blocks.append({"type": "header", "lines": header_lines})

        current_block: str | None = None
        current_lines: list[dict] = []

        for line in content_lines:
            line_type = self._classify_line(line)

            if current_block is None:
                current_block = line_type
                current_lines = [line]
            elif line_type == current_block:
                current_lines.append(line)
            else:
                self._finalize_block(current_block, current_lines)
                current_block = line_type
                current_lines = [line]

        if current_lines:
            self._finalize_block(current_block, current_lines)

        if footer_lines:
            self.blocks.append({"type": "footer", "lines": footer_lines})

    def _classify_line(self, line: dict) -> str:
        """Classify a line as 'heading', 'table', or 'paragraph'."""
        if line.get("is_table_row"):
            return "table"
        if line.get("is_heading"):
            return "heading"
        return "paragraph"

    def _finalize_block(self, block_type: str | None, lines: list[dict]) -> None:
        """Finalize a block and add to results."""
        if block_type == "table" and len(lines) >= 2:
            table_block = self._process_table_block(lines)
            if table_block:
                self.blocks.append(table_block)
            else:
                self._add_paragraph_blocks(lines)
        elif block_type == "heading":
            for line in lines:
                self.blocks.append({"type": "heading", "lines": [line]})
        else:
            self._add_paragraph_blocks(lines)

    def _add_paragraph_blocks(self, lines: list[dict]) -> None:
        """Group lines into paragraphs based on vertical spacing and alignment.

        Enhanced paragraph detection with:
        - Statistical spacing analysis for paragraph breaks
        - First-line indent detection for paragraph starts
        - Short line detection (line ending early indicates paragraph end)
        - Proper alignment detection (left, center, right, justify)
        - Improved centering detection with margin analysis
        """
        if not lines:
            return

        all_min_x = [ln.get("min_x", 0) for ln in lines if ln.get("min_x", 0) > 0]
        typical_left_margin = np.percentile(all_min_x, 25) if all_min_x else 5.0

        def is_centered_line(line: dict) -> bool:
            min_x = line.get("min_x", 0)
            max_x = line.get("max_x", 100)
            width = max_x - min_x
            if width > 60:
                return False
            left_margin = min_x
            right_margin = 100 - max_x
            both_large = left_margin > 20 and right_margin > 20
            symmetric = abs(left_margin - right_margin) < 10
            return both_large and symmetric

        def get_alignment(line: dict, text_width_pct: float = 0) -> str:
            if line.get("is_right_aligned"):
                return "right"
            if is_centered_line(line) or line.get("is_centered"):
                return "center"
            if text_width_pct > 70:
                return "justify"
            return "left"

        def is_short_line(line: dict) -> bool:
            max_x = line.get("max_x", 100)
            min_x = line.get("min_x", 0)
            width = max_x - min_x
            return max_x < 70 and width > 10

        def detect_first_line_indent(curr_line: dict, prev_line: dict | None) -> float:
            curr_min_x = curr_line.get("min_x", 0)
            indent_amount = curr_min_x - typical_left_margin
            if indent_amount > 1.5:
                return indent_amount
            return 0.0

        def detect_paragraph_start(curr_line: dict, prev_line: dict | None) -> bool:
            if prev_line is None:
                return True
            curr_min_x = curr_line.get("min_x", 0)
            prev_min_x = prev_line.get("min_x", 0)
            return curr_min_x > prev_min_x + 1.5

        if len(lines) == 1:
            width_pct = lines[0].get("max_x", 100) - lines[0].get("min_x", 0)
            indent = detect_first_line_indent(lines[0], None)
            self.blocks.append({
                "type": "paragraph",
                "lines": lines,
                "alignment": get_alignment(lines[0], width_pct),
                "first_line_indent": indent,
            })
            return

        spacings = []
        for i in range(1, len(lines)):
            spacing = lines[i]["y"] - lines[i - 1]["y"]
            spacings.append(spacing)

        if not spacings:
            width_pct = lines[0].get("max_x", 100) - lines[0].get("min_x", 0)
            self.blocks.append({
                "type": "paragraph",
                "lines": lines,
                "alignment": get_alignment(lines[0], width_pct),
            })
            return

        median_spacing = np.median(spacings)
        para_threshold = median_spacing * 1.3

        current_para = [lines[0]]
        first_width = lines[0].get("max_x", 100) - lines[0].get("min_x", 0)
        current_alignment = get_alignment(lines[0], first_width)
        current_indent = detect_first_line_indent(lines[0], None)

        for i in range(1, len(lines)):
            spacing = spacings[i - 1]
            prev_line = lines[i - 1]
            curr_line = lines[i]
            curr_width = curr_line.get("max_x", 100) - curr_line.get("min_x", 0)
            line_alignment = get_alignment(curr_line, curr_width)

            is_spacing_break = spacing > para_threshold
            is_heading_break = curr_line.get("is_heading") or prev_line.get("is_heading")
            is_alignment_break = line_alignment != current_alignment
            is_indent_break = detect_paragraph_start(curr_line, prev_line)
            is_prev_short = is_short_line(prev_line)
            is_separator_break = curr_line.get("is_after_separator", False)

            should_break = (
                is_spacing_break
                or is_heading_break
                or is_alignment_break
                or is_separator_break
                or (is_indent_break and spacing >= median_spacing * 0.7)
                or (is_prev_short and not is_centered_line(curr_line))
            )

            if should_break:
                self.blocks.append({
                    "type": "paragraph",
                    "lines": current_para,
                    "alignment": current_alignment,
                    "first_line_indent": current_indent,
                })
                current_para = [curr_line]
                current_alignment = line_alignment
                current_indent = detect_first_line_indent(curr_line, prev_line)
            else:
                current_para.append(curr_line)

        if current_para:
            self.blocks.append({
                "type": "paragraph",
                "lines": current_para,
                "alignment": current_alignment,
                "first_line_indent": current_indent,
            })

    def _cluster_items_by_vertical_lines(
        self, all_items: list[dict], vertical_lines: list[float]
    ) -> tuple[dict, list]:
        """Cluster items using visual vertical line boundaries."""
        sorted_lines = sorted(vertical_lines)
        clusters: dict[int, list[float]] = {}

        for item in all_items:
            x = item["x"]
            col_idx = 0
            for line_x in sorted_lines:
                if x < line_x:
                    break
                col_idx += 1

            cluster_id = col_idx + 1
            item["cluster"] = cluster_id
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(x)

        column_centers = [(np.mean(xs), c) for c, xs in clusters.items()]
        column_centers.sort(key=lambda x: x[0])
        return clusters, column_centers

    def _cluster_items_by_coordinates(self, all_items: list[dict]) -> tuple[dict, list] | None:
        """Cluster items using simple X-coordinate binning for efficiency.

        Replaces scipy fclusterdata with a simpler O(n) algorithm.
        """
        if not all_items:
            return None

        sorted_items = sorted(all_items, key=lambda x: x["x"])

        threshold = 6.0
        clusters: dict[int, list[float]] = {}
        current_cluster = 1
        current_items = [sorted_items[0]]
        sorted_items[0]["cluster"] = current_cluster

        for item in sorted_items[1:]:
            if item["x"] - current_items[-1]["x"] <= threshold:
                item["cluster"] = current_cluster
                current_items.append(item)
            else:
                clusters[current_cluster] = [i["x"] for i in current_items]
                current_cluster += 1
                item["cluster"] = current_cluster
                current_items = [item]

        clusters[current_cluster] = [i["x"] for i in current_items]

        column_centers = [(np.mean(xs), c) for c, xs in clusters.items()]
        column_centers.sort(key=lambda x: x[0])

        return clusters, column_centers

    def _validate_table_structure(
        self, all_items: list[dict], cluster_to_col: dict, num_lines: int
    ) -> bool:
        """Validate that most rows have multiple columns."""
        row_cols: dict[int, set] = {}
        for item in all_items:
            li = item["line_idx"]
            if li not in row_cols:
                row_cols[li] = set()
            row_cols[li].add(cluster_to_col[item["cluster"]])

        multi_col_rows = sum(1 for cols in row_cols.values() if len(cols) >= 2)
        return multi_col_rows >= num_lines * 0.4

    def _build_table_data(
        self,
        all_items: list[dict],
        cluster_to_col: dict,
        num_columns: int,
        num_lines: int,
    ) -> list[dict]:
        """Build table rows data from clustered items."""
        rows = []
        for line_idx in range(num_lines):
            cells = [""] * num_columns
            for item in all_items:
                if item["line_idx"] == line_idx:
                    col_idx = cluster_to_col[item["cluster"]]
                    if cells[col_idx]:
                        cells[col_idx] += " " + item["text"]
                    else:
                        cells[col_idx] = item["text"]
            rows.append({"cells": [c.strip() for c in cells]})
        return rows

    def _process_table_block(self, lines: list[dict]) -> dict | None:
        """Process lines as a table using X-coordinate clustering.

        When OpenCV visual data is available, uses detected table grid lines
        for more precise column identification.
        """
        all_items = [
            {"x": col["x"], "text": col["text"], "line_idx": line_idx}
            for line_idx, line in enumerate(lines)
            for col in line.get("columns", [])
        ]

        if len(all_items) < 4:
            return None

        # Try OpenCV-detected table grid lines first
        visual_v_lines = self._get_visual_table_columns_for_lines(lines)
        if visual_v_lines and len(visual_v_lines) >= 2:
            _, column_centers = self._cluster_items_by_vertical_lines(all_items, visual_v_lines)
        elif self.vertical_lines:
            _, column_centers = self._cluster_items_by_vertical_lines(
                all_items, self.vertical_lines
            )
        else:
            result = self._cluster_items_by_coordinates(all_items)
            if result is None:
                return None
            _, column_centers = result

        num_columns = len(column_centers)
        if num_columns < 2:
            return None

        cluster_to_col = {c: idx for idx, (_, c) in enumerate(column_centers)}

        if not self._validate_table_structure(all_items, cluster_to_col, len(lines)):
            return None

        rows = self._build_table_data(all_items, cluster_to_col, num_columns, len(lines))

        return {
            "type": "table",
            "num_columns": num_columns,
            "columns": [x for x, _ in column_centers],
            "rows": rows,
            "lines": lines,
        }

    def _get_visual_table_columns_for_lines(self, lines: list[dict]) -> list[float] | None:
        """Get OpenCV-detected vertical column positions for lines within a table region.

        Args:
            lines: Lines to check

        Returns:
            List of column X positions as percentages, or None
        """
        if not self.visual_tables or not self.visual_data:
            return None

        page_w = self.visual_data.page_width
        page_h = self.visual_data.page_height
        if page_w <= 0 or page_h <= 0:
            return None

        # Find the visual table region that overlaps with these lines
        avg_y = np.mean([ln["y"] for ln in lines])

        for table in self.visual_tables:
            table_y_pct = (table.y / page_h) * 100
            table_end_pct = ((table.y + table.height) / page_h) * 100

            if table_y_pct - 2 <= avg_y <= table_end_pct + 2:
                # Convert table's vertical line pixel positions to percentages
                return [(vx / page_w) * 100 for vx in table.v_lines]

        return None
