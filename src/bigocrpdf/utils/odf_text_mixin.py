"""ODF Text Pipeline Mixin

Methods for text analysis, classification, paragraph
and table generation from raw text input.
"""

from __future__ import annotations

import re

from odf.table import Table, TableCell, TableColumn, TableRow
from odf.text import P

from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.odf_types import (
    DocumentLayout,
    TextBlock,
    TextBlockType,
)


class ODFTextPipelineMixin:
    """Mixin providing text analysis pipeline for ODFExporter."""

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text to add line breaks before common section patterns.

        Args:
            text: Raw text without proper line breaks

        Returns:
            Text with added line breaks
        """
        # If text already has line breaks, don't modify it
        if text.count("\n") > 5:
            return text

        # Patterns that indicate new sections/paragraphs (language-agnostic)
        section_patterns = [
            r"(\d+\.\s+[A-Z][A-Za-zÀ-ú\s]+)",  # "1. SECTION TITLE"
            r"(\d+\.\d+\s+[A-Z])",  # "1.1 Subsection"
            r"([A-ZÀ-Ú][a-zà-ú]+(?: [a-zà-ú]+)*:)",  # "Field name:" form labels
            r"(Dra?\.\s+[A-Z]|Mrs?\.\s+[A-Z]|Prof\.\s+[A-Z])",  # Titles: Dr./Dra./Mr./Mrs./Prof.
            r"([A-Z]{4,}(?:\s+[A-Z]{4,})*)",  # ALL-CAPS SECTION HEADERS
        ]

        result = text
        for pattern in section_patterns:
            result = re.sub(pattern, r"\n\n\1", result)

        # Also add breaks after common endings
        result = re.sub(r"(\.\s+)(\d+\.)", r"\1\n\n\2", result)  # Period before numbered section

        # Clean up multiple newlines
        result = re.sub(r"\n{3,}", "\n\n", result)
        result = result.strip()

        logger.info(f"Preprocessed text: added {result.count(chr(10))} line breaks")
        return result

    def _analyze_text(self, text: str) -> DocumentLayout:
        """Analyze text structure and detect formatting patterns.

        Args:
            text: Raw text to analyze

        Returns:
            DocumentLayout with detected blocks
        """
        layout = DocumentLayout()
        lines = text.split("\n")

        # Calculate statistics for heuristic detection
        line_lengths = [len(line.strip()) for line in lines if line.strip()]
        avg_length = sum(line_lengths) / len(line_lengths) if line_lengths else 50
        max_length = max(line_lengths) if line_lengths else 80

        for i, line in enumerate(lines):
            stripped = line.strip()

            if not stripped:
                # Empty line - paragraph separator
                layout.blocks.append(TextBlock(text="", block_type=TextBlockType.EMPTY))
                continue

            block = self._classify_line(stripped, line, avg_length, max_length, i)
            layout.blocks.append(block)

        return layout

    def _classify_line(
        self,
        stripped: str,
        original: str,
        avg_length: float,
        max_length: float,
        _line_num: int,
    ) -> TextBlock:
        """Classify a single line of text.

        Args:
            stripped: Stripped line text
            original: Original line with whitespace
            avg_length: Average line length in document
            max_length: Maximum line length
            line_num: Line number

        Returns:
            Classified TextBlock
        """
        block = TextBlock(text=stripped)
        line_len = len(stripped)

        # Check for leading whitespace (potential centering)
        leading_spaces = len(original) - len(original.lstrip())
        if leading_spaces > 5 and line_len < avg_length * 0.7:
            block.is_centered = True
            block.block_type = TextBlockType.CENTERED

        # Check for all caps (likely heading)
        if stripped.isupper() and len(stripped) > 3:
            block.is_bold = True
            if line_len < avg_length * 0.5:
                block.block_type = TextBlockType.HEADING
                block.font_size = 16.0
            elif line_len < avg_length * 0.3:
                block.block_type = TextBlockType.TITLE
                block.font_size = 24.0
                block.is_centered = True

        # Check for list patterns
        list_patterns = [
            r"^[\d]+[.)]\s",  # Numbered: 1. 2) etc
            r"^[•●○■□▪▸►-]\s",  # Bullets
            r"^[a-z][.)]\s",  # Lettered: a. b) etc
            r"^[A-Z][.)]\s",  # Capital lettered
            r"^\*\s",  # Asterisk
        ]
        for pattern in list_patterns:
            if re.match(pattern, stripped):
                block.block_type = TextBlockType.LIST_ITEM
                break

        # Check for short lines (potential titles/headers)
        if (
            line_len < 50
            and line_len < avg_length * 0.4
            and not block.block_type == TextBlockType.LIST_ITEM
        ):
            # Short line - check if it's a title pattern
            if re.match(r"^[A-Z][A-Za-z\s]+$", stripped) and line_len < 40:
                block.block_type = TextBlockType.HEADING
                block.font_size = 14.0
                block.is_bold = True

        # Check for table-like patterns (tab or multiple space separation)
        if "\t" in stripped or re.search(r"\s{3,}", stripped):
            parts = re.split(r"\t|\s{3,}", stripped)
            if len(parts) >= 2:
                block.block_type = TextBlockType.TABLE_ROW

        # Default large text detection based on surrounding context
        # (would need more context for accurate detection)

        return block

    def _generate_content(self, layout: DocumentLayout) -> None:
        """Generate ODF content from analyzed layout.

        Args:
            layout: Analyzed document layout
        """
        table_rows = []

        for block in layout.blocks:
            # Handle table accumulation
            if block.block_type == TextBlockType.TABLE_ROW:
                table_rows.append(block)
                continue
            elif table_rows:
                # Flush accumulated table rows
                self._create_table(table_rows)
                table_rows = []

            # Handle other block types
            if block.block_type == TextBlockType.EMPTY:
                # Add empty paragraph for spacing
                p = P()
                self.doc.text.addElement(p)

            elif block.block_type == TextBlockType.TITLE:
                p = P(stylename=self.styles["title"], text=block.text)
                self.doc.text.addElement(p)

            elif block.block_type == TextBlockType.HEADING:
                style_name = "h1" if block.font_size >= 18 else "h2"
                p = P(stylename=self.styles.get(style_name, self.styles["h2"]))
                p.addText(block.text)
                self.doc.text.addElement(p)

            elif block.block_type == TextBlockType.CENTERED:
                p = P(stylename=self.styles["centered"], text=block.text)
                self.doc.text.addElement(p)

            elif block.block_type == TextBlockType.LIST_ITEM:
                p = P(stylename=self.styles["list"], text=block.text)
                self.doc.text.addElement(p)

            else:  # PARAGRAPH
                p = P(stylename=self.styles["normal"], text=block.text)
                self.doc.text.addElement(p)

        # Flush any remaining table rows
        if table_rows:
            self._create_table(table_rows)

    def _create_table(self, rows: list[TextBlock]) -> None:
        """Create a table from table row blocks.

        Args:
            rows: List of table row blocks
        """
        if not rows:
            return

        # Parse rows into cells
        parsed_rows = []
        max_cols = 0
        for row in rows:
            cells = re.split(r"\t|\s{3,}", row.text)
            cells = [c.strip() for c in cells if c.strip()]
            parsed_rows.append(cells)
            max_cols = max(max_cols, len(cells))

        if max_cols < 2:
            # Not really a table, treat as paragraphs
            for row in rows:
                p = P(stylename=self.styles["normal"], text=row.text)
                self.doc.text.addElement(p)
            return

        # Create table
        table = Table(stylename=self.styles["table"])

        # Add columns
        for _ in range(max_cols):
            col = TableColumn()
            table.addElement(col)

        # Add rows
        for cells in parsed_rows:
            tr = TableRow()
            for i in range(max_cols):
                tc = TableCell(stylename=self.styles["cell"])
                cell_text = cells[i] if i < len(cells) else ""
                p = P(text=cell_text)
                tc.addElement(p)
                tr.addElement(tc)
            table.addElement(tr)

        self.doc.text.addElement(table)
