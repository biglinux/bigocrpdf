"""ODF Frame Renderer Mixin

Methods for frame-based page rendering: visual blocks,
headings, headers, footers, and table blocks.
"""

from __future__ import annotations

from odf.style import (
    GraphicProperties,
    ParagraphProperties,
    Style,
    TableCellProperties,
    TableProperties,
    TextProperties,
)
from odf.table import Table, TableCell, TableColumn, TableRow
from odf.text import P

from bigocrpdf.utils.layout_analyzer import LayoutAnalyzer
from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.odf_types import OCRTextData


class ODFFrameRendererMixin:
    """Mixin providing frame-based block rendering for ODFExporter."""

    def _create_frame_styles(self) -> None:
        """Create styles for text frames used in absolute positioning."""

        # Basic frame style - no border, transparent background
        frame_style = Style(name="FrameStyle", family="graphic")
        frame_style.addElement(
            GraphicProperties(
                wrap="none",
                border="none",
                padding="0cm",
                margin="0cm",
            )
        )
        self.doc.automaticstyles.addElement(frame_style)
        self.styles["frame"] = frame_style

    def _process_page_with_frames(
        self,
        page_data: list[OCRTextData],
        page_num: int,
    ) -> None:
        """Process page data using layout analysis.

        Args:
            page_data: OCR data for one page
            page_num: Page number (1-indexed)
        """
        if not page_data:
            return

        logger.debug(f"Page {page_num}: Processing {len(page_data)} OCR boxes with LayoutAnalyzer")

        # Use the LayoutAnalyzer for statistical layout analysis
        analyzer = LayoutAnalyzer(page_data)
        blocks = analyzer.analyze()

        logger.debug(f"Page {page_num}: LayoutAnalyzer detected {len(blocks)} blocks")

        # Separate header, footer blocks from content blocks
        header_block = None
        content_blocks = []
        footer_block = None

        for block in blocks:
            if block.get("type") == "header":
                header_block = block
            elif block.get("type") == "footer":
                footer_block = block
            else:
                content_blocks.append(block)

        # Add header first (if any)
        if header_block and header_block.get("lines"):
            self._add_header_block(header_block["lines"])

        # Generate ODF content from detected blocks
        for block in content_blocks:
            block_type = block.get("type", "paragraph")
            lines = block.get("lines", [])
            alignment = block.get("alignment", "left")
            first_line_indent = block.get("first_line_indent", 0.0)

            if block_type == "table":
                # Generate ODF table using pre-computed cell data
                self._add_table_block(block)
            elif block_type == "heading":
                # Generate heading paragraph with appropriate style
                if lines:
                    self._add_heading_block(lines)
            else:
                # Generate normal paragraph with alignment and indent
                if lines:
                    self._add_paragraph_from_lines(
                        lines, alignment=alignment, first_line_indent=first_line_indent
                    )

        # Add footer at the bottom of the page content
        if footer_block and footer_block.get("lines"):
            self._add_footer_block(footer_block["lines"])

    def _add_table_block(self, block: dict) -> None:
        """Generate ODF table from analyzed block with pre-computed cell data.

        Args:
            block: Table block from LayoutAnalyzer containing:
                - num_columns: Number of columns
                - columns: List of column X positions (centers)
                - rows: List of rows with 'cells' list
        """
        num_columns = block.get("num_columns", 0)
        rows = block.get("rows", [])

        if not rows or num_columns < 2:
            return

        # Create table style
        table_style_name = f"Table{self._table_counter}"
        self._table_counter += 1

        table_style = Style(name=table_style_name, family="table")
        table_style.addElement(TableProperties(width="17cm", align="margins"))
        self.doc.automaticstyles.addElement(table_style)

        # Create cell style (centered, with borders)
        cell_style_name = f"TableCell{self._table_counter}"
        cell_style = Style(name=cell_style_name, family="table-cell")
        cell_style.addElement(
            TableCellProperties(
                padding="0.05cm", border="0.5pt solid #000000", verticalalign="middle"
            )
        )
        self.doc.automaticstyles.addElement(cell_style)

        # Create table
        table = Table(name=f"Table{self._table_counter}", stylename=table_style_name)

        # Define columns
        for _ in range(num_columns):
            table.addElement(TableColumn())

        # Process each row
        for row_data in rows:
            row = TableRow()
            cells = row_data.get("cells", [])

            for col_idx in range(num_columns):
                cell = TableCell(stylename=cell_style_name)

                cell_text = cells[col_idx] if col_idx < len(cells) else ""
                p = P()
                if cell_text:
                    p.addText(cell_text)
                cell.addElement(p)
                row.addElement(cell)

            table.addElement(row)

        self.doc.text.addElement(table)

        # Add empty line after table
        self.doc.text.addElement(P())

    def _add_heading_block(self, lines: list[dict]) -> None:
        """Generate ODF heading paragraph from analyzed block.

        Args:
            lines: List of line dictionaries (typically just one for heading)
        """
        if not lines:
            return

        # Join all lines into heading text and find max height
        text_parts = []
        max_height = 12.0  # Default fallback

        for line in lines:
            line_text = line.get("text", "")
            if line_text:
                text_parts.append(line_text)

            # Track largest font for heading style
            items = line.get("items", [])
            for item in items:
                if item.height > max_height:
                    max_height = item.height

        if not text_parts:
            return

        heading_text = " ".join(text_parts)

        # Calculate font size using consistent formula
        font_size = self._calculate_font_size(max_height, is_heading=True)

        # Determine heading level based on font size (adjusted for smaller buckets)
        if font_size >= 16:
            heading_level = 1
        elif font_size >= 14:
            heading_level = 2
        else:
            heading_level = 3

        # Create heading style
        heading_style_name = f"Heading{heading_level}_{self._table_counter}"
        self._table_counter += 1

        heading_style = Style(name=heading_style_name, family="paragraph")
        heading_style.addElement(ParagraphProperties(margintop="0.3cm", marginbottom="0.2cm"))
        heading_style.addElement(TextProperties(fontsize=f"{font_size}pt", fontweight="bold"))
        self.doc.automaticstyles.addElement(heading_style)

        # Create heading paragraph
        p = P(stylename=heading_style_name)
        p.addText(heading_text)
        self.doc.text.addElement(p)

    def _add_header_block(self, lines: list[dict]) -> None:
        """Generate ODF header section from detected header lines.

        The header typically contains letterhead information like company name,
        logo text, address, etc. It's displayed smaller and with special styling.

        Args:
            lines: List of line dictionaries from the header area
        """
        if not lines:
            return

        # Create header style - smaller font, special formatting
        header_style_name = f"Header_{self._table_counter}"
        self._table_counter += 1

        header_style = Style(name=header_style_name, family="paragraph")
        header_style.addElement(
            ParagraphProperties(
                margintop="0.1cm",
                marginbottom="0.1cm",
            )
        )
        header_style.addElement(
            TextProperties(
                fontsize="11pt",
                fontweight="bold",
                color="#333333",
            )
        )
        self.doc.automaticstyles.addElement(header_style)

        # Create subtitle style for secondary header lines
        subtitle_style_name = f"HeaderSub_{self._table_counter}"
        subtitle_style = Style(name=subtitle_style_name, family="paragraph")
        subtitle_style.addElement(
            ParagraphProperties(
                margintop="0.05cm",
                marginbottom="0.2cm",
            )
        )
        subtitle_style.addElement(
            TextProperties(
                fontsize="9pt",
                color="#666666",
            )
        )
        self.doc.automaticstyles.addElement(subtitle_style)

        # Add header lines
        for i, line in enumerate(lines):
            text = line.get("text", "").strip()
            if text:
                # First line is main header, rest are subtitles
                if i == 0:
                    p = P(stylename=header_style_name)
                else:
                    p = P(stylename=subtitle_style_name)
                p.addText(text)
                self.doc.text.addElement(p)

        # Add separator after header
        sep_style_name = f"HeaderSep_{self._table_counter}"
        sep_style = Style(name=sep_style_name, family="paragraph")
        sep_style.addElement(
            ParagraphProperties(
                marginbottom="0.3cm",
                borderbottom="0.5pt solid #cccccc",
            )
        )
        self.doc.automaticstyles.addElement(sep_style)
        self.doc.text.addElement(P(stylename=sep_style_name))

    def _add_footer_block(self, lines: list[dict]) -> None:
        """Generate ODF footer section from detected footer lines.

        The footer is added as a styled paragraph at the bottom of page content,
        with a separator line above it.

        Args:
            lines: List of line dictionaries from the footer area
        """
        if not lines:
            return

        # Add a visual separator before footer
        separator_style_name = f"FooterSep_{self._table_counter}"
        sep_style = Style(name=separator_style_name, family="paragraph")
        sep_style.addElement(
            ParagraphProperties(
                margintop="0.5cm",
                marginbottom="0.1cm",
                borderbottom="0.5pt solid #cccccc",
            )
        )
        sep_style.addElement(TextProperties(fontsize="8pt"))
        self.doc.automaticstyles.addElement(sep_style)

        # Add separator paragraph
        sep_p = P(stylename=separator_style_name)
        self.doc.text.addElement(sep_p)

        # Create footer style - smaller font, centered
        footer_style_name = f"Footer_{self._table_counter}"
        self._table_counter += 1

        footer_style = Style(name=footer_style_name, family="paragraph")
        footer_style.addElement(
            ParagraphProperties(
                margintop="0.1cm",
                marginbottom="0.05cm",
                textalign="center",
            )
        )
        footer_style.addElement(
            TextProperties(
                fontsize="9pt",
                color="#666666",
            )
        )
        self.doc.automaticstyles.addElement(footer_style)

        # Combine footer lines
        for line in lines:
            text = line.get("text", "").strip()
            if text:
                p = P(stylename=footer_style_name)
                p.addText(text)
                self.doc.text.addElement(p)
