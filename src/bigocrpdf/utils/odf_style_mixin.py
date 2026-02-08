"""ODF Style and Table Mixin

Methods for style management, table building, paragraph
formatting, and alignment detection.
"""

from __future__ import annotations

import re

from odf.style import (
    ParagraphProperties,
    Style,
    TableCellProperties,
    TableProperties,
    TextProperties,
)
from odf.text import P, Tab


class ODFStyleTableMixin:
    """Mixin providing style and table building for ODFExporter."""

    def _calculate_font_size(self, box_height: float, is_heading: bool = False) -> float:
        """Calculate font size from OCR box height.

        The bounding box height from pdftotext -bbox is typically 3-4x larger
        than the actual font size. We use a divisor of 3.5 for accurate conversion.

        Standard size buckets:
        - Body: 9, 10, 11, 12pt
        - Subheadings: 12, 13, 14pt
        - Headings: 14, 16, 18pt (reduced from larger values)

        Args:
            box_height: OCR text box height in points (bounding box)
            is_heading: Whether this is heading text (uses larger buckets)

        Returns:
            Normalized font size in points
        """
        # Bounding box height is ~3.5x font size
        raw_size = max(4, min(box_height / 3.5, 48))

        # Standard size buckets for consistency (smaller headings)
        if is_heading:
            buckets = [12, 13, 14, 16, 18]  # Reduced from [14, 16, 18, 20, 24]
        else:
            buckets = [9, 10, 11, 12]  # Normal body text

        # Find closest standard size
        return min(buckets, key=lambda x: abs(x - raw_size))

    def _add_paragraph_from_lines(
        self, lines: list[dict], alignment: str = "left", first_line_indent: float = 0.0
    ) -> None:
        """Add a paragraph to the ODF document from grouped lines.

        Calculates font size based on box height using PDF's formula,
        then normalizes to standard document sizes for consistency.

        Args:
            lines: List of line dicts that form this paragraph
            alignment: Text alignment - 'left', 'center', 'right', or 'justify'
            first_line_indent: First line indent percentage - if > 0, adds TAB at start
        """
        if not lines:
            return

        # Combine all line texts
        text_parts = [line["text"] for line in lines]
        combined_text = " ".join(text_parts)
        combined_text = re.sub(r"\s{2,}", " ", combined_text).strip()

        if not combined_text:
            return

        # Check if we need TAB at start (>1.5% indent = add tab)
        needs_tab = first_line_indent > 1.5 and alignment not in ("center", "right")

        # Detect alignment from line positions
        first_line = lines[0]
        detected_alignment = self._detect_alignment(first_line, combined_text)
        if detected_alignment != "left":
            alignment = detected_alignment

        # Determine style based on first line characteristics
        is_heading = first_line.get("is_heading", False)

        if is_heading:
            style_name = "h1" if re.match(r"^\d+\.", combined_text) else "h2"
        elif first_line.get("is_centered") and len(combined_text) < 60:
            style_name = "centered"
            alignment = "center"
        elif first_line.get("is_short_line") and len(lines) == 1:
            style_name = "field"
        else:
            style_name = "normal"

        # Calculate font size using PDF's proven formula
        avg_height = sum(line["height"] for line in lines) / len(lines)
        font_size_pt = self._calculate_font_size(avg_height, is_heading)

        # Get or create style with font size and alignment
        style = self._get_or_create_sized_style(style_name, font_size_pt, alignment)

        # Create paragraph - add ODF Tab element if indent needed
        p = P(stylename=style)
        if needs_tab:
            p.addElement(Tab())
        p.addText(combined_text)
        self.doc.text.addElement(p)

    def _detect_alignment(self, line: dict, text: str) -> str:
        """Detect text alignment from line position.

        Uses position percentages to determine alignment:
        - Right: ends near right margin (>80%) and starts after 30%
        - Center: content centered in middle 50% of page
        - Justify: long paragraphs (default for body text)
        - Left: default

        Args:
            line: Line dict with min_x, max_x positions
            text: Combined text for length analysis

        Returns:
            Alignment string: 'left', 'center', 'right', or 'justify'
        """
        min_x = line.get("min_x", 0)
        max_x = line.get("max_x", 100)
        width = max_x - min_x
        left_margin = min_x
        right_margin = 100 - max_x

        # Right-aligned: ends near right margin but doesn't start from left
        if line.get("is_right_aligned") or (max_x > 85 and min_x > 35):
            return "right"

        # STRICT Centered detection - only true center, not indented first lines
        # Requires: both margins >20%, symmetric within 10%, narrow (<60%)
        is_truly_centered = line.get("is_centered") or (
            left_margin > 20
            and right_margin > 20
            and abs(left_margin - right_margin) < 10
            and width < 60
        )
        if is_truly_centered:
            return "center"

        # Justify for longer body text (more than 60% of page width)
        if width > 60 and len(text) > 80:
            return "justify"

        return "left"

    def _get_or_create_sized_style(
        self,
        base_style: str,
        font_size_pt: float,
        alignment: str = "left",
    ) -> Style:
        """Get or create a paragraph style with specific font size and alignment.

        Caches styles to avoid duplication. Maps alignments to ODF values:
        - left -> 'left'
        - center -> 'center'
        - right -> 'end'
        - justify -> 'justify'

        Args:
            base_style: Base style name ('normal', 'h1', etc.)
            font_size_pt: Font size in points
            alignment: Text alignment

        Returns:
            Style object with correct font size and alignment
        """
        # Create unique style name including alignment
        size_str = str(font_size_pt).replace(".", "_")
        align_suffix = "" if alignment == "left" else f"_{alignment}"
        style_name = f"{base_style}_{size_str}pt{align_suffix}"

        if style_name in self.styles:
            return self.styles[style_name]

        # Map alignment to ODF text-align value
        align_map = {
            "left": "left",
            "center": "center",
            "right": "end",
            "justify": "justify",
        }
        odf_align = align_map.get(alignment, "left")

        # For centered styles, force center alignment
        if base_style in ("centered", "title"):
            odf_align = "center"

        # Determine text properties based on base style
        is_bold = base_style in ("h1", "h2", "title")

        # Create new style with appropriate properties
        new_style = Style(name=style_name, family="paragraph")

        # Set margins based on style type
        margin_top = "0.3cm" if is_bold else "0.1cm"
        margin_bottom = "0.2cm" if is_bold else "0.15cm"

        new_style.addElement(
            ParagraphProperties(
                marginbottom=margin_bottom,
                margintop=margin_top,
                textalign=odf_align,
            )
        )

        # Text properties
        text_props = {"fontsize": f"{font_size_pt}pt"}
        if is_bold:
            text_props["fontweight"] = "bold"

        new_style.addElement(TextProperties(**text_props))

        self.doc.automaticstyles.addElement(new_style)
        self.styles[style_name] = new_style

        return new_style

    def _create_styles(self) -> None:
        """Create document styles for formatting."""
        # Title style - large, centered, bold
        title_style = Style(name="Title", family="paragraph")
        title_style.addElement(ParagraphProperties(textalign="center", marginbottom="0.5cm"))
        title_style.addElement(TextProperties(fontsize="24pt", fontweight="bold"))
        self.doc.styles.addElement(title_style)
        self.styles["title"] = title_style

        # Heading styles (H1-H3) - for numbered sections and headers
        for level, size, margin_top, margin_bottom in [
            (1, "14pt", "0.5cm", "0.3cm"),  # Main section headers
            (2, "13pt", "0.4cm", "0.2cm"),  # Sub-section headers
            (3, "12pt", "0.3cm", "0.2cm"),  # Minor headers
        ]:
            h_style = Style(name=f"Heading{level}", family="paragraph")
            h_style.addElement(
                ParagraphProperties(
                    marginbottom=margin_bottom,
                    margintop=margin_top,
                    keepwithnext="always",
                )
            )
            h_style.addElement(TextProperties(fontsize=size, fontweight="bold"))
            self.doc.styles.addElement(h_style)
            self.styles[f"h{level}"] = h_style

        # Centered text style
        centered_style = Style(name="Centered", family="paragraph")
        centered_style.addElement(ParagraphProperties(textalign="center"))
        centered_style.addElement(TextProperties(fontsize="12pt"))
        self.doc.styles.addElement(centered_style)
        self.styles["centered"] = centered_style

        # Normal paragraph style with proper spacing
        normal_style = Style(name="Normal", family="paragraph")
        normal_style.addElement(
            ParagraphProperties(
                marginbottom="0.15cm",
                margintop="0.1cm",
                textindent="0cm",
                textalign="left",
            )
        )
        normal_style.addElement(TextProperties(fontsize="11pt"))
        self.doc.styles.addElement(normal_style)
        self.styles["normal"] = normal_style

        # Field label style (for form fields like "Nome:", "Data:")
        field_style = Style(name="FieldLabel", family="paragraph")
        field_style.addElement(ParagraphProperties(marginbottom="0.1cm", margintop="0.1cm"))
        field_style.addElement(TextProperties(fontsize="11pt"))
        self.doc.styles.addElement(field_style)
        self.styles["field"] = field_style

        # List item style
        list_style = Style(name="ListItem", family="paragraph")
        list_style.addElement(ParagraphProperties(marginleft="1cm", marginbottom="0.1cm"))
        list_style.addElement(TextProperties(fontsize="11pt"))
        self.doc.styles.addElement(list_style)
        self.styles["list"] = list_style

        # Large text styles for varying font sizes
        for size in [10, 14, 16, 18, 20, 24, 28, 32]:
            size_style = Style(name=f"Size{size}", family="paragraph")
            size_style.addElement(TextProperties(fontsize=f"{size}pt"))
            self.doc.styles.addElement(size_style)
            self.styles[f"size{size}"] = size_style

        # Table styles
        table_style = Style(name="TableStyle", family="table")
        table_style.addElement(TableProperties(width="100%", align="margins"))
        self.doc.automaticstyles.addElement(table_style)
        self.styles["table"] = table_style

        cell_style = Style(name="CellStyle", family="table-cell")
        cell_style.addElement(TableCellProperties(padding="0.2cm", border="0.1pt solid #000000"))
        self.doc.automaticstyles.addElement(cell_style)
        self.styles["cell"] = cell_style
