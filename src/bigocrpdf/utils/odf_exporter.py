"""ODF Exporter Module

High-fidelity export of extracted text to OpenDocument Format (ODF).
Analyzes text structure and applies appropriate formatting.

Key features:
- Dynamic line grouping based on actual text height
- Paragraph detection using vertical spacing analysis
- Section/heading detection for uppercase text and numbered sections
- Preservation of document structure (titles, paragraphs, lists)
- Table detection using efficient X-coordinate binning
"""

from odf.draw import Frame, Image
from odf.opendocument import OpenDocumentText
from odf.style import (
    ParagraphProperties,
    Style,
)
from odf.text import P

from bigocrpdf.constants import MIN_IMAGE_FILE_SIZE_BYTES
from bigocrpdf.utils.layout_analyzer import LayoutAnalyzer
from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.odf_frame_mixin import ODFFrameRendererMixin
from bigocrpdf.utils.odf_simple_mixin import ODFSimpleContentMixin
from bigocrpdf.utils.odf_style_mixin import ODFStyleTableMixin
from bigocrpdf.utils.odf_text_mixin import ODFTextPipelineMixin
from bigocrpdf.utils.odf_types import (
    DocumentLayout,
    DocumentParagraph,
    OCRLine,
    OCRTextData,
    TextBlock,
    TextBlockType,
)

# Re-export for backward compatibility
__all__ = [
    "DocumentLayout",
    "DocumentParagraph",
    "LayoutAnalyzer",
    "OCRLine",
    "OCRTextData",
    "ODFExporter",
    "TextBlock",
    "TextBlockType",
]


class ODFExporter(
    ODFSimpleContentMixin,
    ODFFrameRendererMixin,
    ODFStyleTableMixin,
    ODFTextPipelineMixin,
):
    """Exports text to ODF with high-fidelity formatting."""

    def __init__(self):
        self.doc = None
        self.styles = {}
        self._table_counter = 0  # Counter for unique table/style names

    def export_text(self, text: str, output_path: str) -> bool:
        """Export plain text to ODF with detected formatting.

        Args:
            text: The extracted text to export
            output_path: Path for the output .odt file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create new document
            self.doc = OpenDocumentText()

            # Create styles
            self._create_styles()

            # Preprocess text to add line breaks if needed
            processed_text = self._preprocess_text(text)

            # Analyze and parse text structure
            layout = self._analyze_text(processed_text)

            # Generate ODF content
            self._generate_content(layout)

            # Save document
            self.doc.save(output_path)
            logger.info(f"ODF exported successfully: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export ODF: {e}")
            return False

    def export_structured_data(
        self,
        ocr_data: list[OCRTextData],
        output_path: str,
        page_images: dict[int, str] | None = None,
        page_image_paths: list[str] | None = None,
    ) -> bool:
        """Export structured OCR data to ODF with simple, clean formatting.

        Uses a simplified approach similar to TXT export:
        1. Sort items by reading order (Y then X within lines)
        2. Group into paragraphs based on vertical spacing
        3. Apply basic font sizes and alignment

        Args:
            ocr_data: List of OCR text data with position information
            output_path: Path for the output .odt file
            page_images: Deprecated, kept for backwards compatibility (ignored)
            page_image_paths: Optional list of page image paths for OpenCV analysis

        Returns:
            True if successful, False otherwise
        """
        try:
            self.doc = OpenDocumentText()
            self._table_counter = 0
            self._create_styles()
            self._create_frame_styles()

            # Group by page
            pages: dict[int, list[OCRTextData]] = {}
            for item in ocr_data:
                if item.page_num not in pages:
                    pages[item.page_num] = []
                pages[item.page_num].append(item)

            # Process each page using the layout analyzer
            for page_num in sorted(pages.keys()):
                page_data = pages[page_num]
                # Get page image for OpenCV analysis
                img_path = None
                if page_image_paths and (page_num - 1) < len(page_image_paths):
                    img_path = page_image_paths[page_num - 1]
                self._process_page_with_frames(page_data, page_num, page_image_path=img_path)

                # Add page break between pages (except last)
                if page_num < max(pages.keys()):
                    self._add_page_break()

            self.doc.save(output_path)
            logger.info(f"Structured ODF exported: {output_path}")
            return True

        except Exception as e:
            import traceback

            logger.error(f"Failed to export structured ODF: {e}")
            logger.debug(traceback.format_exc())
            return False

    def export_mixed_content(
        self,
        native_text: str,
        images: list[str],
        ocr_texts: list[str],
        output_path: str,
    ) -> bool:
        """Export mixed content PDF (text + images) to ODF.

        Creates an ODF document with:
        1. Native text from the PDF preserved with formatting
        2. Images embedded in the document
        3. OCR text for each image (if available) below the image

        Args:
            native_text: Native text from the PDF
            images: List of paths to extracted images
            ocr_texts: List of OCR text corresponding to each image
            output_path: Path for the output .odt file

        Returns:
            True if successful, False otherwise
        """
        try:
            self.doc = OpenDocumentText()
            self._create_styles()

            # First, add the native text
            if native_text:
                self._add_native_text_section(native_text)

            # Then add images with their OCR text
            if images:
                self._add_images_section(images, ocr_texts)

            self.doc.save(output_path)
            logger.info(f"Mixed content ODF exported: {output_path}")
            return True

        except Exception as e:
            import traceback

            logger.error(f"Failed to export mixed content ODF: {e}")
            logger.debug(traceback.format_exc())
            return False

    def export_paged_images(
        self,
        page_images: list[str],
        page_texts: list[str],
        output_path: str,
    ) -> bool:
        """Export images with OCR text, one image per page with its text below.

        Creates an ODF document where each page's image is followed by its OCR text.

        Args:
            page_images: List of image paths, one per page (sorted by page order)
            page_texts: List of OCR text strings, one per page
            output_path: Path for the output .odt file

        Returns:
            True if successful, False otherwise
        """
        import os

        from PIL import Image as PILImage

        try:
            self.doc = OpenDocumentText()
            self._create_styles()

            # Filter out small images (masks) and match with texts
            MIN_IMAGE_SIZE = MIN_IMAGE_FILE_SIZE_BYTES
            real_images = []
            for img_path in page_images:
                if os.path.exists(img_path):
                    file_size = os.path.getsize(img_path)
                    if file_size >= MIN_IMAGE_SIZE:
                        real_images.append(img_path)

            # Process each page
            for i, img_path in enumerate(real_images):
                try:
                    with PILImage.open(img_path) as pil_img:
                        img_width, img_height = pil_img.size

                    # Calculate display size (max 15cm width)
                    max_width_cm = 15
                    aspect = img_height / img_width
                    display_width = min(max_width_cm, img_width / 37.8)
                    display_height = display_width * aspect

                    # Add page header
                    if i > 0:
                        # Page separator
                        sep_p = P(stylename="Normal")
                        sep_p.addText("─" * 30)
                        self.doc.text.addElement(sep_p)

                    page_header = P(stylename="Bold")
                    page_header.addText(f"Página {i + 1}")
                    self.doc.text.addElement(page_header)

                    # Add image
                    img_para = P(stylename="Normal")
                    frame = Frame(
                        width=f"{display_width:.2f}cm",
                        height=f"{display_height:.2f}cm",
                    )
                    img_href = self.doc.addPicture(img_path)
                    img_element = Image(href=img_href)
                    frame.addElement(img_element)
                    img_para.addElement(frame)
                    self.doc.text.addElement(img_para)

                    # Add OCR text for this page
                    if i < len(page_texts) and page_texts[i]:
                        self._add_formatted_ocr_text(page_texts[i])

                    # Add spacing
                    space_p = P(stylename="Normal")
                    space_p.addText("")
                    self.doc.text.addElement(space_p)

                except Exception as e:
                    logger.warning(f"Error adding page {i + 1} image: {e}")
                    continue

            self.doc.save(output_path)
            logger.info(f"Paged images ODF exported: {output_path}")
            return True

        except Exception as e:
            import traceback

            logger.error(f"Failed to export paged images ODF: {e}")
            logger.debug(traceback.format_exc())
            return False

    def export_formatted_with_images(
        self,
        ocr_data: list[OCRTextData],
        page_images: list[str],
        output_path: str,
    ) -> bool:
        """Export with images and structured/formatted OCR text per page.

        Combines page images with properly formatted OCR text using layout analysis.

        Args:
            ocr_data: List of OCR text data with position information
            page_images: List of image paths, sorted by page order
            output_path: Path for the output .odt file

        Returns:
            True if successful, False otherwise
        """
        import os

        from PIL import Image as PILImage

        try:
            self.doc = OpenDocumentText()
            self._table_counter = 0
            self._create_styles()
            self._create_frame_styles()

            # Filter out small images (masks)
            MIN_IMAGE_SIZE = MIN_IMAGE_FILE_SIZE_BYTES
            real_images = []
            for img_path in page_images:
                if os.path.exists(img_path):
                    if os.path.getsize(img_path) >= MIN_IMAGE_SIZE:
                        real_images.append(img_path)

            # Group OCR data by page
            pages: dict[int, list[OCRTextData]] = {}
            for item in ocr_data:
                if item.page_num not in pages:
                    pages[item.page_num] = []
                pages[item.page_num].append(item)

            max_pages = max(len(real_images), max(pages.keys()) if pages else 0)

            # Process each page
            for page_idx in range(max_pages):
                page_num = page_idx + 1

                # Add page separator (except first page)
                if page_idx > 0:
                    sep_p = P(stylename="Normal")
                    sep_p.addText("─" * 30)
                    self.doc.text.addElement(sep_p)

                # Add page header
                page_header = P(stylename="Bold")
                page_header.addText(f"Página {page_num}")
                self.doc.text.addElement(page_header)

                # Add image if available
                if page_idx < len(real_images):
                    img_path = real_images[page_idx]
                    try:
                        with PILImage.open(img_path) as pil_img:
                            img_width, img_height = pil_img.size

                        max_width_cm = 15
                        aspect = img_height / img_width
                        display_width = min(max_width_cm, img_width / 37.8)
                        display_height = display_width * aspect

                        img_para = P(stylename="Normal")
                        frame = Frame(
                            width=f"{display_width:.2f}cm",
                            height=f"{display_height:.2f}cm",
                        )
                        img_href = self.doc.addPicture(img_path)
                        img_element = Image(href=img_href)
                        frame.addElement(img_element)
                        img_para.addElement(frame)
                        self.doc.text.addElement(img_para)
                    except Exception as e:
                        logger.warning(f"Error adding image for page {page_num}: {e}")

                # Add formatted OCR text for this page
                if page_num in pages:
                    page_data = pages[page_num]
                    # Pass page image for OpenCV analysis
                    img_for_analysis = (
                        real_images[page_idx] if page_idx < len(real_images) else None
                    )
                    self._process_page_with_frames(
                        page_data, page_num, page_image_path=img_for_analysis
                    )

                # Add spacing
                space_p = P(stylename="Normal")
                space_p.addText("")
                self.doc.text.addElement(space_p)

            self.doc.save(output_path)
            logger.info(f"Formatted with images ODF exported: {output_path}")
            return True

        except Exception as e:
            import traceback

            logger.error(f"Failed to export formatted with images ODF: {e}")
            logger.debug(traceback.format_exc())
            return False

    def _get_page_dimensions(self, page_idx, real_images, PILImage):
        """Return (page_w_cm, page_h_cm) for given page from image aspect ratio."""
        page_w_cm = 17.0
        page_h_cm = 25.7
        if page_idx < len(real_images):
            try:
                with PILImage.open(real_images[page_idx]) as pil_img:
                    img_w_px, img_h_px = pil_img.size
                if img_w_px > 0:
                    page_h_cm = page_w_cm * (img_h_px / img_w_px)
            except Exception as e:
                logger.warning(f"Error reading page image {page_idx + 1}: {e}")
        return page_w_cm, page_h_cm

    def export_frame_layout(
        self,
        ocr_data: list[OCRTextData],
        page_images: list[str],
        output_path: str,
    ) -> bool:
        """Export using frame-based absolute positioning for maximum fidelity."""
        import os

        from PIL import Image as PILImage

        try:
            self.doc = OpenDocumentText()
            self._table_counter = 0
            self._create_styles()
            self._create_frame_layout_styles()

            PILImage.MAX_IMAGE_PIXELS = None

            real_images = [
                p
                for p in page_images
                if os.path.exists(p) and os.path.getsize(p) >= MIN_IMAGE_FILE_SIZE_BYTES
            ]

            pages: dict[int, list[OCRTextData]] = {}
            for item in ocr_data:
                pages.setdefault(item.page_num, []).append(item)

            all_valid = [it for it in ocr_data if it.text.strip()]
            global_font_map = _cluster_font_sizes(all_valid) if all_valid else {}
            global_body_pt = _get_body_font_size(global_font_map) if global_font_map else 11.0

            max_pages = max(len(real_images), max(pages.keys()) if pages else 0)

            for page_idx in range(max_pages):
                page_num = page_idx + 1
                if page_idx > 0:
                    self._add_page_break()

                page_w_cm, page_h_cm = self._get_page_dimensions(
                    page_idx,
                    real_images,
                    PILImage,
                )

                if page_num in pages and page_idx < len(real_images):
                    try:
                        from bigocrpdf.utils.visual_analyzer import analyze_text_styles

                        analyze_text_styles(real_images[page_idx], pages[page_num])
                    except Exception as e:
                        logger.debug(f"Style analysis skipped for page {page_num}: {e}")

                if page_num in pages:
                    self._place_text_frames(
                        pages[page_num],
                        page_w_cm,
                        page_h_cm,
                        0,
                        0,
                        global_font_map=global_font_map,
                        global_body_pt=global_body_pt,
                    )

            self.doc.save(output_path)
            logger.info(f"Frame layout ODF exported: {output_path}")
            return True

        except Exception as e:
            import traceback

            logger.error(f"Failed to export frame layout ODF: {e}")
            logger.debug(traceback.format_exc())
            return False

    def _create_frame_layout_styles(self) -> None:
        """Create styles specific to frame-based layout with absolute positioning."""
        from odf.style import (
            FontFace,
            GraphicProperties,
            MasterPage,
            PageLayout,
            PageLayoutProperties,
        )

        # Explicit A4 page layout with 2cm margins
        pl = PageLayout(name="FramePageLayout")
        pl.addElement(
            PageLayoutProperties(
                pagewidth="21cm",
                pageheight="29.7cm",
                margintop="2cm",
                marginbottom="2cm",
                marginleft="2cm",
                marginright="2cm",
            )
        )
        self.doc.automaticstyles.addElement(pl)

        mp = MasterPage(name="Standard", pagelayoutname="FramePageLayout")
        self.doc.masterstyles.addElement(mp)

        # Font face declaration for consistent rendering
        ff = FontFace(
            name="Liberation Sans",
            fontfamily="Liberation Sans",
            fontfamilygeneric="swiss",
            fontpitch="variable",
        )
        self.doc.fontfacedecls.addElement(ff)

        # Transparent positioned text frame style
        frame_text_style = Style(name="FrameTextStyle", family="graphic")
        frame_text_style.addElement(
            GraphicProperties(
                stroke="none",
                fill="none",
                padding="0cm",
                margin="0cm",
                verticalpos="from-top",
                verticalrel="paragraph",
                horizontalpos="from-left",
                horizontalrel="paragraph",
            )
        )
        self.doc.automaticstyles.addElement(frame_text_style)
        self.styles["frame_text"] = frame_text_style

    def _place_text_frames(
        self,
        items: list[OCRTextData],
        page_w_cm: float,
        page_h_cm: float,
        img_w_px: int,
        img_h_px: int,
        *,
        global_font_map: dict[int, float] | None = None,
        global_body_pt: float | None = None,
    ) -> None:
        """Place OCR text as flowing paragraphs with content-fitting layout."""
        if not items:
            return

        content_w_cm = 17.0
        content_h_cm = 25.7

        sorted_items = sorted(items, key=lambda it: (it.y, it.x))
        valid_items = [it for it in sorted_items if it.text.strip()]
        if not valid_items:
            return

        if global_font_map is not None:
            item_font_map = {id(it): global_font_map.get(id(it), 11.0) for it in valid_items}
            body_font_pt = global_body_pt or _get_body_font_size(item_font_map)
        else:
            item_font_map = _cluster_font_sizes(valid_items)
            body_font_pt = _get_body_font_size(item_font_map)
        is_justified = _detect_justified(valid_items)

        lines = _group_into_visual_lines(valid_items)
        if not lines:
            return

        pt_to_cm = 2.54 / 72.0
        avg_char_width_cm = body_font_pt * pt_to_cm * 0.55
        chars_per_line = max(1, content_w_cm / avg_char_width_cm)

        line_info = _build_line_metrics(
            lines,
            item_font_map,
            body_font_pt,
            chars_per_line,
            pt_to_cm,
            content_h_cm,
        )

        scale = _compute_page_scale(line_info, content_h_cm)

        for li in line_info:
            self._emit_paragraph(
                li,
                item_font_map,
                body_font_pt,
                is_justified,
                content_w_cm,
                content_h_cm,
                scale,
            )

    def _emit_paragraph(
        self,
        li: dict,
        item_font_map: dict[int, float],
        body_font_pt: float,
        is_justified: bool,
        content_w_cm: float,
        content_h_cm: float,
        scale: float,
    ) -> None:
        """Render a single visual line as an ODF paragraph with styled spans."""
        from odf.style import ParagraphProperties, TextProperties
        from odf.text import Span

        MIN_FONT = 6.0
        line_items = li["items"]
        x_cm = (li["min_x"] / 100.0) * content_w_cm
        gap_cm = max((li["gap_pct"] / 100.0) * content_h_cm * scale, 0.02)

        line_width = max(it.width for it in line_items)
        align = "justify" if is_justified and line_width > 60 else "left"

        p_style_name = f"FrLn_{self._table_counter}"
        self._table_counter += 1

        margin_top = f"{gap_cm:.2f}cm" if li["is_para_gap"] else f"{min(gap_cm, 0.15):.2f}cm"
        p_style = Style(name=p_style_name, family="paragraph")
        p_style.addElement(
            ParagraphProperties(
                textalign=align,
                marginbottom="0cm",
                marginleft=f"{x_cm:.2f}cm",
                margintop=margin_top,
            )
        )
        self.doc.automaticstyles.addElement(p_style)

        p = P(stylename=p_style_name)
        for i, item in enumerate(line_items):
            text = item.text.strip()
            if not text:
                continue

            font_size_pt = item_font_map.get(id(item), body_font_pt)
            font_size_pt = max(MIN_FONT, round(font_size_pt * scale * 2) / 2)

            span_style_name = f"FrSp_{self._table_counter}"
            self._table_counter += 1
            span_style = Style(name=span_style_name, family="text")
            text_props: dict = {"fontsize": f"{font_size_pt:.1f}pt"}
            if item.is_bold:
                text_props["fontweight"] = "bold"
            if item.is_underlined:
                text_props["textunderlinestyle"] = "solid"
                text_props["textunderlinewidth"] = "auto"
            span_style.addElement(TextProperties(**text_props))
            self.doc.automaticstyles.addElement(span_style)

            if i > 0:
                p.addText("  ")
            span = Span(stylename=span_style_name)
            span.addText(text)
            p.addElement(span)

        self.doc.text.addElement(p)

    def _add_page_break(self) -> None:
        """Add a page break to the document."""
        # Create a style with page break
        break_style = Style(name="PageBreak", family="paragraph")
        break_style.addElement(ParagraphProperties(breakbefore="page"))
        self.doc.automaticstyles.addElement(break_style)

        p = P(stylename=break_style)
        self.doc.text.addElement(p)


# ---------------------------------------------------------------------------
# Font size clustering (module-level helpers for _place_text_frames)
# ---------------------------------------------------------------------------

# Standard typographic scale (point sizes found in professional documents)
_TYPO_SCALE = [6, 7, 8, 9, 10, 10.5, 11, 12, 14, 16, 18, 20, 24, 28, 32, 36, 48]

# Tolerance for grouping raw heights into the same cluster (15%)
_CLUSTER_TOLERANCE = 0.15

# Scale factor from single-line OCR box height → font size.
# OCR bounding boxes include ~40% padding above/below glyphs.
_INITIAL_SCALE = 0.60

# Minimum height ratio to consider a box as multi-line
_MULTILINE_RATIO_THRESHOLD = 1.7

# Characters below which a box is assumed single-line regardless of height
_SHORT_TEXT_THRESHOLD = 15


def _group_into_visual_lines(valid_items: list) -> list[list]:
    """Group OCR items into visual lines based on Y proximity."""
    lines: list[list] = []
    current_line: list = []
    current_y = -999.0
    for item in valid_items:
        if current_line and abs(item.y - current_y) > 1.5:
            lines.append(current_line)
            current_line = []
        current_line.append(item)
        current_y = sum(it.y for it in current_line) / len(current_line)
    if current_line:
        lines.append(current_line)
    return lines


def _build_line_metrics(
    lines: list[list],
    item_font_map: dict[int, float],
    body_font_pt: float,
    chars_per_line: float,
    pt_to_cm: float,
    content_h_cm: float,
) -> list[dict]:
    """Compute per-line metrics: font size, height, gap, wrapping, para detection."""
    line_info: list[dict] = []
    prev_y_pct = 0.0
    for line_items in lines:
        min_y = min(it.y for it in line_items)
        line_font_pt = max(item_font_map.get(id(it), body_font_pt) for it in line_items)
        line_h_cm = line_font_pt * pt_to_cm * 1.35
        total_chars = sum(len(it.text) for it in line_items) + len(line_items) - 1
        wrapped = max(1, total_chars / chars_per_line)
        gap_pct = max(min_y - prev_y_pct, 0.0)
        prev_y_pct = min_y + (line_font_pt * pt_to_cm / content_h_cm) * 100
        line_info.append({
            "items": line_items,
            "font_pt": line_font_pt,
            "line_h_cm": line_h_cm,
            "wrapped": wrapped,
            "gap_pct": gap_pct,
            "min_x": min(it.x for it in line_items),
            "is_para_gap": gap_pct > 4.0,
        })
    return line_info


def _compute_page_scale(line_info: list[dict], content_h_cm: float) -> float:
    """Compute a scale factor to fit all lines within the page content height."""
    total_text_cm = sum(li["line_h_cm"] * li["wrapped"] for li in line_info)
    total_gap_cm = sum((li["gap_pct"] / 100.0) * content_h_cm for li in line_info)
    total_cm = total_text_cm + total_gap_cm
    if total_cm > content_h_cm and total_cm > 0:
        return max(content_h_cm / total_cm, 0.5)
    return 1.0


def _snap_to_typo_scale(pt: float) -> float:
    """Snap a point size to the nearest standard typographic size."""
    best = _TYPO_SCALE[0]
    best_dist = abs(pt - best)
    for s in _TYPO_SCALE[1:]:
        d = abs(pt - s)
        if d < best_dist:
            best = s
            best_dist = d
    return best


def _find_single_line_baseline(heights: list[float]) -> float:
    """Find a representative single-line box height from a set of OCR heights.

    Uses the lower portion of the height distribution, since the shortest
    items are guaranteed to be single-line.  We pick the median of the
    bottom-third to be robust against outliers.
    """
    if not heights:
        return 0.0
    s = sorted(heights)
    n = max(1, len(s) // 3)
    return s[n // 2]


def _estimate_line_count(height: float, baseline: float, text_len: int) -> int:
    """Estimate how many visual text lines an OCR bounding box spans.

    Uses the ratio of box height to the single-line baseline.  Short text
    snippets (< _SHORT_TEXT_THRESHOLD characters) are assumed single-line
    even when tall (large-font headings / logos).
    """
    if baseline <= 0 or height <= 0:
        return 1
    ratio = height / baseline
    if ratio < _MULTILINE_RATIO_THRESHOLD:
        return 1
    if text_len < _SHORT_TEXT_THRESHOLD:
        return 1
    return max(1, round(ratio))


def _cluster_font_sizes(items: list) -> dict[int, float]:
    """Cluster OCR box heights into standard typographic sizes.

    Multi-line boxes are normalised to their per-line height before
    clustering so that body-text lines with 2-3 merged lines do not
    skew the result.

    Returns a dict mapping ``id(item)`` → ``font_size_pt`` for every item.
    """
    if not items:
        return {}

    raw_heights = [it.height for it in items]
    baseline = _find_single_line_baseline(raw_heights)

    # Step 1: normalise multi-line boxes then apply scale
    raw_pts: list[tuple[int, float]] = []
    for it in items:
        n_lines = _estimate_line_count(it.height, baseline, len(it.text.strip()))
        norm_h = it.height / n_lines
        raw_pts.append((id(it), norm_h * _INITIAL_SCALE))

    # Step 2: sort and cluster (values within TOLERANCE of each other)
    sorted_pts = sorted(raw_pts, key=lambda x: x[1])
    clusters: list[list[tuple[int, float]]] = [[sorted_pts[0]]]

    for item_id, pt in sorted_pts[1:]:
        cluster_avg = sum(p for _, p in clusters[-1]) / len(clusters[-1])
        if abs(pt - cluster_avg) / max(cluster_avg, 0.1) <= _CLUSTER_TOLERANCE:
            clusters[-1].append((item_id, pt))
        else:
            clusters.append([(item_id, pt)])

    # Step 3: find the body text cluster (most items)
    body_cluster = max(clusters, key=len)
    body_avg = sum(p for _, p in body_cluster) / len(body_cluster)
    body_snapped = _snap_to_typo_scale(body_avg)

    # Step 4: derive calibration factor from body cluster
    if body_avg > 0:
        cal = body_snapped / body_avg
    else:
        cal = 1.0

    # Step 5: apply calibration to all clusters and snap
    result: dict[int, float] = {}
    for cluster in clusters:
        cluster_avg = sum(p for _, p in cluster) / len(cluster)
        calibrated = cluster_avg * cal
        snapped = _snap_to_typo_scale(calibrated)
        snapped = max(6.0, min(snapped, 48.0))
        for item_id, _ in cluster:
            result[item_id] = snapped

    return result


def _get_body_font_size(font_map: dict[int, float]) -> float:
    """Return the most common (body text) font size from the cluster map."""
    if not font_map:
        return 11.0
    from collections import Counter

    counts = Counter(font_map.values())
    return counts.most_common(1)[0][0]


def _detect_justified(items: list) -> bool:
    """Detect whether body text appears to use justified alignment.

    Considers full-width body paragraphs (width > 60%) and checks if
    their right edges cluster within a narrow band, which indicates
    justified or right-aligned (hence justified) text.
    """
    body = [it for it in items if it.width > 60]
    if len(body) < 5:
        return False
    right_edges = [it.x + it.width for it in body]
    right_edges.sort()
    median_r = right_edges[len(right_edges) // 2]
    tight = sum(1 for r in right_edges if abs(r - median_r) < 3.0)
    return tight / len(body) > 0.6
