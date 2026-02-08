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
        self, ocr_data: list[OCRTextData], output_path: str, page_images: dict[int, str] = None
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
                self._process_page_with_frames(page_data, page_num)

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
            MIN_IMAGE_SIZE = 2048  # 2KB minimum
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
            MIN_IMAGE_SIZE = 2048
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
                    self._process_page_with_frames(page_data, page_num)

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

    def _add_page_break(self) -> None:
        """Add a page break to the document."""
        # Create a style with page break
        break_style = Style(name="PageBreak", family="paragraph")
        break_style.addElement(ParagraphProperties(breakbefore="page"))
        self.doc.automaticstyles.addElement(break_style)

        p = P(stylename=break_style)
        self.doc.text.addElement(p)
