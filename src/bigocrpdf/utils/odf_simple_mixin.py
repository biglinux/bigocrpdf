"""ODF Simple Content Mixin

Methods for simple content rendering: native text sections,
formatted OCR text, images, and simple paragraph grouping.
"""

from __future__ import annotations

import re

from odf.draw import Frame, Image
from odf.text import P

from bigocrpdf.constants import MIN_IMAGE_FILE_SIZE_BYTES
from bigocrpdf.utils.logger import logger


class ODFSimpleContentMixin:
    """Mixin providing simple content rendering for ODFExporter."""

    def _add_native_text_section(self, text: str) -> None:
        """Add native text section to the document with proper formatting.

        Args:
            text: Native text from the PDF
        """

        # Split text into lines while preserving structure
        lines = text.split("\n")

        # Track consecutive blank lines to add spacing
        blank_count = 0
        current_para_lines: list[str] = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                # Blank line
                blank_count += 1

                # If we have content accumulated, output it
                if current_para_lines:
                    self._output_paragraph(current_para_lines)
                    current_para_lines = []

                    # Add extra spacing for multiple blank lines
                    if blank_count >= 2:
                        space_p = P(stylename="Normal")
                        space_p.addText("")
                        self.doc.text.addElement(space_p)
            else:
                # Non-blank line
                # Check if this looks like a title (short, possibly uppercase)
                is_title = len(stripped) < 50 and (
                    stripped.isupper()
                    or stripped.endswith(":")
                    or re.match(r"^[A-Z][^.]*$", stripped)  # Starts uppercase, no periods
                )

                if blank_count >= 2 and current_para_lines:
                    # New section after multiple blank lines
                    self._output_paragraph(current_para_lines)
                    current_para_lines = []
                    # Add spacing
                    space_p = P(stylename="Normal")
                    space_p.addText("")
                    self.doc.text.addElement(space_p)

                if is_title and not current_para_lines:
                    # Output as a heading/bold line
                    p = P(stylename="Bold")
                    p.addText(stripped)
                    self.doc.text.addElement(p)
                else:
                    # Add to current paragraph
                    current_para_lines.append(stripped)

                blank_count = 0

        # Output any remaining content
        if current_para_lines:
            self._output_paragraph(current_para_lines)

    def _output_paragraph(self, lines: list[str]) -> None:
        """Output a paragraph with proper line handling.

        Args:
            lines: List of lines in the paragraph
        """
        if not lines:
            return

        p = P(stylename="Normal")
        for i, line in enumerate(lines):
            if i > 0:
                p.addText(" ")  # Add space between lines in same paragraph
            p.addText(line)
        self.doc.text.addElement(p)

    def _add_formatted_ocr_text(self, text: str) -> None:
        """Add OCR text with proper paragraph formatting, preserving line breaks.

        Args:
            text: OCR text to format and add
        """
        from odf.text import LineBreak

        # Split into lines and process each
        lines = text.split("\n")
        current_para = P(stylename="Normal")
        line_count = 0

        for line in lines:
            stripped = line.strip()

            if not stripped:
                # Empty line - output current paragraph and start new one
                if line_count > 0:
                    self.doc.text.addElement(current_para)
                    current_para = P(stylename="Normal")
                    line_count = 0
                continue

            # Check if this looks like a heading
            is_heading = (
                len(stripped) < 60
                and line_count == 0
                and (
                    stripped.isupper()
                    or stripped.endswith(":")
                    or stripped.startswith(("1.", "2.", "3.", "4.", "5."))
                )
            )

            if is_heading:
                # Output any pending content first
                if line_count > 0:
                    self.doc.text.addElement(current_para)
                    current_para = P(stylename="Normal")
                    line_count = 0

                # Add heading as bold
                heading_p = P(stylename="Bold")
                heading_p.addText(stripped)
                self.doc.text.addElement(heading_p)
            else:
                # Add to current paragraph with line break if needed
                if line_count > 0:
                    current_para.addElement(LineBreak())
                current_para.addText(stripped)
                line_count += 1

        # Output any remaining content
        if line_count > 0:
            self.doc.text.addElement(current_para)

    def _add_images_section(self, images: list[str], ocr_texts: list[str]) -> None:
        """Add images section with OCR text to the document.

        Args:
            images: List of paths to images
            ocr_texts: List of OCR text corresponding to each image
        """
        import os

        from PIL import Image as PILImage

        # Filter out small images (likely alpha masks, < 2KB)
        MIN_IMAGE_SIZE = MIN_IMAGE_FILE_SIZE_BYTES
        real_images = []
        largest_image_idx = -1
        largest_image_size = 0

        for img_path in images:
            if os.path.exists(img_path):
                file_size = os.path.getsize(img_path)
                if file_size >= MIN_IMAGE_SIZE:
                    real_images.append(img_path)
                    if file_size > largest_image_size:
                        largest_image_size = file_size
                        largest_image_idx = len(real_images) - 1

        if not real_images:
            logger.info("No real images found (all were masks)")
            return

        # Get OCR text (combine all available texts)
        combined_ocr_text = "\n\n".join(t for t in ocr_texts if t and t.strip())

        # Add separator
        sep_p = P(stylename="Normal")
        sep_p.addText("─" * 30)
        self.doc.text.addElement(sep_p)

        header_p = P(stylename="Bold")
        header_p.addText("Imagens do documento:")
        self.doc.text.addElement(header_p)

        for i, img_path in enumerate(real_images):
            # Get image dimensions
            try:
                with PILImage.open(img_path) as pil_img:
                    img_width, img_height = pil_img.size

                # Calculate display size (max 15cm width)
                max_width_cm = 15
                aspect = img_height / img_width
                display_width = min(max_width_cm, img_width / 37.8)  # ~96 DPI
                display_height = display_width * aspect

                # Add image to document
                img_para = P(stylename="Normal")

                # Create frame for image (no style, just dimensions)
                frame = Frame(
                    width=f"{display_width:.2f}cm",
                    height=f"{display_height:.2f}cm",
                )

                # Add image to frame
                img_href = self.doc.addPicture(img_path)
                img_element = Image(href=img_href)
                frame.addElement(img_element)

                img_para.addElement(frame)
                self.doc.text.addElement(img_para)

                # Add OCR text below the largest image (most likely to contain text)
                if i == largest_image_idx and combined_ocr_text:
                    ocr_para = P(stylename="Small")
                    ocr_para.addText("Texto reconhecido nesta imagem:")
                    self.doc.text.addElement(ocr_para)

                    # Format OCR text with proper paragraph structure
                    self._add_formatted_ocr_text(combined_ocr_text)

                # Add space between images
                space_p = P(stylename="Normal")
                space_p.addText("")
                self.doc.text.addElement(space_p)

            except Exception as e:
                logger.warning(f"Error adding image {img_path}: {e}")
                continue
