"""
Text Layer Rendering for Searchable PDFs.

This module overlays invisible OCR text onto PDF pages using ReportLab,
creating searchable PDFs while preserving the original image appearance.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from bigocrpdf.constants import FONT_SIZE_SCALE_FACTOR, MAX_FONT_SIZE, MIN_FONT_SIZE

if TYPE_CHECKING:
    from bigocrpdf.services.rapidocr_service.config import OCRConfig, OCRResult

logger = logging.getLogger(__name__)


@dataclass
class TextBox:
    """Represents a text box with position and content.

    All coordinates are in PDF points (1/72 inch), with origin at bottom-left.

    Attributes:
        text: The text content
        x: X coordinate (left edge)
        y: Y coordinate (bottom edge)
        width: Box width
        height: Box height
        confidence: OCR confidence score (0-1)
        font_size: Calculated font size in points
    """

    text: str
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0
    font_size: float = 12.0


@dataclass
class PageTextLayer:
    """Text layer data for one page.

    Attributes:
        page_num: 1-based page number
        boxes: List of text boxes
        width_pts: Page width in points
        height_pts: Page height in points
        image_width_px: Original image width in pixels
        image_height_px: Original image height in pixels
    """

    page_num: int
    boxes: list[TextBox] = field(default_factory=list)
    width_pts: float = A4[0]
    height_pts: float = A4[1]
    image_width_px: int = 0
    image_height_px: int = 0


class TextLayerRenderer:
    """Creates invisible text layers for searchable PDFs.

    The renderer creates PDF pages with transparent text positioned
    exactly where OCR detected text in the original image. The text
    is invisible (render mode 3) but selectable and searchable.
    """

    # Register fonts as class variable to avoid re-registration
    _registered_fonts: set[str] = set()

    def __init__(self, config: "OCRConfig") -> None:
        """Initialize the renderer.

        Args:
            config: OCR configuration object
        """
        self.config = config
        self._font_name: str | None = None
        self._setup_font()

    def _setup_font(self) -> None:
        """Register the font for the current language."""
        font_path = self.config.get_font_path()
        if not font_path:
            logger.warning("No font configured, using Helvetica")
            self._font_name = "Helvetica"
            return

        font_path = Path(font_path)
        if not font_path.exists():
            logger.warning(f"Font not found: {font_path}, using Helvetica")
            self._font_name = "Helvetica"
            return

        # Create a unique font name
        font_name = f"OCRFont_{font_path.stem}"

        if font_name not in self._registered_fonts:
            try:
                pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
                self._registered_fonts.add(font_name)
                logger.debug(f"Registered font: {font_name} from {font_path}")
            except Exception as e:
                logger.warning(f"Failed to register font {font_path}: {e}")
                self._font_name = "Helvetica"
                return

        self._font_name = font_name

    def create_text_layer(
        self,
        ocr_result: "OCRResult",
        image_width_px: int,
        image_height_px: int,
    ) -> PageTextLayer:
        """Convert OCR results to a text layer.

        Transforms OCR bounding box coordinates from image pixel space
        to PDF point space using the target DPI.

        Args:
            ocr_result: OCR result containing text boxes
            image_width_px: Image width in pixels
            image_height_px: Image height in pixels

        Returns:
            PageTextLayer with positioned text boxes
        """
        # Calculate page dimensions in points
        dpi = float(self.config.dpi)
        width_pts = image_width_px / dpi * 72.0
        height_pts = image_height_px / dpi * 72.0

        layer = PageTextLayer(
            page_num=1,
            width_pts=width_pts,
            height_pts=height_pts,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
        )

        if not ocr_result.boxes:
            return layer

        for box in ocr_result.boxes:
            text = box.get("text", "").strip()
            if not text:
                continue

            coords = box.get("box", [])
            confidence = box.get("confidence", 1.0)

            if len(coords) < 4:
                continue

            # Convert coordinates from pixels to points
            # RapidOCR returns [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # as top-left, top-right, bottom-right, bottom-left
            try:
                if isinstance(coords[0], (list, tuple)):
                    # Quadrilateral format
                    x_coords = [pt[0] for pt in coords]
                    y_coords = [pt[1] for pt in coords]
                else:
                    # [x1, y1, x2, y2] format
                    x_coords = [coords[0], coords[2]]
                    y_coords = [coords[1], coords[3]]

                x_min = min(x_coords) / dpi * 72.0
                x_max = max(x_coords) / dpi * 72.0
                y_min = min(y_coords) / dpi * 72.0
                y_max = max(y_coords) / dpi * 72.0

                width = x_max - x_min
                height = y_max - y_min

                # Convert Y coordinate (PDF origin is bottom-left)
                y_pdf = height_pts - y_max

                # Calculate font size based on box height
                font_size = max(MIN_FONT_SIZE, min(height * FONT_SIZE_SCALE_FACTOR, MAX_FONT_SIZE))

                layer.boxes.append(
                    TextBox(
                        text=text,
                        x=x_min,
                        y=y_pdf,
                        width=width,
                        height=height,
                        confidence=confidence,
                        font_size=font_size,
                    )
                )
            except (IndexError, TypeError, ValueError) as e:
                logger.debug(f"Failed to process box: {e}")
                continue

        return layer

    def render(
        self,
        canvas: canvas.Canvas,
        ocr_result: "OCRResult",
        image_size: tuple[int, int],
        rotation: int = 0,
    ) -> int:
        """Render text layer directly to an existing canvas.

        Args:
            canvas: ReportLab canvas to draw on
            ocr_result: OCR result containing text boxes
            image_size: Tuple of (width, height) in pixels
            rotation: Rotation angle in degrees (0, 90, 180, 270)
        """
        # Create text layer (computes boxes in Points relative to image_size)
        layer = self.create_text_layer(ocr_result, image_size[0], image_size[1])

        canvas.saveState()

        # Apply transform if rotation is needed to match PDF coordinate system
        # layer.width/height correspond to the View dimensions (Upright)

        if rotation == 90:
            # Rotate 90 deg CW (relative to PDF coords)
            # Map Text(x,y) -> Page(W-y, x)
            # layer.height_pts is approx Page Width
            canvas.translate(layer.height_pts, 0)
            canvas.rotate(90)
        elif rotation == 180:
            # Rotate 180
            # Map Text(x,y) -> Page(W-x, H-y)
            canvas.translate(layer.width_pts, layer.height_pts)
            canvas.rotate(180)
        elif rotation == 270:
            # Rotate 270 (90 CCW)
            # Map Text(x,y) -> Page(y, H-x)
            # layer.width_pts is approx Page Height
            canvas.translate(0, layer.width_pts)
            canvas.rotate(270)

        # Draw invisible text
        canvas.setFillColorRGB(0, 0, 0, 0)  # Transparent fill
        canvas.setStrokeColorRGB(0, 0, 0, 0)  # Transparent stroke

        for box in layer.boxes:
            try:
                canvas.setFont(self._font_name or "Helvetica", box.font_size)

                # Set text render mode to invisible (mode 3)
                canvas.saveState()
                canvas._code.append("3 Tr")  # Invisible text render mode

                # Draw the text
                canvas.drawString(box.x, box.y, box.text)

                canvas.restoreState()
            except Exception as e:
                logger.debug(f"Failed to render text '{box.text[:20]}...': {e}")
                continue

        canvas.restoreState()
