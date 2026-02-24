"""
Text Layer Rendering for Searchable PDFs.

This module overlays invisible OCR text onto PDF pages using ReportLab,
creating searchable PDFs while preserving the original image appearance.
"""

import logging
import math
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


def _escape_pdf(text: str) -> str:
    """Escape special PDF string characters."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


@dataclass
class TextBox:
    """Represents a text box with position and content.

    All coordinates are in PDF points (1/72 inch), with origin at bottom-left.

    Attributes:
        text: The text content
        x: X coordinate (left edge)
        y: Y coordinate (baseline)
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

    Accepts an ``OCRConfig`` for font management and DPI-based
    coordinate conversion.
    """

    # Register fonts as class variable to avoid re-registration
    _registered_fonts: set[str] = set()

    def __init__(self, config: "OCRConfig | int") -> None:
        """Initialize the renderer.

        Args:
            config: OCR configuration object (or plain ``int`` DPI
                    for backward compatibility).
        """
        # Backward-compat: accept a bare int (DPI) as in the old API
        if isinstance(config, int):
            from bigocrpdf.services.rapidocr_service.config import OCRConfig as _Cfg

            _cfg = _Cfg(dpi=config)
            self.config = _cfg
        else:
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
        ocr_results: list["OCRResult"],
        image_width_px: int,
        image_height_px: int,
    ) -> PageTextLayer:
        """Convert OCR results to a text layer.

        Uses the full OCR quadrilateral for accurate height/width
        measurement and baseline positioning.  Text is kept horizontal
        (no rotation) so that PDF text extractors (pdfgrep, pdftotext,
        copy-paste) correctly reconstruct line order.

        Args:
            ocr_results: List of OCR results for one page.
            image_width_px: Image width in pixels.
            image_height_px: Image height in pixels.

        Returns:
            PageTextLayer with positioned text boxes.
        """
        dpi = float(self.config.dpi)
        px_to_pt = 72.0 / dpi
        width_pts = image_width_px * px_to_pt
        height_pts = image_height_px * px_to_pt

        layer = PageTextLayer(
            page_num=1,
            width_pts=width_pts,
            height_pts=height_pts,
            image_width_px=image_width_px,
            image_height_px=image_height_px,
        )

        # Helvetica descent as fraction of em-size (207/1000)
        DESCENT_FRAC = 0.207

        for result in ocr_results:
            text = result.text.strip()
            if not text:
                continue

            coords = result.box
            confidence = result.confidence

            if len(coords) < 4:
                continue

            try:
                if isinstance(coords[0], (list, tuple)):
                    # Quadrilateral: TL, TR, BR, BL (RapidOCR order)
                    tl, tr, br, bl = coords[0], coords[1], coords[2], coords[3]
                else:
                    # Flat [x1,y1,x2,y2] — build axis-aligned quad
                    tl = [coords[0], coords[1]]
                    tr = [coords[2], coords[1]]
                    br = [coords[2], coords[3]]
                    bl = [coords[0], coords[3]]

                # Text width along reading direction (bottom edge length)
                dx = br[0] - bl[0]
                dy = br[1] - bl[1]
                text_width_px = math.hypot(dx, dy)

                # Text height perpendicular to reading direction
                left_h = math.hypot(tl[0] - bl[0], tl[1] - bl[1])
                right_h = math.hypot(tr[0] - br[0], tr[1] - br[1])
                text_height_px = (left_h + right_h) / 2.0

                width = text_width_px * px_to_pt
                height = text_height_px * px_to_pt

                font_size = max(
                    MIN_FONT_SIZE,
                    min(height * FONT_SIZE_SCALE_FACTOR, MAX_FONT_SIZE),
                )

                # Horizontal position: left edge of the quadrilateral
                x_min = min(tl[0], bl[0])
                x_pdf = x_min * px_to_pt

                # Vertical position: baseline with descent offset.
                # Use the average of the bottom edge y-coords → PDF y-flip.
                y_bottom_avg = (bl[1] + br[1]) / 2.0
                y_bottom_pts = y_bottom_avg * px_to_pt
                descent_pts = DESCENT_FRAC * font_size
                y_pdf = height_pts - y_bottom_pts + descent_pts

                layer.boxes.append(
                    TextBox(
                        text=text,
                        x=x_pdf,
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

    def _sort_for_reading_order(
        self, results: list["OCRResult"], page_width: float = 0
    ) -> list["OCRResult"]:
        """Sort OCR results in reading order (top-to-bottom, left-to-right).

        Args:
            results: OCR results with bounding boxes.
            page_width: Page width in pixels (unused, kept for API compat).

        Returns:
            Sorted list of OCR results.
        """
        return sorted(
            results,
            key=lambda r: (min(p[1] for p in r.box), min(p[0] for p in r.box)),
        )

    @staticmethod
    def _snap_baselines(boxes: list[TextBox]) -> None:
        """Align text boxes on the same visual line to the same y-coordinate.

        PDF text extractors (pdftotext, pdfgrep) use vertical position to
        group characters into lines.  When OCR boxes on the same visual line
        have slightly different y-coords (scan curvature), extractors split
        them into separate lines.

        Uses sequential clustering by y-proximity with cluster centroid
        comparison.  Each cluster is snapped to its median y value.
        """
        if not boxes:
            return

        # Sort by y descending (top of page = largest PDF y first)
        sorted_boxes = sorted(boxes, key=lambda b: -b.y)

        clusters: list[list[TextBox]] = []
        current_cluster: list[TextBox] = [sorted_boxes[0]]

        for box in sorted_boxes[1:]:
            cluster_y = sum(b.y for b in current_cluster) / len(current_cluster)
            cluster_min_h = min(b.height for b in current_cluster)
            # Threshold: 35% of the smaller height between cluster min and
            # current box.  For 12pt text this is ~4.2pt — catches scan
            # curvature (≤4pt) but stays below typical inter-line gap (≥8pt).
            threshold = min(cluster_min_h, box.height) * 0.35
            if abs(cluster_y - box.y) <= threshold:
                current_cluster.append(box)
            else:
                clusters.append(current_cluster)
                current_cluster = [box]
        clusters.append(current_cluster)

        # Snap each cluster to its median y
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            ys = sorted(b.y for b in cluster)
            median_y = ys[len(ys) // 2]
            for b in cluster:
                b.y = median_y

    def render(
        self,
        canvas: canvas.Canvas,
        ocr_results: list["OCRResult"],
        image_size: tuple[int, int],
        rotation: int = 0,
    ) -> int:
        """Render invisible text layer directly to an existing canvas.

        Args:
            canvas: ReportLab canvas to draw on.
            ocr_results: List of OCR results for one page.
            image_size: Tuple of ``(width, height)`` in pixels.
            rotation: Rotation angle in degrees (0, 90, 180, 270).

        Returns:
            Number of text regions rendered.
        """
        layer = self.create_text_layer(ocr_results, image_size[0], image_size[1])

        # Snap baselines so text extractors group same-line boxes together
        self._snap_baselines(layer.boxes)

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

        count = 0
        font_name = self._font_name or "Helvetica"

        # Group boxes by snapped baseline (same y = same line).
        # Rendering all same-line text in one BT/ET block ensures
        # that pdftotext treats them as a single line.
        from collections import defaultdict

        lines: dict[float, list[TextBox]] = defaultdict(list)
        for box in layer.boxes:
            lines[box.y].append(box)

        for y_val, line_boxes in sorted(lines.items(), reverse=True):
            # Sort by x position (left to right reading order)
            line_boxes.sort(key=lambda b: b.x)

            try:
                canvas.saveState()

                text_obj = canvas.beginText()
                # Invisible text render mode — must be inside BT/ET
                text_obj._code.append("3 Tr")

                first = True
                line_start_x = 0.0
                line_start_y = 0.0

                for box in line_boxes:
                    natural_w = pdfmetrics.stringWidth(box.text, font_name, box.font_size)
                    if natural_w > 0 and box.width > 0:
                        h_scale = box.width / natural_w * 100.0
                    else:
                        h_scale = 100.0

                    if first:
                        text_obj.setTextOrigin(box.x, box.y)
                        line_start_x = box.x
                        line_start_y = box.y
                        first = False
                    else:
                        # Td is relative to the text line matrix (set by
                        # the last Td/Tm), NOT the text matrix cursor
                        # (which textOut advances).
                        dx = box.x - line_start_x
                        dy = box.y - line_start_y
                        text_obj._code.append(f"{dx:.2f} {dy:.2f} Td")
                        line_start_x = box.x
                        line_start_y = box.y

                    # setFont for TTFonts defers Tf emission — the
                    # actual /Subset Tf operator is emitted later by
                    # textOut() via _formatText(), which handles font
                    # subsetting correctly.
                    text_obj.setFont(font_name, box.font_size)
                    text_obj.setHorizScale(h_scale)
                    # textOut handles TTFont subset encoding and emits
                    # the correct Tf + Tj operators.
                    text_obj.textOut(box.text)
                    count += 1

                canvas.drawText(text_obj)
                canvas.restoreState()
            except Exception as e:
                logger.debug(f"Failed to render line at y={y_val:.1f}: {e}")
                continue

        canvas.restoreState()
        return count
