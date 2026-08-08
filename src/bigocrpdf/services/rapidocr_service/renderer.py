"""
Text Layer Rendering for Searchable PDFs.

This module overlays invisible OCR text onto PDF pages using ReportLab,
creating searchable PDFs while preserving the original image appearance.
"""

import logging
import math
from collections import defaultdict
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


def _line_text_with_spacing(
    line_boxes: list[TextBox],
    font_name: str,
    line_font_size: float,
) -> str:
    space_width = pdfmetrics.stringWidth(" ", font_name, line_font_size)
    if space_width <= 0:
        space_width = line_font_size * 0.25

    parts: list[str] = []
    for i, box in enumerate(line_boxes):
        parts.append(box.text)
        if i < len(line_boxes) - 1:
            next_box = line_boxes[i + 1]
            gap = next_box.x - (box.x + box.width)
            num_spaces = max(1, round(gap / space_width)) if gap > 0 else 1
            parts.append(" " * num_spaces)
    return "".join(parts)


def _line_horizontal_scale(
    line_text: str,
    line_width: float,
    font_name: str,
    line_font_size: float,
) -> float:
    natural_width = pdfmetrics.stringWidth(line_text, font_name, line_font_size)
    if natural_width > 0 and line_width > 0:
        return line_width / natural_width * 100.0
    return 100.0


def _normalize_ocr_quadrilateral(
    box: object,
) -> (
    tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]] | None
):
    """Validate nested or legacy flat OCR coordinates as four numeric points."""
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    raw_points = (
        box[:4]
        if isinstance(box[0], (list, tuple))
        else ((box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3]))
    )
    try:
        points = tuple((float(point[0]), float(point[1])) for point in raw_points)
    except (IndexError, TypeError, ValueError):
        return None
    if len(points) != 4 or not all(math.isfinite(value) for point in points for value in point):
        return None
    return points[0], points[1], points[2], points[3]


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
        page_size_pts: tuple[float, float] | None = None,
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
            page_size_pts: If provided, use these page dimensions (width, height)
                in points for coordinate mapping instead of computing from DPI.
                Required for overlay mode where the image DPI may differ from
                the configured DPI.

        Returns:
            PageTextLayer with positioned text boxes.
        """
        if page_size_pts:
            width_pts, height_pts = page_size_pts
            px_to_pt_x = width_pts / image_width_px
            px_to_pt_y = height_pts / image_height_px
            # Use average for font size scaling
            px_to_pt = (px_to_pt_x + px_to_pt_y) / 2.0
        else:
            dpi = float(self.config.dpi)
            px_to_pt = 72.0 / dpi
            px_to_pt_x = px_to_pt
            px_to_pt_y = px_to_pt
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

            confidence = result.confidence
            quadrilateral = _normalize_ocr_quadrilateral(result.box)
            if quadrilateral is None:
                continue

            try:
                # Quadrilateral: TL, TR, BR, BL (RapidOCR order)
                tl, tr, br, bl = quadrilateral

                # Text width along reading direction (bottom edge length)
                dx = br[0] - bl[0]
                dy = br[1] - bl[1]
                text_width_px = math.hypot(dx, dy)

                # Text height perpendicular to reading direction
                left_h = math.hypot(tl[0] - bl[0], tl[1] - bl[1])
                right_h = math.hypot(tr[0] - br[0], tr[1] - br[1])
                text_height_px = (left_h + right_h) / 2.0

                width = text_width_px * px_to_pt_x
                height = text_height_px * px_to_pt_y

                font_size = max(
                    MIN_FONT_SIZE,
                    min(height * FONT_SIZE_SCALE_FACTOR, MAX_FONT_SIZE),
                )

                # Horizontal position: left edge of the quadrilateral
                x_min = min(tl[0], bl[0])
                x_pdf = x_min * px_to_pt_x

                # Vertical position: baseline with descent offset.
                # Use the average of the bottom edge y-coords → PDF y-flip.
                y_bottom_avg = (bl[1] + br[1]) / 2.0
                y_bottom_pts = y_bottom_avg * px_to_pt_y
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
        page_size_pts: tuple[float, float] | None = None,
        image_offset: tuple[float, float] | None = None,
    ) -> int:
        """Render invisible text layer directly to an existing canvas.

        Args:
            canvas: ReportLab canvas to draw on.
            ocr_results: List of OCR results for one page.
            image_size: Tuple of ``(width, height)`` in pixels.
            rotation: Rotation angle in degrees (0, 90, 180, 270).
            page_size_pts: If provided, map pixel coordinates to this page
                size instead of using DPI-based calculation.
            image_offset: If provided, translate the text layer by (x, y)
                points so OCR text aligns with an image that does not start
                at the page origin.

        Returns:
            Number of text regions rendered.
        """
        layer = self.create_text_layer(
            ocr_results, image_size[0], image_size[1], page_size_pts=page_size_pts
        )

        # Snap baselines so text extractors group same-line boxes together
        self._snap_baselines(layer.boxes)

        canvas.saveState()

        # Shift text layer to the image position when the image does not
        # start at the page origin (e.g. centred or offset images).
        if image_offset:
            canvas.translate(image_offset[0], image_offset[1])

        self._apply_page_rotation(canvas, layer, rotation)

        # PDF text render mode 3 makes the text invisible without alpha states.
        font_name = self._font_name or "Helvetica"
        count = self._render_text_lines(canvas, layer.boxes, font_name)
        canvas.restoreState()
        return count

    @staticmethod
    def _apply_page_rotation(
        canvas: canvas.Canvas,
        layer: PageTextLayer,
        rotation: int,
    ) -> None:
        if rotation == 90:
            canvas.translate(layer.height_pts, 0)
            canvas.rotate(90)
        elif rotation == 180:
            canvas.translate(layer.width_pts, layer.height_pts)
            canvas.rotate(180)
        elif rotation == 270:
            canvas.translate(0, layer.width_pts)
            canvas.rotate(270)

    @staticmethod
    def _text_lines(boxes: list[TextBox]) -> list[tuple[float, list[TextBox]]]:
        lines: dict[float, list[TextBox]] = defaultdict(list)
        for box in boxes:
            lines[box.y].append(box)
        return sorted(lines.items(), reverse=True)

    def _render_text_lines(
        self,
        canvas: canvas.Canvas,
        boxes: list[TextBox],
        font_name: str,
    ) -> int:
        count = 0
        for y_val, line_boxes in self._text_lines(boxes):
            line_boxes.sort(key=lambda b: b.x)
            try:
                count += self._render_text_line(canvas, line_boxes, font_name)
            except Exception as e:
                logger.debug(f"Failed to render line at y={y_val:.1f}: {e}")
        return count

    @staticmethod
    def _render_text_line(
        canvas: canvas.Canvas,
        line_boxes: list[TextBox],
        font_name: str,
    ) -> int:
        canvas.saveState()
        line_font_size = sum(b.font_size for b in line_boxes) / len(line_boxes)
        line_text = _line_text_with_spacing(line_boxes, font_name, line_font_size)
        line_x = line_boxes[0].x
        line_y = line_boxes[0].y
        line_width = (line_boxes[-1].x + line_boxes[-1].width) - line_boxes[0].x
        h_scale = _line_horizontal_scale(line_text, line_width, font_name, line_font_size)

        text_obj = canvas.beginText()
        text_obj.setTextRenderMode(3)
        text_obj.setTextOrigin(line_x, line_y)
        text_obj.setFont(font_name, line_font_size)
        text_obj.setHorizScale(h_scale)
        text_obj.textOut(line_text)

        canvas.drawText(text_obj)
        canvas.restoreState()
        return len(line_boxes)
