"""
ODF Type Definitions

Data classes used throughout the ODF export pipeline for representing
text blocks, OCR data, document paragraphs, and layout structures.
"""

from dataclasses import dataclass, field
from enum import Enum, auto


class TextBlockType(Enum):
    """Types of text blocks detected in the document."""

    TITLE = auto()  # Large, centered text
    HEADING = auto()  # Bold or uppercase headers
    NUMBERED_HEADING = auto()  # Numbered section like "1. DADOS"
    PARAGRAPH = auto()  # Normal text paragraphs
    LIST_ITEM = auto()  # Bulleted or numbered items
    TABLE_ROW = auto()  # Data arranged in columns
    CENTERED = auto()  # Centered text
    EMPTY = auto()  # Empty line (paragraph separator)
    FIELD_LABEL = auto()  # Form field like "Nome:", "Data:"


@dataclass
class TextBlock:
    """Represents a block of text with formatting information."""

    text: str
    block_type: TextBlockType = TextBlockType.PARAGRAPH
    font_size: float = 12.0  # in points
    is_bold: bool = False
    is_centered: bool = False


@dataclass
class OCRTextData:
    """Structured OCR data with position and formatting information."""

    text: str
    x: float = 0.0  # horizontal position (0-100 as percentage of page width)
    y: float = 0.0  # vertical position
    width: float = 100.0  # width as percentage of page width
    height: float = 12.0  # estimated font height in points
    confidence: float = 1.0
    page_num: int = 1
    is_bold: bool = False
    is_underlined: bool = False


@dataclass
class OCRLine:
    """Represents a single line of text composed of multiple OCR boxes."""

    items: list[OCRTextData] = field(default_factory=list)
    y_position: float = 0.0  # average Y position
    avg_height: float = 12.0  # average text height
    text: str = ""  # combined text
    is_title: bool = False
    is_heading: bool = False
    is_centered: bool = False


@dataclass
class DocumentParagraph:
    """Represents a paragraph composed of one or more lines."""

    lines: list[OCRLine] = field(default_factory=list)
    block_type: TextBlockType = TextBlockType.PARAGRAPH
    text: str = ""  # combined text from all lines
    font_size: float = 12.0


@dataclass
class DocumentLayout:
    """Represents the analyzed document layout."""

    blocks: list[TextBlock] = field(default_factory=list)
    num_columns: int = 1
    page_width: float = 595.0  # A4 width in points
    page_height: float = 842.0  # A4 height in points
    footer_y_threshold: float | None = None  # Detected visual footer line
