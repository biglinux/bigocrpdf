"""TSV parsing utilities for pdftotext output.

Provides data models, constants, and functions for parsing pdftotext -tsv
output into structured word / line objects, plus lightweight text classifiers.
"""

import csv
import io
import re
import subprocess
from dataclasses import dataclass, field

from bigocrpdf.utils.logger import logger  # noqa: I001

# ── Data Model ──


@dataclass
class Word:
    text: str
    left: float
    top: float
    width: float
    height: float
    right: float = 0.0

    def __post_init__(self):
        self.right = self.left + self.width


@dataclass
class TextLine:
    words: list[Word]
    y: float
    min_x: float = 0.0
    max_x: float = 0.0
    text: str = ""

    def __post_init__(self):
        if self.words:
            self.words.sort(key=lambda w: w.left)
            self.min_x = self.words[0].left
            self.max_x = max(w.right for w in self.words)
            self.text = self._assemble_text()

    def _assemble_text(self) -> str:
        if not self.words:
            return ""
        parts = [self.words[0].text]
        for i in range(1, len(self.words)):
            gap = self.words[i].left - self.words[i - 1].right
            if gap > self.words[i - 1].width * 2:
                parts.append("  ")
            else:
                parts.append(" ")
            parts.append(self.words[i].text)
        return "".join(parts)


@dataclass
class DocElement:
    kind: str
    text: str = ""
    rows: list[list[str]] = field(default_factory=list)
    # TXT-only metadata (ignored by ODF generation)
    raw_lines: list[str] = field(default_factory=list)
    indent_chars: int = 0
    y_top: float = 0.0  # Top-down Y position (matches pdftotext TSV coordinates)


# ── Constants ──

FOOTER_REGION_Y = 780.0
Y_TOLERANCE = 5.0
X_GAP_SPLIT = 300.0
COLUMN_GAP_THRESHOLD = 30.0
MAX_TABLE_GAP = 200.0
HEADER_MISALIGN_TOLERANCE = 100.0
CENTER_CLUSTER_TOLERANCE = 30.0
BOUNDARY_SPLIT_MIN_GAP = 2.0
MIN_WORD_WIDTH = 2.0  # Words narrower than this are OCR artifacts
COLUMN_VALLEY_BIN = 10.0  # Bin width for column histogram
COLUMN_MIN_VALLEY_WIDTH = 30.0  # Minimum gap for column split
COLUMN_CENTER_RANGE = (0.25, 0.75)  # Page fraction where column gap expected
TABLE_TWO_COL_GAP = 100.0  # Larger gap required for 2-column table lines
MIN_TABLE_ROWS = 2  # Minimum rows to confirm a table region
MAX_TABLE_CELL_LEN = 14  # Median cell text length above this rejects as table
HEADER_WORD_GAP = 8.0  # Minimum gap for header line word groups
PARA_INDENT_THRESHOLD = 20.0  # min_x shift to detect first-line indent (new paragraph)
PARA_INDENT_MAX = 60.0  # max_x shift — beyond this, it's not a first-line indent


# ── TSV Parsing ──


def parse_tsv_pages(pdf_path: str) -> dict[int, list[Word]]:
    """Parse pdftotext -tsv output into words per page."""
    result = subprocess.run(
        ["pdftotext", "-tsv", pdf_path, "-"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    pages: dict[int, list[Word]] = {}
    reader = csv.reader(io.StringIO(result.stdout), delimiter="\t", quoting=csv.QUOTE_NONE)
    next(reader)  # skip header

    for row in reader:
        if len(row) < 12 or row[0] != "5":
            continue
        text = row[11].strip() if len(row) > 11 else ""
        if not text or "\t" in text or "\n" in text:
            continue
        page = int(row[1])
        w = Word(
            text=text,
            left=float(row[6]),
            top=float(row[7]),
            width=float(row[8]),
            height=float(row[9]),
        )
        if page not in pages:
            pages[page] = []
        pages[page].append(w)

    return pages


# ── Noise Filtering ──


def filter_words(words: list[Word], page_num: int) -> list[Word]:
    """Remove clear OCR artifacts."""
    result = []
    for w in words:
        # Skip words with tiny width (OCR artifacts from lines/patterns)
        if w.width < MIN_WORD_WIDTH and len(w.text) > 1:
            continue
        result.append(w)
    return result


# ── Line Grouping ──


def group_into_lines(words: list[Word]) -> list[TextLine]:
    """Group words into lines by y-proximity, splitting at large x-gaps."""
    if not words:
        return []

    sorted_words = sorted(words, key=lambda w: (w.top, w.left))
    lines: list[TextLine] = []
    current_words = [sorted_words[0]]
    current_y = sorted_words[0].top

    for w in sorted_words[1:]:
        if abs(w.top - current_y) <= Y_TOLERANCE:
            current_words.append(w)
            current_y = sum(ww.top for ww in current_words) / len(current_words)
        else:
            lines.extend(_split_line_by_x_gap(current_words, current_y))
            current_words = [w]
            current_y = w.top

    if current_words:
        lines.extend(_split_line_by_x_gap(current_words, current_y))

    lines.sort(key=lambda ln: ln.y)
    return lines


def _split_line_by_x_gap(words: list[Word], y: float) -> list[TextLine]:
    """Split a line at huge x-gaps (watermark separation)."""
    if len(words) <= 1:
        return [TextLine(words=words, y=y)]

    sorted_w = sorted(words, key=lambda w: w.left)
    segments: list[list[Word]] = [[sorted_w[0]]]

    for w in sorted_w[1:]:
        if w.left - segments[-1][-1].right > X_GAP_SPLIT:
            segments.append([w])
        else:
            segments[-1].append(w)

    return [TextLine(words=seg, y=y) for seg in segments]


# ── Text Classification ──


def is_heading_text(text: str) -> str | None:
    """Classify text as a heading level, or None."""
    s = text.strip()
    if not s or len(s) > 120 or len(s) < 3:
        return None
    if sum(1 for c in s if c.isalpha()) < 2:
        return None

    # Numbered section
    if re.match(r"^\d+(\.\d+)*[.)]\s+\S", s) and len(s) < 80:
        return "heading1"

    # ALL CAPS
    alpha_only = re.sub(r"[^a-zA-ZÀ-ÿ]", "", s)
    if (
        alpha_only
        and alpha_only.isupper()
        and len(alpha_only) >= 5
        and len(s) < 60
        and len(s.split()) <= 5
    ):
        return "heading2"

    # Code + separator pattern
    if re.match(r"^[A-Z]\d+\s*[-–—.]\s*\w", s) and len(s) < 60:
        return "heading3"

    return None


def is_kv_line(text: str) -> bool:
    """Check if text is a key: value pair."""
    return bool(re.match(r"^[^:]{3,40}:\s+.+$", text)) and len(text) < 120
