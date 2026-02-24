"""pdftotext TSV → ODF converter.

Public API module.  Delegates parsing, layout analysis, ODF generation
and plain-text formatting to:
  - tsv_parser        (data models, constants, TSV parsing, line grouping)
  - column_detector   (multi-column / table detection)
  - odf_builder       (ODF document generation)
"""

import re

from bigocrpdf.utils.column_detector import (
    detect_page_columns,
    detect_table_region,
    is_table_line,
    split_words_by_columns,
)
from bigocrpdf.utils.odf_builder import _extract_pdf_images, create_odf
from bigocrpdf.utils.tsv_parser import (
    MIN_TABLE_ROWS,
    PARA_INDENT_MAX,
    PARA_INDENT_THRESHOLD,
    DocElement,
    Word,
    filter_words,
    group_into_lines,
    is_heading_text,
    is_kv_line,
    parse_tsv_pages,
)

from bigocrpdf.utils.logger import logger  # noqa: I001


# ── Page Processing ──


def process_page(words: list[Word], page_num: int) -> list[DocElement]:
    """Full pipeline for one page, with automatic column detection."""
    clean = filter_words(words, page_num)
    if not clean:
        return []

    # Detect multi-column layout
    columns = detect_page_columns(clean)
    if columns and len(columns) > 1:
        all_elements: list[DocElement] = []
        col_words = split_words_by_columns(clean, columns)
        for col_w in col_words:
            if col_w:
                all_elements.extend(_process_single_column(col_w))
        return all_elements

    return _process_single_column(clean)


def _classify_paragraph_text(text: str, para_is_indented: bool) -> str:
    """Return DocumentElement kind for paragraph text."""
    ht = is_heading_text(text)
    if ht:
        return ht
    if is_kv_line(text) and not re.match(r"^[A-Z]\d*(\.\d+)*\s*[-\u2013\u2014.]", text):
        return "kv"
    if para_is_indented:
        return "paragraph_indent"
    return "paragraph"


def _strip_pre_headers(
    para_buf: list[str],
    para_line_idx: list[int],
    line_idx: int,
    n_pre_headers: int,
) -> None:
    """Remove header lines absorbed into para_buf before a table."""
    header_indices = set(range(line_idx - n_pre_headers, line_idx))
    new_buf = [t for t, li in zip(para_buf, para_line_idx) if li not in header_indices]
    new_idx = [li for li in para_line_idx if li not in header_indices]
    para_buf[:] = new_buf
    para_line_idx[:] = new_idx


def _try_consume_table(
    lines: list, i: int, para_buf: list[str], para_line_idx: list[int]
) -> tuple[DocElement | None, int, bool]:
    """Try to consume a table starting at line *i*.

    Returns (table_element, next_index, was_table_line).
    """
    if not is_table_line(lines[i]):
        return None, i, False
    table_rows, end_idx, n_pre_headers = detect_table_region(lines, i)
    if table_rows and len(table_rows) >= MIN_TABLE_ROWS:
        if n_pre_headers > 0:
            _strip_pre_headers(para_buf, para_line_idx, i, n_pre_headers)
        return DocElement("table", rows=table_rows, y_top=lines[i].y), end_idx, True
    return None, i, True


_SECTION_RE = re.compile(r"^[A-Z]\d*(\.\d+)*\s*[-\u2013\u2014.]")


def _classify_standalone_line(
    text: str,
    line,
    body_margin: float | None,
    page_right: float,
    has_para_buf: bool,
    is_section: bool,
) -> str | None:
    """Return element kind if line should be emitted as a standalone element."""
    ht = is_heading_text(text)
    if ht:
        return ht
    if not has_para_buf and not is_section and is_kv_line(text):
        return "kv"
    if not has_para_buf and body_margin is not None and line.min_x > body_margin + PARA_INDENT_MAX:
        line_center = (line.min_x + line.max_x) / 2
        page_center = (body_margin + page_right) / 2
        return "paragraph_right" if line_center > page_center else "paragraph_center"
    return None


def _build_paragraph_element(
    para_buf: list[str],
    para_line_idx: list[int],
    lines: list,
    indent_fn,
    para_is_indented: bool,
) -> DocElement | None:
    """Build a DocElement from accumulated paragraph lines."""
    text = re.sub(r"\s{2,}", " ", " ".join(para_buf)).strip()
    if not text:
        return None
    ind = 0
    rlines: list[str] = []
    y = 0.0
    if para_line_idx:
        ind = indent_fn(lines[para_line_idx[0]].min_x)
        y = lines[para_line_idx[0]].y
        for idx in para_line_idx:
            li = indent_fn(lines[idx].min_x)
            rlines.append(" " * li + lines[idx].text.strip())
    kind = _classify_paragraph_text(text, para_is_indented)
    return DocElement(kind, text, raw_lines=rlines, indent_chars=ind, y_top=y)


def _process_single_column(words: list[Word]) -> list[DocElement]:
    """Process a single column of words into document elements."""
    lines = group_into_lines(words)
    if not lines:
        return []

    text_min_xs = sorted(ln.min_x for ln in lines if len(ln.text.strip()) > 20)
    body_margin = text_min_xs[len(text_min_xs) // 4] if len(text_min_xs) >= 3 else None
    page_right = max(w.right for w in words)
    page_left = min(ln.min_x for ln in lines)
    _char_w = (page_right - page_left) / 90 if page_right > page_left else 6.0

    def _indent(min_x: float) -> int:
        return max(0, round((min_x - page_left) / _char_w))

    elements: list[DocElement] = []
    i = 0
    para_buf: list[str] = []
    para_line_idx: list[int] = []
    para_is_indented = False

    def flush():
        nonlocal para_is_indented
        if para_buf:
            elem = _build_paragraph_element(
                para_buf, para_line_idx, lines, _indent, para_is_indented
            )
            if elem:
                elements.append(elem)
            para_buf.clear()
            para_line_idx.clear()
        para_is_indented = False

    while i < len(lines):
        line = lines[i]
        text = line.text.strip()
        if not text:
            i += 1
            continue

        # Table detection (highest priority)
        table_elem, new_i, was_table = _try_consume_table(lines, i, para_buf, para_line_idx)
        if table_elem is not None:
            flush()
            elements.append(table_elem)
            i = new_i
            continue
        if was_table:
            flush()
            para_buf.append(text)
            para_line_idx.append(i)
            i += 1
            continue

        # Section identifier flushes current paragraph
        is_section = bool(_SECTION_RE.match(text))
        if para_buf and is_section:
            flush()

        # Standalone element (heading, KV, centered/right)
        kind = _classify_standalone_line(
            text, line, body_margin, page_right, bool(para_buf), is_section
        )
        if kind:
            flush()
            ind = _indent(line.min_x)
            elements.append(
                DocElement(
                    kind,
                    text,
                    raw_lines=[" " * ind + text],
                    indent_chars=ind,
                    y_top=line.y,
                )
            )
            i += 1
            continue

        # Detect first-line indent as paragraph boundary
        if (
            body_margin is not None
            and body_margin + PARA_INDENT_THRESHOLD < line.min_x < body_margin + PARA_INDENT_MAX
        ):
            if para_buf:
                flush()
            para_is_indented = True

        para_buf.append(text)
        para_line_idx.append(i)
        if i + 1 < len(lines):
            y_gap = lines[i + 1].y - line.y
            if y_gap > line.words[0].height * 2.0:
                flush()
        i += 1

    flush()
    return elements


def fix_cross_page_breaks(
    all_pages: list[list[DocElement]],
) -> list[list[DocElement]]:
    """Merge paragraphs split across page boundaries."""
    for i in range(len(all_pages) - 1):
        if not all_pages[i] or not all_pages[i + 1]:
            continue
        last = all_pages[i][-1]
        first = all_pages[i + 1][0]
        if (
            last.kind in ("paragraph", "paragraph_indent")
            and first.kind in ("paragraph", "paragraph_indent")
            and last.text
            and first.text
            and last.text.rstrip()[-1:] in (",", "-", "\u2013", "\u2014")
            and first.text.lstrip()[:1].islower()
        ):
            last.text = last.text.rstrip() + " " + first.text.lstrip()
            all_pages[i + 1].pop(0)
    return all_pages


# ── Public API ──


def convert_pdf_to_odf(
    pdf_path: str,
    odf_path: str,
    include_images: bool = False,
    cancel_event: "threading.Event | None" = None,
) -> str:
    """Convert an OCR'd PDF to a structured ODF document."""
    pages_words = parse_tsv_pages(pdf_path)
    if not pages_words:
        logger.warning("No text found in PDF: %s", pdf_path)
        from odf.opendocument import OpenDocumentText

        doc = OpenDocumentText()
        doc.save(odf_path)
        return odf_path

    all_elements: list[list[DocElement]] = []
    for page_num in sorted(pages_words.keys()):
        elements = process_page(pages_words[page_num], page_num)
        all_elements.append(elements)

    all_elements = fix_cross_page_breaks(all_elements)

    page_images: dict[int, list[tuple[bytes, str, int, int]]] | None = None
    if include_images:
        page_images = _extract_pdf_images(pdf_path, cancel_event=cancel_event)
        if not page_images:
            logger.warning("Could not extract images; proceeding without images")

    create_odf(all_elements, odf_path, page_images=page_images, cancel_event=cancel_event)
    return odf_path


# ── Plain-text Generation ──


def _format_table_text(rows: list[list[str]]) -> list[str]:
    """Format table rows as aligned plain-text columns."""
    if not rows:
        return []
    n_cols = max(len(r) for r in rows)
    widths = [0] * n_cols
    for row in rows:
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(cell))
    out: list[str] = []
    for i, row in enumerate(rows):
        cells = [(row[j] if j < len(row) else "").ljust(widths[j]) for j in range(n_cols)]
        out.append(" | ".join(cells))
        if i == 0:
            out.append("-+-".join("-" * w for w in widths))
    return out


def create_text(pages_elements: list[list[DocElement]]) -> str:
    """Generate formatted plain text preserving original visual layout."""
    lines: list[str] = []

    for page_idx, elements in enumerate(pages_elements):
        if page_idx > 0:
            lines.extend([""] * 5)
            lines.append(f"--- Page {page_idx + 1} ---")
            lines.append("")
        else:
            lines.append("--- Page 1 ---")
            lines.append("")

        prev_kind = ""
        for elem in elements:
            kind = elem.kind

            if kind in ("heading1", "heading2", "heading3"):
                if lines and lines[-1] != "":
                    lines.append("")
                if elem.raw_lines:
                    lines.extend(elem.raw_lines)
                else:
                    lines.append(elem.text)
                lines.append("")
            elif kind == "kv":
                if prev_kind not in ("kv", ""):
                    lines.append("")
                if elem.raw_lines:
                    lines.extend(elem.raw_lines)
                else:
                    lines.append(elem.text)
            elif kind == "table":
                if lines and lines[-1] != "":
                    lines.append("")
                lines.extend(_format_table_text(elem.rows))
                lines.append("")
            else:
                if lines and lines[-1] != "":
                    lines.append("")
                if elem.raw_lines:
                    lines.extend(elem.raw_lines)
                else:
                    prefix = " " * elem.indent_chars if elem.indent_chars else ""
                    lines.append(prefix + elem.text)

            prev_kind = kind

    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines) + "\n"


def convert_pdf_to_text(pdf_path: str) -> str:
    """Convert an OCR'd PDF to structured plain text."""
    pages_words = parse_tsv_pages(pdf_path)
    if not pages_words:
        return ""

    all_elements: list[list[DocElement]] = []
    for page_num in sorted(pages_words.keys()):
        elements = process_page(pages_words[page_num], page_num)
        all_elements.append(elements)

    all_elements = fix_cross_page_breaks(all_elements)
    return create_text(all_elements)
