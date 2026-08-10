"""pdftotext TSV → ODF converter.

Public API module.  Delegates parsing, layout analysis, ODF generation
and plain-text formatting to:
  - tsv_parser        (data models, constants, TSV parsing, line grouping)
  - column_detector   (multi-column / table detection)
  - odf_builder       (ODF document generation)
"""

import re
import threading
from collections.abc import Callable
from statistics import median

from bigocrpdf.utils.column_detector import (
    detect_page_columns,
    detect_table_region,
    is_table_line,
    split_words_by_columns,
)
from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.odf_builder import create_odf, create_positioned_text_odf
from bigocrpdf.utils.tsv_parser import (
    MIN_TABLE_ROWS,
    PARA_INDENT_MAX,
    PARA_INDENT_THRESHOLD,
    DocElement,
    TextLine,
    Word,
    filter_words,
    group_into_lines,
    is_heading_text,
    is_kv_line,
    parse_tsv_pages,
)

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
    if _is_preformatted_text(text):
        return "preformatted"
    ht = is_heading_text(text)
    if ht and _is_plausible_heading_text(text):
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
    new_buf = [t for t, li in zip(para_buf, para_line_idx, strict=True) if li not in header_indices]
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
    if _is_preformatted_text(text):
        return "preformatted"
    ht = is_heading_text(text)
    if ht and _is_plausible_heading_text(text):
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


IndentCalculator = Callable[[float], int]


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

    while i < len(lines):
        i, para_is_indented = _process_column_line(
            elements,
            lines,
            i,
            para_buf,
            para_line_idx,
            _indent,
            para_is_indented,
            body_margin,
            page_right,
        )

    _flush_paragraph(elements, para_buf, para_line_idx, lines, _indent, para_is_indented)
    return elements


def _process_column_line(
    elements: list[DocElement],
    lines: list[TextLine],
    line_index: int,
    para_buf: list[str],
    para_line_idx: list[int],
    indent_fn: IndentCalculator,
    para_is_indented: bool,
    body_margin: float | None,
    page_right: float,
) -> tuple[int, bool]:
    line = lines[line_index]
    text = line.text.strip()
    if not text:
        return line_index + 1, para_is_indented

    table_result = _consume_table_line(
        elements, lines, line_index, para_buf, para_line_idx, indent_fn, para_is_indented
    )
    if table_result is not None:
        return table_result

    is_section = bool(_SECTION_RE.match(text))
    if para_buf and is_section:
        para_is_indented = _flush_paragraph(
            elements, para_buf, para_line_idx, lines, indent_fn, para_is_indented
        )

    kind = _classify_standalone_line(
        text, line, body_margin, page_right, bool(para_buf), is_section
    )
    if kind:
        return line_index + 1, _append_standalone_line(
            elements, para_buf, para_line_idx, lines, indent_fn, para_is_indented, kind, text, line
        )

    para_is_indented = _start_indented_paragraph_if_needed(
        elements,
        lines,
        line_index,
        para_buf,
        para_line_idx,
        indent_fn,
        para_is_indented,
        body_margin,
    )
    para_buf.append(text)
    para_line_idx.append(line_index)
    if _has_paragraph_gap_after(lines, line_index):
        para_is_indented = _flush_paragraph(
            elements, para_buf, para_line_idx, lines, indent_fn, para_is_indented
        )
    return line_index + 1, para_is_indented


def _consume_table_line(
    elements: list[DocElement],
    lines: list[TextLine],
    line_index: int,
    para_buf: list[str],
    para_line_idx: list[int],
    indent_fn: IndentCalculator,
    para_is_indented: bool,
) -> tuple[int, bool] | None:
    text = lines[line_index].text.strip()
    table_elem, new_index, was_table = _try_consume_table(
        lines, line_index, para_buf, para_line_idx
    )
    if table_elem is not None:
        para_is_indented = _flush_paragraph(
            elements, para_buf, para_line_idx, lines, indent_fn, para_is_indented
        )
        elements.append(table_elem)
        return new_index, para_is_indented
    if not was_table:
        return None
    para_is_indented = _flush_paragraph(
        elements, para_buf, para_line_idx, lines, indent_fn, para_is_indented
    )
    para_buf.append(text)
    para_line_idx.append(line_index)
    return line_index + 1, para_is_indented


def _append_standalone_line(
    elements: list[DocElement],
    para_buf: list[str],
    para_line_idx: list[int],
    lines: list[TextLine],
    indent_fn: IndentCalculator,
    para_is_indented: bool,
    kind: str,
    text: str,
    line: TextLine,
) -> bool:
    para_is_indented = _flush_paragraph(
        elements, para_buf, para_line_idx, lines, indent_fn, para_is_indented
    )
    indent_chars = indent_fn(line.min_x)
    elements.append(
        DocElement(
            kind,
            text,
            raw_lines=[" " * indent_chars + text],
            indent_chars=indent_chars,
            y_top=line.y,
        )
    )
    return para_is_indented


def _start_indented_paragraph_if_needed(
    elements: list[DocElement],
    lines: list[TextLine],
    line_index: int,
    para_buf: list[str],
    para_line_idx: list[int],
    indent_fn: IndentCalculator,
    para_is_indented: bool,
    body_margin: float | None,
) -> bool:
    if body_margin is None:
        return para_is_indented

    min_x = lines[line_index].min_x
    if not (body_margin + PARA_INDENT_THRESHOLD < min_x < body_margin + PARA_INDENT_MAX):
        return para_is_indented

    if para_buf:
        _flush_paragraph(elements, para_buf, para_line_idx, lines, indent_fn, para_is_indented)
    return True


def _has_paragraph_gap_after(lines: list[TextLine], line_index: int) -> bool:
    if line_index + 1 >= len(lines):
        return False
    line = lines[line_index]
    return lines[line_index + 1].y - line.y > line.words[0].height * 2.0


def _flush_paragraph(
    elements: list[DocElement],
    para_buf: list[str],
    para_line_idx: list[int],
    lines: list[TextLine],
    indent_fn: IndentCalculator,
    para_is_indented: bool,
) -> bool:
    if para_buf:
        elem = _build_paragraph_element(para_buf, para_line_idx, lines, indent_fn, para_is_indented)
        if elem:
            elements.append(elem)
        para_buf.clear()
        para_line_idx.clear()
    return False


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
            if last.raw_lines:
                last.raw_lines.extend(first.raw_lines or [first.text.lstrip()])
            all_pages[i + 1].pop(0)
    return all_pages


# ── Public API ──


def convert_pdf_to_odf(
    pdf_path: str,
    odf_path: str,
    include_images: bool = False,
    cancel_event: "threading.Event | None" = None,
) -> str:
    """Convert a PDF to a fixed-layout or reflowable structured ODF document."""
    pages_words = parse_tsv_pages(pdf_path)
    page_geometries = _pdf_page_geometries(pdf_path)
    if include_images:
        if not page_geometries:
            raise ValueError(f"Could not read PDF page geometry for positioned ODT: {pdf_path}")
        return create_positioned_text_odf(pages_words, odf_path, page_geometries, cancel_event)

    if not pages_words and not page_geometries:
        logger.warning("No text found in PDF: %s", pdf_path)
        create_odf([], odf_path, cancel_event=cancel_event)
        return odf_path

    all_elements: list[list[DocElement]] = []
    page_numbers = range(1, len(page_geometries) + 1) if page_geometries else sorted(pages_words)
    for page_num in page_numbers:
        elements = process_page(pages_words.get(page_num, []), page_num)
        all_elements.append(elements)

    all_elements = fix_cross_page_breaks(all_elements)

    create_odf(
        all_elements,
        odf_path,
        page_size_cm=page_geometries[0][2:] if page_geometries else None,
        body_font_size_pt=_body_font_size_from_words(pages_words),
        cancel_event=cancel_event,
    )
    return odf_path


def _body_font_size_from_words(pages_words: dict[int, list[Word]]) -> float:
    heights = [word.height for words in pages_words.values() for word in words if word.height > 0]
    return min(max(median(heights), 6.5), 10.5) if heights else 9.0


# Longest side an exported page may have, in centimetres: A4's height. Photo
# and scan PDFs routinely declare page boxes of a metre or more, which is a
# valid page but not a usable document.
_MAX_PAGE_SIDE_CM = 29.7
_CM_PER_POINT = 2.54 / 72.0


def _page_size_cm(width_pt: float, height_pt: float) -> tuple[float, float]:
    """The page's real size in centimetres, shrunk only if it is absurd.

    Every page used to be stretched to 29.7 cm on its longer side whatever it
    measured, so a US Letter document came out 6.3% too large and every word in
    the fixed-layout export landed 6.3% away from where it belongs -- 42 points
    off by the foot of the page. Scaling down an oversized page is still
    needed; scaling a normal one up never was.
    """
    width_cm = width_pt * _CM_PER_POINT
    height_cm = height_pt * _CM_PER_POINT
    longest = max(width_cm, height_cm)
    if longest > _MAX_PAGE_SIDE_CM:
        shrink = _MAX_PAGE_SIDE_CM / longest
        width_cm *= shrink
        height_cm *= shrink
    return width_cm, height_cm


def _pdf_page_geometries(pdf_path: str) -> list[tuple[float, float, float, float]]:
    """Read every PDF page ratio without invoking an office suite."""
    import pikepdf

    try:
        with pikepdf.open(pdf_path) as pdf:
            sizes = []
            for page in pdf.pages:
                media_box = page.mediabox
                width = abs(float(media_box[2]) - float(media_box[0]))
                height = abs(float(media_box[3]) - float(media_box[1]))
                if width <= 0 or height <= 0:
                    return []
                sizes.append((width, height, *_page_size_cm(width, height)))
            return sizes
    except (OSError, ValueError, pikepdf.PdfError):
        return []


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
            _append_text_element(lines, elem, prev_kind)
            prev_kind = kind

    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines) + "\n"


def _append_text_element(lines: list[str], elem: DocElement, prev_kind: str) -> None:
    if elem.kind in ("heading1", "heading2", "heading3"):
        _append_blank_before(lines)
        _append_doc_element_lines(lines, elem)
        lines.append("")
    elif elem.kind == "kv":
        if prev_kind not in ("kv", ""):
            lines.append("")
        _append_doc_element_lines(lines, elem)
    elif elem.kind == "table":
        _append_blank_before(lines)
        lines.extend(_format_table_text(elem.rows))
        lines.append("")
    else:
        _append_blank_before(lines)
        _append_doc_element_lines(lines, elem, with_indent=True)


def _append_blank_before(lines: list[str]) -> None:
    if lines and lines[-1] != "":
        lines.append("")


def _append_doc_element_lines(
    lines: list[str],
    elem: DocElement,
    with_indent: bool = False,
) -> None:
    if elem.raw_lines:
        lines.extend(elem.raw_lines)
        return

    prefix = " " * elem.indent_chars if with_indent and elem.indent_chars else ""
    lines.append(prefix + elem.text)


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


# ── Markdown Generation ──

# Inline characters that always need escaping. Line-start punctuation like
# '#', '-', '+', '>' and 'N.' lists is handled separately so we don't uglify
# mid-paragraph text (e.g. CPF/phone numbers full of hyphens).
_MD_INLINE_ESCAPE_RE = re.compile(r"([\\`*_\[\]<>|])")
# Ordered-list marker requires whitespace after the dot in CommonMark; without
# the lookahead a paragraph starting with a decimal like "1.5 million" would be
# wrongly escaped to "\1.5 million".
_MD_LINE_START_RE = re.compile(r"^([#\-+>]|\d+\.(?=\s|$))")


def _escape_md(text: str) -> str:
    """Escape Markdown control characters in inline text.

    Escapes characters that have meaning anywhere in a line (``*``, ``_``,
    ``[``, ``]``, ``<``, ``>``, ``|``, backticks, backslashes) and, only at
    the start of the string, the block-level markers (``#``, ``-``, ``+``,
    ``>`` and ordered-list ``N.``).
    """
    escaped = _MD_INLINE_ESCAPE_RE.sub(r"\\\1", text)
    return _MD_LINE_START_RE.sub(r"\\\1", escaped)


def _escape_md_cell(text: str) -> str:
    """Escape inline Markdown specials inside a table cell.

    Same rules as :func:`_escape_md` minus the line-start markers (cells are
    rendered inline, not at the start of a block) plus an explicit ``|``
    escape so the cell does not break the table.
    """
    return _MD_INLINE_ESCAPE_RE.sub(r"\\\1", text)


def _format_table_markdown(rows: list[list[str]]) -> list[str]:
    """Format table rows as a GitHub-flavored Markdown pipe table."""
    if not rows:
        return []
    n_cols = max(len(r) for r in rows)

    def _cell(value: str) -> str:
        return _escape_md_cell(value).strip() or " "

    out: list[str] = []
    header = rows[0]
    out.append(
        "| " + " | ".join(_cell(header[j] if j < len(header) else "") for j in range(n_cols)) + " |"
    )
    out.append("|" + "|".join(["---"] * n_cols) + "|")
    for row in rows[1:]:
        out.append(
            "| " + " | ".join(_cell(row[j] if j < len(row) else "") for j in range(n_cols)) + " |"
        )
    return out


def _yaml_escape(value: str) -> str:
    """Escape a value for safe inclusion in a single-line YAML scalar."""
    # Strip control characters (including NUL, newlines, tabs) — they can't
    # appear in a single-line YAML scalar regardless of quoting.
    sanitized = "".join(c for c in value if ord(c) >= 0x20 or c == " ")
    return sanitized.replace("\\", "\\\\").replace('"', '\\"')


def _build_front_matter(pdf_path: str, page_count: int) -> list[str]:
    """Build YAML front-matter lines for a Markdown export."""
    import datetime
    import os

    title = os.path.splitext(os.path.basename(pdf_path))[0]
    # UTC so the date doesn't drift across timezones (and tests stay stable).
    today = datetime.datetime.now(datetime.UTC).date().isoformat()
    return [
        "---",
        f'title: "{_yaml_escape(title)}"',
        f'source: "{_yaml_escape(os.path.abspath(pdf_path))}"',
        f"pages: {page_count}",
        f"date: {today}",
        'generator: "bigocrpdf"',
        "---",
        "",
    ]


_MD_HEADING_PREFIX = {"heading1": "# ", "heading2": "## ", "heading3": "### "}
_BOX_DRAWING_CHARS = frozenset("┌┐└┘├┤┬┴┼│─═╔╗╚╝║")
_NON_HEADING_LABEL_RE = re.compile(r"^(?:page\s+\d|fax\s+(?:no|number)\b)", re.IGNORECASE)


def _is_preformatted_text(text: str) -> bool:
    box_drawing_count = sum(character in _BOX_DRAWING_CHARS for character in text)
    return box_drawing_count >= 3 or ("{" in text and ";" in text and "}" in text)


def _is_plausible_heading_text(text: str) -> bool:
    text = text.strip()
    if not text or _NON_HEADING_LABEL_RE.match(text):
        return False
    if ":" in text and not text.endswith(":"):
        return False
    letters = [character for character in text if character.isalpha()]
    digits = sum(character.isdigit() for character in text)
    if digits and digits / max(len(letters) + digits, 1) >= 0.1:
        return False
    if len(text.split()) == 1 and letters and not re.search(r"[AEIOUY]", text, re.IGNORECASE):
        return False
    return True


def _ensure_blank_line(lines: list[str]) -> None:
    """Append a blank line unless the previous one already is blank."""
    if lines and lines[-1] != "":
        lines.append("")


def _emit_heading(lines: list[str], elem: DocElement) -> None:
    _ensure_blank_line(lines)
    lines.append(_MD_HEADING_PREFIX[elem.kind] + _escape_md(elem.text.strip()))
    lines.append("")


def _is_plausible_markdown_heading(elem: DocElement) -> bool:
    return _is_plausible_heading_text(elem.text)


def _emit_table(lines: list[str], elem: DocElement) -> None:
    _ensure_blank_line(lines)
    lines.extend(_format_table_markdown(elem.rows))
    lines.append("")


def _emit_kv(lines: list[str], elem: DocElement) -> None:
    """Bold the key portion (before the first colon) for readability."""
    _ensure_blank_line(lines)
    text = elem.text.strip()
    if ":" not in text:
        lines.append(_escape_md(text))
    else:
        key, _sep, value = text.partition(":")
        lines.append(f"**{_escape_md(key.strip())}:** {_escape_md(value.strip())}")
    lines.append("")


def _emit_paragraph(lines: list[str], elem: DocElement) -> None:
    """Paragraph variants (paragraph, paragraph_indent, paragraph_right, …).

    When the OCR layer preserved per-line breaks in ``raw_lines`` (multi-line
    addresses, poetry, etc.), emit each line separately with a CommonMark
    hard break (two trailing spaces) so the rendered output keeps the
    original line geometry instead of collapsing into one run-on paragraph.
    """
    _ensure_blank_line(lines)
    raw = [line for line in (elem.raw_lines or []) if line.strip()]
    if len(raw) > 1:
        for i, line in enumerate(raw):
            suffix = "  " if i < len(raw) - 1 else ""
            lines.append(_escape_md(line.strip()) + suffix)
    else:
        lines.append(_escape_md(elem.text.strip()))
    lines.append("")


def _is_preformatted(elem: DocElement) -> bool:
    text = "\n".join(elem.raw_lines) if elem.raw_lines else elem.text
    return elem.kind == "preformatted" or _is_preformatted_text(text)


def _emit_code_block(lines: list[str], elem: DocElement) -> None:
    _ensure_blank_line(lines)
    raw_lines = elem.raw_lines or elem.text.splitlines() or [elem.text]
    lines.append("```")
    lines.extend(line.rstrip() for line in raw_lines)
    lines.append("```")
    lines.append("")


def _emit_code_group(lines: list[str], elements: list[DocElement]) -> None:
    _ensure_blank_line(lines)
    lines.append("```")
    for element in elements:
        raw_lines = element.raw_lines or element.text.splitlines() or [element.text]
        lines.extend(line.rstrip() for line in raw_lines)
    lines.append("```")
    lines.append("")


def _emit_element(lines: list[str], elem: DocElement) -> None:
    """Dispatch a single DocElement to the appropriate Markdown emitter."""
    if _is_preformatted(elem):
        _emit_code_block(lines, elem)
    elif elem.kind in _MD_HEADING_PREFIX and _is_plausible_markdown_heading(elem):
        _emit_heading(lines, elem)
    elif elem.kind == "table":
        _emit_table(lines, elem)
    elif elem.kind == "kv":
        _emit_kv(lines, elem)
    else:
        _emit_paragraph(lines, elem)


def create_markdown(pages_elements: list[list[DocElement]]) -> str:
    """Generate Markdown preserving document structure (headings, tables, paragraphs)."""
    lines: list[str] = []

    for page_idx, elements in enumerate(pages_elements):
        if page_idx > 0:
            _ensure_blank_line(lines)
            lines.append("---")
            lines.append("")
        element_index = 0
        while element_index < len(elements):
            if not _is_preformatted(elements[element_index]):
                _emit_element(lines, elements[element_index])
                element_index += 1
                continue
            group_end = element_index + 1
            while group_end < len(elements) and _is_preformatted(elements[group_end]):
                group_end += 1
            _emit_code_group(lines, elements[element_index:group_end])
            element_index = group_end

    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines) + "\n"


def convert_pdf_to_markdown(
    pdf_path: str,
    include_front_matter: bool = False,
    cancel_event: "threading.Event | None" = None,
) -> str:
    """Convert an OCR'd PDF to a structured Markdown document.

    Args:
        pdf_path: Path to a PDF containing a text layer (post-OCR).
        include_front_matter: If True, prepend YAML front-matter with title,
            source path, page count and date — handy for ingesting into
            Obsidian/Hugo or as LLM context.
        cancel_event: Optional event polled between pages so long batches
            stay responsive to a cancel button.  Raises
            :class:`ExportCancelled` if set mid-conversion, matching the
            ODF path's contract.
    """
    from bigocrpdf.utils.odf_builder import ExportCancelled

    pages_words = parse_tsv_pages(pdf_path)
    if cancel_event is not None and cancel_event.is_set():
        raise ExportCancelled
    if not pages_words:
        if include_front_matter:
            return "\n".join(_build_front_matter(pdf_path, 0)) + "\n"
        return ""

    all_elements: list[list[DocElement]] = []
    for page_num in sorted(pages_words.keys()):
        if cancel_event is not None and cancel_event.is_set():
            raise ExportCancelled
        elements = process_page(pages_words[page_num], page_num)
        all_elements.append(elements)

    if cancel_event is not None and cancel_event.is_set():
        raise ExportCancelled

    all_elements = fix_cross_page_breaks(all_elements)

    body = create_markdown(all_elements)
    if include_front_matter:
        return "\n".join(_build_front_matter(pdf_path, len(all_elements))) + body
    return body
