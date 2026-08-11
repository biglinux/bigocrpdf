"""ODF document generation from structured DocElements.

Handles ODF style setup, element rendering, table formatting,
and image extraction/embedding from source PDFs.
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from bigocrpdf.utils.durable_writes import publish_file_atomically
from bigocrpdf.utils.logger import logger  # noqa: I001
from bigocrpdf.utils.tsv_parser import DocElement


class ExportCancelled(Exception):
    """Raised when the user cancels the ODF export."""


def _make_odf_paragraph_style(doc, name, para_kw, text_kw):
    from odf.style import ParagraphProperties, Style, TextProperties

    style = Style(name=name, family="paragraph")
    style.addElement(ParagraphProperties(**para_kw))
    style.addElement(TextProperties(**text_kw))
    doc.automaticstyles.addElement(style)
    return style


def _configure_odf_document(
    doc,
    page_size_cm: tuple[float, float] | None = None,
    body_font_size_pt: float = 9.0,
) -> dict:
    from odf.style import (
        Columns,
        FontFace,
        MasterPage,
        PageLayout,
        PageLayoutProperties,
        SectionProperties,
        Style,
        TableProperties,
        TextProperties,
    )

    page_width_cm, page_height_cm = page_size_cm or (21.0, 29.7)
    margin_cm = 1.5
    content_width_cm = max(page_width_cm - 2 * margin_cm, 4.0)
    page_layout = PageLayout(name="SourcePage")
    page_layout.addElement(
        PageLayoutProperties(
            pagewidth=f"{page_width_cm:.2f}cm",
            pageheight=f"{page_height_cm:.2f}cm",
            margintop=f"{margin_cm:.2f}cm",
            marginbottom=f"{margin_cm:.2f}cm",
            marginleft=f"{margin_cm:.2f}cm",
            marginright=f"{margin_cm:.2f}cm",
        )
    )
    doc.automaticstyles.addElement(page_layout)
    doc.masterstyles.addElement(MasterPage(name="Standard", pagelayoutname="SourcePage"))

    font_face = FontFace(
        name="Liberation Sans",
        fontfamily="Liberation Sans",
        fontfamilygeneric="swiss",
        fontpitch="variable",
    )
    doc.fontfacedecls.addElement(font_face)

    styles = _odf_paragraph_styles(doc, body_font_size_pt)
    styles["bold"] = Style(name="Bold", family="text")
    styles["bold"].addElement(TextProperties(fontweight="bold"))
    doc.automaticstyles.addElement(styles["bold"])

    styles["page_break"] = _make_odf_paragraph_style(doc, "PB", {"breakbefore": "page"}, {})
    styles["column_break"] = _make_odf_paragraph_style(
        doc, "ColumnBreak", {"breakbefore": "column"}, {}
    )
    styles["two_columns"] = Style(name="TwoColumns", family="section")
    section_properties = SectionProperties()
    section_properties.addElement(Columns(columncount=2, columngap="0.6cm"))
    styles["two_columns"].addElement(section_properties)
    doc.automaticstyles.addElement(styles["two_columns"])
    styles["table"] = Style(name="Tbl", family="table")
    styles["table"].addElement(TableProperties(width=f"{content_width_cm:.2f}cm", align="center"))
    styles["table_width_cm"] = content_width_cm
    doc.automaticstyles.addElement(styles["table"])

    styles["cell"] = _odf_table_cell_style(doc, "Cell", "0.5pt solid #dddddd")
    styles["header_cell"] = _odf_table_cell_style(doc, "HCell", "1pt solid #888888")
    styles["cell_text"] = _make_odf_paragraph_style(
        doc,
        "CellText",
        {"textalign": "center", "marginbottom": "0cm"},
        {"fontsize": f"{max(body_font_size_pt - 0.5, 6.0):.2f}pt", "fontfamily": "Liberation Sans"},
    )
    styles["cell_text_left"] = _make_odf_paragraph_style(
        doc,
        "CellTextL",
        {"textalign": "left", "marginbottom": "0cm"},
        {"fontsize": f"{max(body_font_size_pt - 0.5, 6.0):.2f}pt", "fontfamily": "Liberation Sans"},
    )
    styles["image_frame"] = Style(name="ImgFrame", family="graphic")
    return styles


def _odf_paragraph_styles(doc, body_font_size_pt: float) -> dict:
    body_size = f"{body_font_size_pt:.2f}pt"
    heading1_size = f"{body_font_size_pt * 4 / 3:.2f}pt"
    heading2_size = f"{body_font_size_pt * 7 / 6:.2f}pt"
    heading3_size = f"{body_font_size_pt * 19 / 18:.2f}pt"
    body_text = {"fontsize": body_size, "fontfamily": "Liberation Sans"}
    styles = {
        "heading1": _make_odf_paragraph_style(
            doc,
            "H1",
            {
                "textalign": "left",
                "margintop": "0.25cm",
                "marginbottom": "0.12cm",
                "keepwithnext": "always",
            },
            {"fontsize": heading1_size, "fontweight": "bold", "fontfamily": "Liberation Sans"},
        ),
        "heading2": _make_odf_paragraph_style(
            doc,
            "H2",
            {
                "textalign": "left",
                "margintop": "0.2cm",
                "marginbottom": "0.1cm",
                "keepwithnext": "always",
            },
            {"fontsize": heading2_size, "fontweight": "bold", "fontfamily": "Liberation Sans"},
        ),
        "heading3": _make_odf_paragraph_style(
            doc,
            "H3",
            {
                "textalign": "left",
                "margintop": "0.15cm",
                "marginbottom": "0.08cm",
                "keepwithnext": "always",
                "marginleft": "0.5cm",
            },
            {"fontsize": heading3_size, "fontweight": "bold", "fontfamily": "Liberation Sans"},
        ),
        "paragraph": _make_odf_paragraph_style(
            doc,
            "Body",
            {"textalign": "left", "marginbottom": "0.05cm", "lineheight": "115%"},
            body_text,
        ),
        "paragraph_indent": _make_odf_paragraph_style(
            doc,
            "BodyI",
            {
                "textalign": "left",
                "marginbottom": "0.05cm",
                "lineheight": "115%",
                "textindent": "1.25cm",
            },
            body_text,
        ),
        "paragraph_center": _make_odf_paragraph_style(
            doc,
            "BodyC",
            {"textalign": "center", "marginbottom": "0.03cm", "lineheight": "115%"},
            body_text,
        ),
        "paragraph_right": _make_odf_paragraph_style(
            doc,
            "BodyR",
            {"textalign": "end", "marginbottom": "0.03cm", "lineheight": "115%"},
            body_text,
        ),
        "kv": _make_odf_paragraph_style(
            doc,
            "KV",
            {"textalign": "left", "marginbottom": "0.03cm", "lineheight": "115%"},
            body_text,
        ),
        "preformatted": _make_odf_paragraph_style(
            doc,
            "Preformatted",
            {"textalign": "left", "marginbottom": "0.08cm", "lineheight": "105%"},
            {"fontsize": body_size, "fontfamily": "Liberation Mono"},
        ),
    }
    styles["heading1_center"] = _make_odf_paragraph_style(
        doc,
        "H1C",
        {
            "textalign": "center",
            "margintop": "0.25cm",
            "marginbottom": "0.12cm",
            "keepwithnext": "always",
        },
        {"fontsize": heading1_size, "fontweight": "bold", "fontfamily": "Liberation Sans"},
    )
    styles["heading2_center"] = _make_odf_paragraph_style(
        doc,
        "H2C",
        {
            "textalign": "center",
            "margintop": "0.2cm",
            "marginbottom": "0.1cm",
            "keepwithnext": "always",
        },
        {"fontsize": heading2_size, "fontweight": "bold", "fontfamily": "Liberation Sans"},
    )
    styles["heading3_center"] = _make_odf_paragraph_style(
        doc,
        "H3C",
        {
            "textalign": "center",
            "margintop": "0.15cm",
            "marginbottom": "0.08cm",
            "keepwithnext": "always",
        },
        {"fontsize": heading3_size, "fontweight": "bold", "fontfamily": "Liberation Sans"},
    )
    return styles


def _odf_table_cell_style(doc, name: str, border_bottom: str):
    from odf.style import Style, TableCellProperties

    style = Style(name=name, family="table-cell")
    style.addElement(
        TableCellProperties(
            padding="0.06cm",
            borderbottom=border_bottom,
            verticalalign="middle",
        )
    )
    doc.automaticstyles.addElement(style)
    return style


def create_odf(
    pages_elements: list[list[DocElement]],
    output_path: str,
    page_images: dict[int, list[tuple[bytes, str, int, int, float]]] | None = None,
    page_size_cm: tuple[float, float] | None = None,
    body_font_size_pt: float = 9.0,
    cancel_event: threading.Event | None = None,
):
    """Generate a structured ODF document."""
    from odf.opendocument import OpenDocumentText

    doc: Any = OpenDocumentText()
    styles = _configure_odf_document(doc, page_size_cm, body_font_size_pt)
    if page_images:
        doc.automaticstyles.addElement(styles["image_frame"])

    counters = {"table": [0], "image": [0]}

    for page_idx, elements in enumerate(pages_elements):
        if cancel_event is not None and cancel_event.is_set():
            raise ExportCancelled()
        _render_odf_page(doc, elements, page_idx, page_images, styles, counters)

    if cancel_event is not None and cancel_event.is_set():
        raise ExportCancelled()

    target = Path(output_path)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=target.suffix or ".odt",
        dir=target.parent,
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        doc.save(str(temp_path))
        publish_file_atomically(temp_path, target, overwrite=True)
    finally:
        temp_path.unlink(missing_ok=True)
    logger.info("Saved ODF: %s", target)


# The fixed-layout export writes every line in Liberation Sans and measures it
# with ReportLab's Helvetica metrics. That is exact rather than approximate:
# across the ASCII range the two differ by at most 0.08% in advance width
# (measured), because Liberation Sans is metric-compatible with Arial, which is
# metric-compatible with Helvetica.
POSITIONED_FONT_FAMILY = "Liberation Sans"
_METRIC_PROXY_FONT = "Helvetica"

# Liberation Sans hhea, measured: ascender 0.9053, descender -0.2119,
# lineGap 0.0327 em, so its natural line box is 1.1499 em and LibreOffice puts
# the first baseline ~0.94 em below the box top. These two are calibrations of
# LibreOffice's layout, not derivations from the format; the round-trip test is
# what pins them.
LIBERATION_LINE_BOX_EM = 1.15
LIBERATION_FIRST_BASELINE_EM = 0.94

# How far the width fit may stray from the height estimate. The width fit is
# exact for Helvetica-metric sources and drifts for others -- Times needs about
# 1.11, monospace less than 1 -- so the band keeps a wrong proxy from producing
# an absurd size while leaving the fit free to do the fine work.
WIDTH_FIT_MIN_RATIO = 0.75
WIDTH_FIT_MAX_RATIO = 1.30
# Below this share of the line's extent being measurable, the fit is not
# representative and the height estimate is used whole.
WIDTH_FIT_MIN_COVERAGE = 0.40
FRAME_WIDTH_SLACK = 1.06
FONT_SIZE_QUANTUM_PT = 0.25
_CM_PER_POINT = 2.54 / 72.0


def _proxy_width(text: str, size_pt: float) -> float | None:
    """Advance width of *text*, or None when the proxy font cannot measure it.

    The encodability check is the measurement, not a guard around it:
    ``stringWidth`` does not refuse text outside the font's encoding, it
    silently returns a number -- 45.66 pt for six CJK characters at 10 pt --
    and a fit built on that number would be confidently wrong. Latin-1 is
    Helvetica's encoding, so text outside it has no advance width to report.
    """
    from reportlab.pdfbase import pdfmetrics

    try:
        text.encode("latin-1")
    except UnicodeEncodeError:
        return None
    try:
        return pdfmetrics.stringWidth(text, _METRIC_PROXY_FONT, size_pt)
    except (KeyError, ValueError):
        return None


def _width_fitted_size(words) -> tuple[float | None, float]:
    """Median per-word size that reproduces the measured widths, and its coverage.

    Per word rather than per line: ``TextLine._assemble_text`` invents one or
    two spaces per gap whatever the gap really measures, so fitting the whole
    line would charge that reconstruction error to the glyph size.
    """
    from statistics import median

    from bigocrpdf.utils.tsv_parser import MIN_WORD_WIDTH

    fits: list[float] = []
    measured_extent = 0.0
    total_extent = 0.0
    for word in words:
        total_extent += max(word.width, 0.0)
        if word.width <= MIN_WORD_WIDTH:
            continue
        unit = _proxy_width(word.text, 1.0)
        if not unit:
            continue
        fits.append(word.width / unit)
        measured_extent += word.width
    if not fits or total_extent <= 0:
        return None, 0.0
    return median(fits), measured_extent / total_extent


def positioned_font_size(line, shrink: float = 1.0) -> tuple[float, float]:
    """Return ``(size in source points, size to write)`` for one line.

    The width is what breaks a layout -- a line 15% too wide overruns its
    column, while one 5% too tall merely looks off -- so the width fit leads and
    the height estimate only bounds it.
    """
    from bigocrpdf.constants import MAX_FONT_SIZE, MIN_FONT_SIZE
    from bigocrpdf.utils.tsv_parser import TSV_BOX_HEIGHT_EM

    line_height_pt = max((word.height for word in line.words), default=9.0)
    from_height = max(line_height_pt, 0.1) / TSV_BOX_HEIGHT_EM

    from_width, coverage = _width_fitted_size(line.words)
    if from_width is None or coverage < WIDTH_FIT_MIN_COVERAGE:
        size_source_pt = from_height
    else:
        size_source_pt = min(
            max(from_width, WIDTH_FIT_MIN_RATIO * from_height),
            WIDTH_FIT_MAX_RATIO * from_height,
        )

    written = min(max(size_source_pt * shrink, MIN_FONT_SIZE), MAX_FONT_SIZE)
    written = round(written / FONT_SIZE_QUANTUM_PT) * FONT_SIZE_QUANTUM_PT
    return size_source_pt, max(written, MIN_FONT_SIZE)


def positioned_frame_geometry(
    line, size_source_pt: float, scale_x: float, scale_y: float
) -> tuple[float, float, float, float]:
    """Frame ``(x, y, width, height)`` in centimetres for one line.

    The frame top is not the line top: ``line.y`` is a metric-box top for the
    *source* font, and LibreOffice will place its own first baseline relative to
    the frame. Recovering the source baseline and subtracting Liberation's
    first-baseline offset is what makes the two coincide.
    """
    from bigocrpdf.utils.tsv_parser import TSV_BASELINE_FRACTION

    line_height_pt = max((word.height for word in line.words), default=9.0)
    baseline_pt = line.y + TSV_BASELINE_FRACTION * line_height_pt
    y_pt = baseline_pt - LIBERATION_FIRST_BASELINE_EM * size_source_pt
    height_pt = LIBERATION_LINE_BOX_EM * size_source_pt

    source_extent_pt = max(line.max_x - line.min_x, 0.0)
    rendered_pt = _proxy_width(line.text, size_source_pt) or 0.0
    width_pt = max(source_extent_pt, rendered_pt) * FRAME_WIDTH_SLACK

    return (
        max(line.min_x * scale_x, 0.0),
        max(y_pt * scale_y, 0.0),
        max(width_pt * scale_x, 0.05),
        max(height_pt * scale_y, 0.05),
    )


def create_positioned_text_odf(
    pages_words,
    output_path: str,
    page_geometries: list[tuple[float, float, float, float]],
    cancel_event: threading.Event | None = None,
) -> str:
    """Create an editable fixed-layout ODT from positioned PDF words."""
    from odf.draw import Frame, TextBox
    from odf.opendocument import OpenDocumentText
    from odf.style import (
        GraphicProperties,
        MasterPage,
        PageLayout,
        PageLayoutProperties,
        ParagraphProperties,
        Style,
        TextProperties,
    )
    from odf.text import P

    from bigocrpdf.utils.tsv_parser import TextLine, filter_words, group_into_lines

    page_numbers = list(range(1, len(page_geometries) + 1))
    doc: Any = OpenDocumentText()
    page_break_styles = []
    for page_index, (_width_pt, _height_pt, width_cm, height_cm) in enumerate(page_geometries):
        page_layout = PageLayout(name=f"PositionedPage{page_index + 1}")
        page_layout.addElement(
            PageLayoutProperties(
                pagewidth=f"{width_cm:.2f}cm",
                pageheight=f"{height_cm:.2f}cm",
                margin="0cm",
            )
        )
        doc.automaticstyles.addElement(page_layout)
        master_name = "Standard" if page_index == 0 else f"PositionedMaster{page_index + 1}"
        doc.masterstyles.addElement(MasterPage(name=master_name, pagelayoutname=page_layout))
        page_break_styles.append(
            Style(
                name=f"PositionedBreak{page_index + 1}",
                family="paragraph",
                masterpagename=master_name,
            )
        )
        page_break_styles[-1].addElement(ParagraphProperties(breakbefore="page"))
        doc.automaticstyles.addElement(page_break_styles[-1])
    frame_style = Style(name="PositionedTextFrame", family="graphic")
    frame_style.addElement(
        GraphicProperties(
            wrap="run-through",
            runthrough="foreground",
            stroke="none",
            fill="none",
            # LibreOffice insets text 0.25cm horizontally and 0.125cm
            # vertically by default, which both shifts every line and steals
            # the width it needs. Per-side values as well as the shorthand,
            # because the import has historically ignored the shorthand alone.
            padding="0cm",
            paddingtop="0cm",
            paddingbottom="0cm",
            paddingleft="0cm",
            paddingright="0cm",
            border="none",
            # This, not the computed width, is what guarantees a line never
            # wraps: when the font is substituted -- CJK, or a host without
            # Liberation Sans -- no width we could compute would be right, and
            # a wrapped line drops its tail onto the line below.
            autogrowwidth="true",
            autogrowheight="true",
            wrapoption="no-wrap",
            textareaverticalalign="top",
            # Without these, svg:x/y are resolved against the default frame of
            # reference rather than the page. It happens to work today only
            # because the page margin is zero.
            horizontalpos="from-left",
            horizontalrel="page",
            verticalpos="from-top",
            verticalrel="page",
            flowwithtext="false",
        )
    )
    doc.automaticstyles.addElement(frame_style)
    text_styles = {}
    anchored_frames = []
    for page_index, page_number in enumerate(page_numbers):
        if cancel_event is not None and cancel_event.is_set():
            raise ExportCancelled()
        width_pt, height_pt, width_cm, height_cm = page_geometries[page_index]
        scale_x = width_cm / width_pt
        scale_y = height_cm / height_pt
        # _page_size_cm shrinks pages longer than A4, and scale_y carries that
        # shrink. The font has to shrink with it, or an oversized photo page
        # keeps full-size text inside frames reduced to a third.
        shrink = scale_y / _CM_PER_POINT
        source_lines = group_into_lines(filter_words(pages_words.get(page_number, []), page_number))
        positioned_runs = []
        for source_line in source_lines:
            run_words = []
            for word in source_line.words:
                if run_words:
                    gap = word.left - run_words[-1].right
                    run_height = max(item.height for item in run_words)
                    if gap > max(run_height * 2.0, 18.0):
                        positioned_runs.append(TextLine(run_words, source_line.y))
                        run_words = []
                run_words.append(word)
            if run_words:
                positioned_runs.append(TextLine(run_words, source_line.y))

        # Reading order, made explicit: a later change to the run splitter must
        # not be able to scramble it silently, because the z-index below -- and
        # with it hit-testing, Tab order and the order LibreOffice emits text
        # when exporting -- follows the order frames are written in.
        positioned_runs.sort(key=lambda run: (round(run.y, 1), run.min_x))

        for line_index, line in enumerate(positioned_runs):
            if not line.text.strip():
                continue
            size_source_pt, font_size_pt = positioned_font_size(line, shrink)
            if font_size_pt not in text_styles:
                style = Style(name=f"PositionedText{len(text_styles) + 1}", family="paragraph")
                style.addElement(
                    ParagraphProperties(
                        margin="0cm",
                        padding="0cm",
                        lineheight="100%",
                        textindent="0cm",
                        # Explicitly left, not start: under an RTL interface
                        # locale "start" right-aligns every line inside its
                        # frame and moves the whole page.
                        textalign="left",
                    )
                )
                style.addElement(
                    TextProperties(
                        fontsize=f"{font_size_pt:.2f}pt", fontfamily=POSITIONED_FONT_FAMILY
                    )
                )
                doc.automaticstyles.addElement(style)
                text_styles[font_size_pt] = style
            x_cm, y_cm, width_cm_frame, height_cm_frame = positioned_frame_geometry(
                line, size_source_pt, scale_x, scale_y
            )
            frame = Frame(
                stylename=frame_style,
                name=f"Page{page_index + 1}Line{line_index + 1}",
                anchortype="page",
                anchorpagenumber=page_index + 1,
                zindex=len(anchored_frames),
                x=f"{x_cm:.3f}cm",
                y=f"{y_cm:.3f}cm",
                width=f"{width_cm_frame:.3f}cm",
                height=f"{height_cm_frame:.3f}cm",
            )
            text_box = TextBox()
            text_box.addElement(P(stylename=text_styles[font_size_pt], text=line.text))
            frame.addElement(text_box)
            anchored_frames.append(frame)

    if cancel_event is not None and cancel_event.is_set():
        raise ExportCancelled()

    for frame in anchored_frames:
        doc.text.addElement(frame)
    doc.text.addElement(P())
    for page_index in range(1, len(page_numbers)):
        doc.text.addElement(P(stylename=page_break_styles[page_index]))

    target = Path(output_path)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=target.suffix or ".odt", dir=target.parent
    )
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        doc.save(str(temp_path))
        publish_file_atomically(temp_path, target, overwrite=True)
    finally:
        temp_path.unlink(missing_ok=True)
    logger.info("Saved positioned-text ODF: %s", target)
    return str(target)


def _render_odf_page(
    doc,
    elements: list[DocElement],
    page_idx: int,
    page_images: dict[int, list[tuple[bytes, str, int, int, float]]] | None,
    styles: dict,
    counters: dict[str, list[int]],
) -> None:
    from odf.text import P

    if page_idx > 0:
        doc.text.addElement(P(stylename=styles["page_break"]))

    img_items = _page_image_items(page_idx + 1, page_images)
    if not img_items:
        column_break = _column_break_index(elements)
        if column_break is not None:
            from odf.text import Section

            section = Section(name=f"PageColumns{page_idx + 1}", stylename=styles["two_columns"])
            doc.text.addElement(section)
            for elem in elements[:column_break]:
                _render_odf_element(doc, elem, styles, counters["table"], section)
            section.addElement(P(stylename=styles["column_break"]))
            for elem in elements[column_break:]:
                _render_odf_element(doc, elem, styles, counters["table"], section)
            return
        for elem in elements:
            _render_odf_element(doc, elem, styles, counters["table"])
        return

    _render_odf_elements_with_images(doc, elements, img_items, styles, counters)


def _page_image_items(
    page_num: int,
    page_images: dict[int, list[tuple[bytes, str, int, int, float]]] | None,
) -> list[tuple[float, tuple[bytes, str, int, int]]]:
    if not page_images or page_num not in page_images:
        return []
    return [
        (y_top, (img_data, mime, width_px, height_px))
        for img_data, mime, width_px, height_px, y_top in page_images[page_num]
    ]


def _render_odf_elements_with_images(
    doc,
    elements: list[DocElement],
    img_items: list[tuple[float, tuple[bytes, str, int, int]]],
    styles: dict,
    counters: dict[str, list[int]],
) -> None:
    img_idx = 0
    img_items.sort(key=lambda x: x[0])
    for elem in elements:
        while img_idx < len(img_items) and img_items[img_idx][0] <= elem.y_top:
            _render_odf_image(doc, img_items[img_idx][1], styles, counters["image"])
            img_idx += 1
        _render_odf_element(doc, elem, styles, counters["table"])
    while img_idx < len(img_items):
        _render_odf_image(doc, img_items[img_idx][1], styles, counters["image"])
        img_idx += 1


def _render_odf_image(
    doc,
    img_tuple: tuple[bytes, str, int, int],
    styles: dict,
    img_counter: list[int],
) -> None:
    from odf.draw import Frame
    from odf.draw import Image as OdfImage
    from odf.text import P

    img_data, mime, width_px, height_px = img_tuple
    try:
        frame_w_cm = min(16.0, width_px * 2.54 / 150)
        frame_h_cm = frame_w_cm * (height_px / width_px) if width_px > 0 else 8.0
        img_counter[0] += 1
        extension = {"image/jpeg": ".jpg", "image/png": ".png"}.get(mime, ".png")
        img_p = P(stylename=styles["paragraph_center"])
        frame = Frame(
            stylename=styles["image_frame"],
            width=f"{frame_w_cm:.2f}cm",
            height=f"{frame_h_cm:.2f}cm",
            anchortype="as-char",
        )
        href = doc.addPicture(f"Pictures/img{img_counter[0]}{extension}", mime, img_data)
        frame.addElement(OdfImage(href=href))
        img_p.addElement(frame)
        doc.text.addElement(img_p)
    except Exception as e:
        logger.debug("Could not embed image: %s", e)


def _column_break_index(elements: list[DocElement]) -> int | None:
    """Find the reading-order reset produced by the column detector."""
    for index in range(1, len(elements)):
        if elements[index - 1].y_top - elements[index].y_top > 100:
            return index
    return None


# Element kinds the layout analyser produces for headings, and the outline
# level each maps to.
_HEADING_OUTLINE_LEVELS = {"heading1": 1, "heading2": 2, "heading3": 3}


def _render_odf_element(
    doc, elem: DocElement, styles: dict, tbl_counter: list[int], container=None
) -> None:
    from odf.text import H, LineBreak, P

    if elem.kind == "table":
        _render_table(
            doc,
            elem.rows,
            styles["table"],
            styles["cell"],
            styles["header_cell"],
            styles["cell_text"],
            styles["cell_text_left"],
            styles["bold"],
            styles["table_width_cm"],
            tbl_counter,
            container,
        )
        return

    style_key = f"{elem.kind}_{elem.text_align}" if elem.text_align else elem.kind
    style = styles.get(style_key, styles.get(elem.kind, styles["paragraph"]))
    outline_level = _HEADING_OUTLINE_LEVELS.get(elem.kind)
    if outline_level is not None:
        # A heading has to be a heading, not a paragraph that happens to be
        # bold and larger. Only text:h gives an outline in the navigator, an
        # automatic table of contents, and heading semantics to a screen
        # reader -- and the analyser had already identified 29 of them in an
        # eighteen-page contract that exported as a flat wall of paragraphs.
        heading = H(outlinelevel=outline_level, stylename=style)
        heading.addText(elem.text)
        (container or doc.text).addElement(heading)
        return
    paragraph = P(stylename=style)
    if elem.kind == "preformatted" and elem.raw_lines and len(elem.raw_lines) > 1:
        for line_index, line_text in enumerate(elem.raw_lines):
            if line_index > 0:
                paragraph.addElement(LineBreak())
            paragraph.addText(line_text.strip())
    else:
        paragraph.addText(elem.text)
    (container or doc.text).addElement(paragraph)


def _render_table(
    doc,
    rows,
    tbl_s,
    cell_s,
    hdr_cell_s,
    cell_txt_s,
    cell_txt_l,
    bold_s,
    table_width_cm,
    counter,
    container=None,
):
    """Render a table into the ODF document."""
    from odf.style import Style, TableColumnProperties
    from odf.table import Table, TableCell, TableColumn, TableRow
    from odf.text import P, Span

    if not rows:
        return

    max_cols = max(len(r) for r in rows)
    counter[0] += 1
    tid = counter[0]
    table = Table(stylename=tbl_s, name=f"Table{tid}")

    col_w = f"{table_width_cm / max_cols:.2f}cm"
    for ci in range(max_cols):
        cs = Style(name=f"T{tid}C{ci}", family="table-column")
        cs.addElement(TableColumnProperties(columnwidth=col_w))
        doc.automaticstyles.addElement(cs)
        table.addElement(TableColumn(stylename=cs))

    for ri, row_data in enumerate(rows):
        row = TableRow()
        is_hdr = ri == 0
        for ci in range(max_cols):
            cell = TableCell(stylename=hdr_cell_s if is_hdr else cell_s)
            p_style = cell_txt_l if ci == 0 else cell_txt_s
            p = P(stylename=p_style)
            text = row_data[ci] if ci < len(row_data) else ""
            if is_hdr:
                span = Span(stylename=bold_s)
                span.addText(text)
                p.addElement(span)
            else:
                p.addText(text)
            cell.addElement(p)
            row.addElement(cell)
        table.addElement(row)

    (container or doc.text).addElement(table)
