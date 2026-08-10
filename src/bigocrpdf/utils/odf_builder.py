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

    from bigocrpdf.utils.tsv_parser import TextLine, group_into_lines

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
    frame_style.addElement(GraphicProperties(wrap="run-through", stroke="none", fill="none"))
    doc.automaticstyles.addElement(frame_style)
    text_styles = {}
    anchored_frames = []
    for page_index, page_number in enumerate(page_numbers):
        if cancel_event is not None and cancel_event.is_set():
            raise ExportCancelled()
        width_pt, height_pt, width_cm, height_cm = page_geometries[page_index]
        scale_x = width_cm / width_pt
        scale_y = height_cm / height_pt
        source_lines = group_into_lines(pages_words.get(page_number, []))
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

        for line_index, line in enumerate(positioned_runs):
            if not line.text.strip():
                continue
            line_height_pt = max((word.height for word in line.words), default=9.0)
            font_size_pt = round(min(max(line_height_pt * 0.75, 5.0), 24.0), 1)
            if font_size_pt not in text_styles:
                style = Style(name=f"PositionedText{len(text_styles) + 1}", family="paragraph")
                style.addElement(
                    ParagraphProperties(margin="0cm", padding="0cm", lineheight="100%")
                )
                style.addElement(
                    TextProperties(fontsize=f"{font_size_pt:.1f}pt", fontfamily="Liberation Sans")
                )
                doc.automaticstyles.addElement(style)
                text_styles[font_size_pt] = style
            frame = Frame(
                stylename=frame_style,
                name=f"Page{page_index + 1}Line{line_index + 1}",
                anchortype="page",
                anchorpagenumber=page_index + 1,
                zindex=0,
                x=f"{max(line.min_x * scale_x, 0):.3f}cm",
                y=f"{max(line.y * scale_y, 0):.3f}cm",
                width=f"{max((line.max_x - line.min_x) * scale_x + line_height_pt * scale_x * 1.5, 0.3):.3f}cm",
                height=f"{max(line_height_pt * scale_y * 1.8, 0.25):.3f}cm",
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
