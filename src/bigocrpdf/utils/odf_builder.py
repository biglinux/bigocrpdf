"""ODF document generation from structured DocElements.

Handles ODF style setup, element rendering, table formatting,
and image extraction/embedding from source PDFs.
"""

from __future__ import annotations

import io
import re

from bigocrpdf.utils.logger import logger  # noqa: I001
from bigocrpdf.utils.tsv_parser import DocElement

# Max pixel width for images embedded in ODF (~180 DPI on A4).
_MAX_IMAGE_WIDTH = 1500
# Pixel-count threshold above which non-JPEG images are saved as JPEG.
_JPEG_THRESHOLD = 256 * 256


def _pil_to_jpeg(pil_img: "Image.Image", w: int, h: int) -> tuple[bytes, str, int, int]:
    """Convert a PIL image to JPEG bytes, downscaling if oversized.

    Returns (jpeg_bytes, "image/jpeg", out_w, out_h).
    """
    from PIL import Image as PILImage

    if pil_img.mode in ("RGBA", "P"):
        pil_img = pil_img.convert("RGB")
    elif pil_img.mode not in ("RGB", "L"):
        pil_img = pil_img.convert("RGB")

    if w > _MAX_IMAGE_WIDTH:
        scale = _MAX_IMAGE_WIDTH / w
        new_w, new_h = int(w * scale), int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
        w, h = new_w, new_h

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), "image/jpeg", w, h


class ExportCancelled(Exception):
    """Raised when the user cancels the ODF export."""


def _extract_pdf_images(
    pdf_path: str,
    cancel_event: "threading.Event | None" = None,
) -> dict[int, list[tuple[bytes, str, int, int, float]]]:
    """Extract actual images from a PDF, mapped by page number.

    Parses each page's content stream to find which XObject names are
    actually referenced (``/Name Do``), then extracts only those image
    streams using pikepdf — no re-encoding for JPEG, no subprocess.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict mapping 1-based page number to list of
        (image_bytes, mime_type, w_px, h_px, y_top) tuples.
        ``y_top`` is the top-down Y position matching pdftotext TSV
        coordinates.
    """
    try:
        import pikepdf
    except ImportError:
        logger.warning("pikepdf not available; cannot embed images")
        return {}

    MIN_DIM = 50  # Skip icons/masks smaller than 50px

    def _image_positions(page: "pikepdf.Page") -> dict[str, float]:
        """Return {XObject_name: y_top} from the content stream."""
        positions: dict[str, float] = {}
        try:
            mediabox = page.get("/MediaBox")
            page_h = float(mediabox[3]) if mediabox else 792.0
        except Exception:
            page_h = 792.0

        try:
            ops = pikepdf.parse_content_stream(page)
        except Exception:
            return positions

        last_cm: list[float] | None = None
        for operands, operator in ops:
            op = str(operator)
            if op == "cm" and len(operands) == 6:
                last_cm = [float(x) for x in operands]
            elif op == "Do" and operands:
                name = str(operands[0]).lstrip("/")
                if last_cm is not None:
                    f_val = last_cm[5]
                    d_val = abs(last_cm[3])
                    positions[name] = page_h - (f_val + d_val)
                else:
                    positions[name] = 0.0
                last_cm = None
        return positions

    def _used_xobject_names(page: "pikepdf.Page") -> set[str]:
        """Return XObject names actually invoked in the page content stream."""
        names: set[str] = set()
        try:
            contents = page.get("/Contents")
            if contents is None:
                return names
            if isinstance(contents, pikepdf.Array):
                raw = b"".join(s.read_bytes() for s in contents)
            else:
                raw = contents.read_bytes()
        except Exception:
            raw = b""
        for m in re.finditer(rb"/(\S+)\s+Do\b", raw):
            names.add(m.group(1).decode("latin-1", errors="replace"))
        return names

    result: dict[int, list[tuple[bytes, str, int, int, float]]] = {}
    try:
        with pikepdf.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                if cancel_event is not None and cancel_event.is_set():
                    raise ExportCancelled()
                used_names = _used_xobject_names(page)
                img_positions = _image_positions(page)
                resources = page.get("/Resources", {})
                xobjects = resources.get("/XObject", {})
                for name, ref in xobjects.items():
                    clean_name = str(name).lstrip("/")
                    if clean_name not in used_names:
                        continue
                    obj = ref
                    if not hasattr(obj, "keys"):
                        continue
                    subtype = str(obj.get("/Subtype", ""))
                    if subtype != "/Image":
                        continue
                    w = int(obj.get("/Width", 0))
                    h = int(obj.get("/Height", 0))
                    if w < MIN_DIM or h < MIN_DIM:
                        continue

                    y_top = img_positions.get(clean_name, 0.0)

                    fltr = obj.get("/Filter", "")
                    if isinstance(fltr, pikepdf.Array):
                        filters = [str(f) for f in fltr]
                    else:
                        filters = [str(fltr)] if fltr else []

                    # Check if inner filter is DCTDecode (JPEG) wrapped
                    # in any transport encoding (ASCII85, Flate, etc.)
                    is_dct = "/DCTDecode" in filters

                    try:
                        if is_dct and len(filters) == 1:
                            # Pure JPEG — extract raw bytes directly
                            raw = obj.read_raw_bytes()
                            result.setdefault(page_num, []).append((raw, "image/jpeg", w, h, y_top))
                        elif is_dct:
                            # JPEG behind transport encoding (ASCII85, Flate…)
                            pil_img = pikepdf.PdfImage(obj).as_pil_image()
                            data, mime, ow, oh = _pil_to_jpeg(pil_img, w, h)
                            result.setdefault(page_num, []).append((data, mime, ow, oh, y_top))
                        else:
                            # Non-JPEG: use JPEG for large photographic images
                            pil_img = pikepdf.PdfImage(obj).as_pil_image()
                            is_photo = w * h >= _JPEG_THRESHOLD and pil_img.mode in (
                                "RGB",
                                "RGBA",
                                "L",
                            )
                            if is_photo:
                                data, mime, ow, oh = _pil_to_jpeg(pil_img, w, h)
                            else:
                                buf = io.BytesIO()
                                pil_img.save(buf, format="PNG")
                                data, mime, ow, oh = buf.getvalue(), "image/png", w, h
                            result.setdefault(page_num, []).append((data, mime, ow, oh, y_top))
                    except Exception as e:
                        logger.debug(
                            "Could not extract image %s on page %d: %s",
                            name,
                            page_num,
                            e,
                        )
    except Exception as e:
        logger.warning("Failed to extract images from PDF: %s", e)
        return {}

    total = sum(len(v) for v in result.values())
    logger.info("Extracted %d images across %d pages for ODF embedding", total, len(result))
    return result


def create_odf(
    pages_elements: list[list[DocElement]],
    output_path: str,
    page_images: dict[int, list[tuple[bytes, str, int, int]]] | None = None,
    cancel_event: "threading.Event | None" = None,
):
    """Generate a structured ODF document."""
    from odf.opendocument import OpenDocumentText
    from odf.style import (
        FontFace,
        MasterPage,
        PageLayout,
        PageLayoutProperties,
        ParagraphProperties,
        Style,
        TableCellProperties,
        TableProperties,
        TextProperties,
    )
    from odf.text import LineBreak, P

    doc = OpenDocumentText()

    # Page layout (A4)
    pl = PageLayout(name="A4")
    pl.addElement(
        PageLayoutProperties(
            pagewidth="21cm",
            pageheight="29.7cm",
            margintop="2cm",
            marginbottom="2cm",
            marginleft="2.5cm",
            marginright="2cm",
        )
    )
    doc.automaticstyles.addElement(pl)
    mp = MasterPage(name="Standard", pagelayoutname="A4")
    doc.masterstyles.addElement(mp)

    # Font
    ff = FontFace(
        name="Liberation Sans",
        fontfamily="Liberation Sans",
        fontfamilygeneric="swiss",
        fontpitch="variable",
    )
    doc.fontfacedecls.addElement(ff)

    # Paragraph styles
    def _make_style(name, para_kw, text_kw):
        s = Style(name=name, family="paragraph")
        s.addElement(ParagraphProperties(**para_kw))
        s.addElement(TextProperties(**text_kw))
        doc.automaticstyles.addElement(s)
        return s

    h1_s = _make_style(
        "H1",
        {
            "textalign": "left",
            "margintop": "0.6cm",
            "marginbottom": "0.3cm",
            "keepwithnext": "always",
        },
        {"fontsize": "13pt", "fontweight": "bold", "fontfamily": "Liberation Sans"},
    )
    h2_s = _make_style(
        "H2",
        {
            "textalign": "left",
            "margintop": "0.5cm",
            "marginbottom": "0.2cm",
            "keepwithnext": "always",
        },
        {"fontsize": "11.5pt", "fontweight": "bold", "fontfamily": "Liberation Sans"},
    )
    h3_s = _make_style(
        "H3",
        {
            "textalign": "left",
            "margintop": "0.3cm",
            "marginbottom": "0.15cm",
            "keepwithnext": "always",
            "marginleft": "0.5cm",
        },
        {"fontsize": "11pt", "fontweight": "bold", "fontfamily": "Liberation Sans"},
    )
    body_s = _make_style(
        "Body",
        {
            "textalign": "left",
            "marginbottom": "0.15cm",
            "lineheight": "140%",
        },
        {"fontsize": "11pt", "fontfamily": "Liberation Sans"},
    )
    body_indent_s = _make_style(
        "BodyI",
        {
            "textalign": "left",
            "marginbottom": "0.15cm",
            "lineheight": "140%",
            "textindent": "1.25cm",
        },
        {"fontsize": "11pt", "fontfamily": "Liberation Sans"},
    )
    body_center_s = _make_style(
        "BodyC",
        {
            "textalign": "center",
            "marginbottom": "0.05cm",
            "lineheight": "130%",
        },
        {"fontsize": "11pt", "fontfamily": "Liberation Sans"},
    )
    body_right_s = _make_style(
        "BodyR",
        {
            "textalign": "end",
            "marginbottom": "0.05cm",
            "lineheight": "130%",
        },
        {"fontsize": "11pt", "fontfamily": "Liberation Sans"},
    )
    kv_s = _make_style(
        "KV",
        {
            "textalign": "left",
            "marginbottom": "0.05cm",
            "lineheight": "130%",
        },
        {"fontsize": "11pt", "fontfamily": "Liberation Sans"},
    )

    bold_s = Style(name="Bold", family="text")
    bold_s.addElement(TextProperties(fontweight="bold"))
    doc.automaticstyles.addElement(bold_s)

    pb_s = Style(name="PB", family="paragraph")
    pb_s.addElement(ParagraphProperties(breakbefore="page"))
    doc.automaticstyles.addElement(pb_s)

    tbl_s = Style(name="Tbl", family="table")
    tbl_s.addElement(TableProperties(width="16.5cm", align="center"))
    doc.automaticstyles.addElement(tbl_s)

    cell_s = Style(name="Cell", family="table-cell")
    cell_s.addElement(
        TableCellProperties(
            padding="0.12cm",
            borderbottom="0.5pt solid #dddddd",
            verticalalign="middle",
        )
    )
    doc.automaticstyles.addElement(cell_s)

    hdr_cell_s = Style(name="HCell", family="table-cell")
    hdr_cell_s.addElement(
        TableCellProperties(
            padding="0.12cm",
            borderbottom="1pt solid #888888",
            verticalalign="middle",
        )
    )
    doc.automaticstyles.addElement(hdr_cell_s)

    cell_txt_s = _make_style(
        "CellText",
        {"textalign": "center", "marginbottom": "0cm"},
        {"fontsize": "10pt", "fontfamily": "Liberation Sans"},
    )
    cell_txt_l = _make_style(
        "CellTextL",
        {"textalign": "left", "marginbottom": "0cm"},
        {"fontsize": "10pt", "fontfamily": "Liberation Sans"},
    )

    style_map = {
        "heading1": h1_s,
        "heading2": h2_s,
        "heading3": h3_s,
        "paragraph": body_s,
        "paragraph_indent": body_indent_s,
        "paragraph_center": body_center_s,
        "paragraph_right": body_right_s,
        "kv": kv_s,
    }

    tbl_counter = [0]
    img_counter = [0]

    # Embed page images if provided
    _embed_images = bool(page_images)
    if _embed_images:
        from odf.draw import Frame
        from odf.draw import Image as OdfImage

        img_frame_s = Style(name="ImgFrame", family="graphic")
        doc.automaticstyles.addElement(img_frame_s)

    _MIME_EXT = {"image/jpeg": ".jpg", "image/png": ".png"}

    for page_idx, elements in enumerate(pages_elements):
        if cancel_event is not None and cancel_event.is_set():
            raise ExportCancelled()
        if page_idx > 0 and elements:
            doc.text.addElement(P(stylename=pb_s))

        page_num = page_idx + 1

        # Build image render items for this page (with Y position)
        img_items: list[tuple[float, tuple[bytes, str, int, int]]] = []
        if _embed_images and page_num in page_images:
            for img_data, mime, w_px, h_px, y_top in page_images[page_num]:
                img_items.append((y_top, (img_data, mime, w_px, h_px)))

        def _render_image(img_tuple: tuple[bytes, str, int, int]) -> None:
            img_data, mime, w_px, h_px = img_tuple
            try:
                frame_w_cm = min(16.0, w_px * 2.54 / 150)
                frame_h_cm = frame_w_cm * (h_px / w_px) if w_px > 0 else 8.0

                img_counter[0] += 1
                ext = _MIME_EXT.get(mime, ".png")
                pic_name = f"Pictures/img{img_counter[0]}{ext}"

                img_p = P(stylename=body_center_s)
                frame = Frame(
                    stylename=img_frame_s,
                    width=f"{frame_w_cm:.2f}cm",
                    height=f"{frame_h_cm:.2f}cm",
                    anchortype="as-char",
                )
                href = doc.addPicture(pic_name, mime, img_data)
                frame.addElement(OdfImage(href=href))
                img_p.addElement(frame)
                doc.text.addElement(img_p)
            except Exception as e:
                logger.debug("Could not embed image for page %d: %s", page_num, e)

        def _render_element(elem: DocElement) -> None:
            if elem.kind == "table":
                _render_table(
                    doc,
                    elem.rows,
                    tbl_s,
                    cell_s,
                    hdr_cell_s,
                    cell_txt_s,
                    cell_txt_l,
                    bold_s,
                    tbl_counter,
                )
                return
            s = style_map.get(elem.kind, body_s)
            p = P(stylename=s)
            if elem.raw_lines and len(elem.raw_lines) > 1:
                for li, line_text in enumerate(elem.raw_lines):
                    if li > 0:
                        p.addElement(LineBreak())
                    p.addText(line_text.strip())
            else:
                p.addText(elem.text)
            doc.text.addElement(p)

        if not img_items:
            for elem in elements:
                _render_element(elem)
        else:
            # Interleave images with text based on Y position
            img_idx = 0
            img_items.sort(key=lambda x: x[0])
            for elem in elements:
                while img_idx < len(img_items) and img_items[img_idx][0] <= elem.y_top:
                    _render_image(img_items[img_idx][1])
                    img_idx += 1
                _render_element(elem)
            while img_idx < len(img_items):
                _render_image(img_items[img_idx][1])
                img_idx += 1

    doc.save(output_path)
    logger.info("Saved ODF: %s", output_path)


def _render_table(doc, rows, tbl_s, cell_s, hdr_cell_s, cell_txt_s, cell_txt_l, bold_s, counter):
    """Render a table into the ODF document."""
    from odf.style import Style, TableColumnProperties  # noqa: F811
    from odf.table import Table, TableCell, TableColumn, TableRow  # noqa: F811
    from odf.text import P, Span  # noqa: F811

    if not rows:
        return

    max_cols = max(len(r) for r in rows)
    counter[0] += 1
    tid = counter[0]
    table = Table(stylename=tbl_s, name=f"Table{tid}")

    col_w = f"{16.5 / max_cols:.2f}cm"
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

    doc.text.addElement(table)
