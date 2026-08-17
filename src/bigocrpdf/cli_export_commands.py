"""Document export command implementations for the BigOcrPdf CLI.

allow-noisy-log: export commands print user-facing output paths.
"""

import argparse
import logging

from bigocrpdf.utils.durable_writes import write_text_atomically


def _structured_document(args: argparse.Namespace, logger: logging.Logger):
    """Return the OCR document from ``--from-json``, or None to read the PDF.

    Without ``--from-json`` the PDF's own text layer is the source, which is
    what every other PDF tool reads and needs no second file to exist.
    """
    from bigocrpdf.services.rapidocr_service.ocr_document_io import load_ocr_document_json

    if args.from_json is None:
        return None
    document = load_ocr_document_json(args.from_json, args.input)
    if document is None:
        logger.warning(
            "%s does not describe %s; exporting from the PDF text layer instead",
            args.from_json,
            args.input,
        )
        return None
    logger.info("Using structured OCR JSON: %s", args.from_json)
    return document


def _cmd_export_odf(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle the 'export-odf' command."""
    from bigocrpdf.services.rapidocr_service.ocr_document_export import (
        convert_ocr_document_to_odf,
    )
    from bigocrpdf.utils.tsv_odf_converter import convert_pdf_to_odf

    if args.output:
        odf_path = str(args.output)
    else:
        odf_path = str(args.input.with_suffix(".odt"))

    logger.info(f"Converting {args.input} → {odf_path}")
    preserve_text_layout = bool(getattr(args, "preserve_text_layout", False))
    if preserve_text_layout:
        logger.info("Preserving PDF text positions")
        result = convert_pdf_to_odf(str(args.input), odf_path, include_images=True)
    elif (document := _structured_document(args, logger)) is not None:
        result = convert_ocr_document_to_odf(document, odf_path)
    else:
        result = convert_pdf_to_odf(str(args.input), odf_path)
    print(f"Saved: {result}")
    return 0


def _cmd_export_txt(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle the 'export-txt' command."""
    from bigocrpdf.services.rapidocr_service.ocr_document_export import (
        convert_ocr_document_to_text,
    )
    from bigocrpdf.utils.tsv_odf_converter import convert_pdf_to_text

    if args.output:
        txt_path = str(args.output)
    else:
        txt_path = str(args.input.with_suffix(".txt"))

    logger.info(f"Converting {args.input} → {txt_path}")
    document = _structured_document(args, logger)
    if document is not None:
        text = convert_ocr_document_to_text(document)
    else:
        text = convert_pdf_to_text(str(args.input))
    write_text_atomically(txt_path, text)
    print(f"Saved: {txt_path}")
    return 0


def _cmd_export_md(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle the 'export-md' command."""
    from bigocrpdf.services.rapidocr_service.ocr_document_export import (
        convert_ocr_document_to_markdown,
    )
    from bigocrpdf.utils.tsv_odf_converter import convert_pdf_to_markdown

    if args.output:
        md_path = str(args.output)
    else:
        md_path = str(args.input.with_suffix(".md"))

    logger.info(f"Converting {args.input} → {md_path}")
    document = _structured_document(args, logger)
    if document is not None:
        text = convert_ocr_document_to_markdown(
            document,
            source_path=str(args.input),
            include_front_matter=args.front_matter,
        )
    else:
        text = convert_pdf_to_markdown(str(args.input), include_front_matter=args.front_matter)
    write_text_atomically(md_path, text)
    print(f"Saved: {md_path}")
    return 0
