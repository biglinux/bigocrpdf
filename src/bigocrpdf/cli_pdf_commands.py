"""PDF operation command implementations for the BigOcrPdf CLI.

allow-noisy-log: PDF operation commands print user-facing results.
"""

import argparse
import logging
import sys

from bigocrpdf.cli_parser import _parse_page_list, _parse_ranges


def _cmd_split(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle the 'split' command."""
    from bigocrpdf.services.pdf_operations import split_by_pages, split_by_ranges, split_by_size

    if args.pages is not None:
        if args.pages < 1:
            print("Error: --pages must be at least 1", file=sys.stderr)
            return 1
        result = split_by_pages(args.input, args.output, args.pages, prefix=args.prefix)
    elif args.size is not None:
        if args.size <= 0:
            print("Error: --size must be greater than 0", file=sys.stderr)
            return 1
        result = split_by_size(args.input, args.output, args.size, prefix=args.prefix)
    elif args.ranges:
        ranges = _parse_ranges(args.ranges)
        result = split_by_ranges(args.input, args.output, ranges, prefix=args.prefix)
    else:
        print("Error: specify --pages, --size, or --ranges", file=sys.stderr)
        return 1

    print(f"Split into {result.parts} parts ({result.total_pages} total pages)")
    for f in result.output_files:
        print(f"  → {f}")
    return 0


def _cmd_merge(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle the 'merge' command."""
    from bigocrpdf.services.pdf_operations import merge_pdfs

    for p in args.inputs:
        if not p.exists():
            print(f"Error: {p} not found", file=sys.stderr)
            return 1

    result = merge_pdfs([str(p) for p in args.inputs], str(args.output))
    if result.success:
        print(f"Merged: {result.message} → {args.output}")
        return 0
    else:
        print(f"Error: {result.message}", file=sys.stderr)
        return 1


def _cmd_compress(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle the 'compress' command."""
    from bigocrpdf.services.pdf_operations import compress_pdf

    result = compress_pdf(
        args.input,
        args.output,
        image_quality=args.quality,
        image_dpi=args.dpi,
    )
    if result.success:
        print(result.message)
        return 0
    else:
        print(f"Error: {result.message}", file=sys.stderr)
        return 1


def _cmd_rotate(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle the 'rotate' command."""
    from bigocrpdf.services.pdf_operations import get_pdf_info, rotate_pages

    if args.pages:
        pages = _parse_page_list(args.pages)
    else:
        info = get_pdf_info(args.input)
        pages = list(range(1, info.page_count + 1))

    result = rotate_pages(args.input, args.output, pages, args.angle)
    if result.success:
        print(f"Rotated: {result.message} → {args.output}")
        return 0
    else:
        print(f"Error: {result.message}", file=sys.stderr)
        return 1


def _cmd_delete(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle the 'delete' command."""
    from bigocrpdf.services.pdf_operations import delete_pages

    pages = _parse_page_list(args.pages)
    result = delete_pages(args.input, args.output, pages)
    if result.success:
        print(f"Deleted: {result.message} → {args.output}")
        return 0
    else:
        print(f"Error: {result.message}", file=sys.stderr)
        return 1


def _cmd_extract(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle the 'extract' command."""
    from bigocrpdf.services.pdf_operations import extract_pages

    pages = _parse_page_list(args.pages)
    result = extract_pages(args.input, args.output, pages)
    if result.success:
        print(f"Extracted: {result.message} → {args.output}")
        return 0
    else:
        print(f"Error: {result.message}", file=sys.stderr)
        return 1


def _cmd_insert(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle the 'insert' command."""
    from bigocrpdf.services.pdf_operations import insert_pages

    if not args.insert_from.exists():
        print(f"Error: {args.insert_from} not found", file=sys.stderr)
        return 1

    source_pages = _parse_page_list(args.pages) if args.pages else None
    result = insert_pages(
        args.input,
        args.insert_from,
        args.output,
        at_page=args.at,
        source_pages=source_pages,
    )
    if result.success:
        print(f"Inserted: {result.message} → {args.output}")
        return 0
    else:
        print(f"Error: {result.message}", file=sys.stderr)
        return 1


def _cmd_reorder(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle the 'reorder' command."""
    from bigocrpdf.services.pdf_operations import reorder_pages, reverse_pages

    if args.reverse:
        result = reverse_pages(args.input, args.output)
    else:
        order = [int(x.strip()) for x in args.order.split(",")]
        result = reorder_pages(args.input, args.output, order)

    if result.success:
        print(f"Reordered: {result.message} → {args.output}")
        return 0
    else:
        print(f"Error: {result.message}", file=sys.stderr)
        return 1


def _cmd_info(args: argparse.Namespace, _logger: logging.Logger) -> int:
    """Handle the 'info' command."""
    from bigocrpdf.services.pdf_operations import get_pdf_info

    info = get_pdf_info(str(args.input))
    print(f"File:       {info.path}")
    print(f"Pages:      {info.page_count}")
    print(f"Size:       {info.file_size_mb:.2f} MB ({info.file_size_bytes:,} bytes)")
    print(f"Version:    PDF {info.pdf_version}")
    print(f"Encrypted:  {'Yes' if info.encrypted else 'No'}")
    if info.title:
        print(f"Title:      {info.title}")
    if info.author:
        print(f"Author:     {info.author}")
    if info.creator:
        print(f"Creator:    {info.creator}")
    return 0
