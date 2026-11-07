#!/usr/bin/env python3
"""BigOcrPdf CLI entry point."""

import logging
import signal
import sys

from bigocrpdf import keep_cwd_out_of_child_imports
from bigocrpdf.cli_editor_commands import _cmd_edit
from bigocrpdf.cli_export_commands import _cmd_export_md, _cmd_export_odf, _cmd_export_txt
from bigocrpdf.cli_ocr_commands import _cmd_ocr
from bigocrpdf.cli_parser import _parse_page_list, _parse_ranges, build_parser
from bigocrpdf.cli_pdf_commands import (
    _cmd_compress,
    _cmd_delete,
    _cmd_extract,
    _cmd_info,
    _cmd_insert,
    _cmd_merge,
    _cmd_reorder,
    _cmd_rotate,
    _cmd_split,
)

__all__ = ["_parse_page_list", "_parse_ranges", "build_parser", "main"]


def _exit_on_termination(signum: int, _frame) -> None:
    """End like Ctrl+C does, so a job stops its workers and removes its files."""
    raise SystemExit(128 + signum)


def _setup_environment() -> None:
    """Minimal setup without GTK dependencies."""
    from bigocrpdf.utils.i18n import setup_i18n

    setup_i18n()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    keep_cwd_out_of_child_imports()
    _setup_environment()
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _exit_on_termination)

    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logger = logging.getLogger("bigocrpdf.cli")

    if hasattr(args, "input") and args.input and not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        return 1

    handlers = {
        "ocr": _cmd_ocr,
        "split": _cmd_split,
        "merge": _cmd_merge,
        "compress": _cmd_compress,
        "rotate": _cmd_rotate,
        "delete": _cmd_delete,
        "extract": _cmd_extract,
        "insert": _cmd_insert,
        "reorder": _cmd_reorder,
        "info": _cmd_info,
        "export-odf": _cmd_export_odf,
        "export-txt": _cmd_export_txt,
        "export-md": _cmd_export_md,
        "edit": _cmd_edit,
    }

    handler = handlers.get(args.command)
    if handler:
        try:
            return handler(args, logger)
        except Exception as e:
            logger.debug("Command failed", exc_info=True)
            print(f"Error: {e}", file=sys.stderr)
            return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
