"""Standalone editor command implementation for the BigOcrPdf CLI.

allow-noisy-log: standalone editor save results are user-facing CLI output.
"""

import argparse
import logging
import sys
from pathlib import Path


def _cmd_edit(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle the 'edit' command — launch GUI editor directly."""
    from bigocrpdf import _check_gtk_dependencies

    if not _check_gtk_dependencies():
        return 1

    from gi.repository import Adw

    from bigocrpdf.ui.pdf_editor.editor_window import PDFEditorWindow

    app = Adw.Application(application_id="com.biglinux.bigocrpdf.editor")

    def on_activate(_app):
        win = PDFEditorWindow(
            application=_app,
            pdf_path=str(args.input.resolve()),
            on_save_callback=lambda doc: _standalone_save(doc, args.input, logger),
        )
        win.present()

    app.connect("activate", on_activate)
    return app.run([])


def _standalone_save(doc: object, original_path: Path, logger: logging.Logger) -> bool:
    """Save callback for standalone editor mode."""
    import os
    import stat
    import tempfile

    from bigocrpdf.ui.pdf_editor.page_model import PDFDocument
    from bigocrpdf.ui.pdf_editor.page_operations import apply_changes_to_pdf
    from bigocrpdf.utils.durable_writes import publish_file_atomically

    if not isinstance(doc, PDFDocument):
        print("Error: failed to save PDF", file=sys.stderr)
        return False

    output = original_path.resolve()
    try:
        output_mode = stat.S_IMODE(output.stat().st_mode) & 0o777
    except OSError:
        output_mode = None
    fd, tmp = tempfile.mkstemp(
        suffix=".pdf",
        prefix="bigocr_edit_",
        dir=output.parent,
    )
    staged_path = Path(tmp)

    try:
        if output_mode is not None:
            os.fchmod(fd, output_mode)
        os.close(fd)
        fd = -1
        if apply_changes_to_pdf(doc, tmp):
            publish_file_atomically(
                staged_path,
                output,
                overwrite=True,
            )
            logger.info("Saved edited PDF: %s", output)
            print(f"Saved: {output}")
            return True
        else:
            print("Error: failed to save PDF", file=sys.stderr)
            return False
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            staged_path.unlink(missing_ok=True)
        except OSError:
            pass
