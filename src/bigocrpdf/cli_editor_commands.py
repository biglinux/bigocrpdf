"""Standalone editor command implementation for the BigOcrPdf CLI."""

import argparse
import logging


def _cmd_edit(args: argparse.Namespace, logger: logging.Logger) -> int:
    """Handle the 'edit' command: launch the GUI editor directly."""
    from bigocrpdf import _check_gtk_dependencies

    if not _check_gtk_dependencies():
        return 1

    from gi.repository import Adw

    from bigocrpdf.config import EDITOR_APP_ID
    from bigocrpdf.ui.pdf_editor.editor_window import PDFEditorWindow

    app = Adw.Application(application_id=EDITOR_APP_ID)

    def on_activate(_app):
        win = PDFEditorWindow(
            application=_app,
            pdf_path=str(args.input.resolve()),
            save_in_place=True,
        )
        win.present()

    app.connect("activate", on_activate)
    return app.run([])
