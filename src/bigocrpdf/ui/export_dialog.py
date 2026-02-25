"""Standalone export dialog for file manager integration.

Shows a format chooser (ODT / TXT) with quality warnings and
an option to apply OCR first when the PDF contains images.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GdkPixbuf, Gtk

from bigocrpdf.utils.i18n import _

_ILLUSTRATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "resources", "illustrations")


def _count_pdf_images(pdf_path: Path) -> int:
    """Count images inside the PDF using pikepdf (fast, no subprocess)."""
    try:
        import pikepdf

        count = 0
        with pikepdf.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                resources = page.get("/Resources")
                if not resources:
                    continue
                xobjects = resources.get("/XObject")
                if not xobjects:
                    continue
                for obj in xobjects.values():
                    if hasattr(obj, "get") and obj.get("/Subtype") == pikepdf.Name.Image:
                        count += 1
        return count
    except Exception:
        return -1  # unknown


def _load_svg(filename: str, size: int = 92) -> Gtk.Image:
    """Load an SVG illustration rendered to a fixed pixel size."""
    path = os.path.join(_ILLUSTRATIONS_DIR, filename)
    if os.path.exists(path):
        pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(path, size, size, True)
        image = Gtk.Image.new_from_pixbuf(pixbuf)
    else:
        image = Gtk.Image()
    image.set_pixel_size(size)
    image.set_halign(Gtk.Align.CENTER)
    image.set_valign(Gtk.Align.CENTER)
    image.set_hexpand(False)
    image.set_vexpand(False)
    image.set_accessible_role(Gtk.AccessibleRole.PRESENTATION)
    return image


def run_export_dialog(pdf_path: Path) -> int:
    """Launch a minimal Adw application that presents the export dialog."""
    app = Adw.Application(application_id="com.biglinux.bigocrpdf.export")

    result: list[int] = [0]

    def on_activate(_app: Adw.Application) -> None:
        win = _ExportWindow(application=_app, pdf_path=pdf_path, result=result)
        win.present()

    app.connect("activate", on_activate)
    app.run([])
    return result[0]


class _ExportWindow(Adw.Window):
    """Dialog-style window for choosing export format."""

    def __init__(
        self,
        *,
        application: Adw.Application,
        pdf_path: Path,
        result: list[int],
    ) -> None:
        super().__init__(application=application)
        self._pdf_path = pdf_path
        self._result = result

        self.set_title(_("Export to ODT or TXT"))
        self.set_default_size(520, -1)
        self.set_resizable(False)
        self.set_modal(True)

        image_count = _count_pdf_images(pdf_path)

        # --- layout ---
        toolbar_view = Adw.ToolbarView()
        header = Adw.HeaderBar()
        header.set_show_title(True)
        toolbar_view.add_top_bar(header)

        clamp = Adw.Clamp()
        clamp.set_maximum_size(520)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=24)
        content.set_margin_top(12)
        content.set_margin_bottom(24)
        content.set_margin_start(16)
        content.set_margin_end(16)

        # --- format cards — each with its own action button ---
        format_group = Adw.PreferencesGroup()
        format_group.set_title(_("Export format"))
        format_group.set_description(
            _("Choose the output format for the extracted text.")
        )

        # ODF row
        row_odt = Adw.ActionRow()
        row_odt.set_title(_("ODF Document (.odt)"))
        row_odt.set_subtitle(
            _(
                "Preserves text formatting, paragraphs and layout. "
                "Editable in LibreOffice, Google Docs and Microsoft Word."
            )
        )
        row_odt.add_prefix(_load_svg("export_odf.svg", 72))
        btn_odt = Gtk.Button(label=_("Export ODF"))
        btn_odt.set_valign(Gtk.Align.CENTER)
        btn_odt.add_css_class("suggested-action")
        btn_odt.set_tooltip_text(_("Export ODF"))
        btn_odt.update_property(
            [Gtk.AccessibleProperty.LABEL],
            [_("Export as ODF document")],
        )
        btn_odt.connect("clicked", lambda _b: self._do_fmt("odf"))
        row_odt.add_suffix(btn_odt)
        format_group.add(row_odt)

        # TXT row
        row_txt = Adw.ActionRow()
        row_txt.set_title(_("Plain Text (.txt)"))
        row_txt.set_subtitle(
            _(
                "Raw text without any formatting. "
                "Lightweight and universal — works everywhere."
            )
        )
        row_txt.add_prefix(_load_svg("export_txt.svg", 72))
        btn_txt = Gtk.Button(label=_("Export TXT"))
        btn_txt.set_valign(Gtk.Align.CENTER)
        btn_txt.add_css_class("suggested-action")
        btn_txt.set_tooltip_text(_("Export TXT"))
        btn_txt.update_property(
            [Gtk.AccessibleProperty.LABEL],
            [_("Export as plain text")],
        )
        btn_txt.connect("clicked", lambda _b: self._do_fmt("txt"))
        row_txt.add_suffix(btn_txt)
        format_group.add(row_txt)

        content.append(format_group)

        # --- OCR notice (only if PDF has images) ---
        if image_count != 0:
            ocr_group = Adw.PreferencesGroup()
            if image_count > 0:
                ocr_group.set_title(
                    _("This PDF contains %d image(s)") % image_count
                )
            else:
                ocr_group.set_title(_("PDF content analysis"))

            ocr_row = Adw.ActionRow()
            ocr_row.set_title(_("Apply OCR first"))
            ocr_row.set_subtitle(
                _(
                    "If the images contain text (scanned pages or photos), "
                    "applying OCR first will produce much better export results."
                )
            )
            ocr_row.add_prefix(_load_svg("export_ocr.svg", 72))

            btn_ocr = Gtk.Button(label=_("Open OCR"))
            btn_ocr.set_valign(Gtk.Align.CENTER)
            btn_ocr.add_css_class("suggested-action")
            btn_ocr.set_tooltip_text(_("Open OCR"))
            btn_ocr.update_property(
                [Gtk.AccessibleProperty.LABEL],
                [_("Open OCR application to recognize text in images")],
            )
            btn_ocr.connect("clicked", self._on_ocr)
            ocr_row.add_suffix(btn_ocr)

            ocr_group.add(ocr_row)
            content.append(ocr_group)

        clamp.set_child(content)
        toolbar_view.set_content(clamp)
        self.set_content(toolbar_view)

    def _do_fmt(self, fmt: str) -> None:
        self._result[0] = _do_export(self._pdf_path, fmt)
        self.close()

    def _on_ocr(self, _btn: Gtk.Button) -> None:
        _open_ocr(self._pdf_path)
        self._result[0] = 0
        self.close()


def _do_export(pdf_path: Path, fmt: str) -> int:
    """Run the CLI export command and return exit code."""
    cmd = [sys.executable, "-m", "bigocrpdf.cli", f"export-{fmt}", str(pdf_path)]
    try:
        proc = subprocess.run(cmd, check=False)
        return proc.returncode
    except Exception as exc:
        print(f"Export error: {exc}", file=sys.stderr)
        return 1


def _open_ocr(pdf_path: Path) -> None:
    """Open the main bigocrpdf GUI with the file in the queue."""
    subprocess.Popen(["bigocrpdf", str(pdf_path)])
