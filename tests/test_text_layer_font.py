"""The font the searchable text layer is written with."""

import subprocess
from pathlib import Path

import pikepdf
import pytest
from reportlab.pdfgen import canvas as reportlab_canvas

from bigocrpdf.services.ocr.config import (
    OCRConfig,
    OCRResult,
    cjk_text_layer_font,
    text_layer_font,
)
from bigocrpdf.services.ocr.renderer import TextLayerRenderer

NOTO = "usr/share/fonts/noto/NotoSans-Regular.ttf"


def _render(pdf_path: Path, lines: list[str], config: OCRConfig) -> tuple[list[str], str]:
    """Render one line per entry; return the page's font names and pdftotext's text."""
    pdf_canvas = reportlab_canvas.Canvas(str(pdf_path), pagesize=(400, 200))
    results = [
        OCRResult(text, [[10, top], [390, top], [390, top + 30], [10, top + 30]], 1.0)
        for text, top in zip(lines, range(10, 200, 50), strict=False)
    ]
    TextLayerRenderer(config).render(pdf_canvas, results, (400, 200), page_size_pts=(400, 200))
    pdf_canvas.showPage()
    pdf_canvas.save()

    with pikepdf.open(pdf_path) as pdf:
        fonts = pdf.pages[0].Resources.Font
        base_fonts = [str(fonts[name].BaseFont) for name in fonts.keys()]
    result = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        check=True,
        capture_output=True,
        encoding="utf-8",
        timeout=10,
    )
    return base_fonts, result.stdout


def test_rendered_text_layer_embeds_noto_sans(tmp_path: Path) -> None:
    sample = "Ελληνικά Кириллица őłğ"
    base_fonts, text = _render(tmp_path / "noto.pdf", [sample], OCRConfig())

    assert any("NotoSans" in name for name in base_fonts), base_fonts
    assert text.split() == sample.split()


@pytest.mark.skipif(
    not cjk_text_layer_font().is_file(), reason=f"{cjk_text_layer_font()} is not installed"
)
def test_cjk_line_uses_fallback_font(tmp_path: Path) -> None:
    lines = ["Ελληνικά őłğ", "中文 日本語 かな カナ 漢字"]
    base_fonts, text = _render(tmp_path / "cjk.pdf", lines, OCRConfig())

    assert any("NotoSans" in name for name in base_fonts), base_fonts
    assert text.split() == " ".join(lines).split()


def test_cjk_without_fallback_keeps_other_lines(tmp_path: Path) -> None:
    lines = ["Ελληνικά őłğ", "中文"]
    config = OCRConfig(cjk_font_path=tmp_path / "absent.ttc")
    _, text = _render(tmp_path / "no_cjk.pdf", lines, config)

    assert text.split()[:2] == lines[0].split()


def test_missing_font_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing.ttf"):
        TextLayerRenderer(OCRConfig(font_path=tmp_path / "missing.ttf"))


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({}, Path("/") / NOTO),
        ({"APPDIR": "/app"}, Path("/app") / NOTO),
        ({"APPDIR": "/app", "BIGOCRPDF_TEXT_FONT": "/nix/font.ttf"}, Path("/nix/font.ttf")),
    ],
)
def test_font_location(monkeypatch, env: dict[str, str], expected: Path) -> None:
    """An AppImage uses its own copy, never the host's; the override wins."""
    for name in ("APPDIR", "BIGOCRPDF_TEXT_FONT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    assert text_layer_font() == expected
