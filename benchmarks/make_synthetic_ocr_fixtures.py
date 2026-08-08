#!/usr/bin/env python3
"""Generate small synthetic OCR fixtures with ground truth."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

from PIL import (
    Image,
    ImageDraw,
    ImageFont,
    features,
)
from PIL import (
    __version__ as pillow_version,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.prepare_benchmark_datasets import (
    _publish_staged_rows,
    _write_text_atomically,
    save_deterministic_image_pdf,
    write_manifest,
)

SAMPLES = [
    (
        "pt_accented",
        "pt_BR",
        ["pt_br", "accented"],
        "Ação nº 123. João pagou R$ 1.234,56 em Brasília.",
    ),
    (
        "pt_legal",
        "pt_BR",
        ["pt_br", "legal"],
        "Processo Administrativo nº 0001234-56.2024.8.07.0001",
    ),
    ("cjk", "zh", ["cjk"], "中文文本 OCR 测试"),
    (
        "greek",
        "el",
        ["greek", "unsupported_ppocrv6"],
        "Ελληνικά κείμενο δοκιμής",
    ),
    (
        "arabic",
        "ar",
        ["arabic", "rtl", "unsupported_ppocrv6"],
        "اختبار التعرف الضوئي على الحروف",
    ),
    ("small_text", "pt_BR", ["small_text"], "Texto pequeno em 8 pt com acentuação."),
    (
        "two_columns",
        "pt_BR",
        ["two_columns"],
        "Coluna esquerda com texto.\nColuna direita com valores.",
    ),
    ("table", "pt_BR", ["table"], "Produto Quantidade Valor\nLivro 2 R$ 80,00"),
]
FIXTURE_DPI = 300


def load_font(
    size: int,
    language: str,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    family = {
        "zh": "Noto Sans CJK SC",
        "ar": "Noto Sans Arabic",
    }.get(language, "Noto Sans")
    try:
        result = subprocess.run(
            ["fc-match", "--format=%{file}", family],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        matched_path = Path(result.stdout.strip())
        if matched_path.is_file():
            return ImageFont.truetype(str(matched_path), size)
    except (OSError, subprocess.SubprocessError):
        pass
    for path in [
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    ]:
        candidate = Path(path)
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def draw_sample(sample_id: str, language: str, text: str, out_path: Path) -> dict[str, object]:
    img = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(img)
    font_size = round(8 * FIXTURE_DPI / 72) if sample_id == "small_text" else 36
    font = load_font(font_size, language)
    missing_characters = _missing_font_characters(font, text)
    if missing_characters:
        raise RuntimeError(f"Font cannot render sample {sample_id}: {''.join(missing_characters)}")

    if sample_id == "two_columns":
        draw.multiline_text(
            (80, 100), "Coluna esquerda\ncom texto.", fill="black", font=font, spacing=14
        )
        draw.multiline_text(
            (650, 100), "Coluna direita\ncom valores.", fill="black", font=font, spacing=14
        )
    elif sample_id == "table":
        rows = [["Produto", "Quantidade", "Valor"], ["Livro", "2", "R$ 80,00"]]
        for row_index, row in enumerate(rows):
            for col_index, cell in enumerate(row):
                draw.text(
                    (80 + col_index * 300, 120 + row_index * 80), cell, fill="black", font=font
                )
    else:
        draw.multiline_text((80, 160), text, fill="black", font=font, spacing=18)

    img.save(out_path)
    font_path = Path(str(getattr(font, "path", "")))
    return {
        "font_path": str(font_path) if font_path.is_file() else None,
        "font_sha256": _sha256(font_path) if font_path.is_file() else None,
    }


def _missing_font_characters(
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    text: str,
) -> list[str]:
    try:
        missing_mask = bytes(cast(Any, font.getmask("\U0010ffff")))
    except (AttributeError, OSError, ValueError):
        return []
    return sorted(
        {
            character
            for character in text
            if not character.isspace() and bytes(cast(Any, font.getmask(character))) == missing_mask
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def generate_rows(out_dir: Path) -> list[dict[str, object]]:
    """Generate deterministic source materials into a private directory."""
    images_dir = out_dir / "images"
    pdfs_dir = out_dir / "pdfs"
    ground_truth_dir = out_dir / "ground_truth"
    for directory in [images_dir, pdfs_dir, ground_truth_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    for sample_id, language, tags, text in SAMPLES:
        image_rel = Path("images") / f"synthetic_{sample_id}.png"
        pdf_rel = Path("pdfs") / f"synthetic_{sample_id}.pdf"
        gt_rel = Path("ground_truth") / f"synthetic_{sample_id}.txt"
        image_path = out_dir / image_rel
        pdf_path = out_dir / pdf_rel

        font_metadata = draw_sample(sample_id, language, text, image_path)
        with Image.open(image_path) as image:
            save_deterministic_image_pdf(
                image,
                pdf_path,
                resolution=float(FIXTURE_DPI),
            )
        (out_dir / gt_rel).write_text(text, encoding="utf-8")
        manifest_rows.append(
            {
                "id": f"synthetic_{sample_id}",
                "dataset": "synthetic",
                "image": str(image_rel),
                "pdf": str(pdf_rel),
                "gt_text": str(gt_rel),
                "language": language,
                "tags": tags,
                "source": {
                    "kind": "synthetic",
                    "generator": "benchmarks/make_synthetic_ocr_fixtures.py",
                    "fixture_dpi": FIXTURE_DPI,
                    "pillow_version": pillow_version,
                    "raqm_available": features.check_feature("raqm"),
                    **font_metadata,
                },
            }
        )
    return manifest_rows


def _write_generated_readme(out_dir: Path) -> None:
    _write_text_atomically(
        out_dir / "README.generated.md",
        (
            "# BigOCRPDF synthetic benchmark fixtures\n\n"
            "Generated by benchmarks/make_synthetic_ocr_fixtures.py.\n"
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--private-stage", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.private_stage:
        manifest_rows = generate_rows(args.out)
    else:
        with tempfile.TemporaryDirectory(
            prefix=".synthetic-source-",
            dir=args.out,
        ) as stage_name:
            stage_dir = Path(stage_name)
            staged_rows = generate_rows(stage_dir)
            manifest_rows = _publish_staged_rows(
                stage_dir,
                args.out,
                staged_rows,
                "synthetic",
            )

    manifest = write_manifest(args.out, manifest_rows)
    _write_generated_readme(args.out)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
