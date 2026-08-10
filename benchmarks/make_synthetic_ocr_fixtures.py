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

from benchmarks.degradations import (
    AXES,
    PHOTO_REALISTIC,
    apply_degradation,
    apply_recipe,
)
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
    ("two_columns", "pt_BR", ["two_columns", "layout_order_uncertain"], ""),
    ("table", "pt_BR", ["table", "layout_order_uncertain"], ""),
]
FIXTURE_DPI = 300

# Which degraded variants each tier generates. The full cross product is 14
# axes x 3 levels x 8 samples, which nobody re-runs; these are the curated
# subsets that make the corpus usable as a gate.
#
#   smoke -- the fast inner loop, one sample, the six axes that discriminate most
#   gate  -- the regression baseline
#   full  -- everything, including the severity-3 rows, which are diagnostic
#            rather than gating: their job is to show where the cliff is
GATE_SAMPLES = ("pt_accented", "pt_legal")
SMOKE_AXES = ("perspective", "illumination", "low_dpi", "blur", "jpeg", "broken_glyphs")
TIERS = {
    "smoke": {"samples": ("pt_accented",), "axes": SMOKE_AXES, "levels": (2,)},
    "gate": {"samples": GATE_SAMPLES, "axes": AXES, "levels": (1, 2)},
    "full": {"samples": None, "axes": AXES, "levels": (1, 2, 3)},
}

# Layout samples declare the segments they draw, and their ground truth is
# derived from those segments rather than written out a second time. The two
# used to be separate literals and had drifted apart, giving both samples a
# permanent non-zero error floor that no OCR change could ever clear.
COLUMN_SEGMENTS = [
    ("Coluna esquerda\ncom texto.", (80, 100)),
    ("Coluna direita\ncom valores.", (650, 100)),
]
TABLE_ROWS = [["Produto", "Quantidade", "Valor"], ["Livro", "2", "R$ 80,00"]]


def sample_ground_truth(sample_id: str, declared_text: str) -> str:
    """The text a sample actually draws, in reading order.

    Reading order across two columns is genuinely ambiguous, which is why both
    layout samples are tagged ``layout_order_uncertain``: gate them on
    ``sorted_line_char_error_rate`` and report plain CER without gating it.
    """
    if sample_id == "two_columns":
        return "\n".join(text for text, _position in COLUMN_SEGMENTS)
    if sample_id == "table":
        return "\n".join(" ".join(row) for row in TABLE_ROWS)
    return declared_text


# More than one family per language because the same typeface ships under
# different names: the CJK font is "Noto Sans CJK SC" on Debian and "Source Han
# Sans CN" on Arch, and asking for the wrong one is not an error -- fc-match
# answers with *something* whatever it is asked, so a missing family silently
# yields a font that cannot write the language.
_FONT_FAMILIES = {
    "zh": ("Noto Sans CJK SC", "Source Han Sans CN", "WenQuanYi Zen Hei"),
    "ar": ("Noto Sans Arabic", "Amiri", "DejaVu Sans"),
}
_FALLBACK_FAMILIES = ("Noto Sans", "DejaVu Sans")


def load_font(
    size: int,
    language: str,
    text: str = "",
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a font that can actually write *text*, not merely one named for it.

    Coverage decides, because the name cannot: the caller draws ground truth,
    and a glyph silently replaced by a box would make the fixture claim
    characters its image does not show.
    """
    last: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None
    for family in _FONT_FAMILIES.get(language, ()) + _FALLBACK_FAMILIES:
        try:
            result = subprocess.run(
                ["fc-match", "--format=%{file}", family],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            matched_path = Path(result.stdout.strip())
            if not matched_path.is_file():
                continue
            last = ImageFont.truetype(str(matched_path), size)
        except (OSError, subprocess.SubprocessError):
            continue
        if not _missing_font_characters(last, text):
            return last
    return last or ImageFont.load_default()


def draw_sample(sample_id: str, language: str, text: str, out_path: Path) -> dict[str, object]:
    img = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(img)
    font_size = round(8 * FIXTURE_DPI / 72) if sample_id == "small_text" else 36
    font = load_font(font_size, language, text)
    missing_characters = _missing_font_characters(font, text)
    if missing_characters:
        raise RuntimeError(f"Font cannot render sample {sample_id}: {''.join(missing_characters)}")

    if sample_id == "two_columns":
        for segment, position in COLUMN_SEGMENTS:
            draw.multiline_text(position, segment, fill="black", font=font, spacing=14)
    elif sample_id == "table":
        for row_index, row in enumerate(TABLE_ROWS):
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
    for sample_id, language, tags, declared_text in SAMPLES:
        text = sample_ground_truth(sample_id, declared_text)
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


def _publish_variant(
    out_dir: Path,
    base_row: dict[str, object],
    suffix: str,
    image: Image.Image,
    degradation: object,
    extra_tags: list[str],
) -> dict[str, object]:
    """Write one degraded variant and its manifest row.

    The ground truth file is shared with the clean sample, not rewritten: the
    degradation touched pixels only, so the expected text is unchanged. That is
    what keeps the corpus exact no matter how severe the damage.
    """
    sample_id = str(base_row["id"])
    variant_id = f"{sample_id}__{suffix}"
    image_rel = Path("images") / f"{variant_id}.png"
    pdf_rel = Path("pdfs") / f"{variant_id}.pdf"

    image.save(out_dir / image_rel)
    with Image.open(out_dir / image_rel) as saved:
        save_deterministic_image_pdf(saved, out_dir / pdf_rel, resolution=float(FIXTURE_DPI))

    source = dict(base_row["source"])  # type: ignore[arg-type]
    source["degradation"] = degradation
    return {
        **base_row,
        "id": variant_id,
        "image": str(image_rel),
        "pdf": str(pdf_rel),
        "tags": [*base_row["tags"], *extra_tags],  # type: ignore[misc]
        "source": source,
    }


def generate_degraded_rows(
    out_dir: Path,
    clean_rows: list[dict[str, object]],
    tier: str,
) -> list[dict[str, object]]:
    """Degraded variants of the clean samples, for the requested tier."""
    spec = TIERS[tier]
    wanted = spec["samples"]
    rows: list[dict[str, object]] = []

    for base_row in clean_rows:
        sample_id = str(base_row["id"]).removeprefix("synthetic_")
        if wanted is not None and sample_id not in wanted:
            continue
        with Image.open(out_dir / str(base_row["image"])) as clean:
            clean = clean.convert("RGB")

            for axis in spec["axes"]:
                for level in spec["levels"]:
                    degraded, params = apply_degradation(clean, axis, level, sample_id)
                    rows.append(
                        _publish_variant(
                            out_dir,
                            base_row,
                            f"{axis}{level}",
                            degraded,
                            params,
                            [f"axis:{axis}", f"level:{level}", f"tier:{tier}"],
                        )
                    )

            composite, records = apply_recipe(clean, PHOTO_REALISTIC, sample_id)
            rows.append(
                _publish_variant(
                    out_dir,
                    base_row,
                    "photo_realistic",
                    composite,
                    records,
                    ["photo_geometry", f"tier:{tier}"],
                )
            )
    return rows


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
    parser.add_argument(
        "--tier",
        choices=(*TIERS, "none"),
        default="none",
        help=(
            "also generate degraded variants: smoke (fast inner loop), "
            "gate (the regression baseline), full (everything, including the "
            "diagnostic severity-3 rows)"
        ),
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.private_stage:
        manifest_rows = generate_rows(args.out)
        if args.tier != "none":
            manifest_rows += generate_degraded_rows(args.out, manifest_rows, args.tier)
    else:
        with tempfile.TemporaryDirectory(
            prefix=".synthetic-source-",
            dir=args.out,
        ) as stage_name:
            stage_dir = Path(stage_name)
            staged_rows = generate_rows(stage_dir)
            if args.tier != "none":
                staged_rows += generate_degraded_rows(stage_dir, staged_rows, args.tier)
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
