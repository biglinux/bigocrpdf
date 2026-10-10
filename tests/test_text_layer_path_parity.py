"""The content-stream text layer places OCR text where the canvas renderer does.

``renderer.TextLayerRenderer`` draws image-only pages through ReportLab;
``pdf_assembly.create_text_layer_commands`` writes raw operators for
mixed-content and embedded-image pages. A document must get the same text
placement whichever route it takes.
"""

import random
import re
from pathlib import Path

import pytest

from bigocrpdf.services.rapidocr_service import pdf_assembly
from bigocrpdf.services.rapidocr_service.config import OCRConfig, OCRResult
from bigocrpdf.services.rapidocr_service.renderer import TextLayerRenderer

SEED = 20260809

PAGE = (612.0, 792.0)
IMAGE = (2550, 3300)
SCALE_X = PAGE[0] / IMAGE[0]
SCALE_Y = PAGE[1] / IMAGE[1]


def _canvas_placement(box: list[list[float]]) -> tuple[float, float, float]:
    renderer = TextLayerRenderer(OCRConfig(font_base_path=Path("/nonexistent-font-dir")))
    layer = renderer.create_text_layer(
        [OCRResult(text="WORD", box=box, confidence=0.9)], IMAGE[0], IMAGE[1], page_size_pts=PAGE
    )
    placed = layer.boxes[0]
    return placed.x, placed.y, placed.font_size


def _stream_commands(box: list[list[float]], offset=(0.0, 0.0), text="WORD") -> str:
    return " ".join(
        pdf_assembly.create_text_layer_commands(
            [OCRResult(text=text, box=box, confidence=0.9)],
            img_x=offset[0],
            img_y=offset[1],
            img_width=PAGE[0],
            img_height=PAGE[1],
            scale_x=SCALE_X,
            scale_y=SCALE_Y,
        )
    )


def _stream_placement(box: list[list[float]], offset=(0.0, 0.0)) -> tuple[float, float, float]:
    commands = _stream_commands(box, offset)
    position = re.search(r"1 0 0 1 (\S+) (\S+) Tm", commands)
    font = re.search(r"/FOcr (\S+) Tf", commands)
    assert position and font, commands
    return float(position[1]), float(position[2]), float(font[1])


def _quad(left: float, top: float, right: float, height: float, shear: float) -> list[list[float]]:
    """A box whose right side is ``shear`` pixels lower than its left."""
    return [
        [left, top],
        [right, top + shear],
        [right, top + shear + height],
        [left, top + height],
    ]


def test_content_stream_matches_canvas_placement():
    rng = random.Random(SEED)

    for _ in range(200):
        left = rng.uniform(0, IMAGE[0] - 200)
        right = rng.uniform(left + 50, IMAGE[0])
        top = rng.uniform(0, IMAGE[1] - 300)
        shear = rng.choice([0.0, rng.uniform(-80, 80)])
        box = _quad(left, top, right, rng.uniform(10, 200), shear)

        x_a, y_a, size_a = _canvas_placement(box)
        x_b, y_b, size_b = _stream_placement(box)

        # The content stream prints two decimals for Tm and one for Tf.
        assert (x_b, y_b) == pytest.approx((x_a, y_a), abs=0.006), f"seed={SEED} box={box}"
        assert size_b == pytest.approx(size_a, abs=0.06), f"seed={SEED} box={box}"


def test_content_stream_is_shifted_by_the_image_origin():
    box = _quad(100.0, 200.0, 900.0, 60.0, 0.0)
    x, y, _ = _stream_placement(box)

    shifted_x, shifted_y, _ = _stream_placement(box, offset=(36.0, 18.0))

    assert (shifted_x - x, shifted_y - y) == pytest.approx((36.0, 18.0), abs=0.011)


def test_content_stream_bounds_the_horizontal_stretch():
    commands = _stream_commands(_quad(0.0, 100.0, 2500.0, 40.0, 0.0), text="W")

    (scale,) = re.findall(r"([\d.]+) Tz", commands)
    assert float(scale) == 300.0


@pytest.mark.parametrize("text", ["", "   "])
def test_blank_ocr_text_produces_no_commands(text):
    assert _stream_commands(_quad(0.0, 0.0, 100.0, 40.0, 0.0), text=text) == ""
