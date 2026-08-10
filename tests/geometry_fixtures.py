"""Synthetic pages and known forward distortions for the geometry tests.

Not a test module -- imported, not collected.

The design point: a corrector is judged by how much of a *known* distortion it
removes. So each function here applies a distortion whose parameters the test
already has, and the corrector's job is to undo it. Nothing here inspects the
corrector.

Two constraints shape the fixtures:

* Generous white margins. A forward warp moves ink outward; without padding the
  glyphs leave the canvas and the test measures clipping rather than recovery.
* Curvature is expressed as the maximum baseline deviation in pixels, the same
  quantity ``dewarp_probmap._MIN_CURVATURE_PX`` and ``_MAX_CURVATURE_PX`` gate
  on, so a test can reason about whether a gate should fire.
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

# A4 at 200 DPI. 300 would be more realistic and twice as slow for no extra
# signal, since every metric here is resolution-normalised.
PAGE_SIZE = (1654, 2339)
MARGIN_FRAC = 0.18

DEFAULT_LINES = [
    "Relatorio tecnico de inspecao predial",
    "Documento emitido em 12 de marco de 2026",
    "Endereco: Avenida Central, numero 1420",
    "Responsavel tecnico: Ana Paula Ribeiro",
    "Registro profissional CREA 55-2291",
    "Area total construida: 384 metros quadrados",
    "Situacao geral da estrutura: adequada",
    "Proxima vistoria prevista para marco de 2027",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    """A real TrueType face, or skip -- bitmap fallbacks are unmeasurable."""
    try:
        matched = subprocess.run(
            ["fc-match", "--format=%{file}", "Noto Sans"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        if matched and Path(matched).is_file():
            return ImageFont.truetype(matched, size)
    except (OSError, subprocess.SubprocessError):
        pass
    for candidate in (
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    pytest.skip("no scalable font available to render geometry fixtures")


def render_synthetic_page(
    lines: list[str] | None = None,
    size: tuple[int, int] = PAGE_SIZE,
    font_size: int = 38,
    margin_frac: float = MARGIN_FRAC,
) -> np.ndarray:
    """A clean page of black text on white, in BGR, with wide margins."""
    lines = lines or DEFAULT_LINES
    width, height = size
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _font(font_size)

    left = int(width * margin_frac)
    top = int(height * margin_frac)
    leading = int(font_size * 2.2)
    for index, line in enumerate(lines):
        draw.text((left, top + index * leading), line, fill="black", font=font)

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


# Fraction of the page width the wide fixture's text must reach. The dewarp
# fits one spline per text line and removes each line's linear trend, so what
# it can see is the bow *across the span the text occupies*. Over a narrow
# column an arc is genuinely indistinguishable from a tilt -- and a tilt is the
# deskew's job -- so only a line reaching both margins exercises the remap.
WIDE_TEXT_COVERAGE = 0.94


def render_wide_text_page(
    size: tuple[int, int] = PAGE_SIZE,
    line_count: int = 12,
) -> np.ndarray:
    """A page whose every line reaches both margins, as a book page does."""
    width, height = size
    text = "documento largo de teste com linhas que alcancam ambas as margens da pagina"
    target = int(width * WIDE_TEXT_COVERAGE)

    font_size = 40
    for _ in range(40):
        measured = _font(font_size).getbbox(text)[2]
        if abs(measured - target) <= max(4, target // 200):
            break
        font_size = max(8, round(font_size * target / max(measured, 1)))
    font = _font(font_size)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left = (width - font.getbbox(text)[2]) // 2
    leading = max(int(font_size * 2.4), (height - 2 * int(height * MARGIN_FRAC)) // line_count)
    top = int(height * MARGIN_FRAC)
    for index in range(line_count):
        draw.text((left, top + index * leading), text, fill="black", font=font)

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def apply_rotation(image: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate about the centre, keeping the canvas size and a white fill."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def apply_homography(
    image: np.ndarray,
    corner_offsets_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """A keystone warp; offsets are a fraction of width so tests are scale-free.

    Returns the warped image and the homography that produced it.
    """
    height, width = image.shape[:2]
    shift = corner_offsets_frac * width
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    destination = np.float32(
        [
            [shift, shift * 0.5],
            [width - shift, 0],
            [width, height],
            [shift * 0.5, height - shift * 0.5],
        ]
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return warped, matrix


def apply_cylindrical_warp(image: np.ndarray, curvature_px: float) -> np.ndarray:
    """Bend text lines into an arc of ``curvature_px`` maximum deviation.

    The deviation is measured the same way ``dewarp_probmap`` measures it, so a
    test can say whether a curvature should clear the remap gate.
    """
    height, width = image.shape[:2]
    xs = np.arange(width, dtype=np.float32)
    # A half-sine across the page: zero at both edges, curvature_px in the middle.
    displacement = curvature_px * np.sin(np.pi * xs / max(width - 1, 1))
    map_x = np.tile(xs, (height, 1))
    map_y = (
        np.tile(np.arange(height, dtype=np.float32).reshape(-1, 1), (1, width))
        - displacement[None, :]
    )
    return cv2.remap(
        image,
        map_x,
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def apply_dark_border(image: np.ndarray, width_frac: float = 0.04, value: int = 20) -> np.ndarray:
    """Frame the page in dark pixels, as a photo on a desk produces."""
    result = image.copy()
    height, width = result.shape[:2]
    band_y = max(1, int(height * width_frac))
    band_x = max(1, int(width * width_frac))
    result[:band_y, :] = value
    result[-band_y:, :] = value
    result[:, :band_x] = value
    result[:, -band_x:] = value
    return result
