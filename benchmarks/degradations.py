#!/usr/bin/env python3
"""Reproducible image degradations for the difficult-image benchmark corpus.

The synthetic fixtures were eight clean renderings: no blur, no noise, no
rotation, no lighting variation. A corpus like that cannot tell a robustness
improvement from a robustness regression, because every sample is easy.

Each function here applies one degradation whose parameters are recorded into
the manifest, so a fixture can always be regenerated and a metric shift can
always be traced to the axis that caused it.

Two rules hold throughout:

* **Ground truth is never transformed.** Degradations touch pixels only; the
  text is authored once and stays exact.
* **Randomness is derived, never global.** The seed comes from the sample id,
  axis and level, so a fixture regenerates identically regardless of the order
  or process it is generated in. No ``random``, no ``np.random.seed``.

cv2 is deliberately not used here. It is already a runtime dependency, and
making it a *fixture identity* dependency too would mean an OpenCV upgrade
silently changes the corpus -- the 4-to-5 upgrade this project just went
through changed interpolation and return shapes, which is exactly the risk.
"""

from __future__ import annotations

import hashlib
import io
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

# Level 0 is the untouched control on every axis; 1-3 run mild to severe.
LEVELS = (0, 1, 2, 3)

AXES = (
    "blur",
    "jpeg",
    "gaussian_noise",
    "salt_pepper",
    "rotate",
    "skew",
    "perspective",
    "illumination",
    "shadow",
    "low_contrast",
    "low_dpi",
    "ink_bleed",
    "faint_glyphs",
    "broken_glyphs",
)

# Composite recipe: what a phone photo of a document actually looks like.
PHOTO_REALISTIC = (
    ("perspective", 2),
    ("illumination", 2),
    ("blur", 1),
    ("gaussian_noise", 1),
    ("jpeg", 2),
)


def derive_seed(sample_id: str, axis: str, level: int) -> int:
    """A stable seed for one (sample, axis, level).

    Derived rather than sequential so that generating a subset, or generating
    in parallel, produces byte-identical fixtures to a full serial run.
    """
    digest = hashlib.blake2s(f"{sample_id}|{axis}|{level}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _rng(sample_id: str, axis: str, level: int) -> np.random.Generator:
    return np.random.default_rng(derive_seed(sample_id, axis, level))


def _as_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def _as_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")


# --- individual axes ---------------------------------------------------------


def _blur(image: Image.Image, level: int, _rng_: np.random.Generator) -> tuple[Image.Image, dict]:
    sigma = {1: 0.8, 2: 1.8, 3: 3.0}[level]
    return image.filter(ImageFilter.GaussianBlur(sigma)), {"sigma": sigma}


def _jpeg(image: Image.Image, level: int, _rng_: np.random.Generator) -> tuple[Image.Image, dict]:
    quality = {1: 60, 2: 30, 3: 12}[level]
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB"), {"quality": quality}


def _gaussian_noise(
    image: Image.Image, level: int, rng: np.random.Generator
) -> tuple[Image.Image, dict]:
    sigma = {1: 8.0, 2: 20.0, 3: 40.0}[level]
    array = _as_array(image)
    return _as_image(array + rng.normal(0.0, sigma, array.shape)), {"sigma": sigma}


def _salt_pepper(
    image: Image.Image, level: int, rng: np.random.Generator
) -> tuple[Image.Image, dict]:
    amount = {1: 0.005, 2: 0.02, 3: 0.06}[level]
    array = _as_array(image)
    mask = rng.random(array.shape[:2])
    array[mask < amount / 2] = 0.0
    array[mask > 1.0 - amount / 2] = 255.0
    return _as_image(array), {"amount": amount}


def _rotate(image: Image.Image, level: int, _rng_: np.random.Generator) -> tuple[Image.Image, dict]:
    degrees = {1: 0.7, 2: 3.0, 3: 12.0}[level]
    rotated = image.rotate(
        degrees, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255)
    )
    return rotated, {"degrees": degrees}


def _skew(image: Image.Image, level: int, _rng_: np.random.Generator) -> tuple[Image.Image, dict]:
    shear = {1: 0.02, 2: 0.06, 3: 0.12}[level]
    width, height = image.size
    # Widen the canvas so the sheared content cannot fall off the edge.
    new_width = width + int(abs(shear) * height) + 2
    return (
        image.transform(
            (new_width, height),
            Image.Transform.AFFINE,
            (1, shear, -shear * height / 2, 0, 1, 0),
            resample=Image.Resampling.BICUBIC,
            fillcolor=(255, 255, 255),
        ),
        {"shear": shear},
    )


def _perspective(
    image: Image.Image, level: int, _rng_: np.random.Generator
) -> tuple[Image.Image, dict]:
    offset_frac = {1: 0.02, 2: 0.06, 3: 0.14}[level]
    width, height = image.size
    shift = offset_frac * width
    source = [(0, 0), (width, 0), (width, height), (0, height)]
    target = [(shift, 0), (width - shift, 0), (width, height), (0, height)]
    coefficients = _perspective_coefficients(target, source)
    return (
        image.transform(
            (width, height),
            Image.Transform.PERSPECTIVE,
            coefficients,
            resample=Image.Resampling.BICUBIC,
            fillcolor=(255, 255, 255),
        ),
        {"offset_frac": offset_frac},
    )


def _perspective_coefficients(
    target: list[tuple[float, float]],
    source: list[tuple[float, float]],
) -> list[float]:
    """Solve for the eight coefficients Pillow's PERSPECTIVE transform wants.

    Pillow maps output to input, so the pairs are passed target-first.
    """
    matrix = []
    for (tx, ty), (sx, sy) in zip(target, source, strict=True):
        matrix.append([tx, ty, 1, 0, 0, 0, -sx * tx, -sx * ty])
        matrix.append([0, 0, 0, tx, ty, 1, -sy * tx, -sy * ty])
    a = np.array(matrix, dtype=np.float64)
    b = np.array(source, dtype=np.float64).reshape(8)
    return list(np.linalg.solve(a, b))


def _illumination(
    image: Image.Image, level: int, _rng_: np.random.Generator
) -> tuple[Image.Image, dict]:
    min_gain = {1: 0.85, 2: 0.60, 3: 0.35}[level]
    array = _as_array(image)
    height, width = array.shape[:2]
    ys = np.linspace(1.0, min_gain, height)[:, None]
    xs = np.linspace(1.0, min_gain, width)[None, :]
    return _as_image(array * np.sqrt(ys * xs)[..., None]), {"min_gain": min_gain}


def _shadow(image: Image.Image, level: int, _rng_: np.random.Generator) -> tuple[Image.Image, dict]:
    drop = {1: 0.15, 2: 0.35, 3: 0.60}[level]
    array = _as_array(image)
    height, width = array.shape[:2]
    edge = np.clip(np.linspace(-4.0, 4.0, width), -4.0, 4.0)
    softness = 1.0 / (1.0 + np.exp(-edge))
    gain = (1.0 - drop) + drop * softness
    return _as_image(array * np.tile(gain, (height, 1))[..., None]), {"drop": drop}


def _low_contrast(
    image: Image.Image, level: int, _rng_: np.random.Generator
) -> tuple[Image.Image, dict]:
    low, high = {1: (30, 225), 2: (70, 185), 3: (100, 160)}[level]
    array = _as_array(image) / 255.0
    return _as_image(low + array * (high - low)), {"low": low, "high": high}


def _low_dpi(
    image: Image.Image, level: int, _rng_: np.random.Generator
) -> tuple[Image.Image, dict]:
    """Resample down to a lower effective DPI and back, losing detail for good."""
    target_dpi = {1: 200, 2: 150, 3: 100}[level]
    scale = target_dpi / 300.0
    width, height = image.size
    small = image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        resample=Image.Resampling.BILINEAR,
    )
    return small.resize((width, height), resample=Image.Resampling.BILINEAR), {
        "target_dpi": target_dpi
    }


def _ink_bleed(
    image: Image.Image, level: int, _rng_: np.random.Generator
) -> tuple[Image.Image, dict]:
    radius = {1: 1, 2: 2, 3: 3}[level]
    return image.filter(ImageFilter.MinFilter(2 * radius + 1)), {"radius": radius}


def _faint_glyphs(
    image: Image.Image, level: int, _rng_: np.random.Generator
) -> tuple[Image.Image, dict]:
    opacity = {1: 0.75, 2: 0.55, 3: 0.35}[level]
    array = _as_array(image)
    return _as_image(255.0 - (255.0 - array) * opacity), {"opacity": opacity}


def _broken_glyphs(
    image: Image.Image, level: int, rng: np.random.Generator
) -> tuple[Image.Image, dict]:
    """Erase small patches of ink, as a worn print or a dirty scanner does."""
    fraction = {1: 0.01, 2: 0.04, 3: 0.09}[level]
    array = _as_array(image)
    ink = array.mean(axis=2) < 128
    erase = ink & (rng.random(ink.shape) < fraction)
    array[erase] = 255.0
    return _as_image(array), {"fraction": fraction}


_AXIS_FUNCTIONS = {
    "blur": _blur,
    "jpeg": _jpeg,
    "gaussian_noise": _gaussian_noise,
    "salt_pepper": _salt_pepper,
    "rotate": _rotate,
    "skew": _skew,
    "perspective": _perspective,
    "illumination": _illumination,
    "shadow": _shadow,
    "low_contrast": _low_contrast,
    "low_dpi": _low_dpi,
    "ink_bleed": _ink_bleed,
    "faint_glyphs": _faint_glyphs,
    "broken_glyphs": _broken_glyphs,
}


def apply_degradation(
    image: Image.Image,
    axis: str,
    level: int,
    sample_id: str,
) -> tuple[Image.Image, dict[str, Any]]:
    """Apply one degradation, returning the image and its recorded parameters.

    Level 0 is the identity on every axis, which gives each sweep a control
    row generated by exactly the same code path as the degraded ones.
    """
    if axis not in _AXIS_FUNCTIONS:
        raise ValueError(f"unknown degradation axis: {axis!r}")
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    if level == 0:
        return image.copy(), {"axis": axis, "level": 0}

    degraded, params = _AXIS_FUNCTIONS[axis](image, level, _rng(sample_id, axis, level))
    return degraded, {
        "axis": axis,
        "level": level,
        "seed": derive_seed(sample_id, axis, level),
        **params,
    }


def apply_recipe(
    image: Image.Image,
    recipe: tuple[tuple[str, int], ...],
    sample_id: str,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Apply several axes in a fixed order, recording each step."""
    result = image
    records = []
    for axis, level in recipe:
        result, params = apply_degradation(result, axis, level, sample_id)
        records.append(params)
    return result, records
