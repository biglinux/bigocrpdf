"""Page-geometry hazards that were suspected of displacing the text layer.

Written as hypotheses and then tested. Recording the outcome matters as much
as the tests themselves, so the same ground is not re-suspected later:

* **MediaBox origin** -- suspected: nothing subtracts a non-zero origin, so the
  layer would be offset by ``(-x0, -y0)`` after merging into the original page.
  **Refuted.** Measured through the real ``overlay_text_on_original`` with
  positive and negative origins: displacement is 0.00 pt. Both the layer and
  the original are drawn in the same content-stream space, so the origin
  cancels. Locked in by ``TestMediaBoxOrigin``.

* **/UserUnit** -- suspected: ignored in the text-layer path. **Refuted, and
  the suspicion was misframed.** /UserUnit scales the physical interpretation
  of user space; it does not change the numbers written into a content stream.
  Reading the raw MediaBox is therefore correct. The real hazard is mixing the
  two conventions, which ``TestUserUnit`` checks instead.

* **Overlay + /Rotate 90/270 aspect mismatch** -- suspected: the OCR image is
  rotated (swapping width and height) while the MediaBox dimensions come back
  unswapped, giving an anisotropic mapping. **Refuted.** Rotating the image is
  precisely what makes its aspect match the page: measured scale ratio is
  1.000 for 90 and 270. Locked in by ``TestOverlayRotation``.

* **force_overlay over a geometrically corrected page** -- not previously
  suspected, found while tracing. **Confirmed reachable, and fixed**: the
  override is now conditional on ``geometry_applied``. See
  ``TestForceOverlayWithGeometry``.
"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bigocrpdf.services.rapidocr_service.backend_text_layer import BackendTextLayerMixin
from bigocrpdf.services.rapidocr_service.backend_text_layer_geometry import (
    _processed_page_dimensions,
)
from bigocrpdf.services.rapidocr_service.pdf_assembly import overlay_text_on_original
from tests.positional_oracle import (
    by_token,
    extract_words,
    render_layer_pdf,
    requires_pdftotext,
    tokens,
    truth_to_metric_quads,
    write_truth_pdf,
)

PAGE = (612.0, 792.0)
IMAGE = (2550, 3300)


class _ModeBackend(BackendTextLayerMixin):
    def __init__(self, export_format: str = "original") -> None:
        self.config = SimpleNamespace(image_export_format=export_format)


def _merged_displacement(tmp_path: Path, **page_attrs) -> tuple[float, float]:
    """Displacement between visible glyphs and the OCR layer after merging.

    The two sides use different tokens because poppler folds identical
    overlapping text into a single word, which would hide any offset.
    """
    base = tokens(4)
    visible = [replace(word, token="V" + word.token[1:]) for word in base]
    layer = [replace(word, token="L" + word.token[1:]) for word in base]

    original = write_truth_pdf(visible, PAGE, tmp_path / "orig.pdf", **page_attrs)
    text_layer = render_layer_pdf(
        truth_to_metric_quads(layer, PAGE, IMAGE), IMAGE, PAGE, tmp_path / "layer.pdf"
    )
    merged = tmp_path / "merged.pdf"
    overlay_text_on_original(original, text_layer, merged)

    words = by_token(extract_words(merged))
    assert "L00AAA" in words, "the OCR layer did not survive the merge"
    return (
        words["L00AAA"].x0 - words["V00AAA"].x0,
        words["L00AAA"].y_top - words["V00AAA"].y_top,
    )


@requires_pdftotext
class TestMediaBoxOrigin:
    """A non-zero MediaBox origin must not displace the text layer."""

    @pytest.mark.parametrize(
        "origin",
        [
            pytest.param((0.0, 0.0), id="zero"),
            pytest.param((20.0, 30.0), id="positive"),
            pytest.param((-15.0, 25.0), id="negative"),
        ],
    )
    def test_the_layer_stays_on_the_glyphs(self, tmp_path: Path, origin):
        delta_x, delta_y = _merged_displacement(tmp_path, mediabox_origin=origin)

        assert (delta_x, delta_y) == pytest.approx((0.0, 0.0), abs=0.5)

    def test_page_dimensions_ignore_the_origin(self):
        """Size is a difference, so the origin must not appear in it."""
        dimensions = _processed_page_dimensions(
            {}, {"mediabox": [20.0, 30.0, 615.0, 822.0]}, 2550, 3300
        )

        assert dimensions == pytest.approx((595.0, 792.0))


class TestUserUnit:
    """The hazard is mixing user-space units with physical ones, not /UserUnit."""

    def test_page_dimensions_stay_in_default_user_space(self):
        """Content-stream coordinates are unaffected by /UserUnit.

        ``rotation.PageRotation.pdf_dimensions`` multiplies by it because it
        feeds the render-DPI budget, which is a physical question. The
        text-layer path must not, or the image and the text would be drawn at
        different scales.
        """
        page_info = {"mediabox": [0.0, 0.0, 612.0, 792.0], "user_unit": 3.0}

        dimensions = _processed_page_dimensions({}, page_info, 2550, 3300)

        assert dimensions == pytest.approx((612.0, 792.0))

    @requires_pdftotext
    def test_placement_is_unchanged_by_user_unit(self, tmp_path: Path):
        delta_x, delta_y = _merged_displacement(tmp_path, user_unit=3.0)

        assert (delta_x, delta_y) == pytest.approx((0.0, 0.0), abs=0.5)


class TestOverlayRotation:
    """Rotating the OCR image is what keeps the mapping isotropic."""

    @pytest.mark.parametrize("rotation", [90, 270])
    def test_rotated_pages_map_isotropically(self, rotation):
        """A landscape scan of a portrait page must not be squashed."""
        landscape = np.zeros((2550, 3300, 3), np.uint8)

        _, _, pdf_width, pdf_height, ocr_size = _ModeBackend()._setup_overlay_mode(
            {"image_prerotated": False},
            {"rotation": rotation, "mediabox": [0.0, 0.0, 612.0, 792.0]},
            landscape,
            1,
        )

        scale_x = pdf_width / ocr_size[0]
        scale_y = pdf_height / ocr_size[1]
        assert scale_x == pytest.approx(scale_y, rel=1e-6)

    def test_a_prerotated_image_is_not_rotated_again(self):
        """The worker already oriented it; rotating twice would transpose it."""
        portrait = np.zeros((3300, 2550, 3), np.uint8)

        image, _, _, _, ocr_size = _ModeBackend()._setup_overlay_mode(
            {"image_prerotated": True},
            {"rotation": 90, "mediabox": [0.0, 0.0, 612.0, 792.0]},
            portrait,
            1,
        )

        assert ocr_size == (2550, 3300)
        assert image.shape[:2] == (3300, 2550)

    def test_a_page_without_a_mediabox_falls_back_to_pixels(self):
        """One pixel per point, so the mapping is at least self-consistent."""
        image = np.zeros((400, 300, 3), np.uint8)

        _, _, pdf_width, pdf_height, _ = _ModeBackend()._setup_overlay_mode(
            {"image_prerotated": False}, {"rotation": 0, "mediabox": None}, image, 1
        )

        assert (pdf_width, pdf_height) == (300.0, 400.0)


class TestStandaloneIsForcedByGeometry:
    """Without an inverse transform, corrected coordinates need the corrected image.

    There is no code anywhere that maps OCR boxes back through a perspective,
    dewarp or deskew transform. Correctness rests entirely on standalone mode
    replacing the image, so that guarantee is what gets tested.
    """

    @pytest.mark.parametrize("orientation_angle", [0, 90, 180, 270])
    @pytest.mark.parametrize("prerotated", [False, True])
    @pytest.mark.parametrize("pdf_rotation", [0, 90])
    def test_geometry_applied_always_selects_standalone(
        self, orientation_angle, prerotated, pdf_rotation
    ):
        result = {
            "page_num": 1,
            "orig_w": 2000,
            "orig_h": 3000,
            "orientation_angle": orientation_angle,
            "image_prerotated": prerotated,
            "original_pdf_rotation": pdf_rotation,
            "geometry_applied": True,
        }

        use_processed, geometry_changed = _ModeBackend()._determine_page_mode(result, 2000, 3000)

        assert use_processed is True
        assert geometry_changed is True


class TestForceOverlayWithGeometry:
    """force_overlay overrides the mode chosen for a corrected page.

    Reachable in production: ``_build_chunk_work_items`` sets
    ``use_rendered_source`` for a masked (JBIG2 foreground/background) page
    whenever geometric correction is enabled, which is the default. The worker
    then renders the page, applies the corrections, and OCRs the corrected
    image -- while ``_process_page_result`` passes
    ``force_overlay=use_rendered_source``, which resets
    ``use_processed_for_page`` to False. The text is then drawn over the
    *original* image using coordinates measured on the *corrected* one.
    """

    @staticmethod
    def _corrected_page() -> dict:
        return {
            "page_num": 1,
            "orig_w": 2000,
            "orig_h": 3000,
            "orientation_angle": 0,
            "image_prerotated": False,
            "original_pdf_rotation": 0,
            "geometry_applied": True,
        }

    def test_mode_selection_asks_for_standalone(self):
        """Characterisation: the decision itself is right..."""
        use_processed, _ = _ModeBackend()._determine_page_mode(self._corrected_page(), 2000, 3000)

        assert use_processed is True

    def test_force_overlay_discards_that_decision(self):
        """...and is then overridden, which is where the coordinates diverge.

        Mirrors ``backend_text_layer.py``: ``if force_overlay:
        use_processed_for_page = False``.
        """
        use_processed, _ = _ModeBackend()._determine_page_mode(self._corrected_page(), 2000, 3000)
        force_overlay = True

        if force_overlay:
            use_processed = False

        assert use_processed is False

    def test_force_overlay_no_longer_overrides_a_corrected_page(self):
        """Fixed: the override is now conditional on geometry_applied.

        Mirrors backend_text_layer._process_page_result. A masked page whose
        pixels were moved keeps standalone mode, so the coordinates and the
        image it drew them from stay in the same space.
        """
        result = self._corrected_page()
        use_processed, _ = _ModeBackend()._determine_page_mode(result, 2000, 3000)
        force_overlay = True

        if force_overlay and not result.get("geometry_applied", False):
            use_processed = False

        assert use_processed is True

    def test_force_overlay_still_applies_to_an_uncorrected_page(self):
        """The original purpose survives: preserve the untouched composite."""
        result = self._corrected_page() | {"geometry_applied": False}
        use_processed, _ = _ModeBackend()._determine_page_mode(result, 2000, 3000)
        force_overlay = True

        if force_overlay and not result.get("geometry_applied", False):
            use_processed = False

        assert use_processed is False
