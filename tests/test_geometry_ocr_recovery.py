"""End-to-end OCR quality: the tier that runs the real engine.

Everything else in the suite is deliberately engine-free -- the mapping from
OCR quadrilaterals to PDF points is provable with synthetic ones, and geometric
recovery is measurable on pixels. This file exists for the two questions those
cannot answer:

* Does a geometric correction actually buy readable text, or is it cost
  without benefit?
* Is the synthetic-quadrilateral shortcut used everywhere else sound?

Opt in with ``pytest -m real_ocr`` or ``tools/quality.sh``. Needs the PP-OCRv6
models on disk, and takes minutes.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from benchmarks.ocr_metrics import char_error_rate

from bigocrpdf.services.rapidocr_service.config import OCRConfig
from tests.geometry_fixtures import (
    DEFAULT_LINES,
    apply_rotation,
    render_synthetic_page,
)

pytestmark = pytest.mark.real_ocr

GROUND_TRUTH = "\n".join(DEFAULT_LINES)


def _models_available() -> bool:
    config = OCRConfig()
    return bool(config.get_det_model_path() and config.get_rec_model_path())


requires_models = pytest.mark.skipif(
    not _models_available(), reason="PP-OCRv6 models are not installed"
)


def _pdf_from_image(image: np.ndarray, path: Path) -> Path:
    """Wrap a rendered page as a one-page PDF, the way a scanner would."""
    from PIL import Image

    png = path.with_suffix(".png")
    cv2.imwrite(str(png), image)
    with Image.open(png) as opened:
        opened.convert("RGB").save(path, "PDF", resolution=200.0)
    return path


def _ocr_text(pdf: Path, out_dir: Path, **config_overrides) -> tuple[str, int]:
    """Run the real pipeline and return (extracted text, region count)."""
    from bigocrpdf.services.rapidocr_service.backend import ProfessionalPDFOCR

    stats = ProfessionalPDFOCR(OCRConfig(**config_overrides)).process(
        pdf, out_dir / f"{pdf.stem}-ocr.pdf"
    )
    return stats.full_text, stats.total_text_regions


@requires_models
class TestCorrectionsBuyAccuracy:
    """A correction must earn its cost in readable characters."""

    @pytest.mark.parametrize("degrees", [4.0, 8.0])
    def test_a_tilted_page_reads_better_with_corrections_on(self, tmp_path: Path, degrees: float):
        tilted = apply_rotation(render_synthetic_page(), degrees)
        pdf = _pdf_from_image(tilted, tmp_path / "tilted.pdf")

        text_on, _ = _ocr_text(pdf, tmp_path)
        text_off, _ = _ocr_text(
            pdf,
            tmp_path / "off",
            enable_deskew=False,
            enable_perspective_correction=False,
            enable_baseline_dewarp=False,
        )

        cer_on = char_error_rate(text_on, GROUND_TRUTH)
        cer_off = char_error_rate(text_off, GROUND_TRUTH)
        assert cer_on <= 0.10, f"corrected page still reads poorly: CER={cer_on:.3f}"
        assert cer_on <= cer_off, f"corrections made it worse: on={cer_on:.3f} off={cer_off:.3f}"

    def test_a_clean_page_is_not_degraded_by_corrections(self, tmp_path: Path):
        """The guard against over-correction, measured in characters.

        Four of the correction paths had no quality validation until recently;
        this is the end-to-end form of that check.
        """
        pdf = _pdf_from_image(render_synthetic_page(), tmp_path / "clean.pdf")

        text_on, _ = _ocr_text(pdf, tmp_path)
        text_off, _ = _ocr_text(
            pdf,
            tmp_path / "off",
            enable_deskew=False,
            enable_perspective_correction=False,
            enable_baseline_dewarp=False,
        )

        cer_on = char_error_rate(text_on, GROUND_TRUTH)
        cer_off = char_error_rate(text_off, GROUND_TRUTH)
        assert cer_on <= cer_off + 0.01


@requires_models
class TestOversizedPageStillProducesText:
    """The regression that started all of this.

    A phone photo converted to PDF has a page box of one point per source
    pixel, which implies 85 MP at 300 DPI. A pre-flight guard rejected the
    whole document before any OCR ran, and no test noticed because none
    measured "did anything come out at all".
    """

    def test_a_photo_sized_page_box_yields_text(self, tmp_path: Path):
        from PIL import Image

        page = render_synthetic_page()
        png = tmp_path / "photo.png"
        cv2.imwrite(str(png), page)
        pdf = tmp_path / "photo.pdf"
        with Image.open(png) as opened:
            # 72 dpi: one point per pixel, exactly what ImageMagick emits.
            opened.convert("RGB").save(pdf, "PDF", resolution=72.0)

        text, regions = _ocr_text(pdf, tmp_path)

        assert regions > 0, "the page produced no OCR regions at all"
        assert char_error_rate(text, GROUND_TRUTH) <= 0.15


@requires_models
class TestSyntheticQuadAssumption:
    """Justifies the shortcut the rest of the suite relies on.

    tests/positional_oracle feeds the renderer quadrilaterals it constructs
    itself. That is only sound if the real engine emits boxes of the same
    shape: one axis-aligned-ish quad per text line, in reading order.
    """

    def test_the_engine_emits_one_quad_per_line_in_reading_order(self, tmp_path: Path):
        from bigocrpdf.services.rapidocr_service.backend import ProfessionalPDFOCR

        pdf = _pdf_from_image(render_synthetic_page(), tmp_path / "clean.pdf")

        stats = ProfessionalPDFOCR(OCRConfig()).process(pdf, tmp_path / "out.pdf")
        page = stats.ocr_document.pages[0]

        assert page.text_results, "no regions returned"
        for result in page.text_results:
            assert len(result.box) == 4, "a region was not a quadrilateral"
        tops = [min(point[1] for point in r.box) for r in page.text_results]
        assert tops == sorted(tops), "regions are not in top-to-bottom order"

    def test_regions_are_close_to_axis_aligned(self, tmp_path: Path):
        """The metric-box fixtures assume so; a heavy shear would invalidate them."""
        from bigocrpdf.services.rapidocr_service.backend import ProfessionalPDFOCR

        pdf = _pdf_from_image(render_synthetic_page(), tmp_path / "clean.pdf")

        stats = ProfessionalPDFOCR(OCRConfig()).process(pdf, tmp_path / "out.pdf")

        for result in stats.ocr_document.pages[0].text_results:
            top_left, top_right = result.box[0], result.box[1]
            width = abs(top_right[0] - top_left[0])
            if width < 10:
                continue
            shear = abs(top_right[1] - top_left[1]) / width
            assert shear < 0.15, f"region is sheared by {shear:.2f} of its width"
