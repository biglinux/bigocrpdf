"""Geometric correction must undo a known distortion, and never invent one.

Four whole modules of correction code -- dewarp_probmap, contour_dewarp,
contour_spans, perspective_skew -- had no test at all. The approach here is to
apply a distortion whose parameters the test already knows and measure how much
of it comes back, using metrics that survive the corrector changing the image
size (see tests/geometry_metrics).

Thresholds are measured, not guessed: every number below came from running the
real pipeline over these fixtures.
"""

import cv2
import numpy as np
import pytest

from bigocrpdf.services.rapidocr_service.config import OCRConfig
from bigocrpdf.services.rapidocr_service.preprocessor import ImagePreprocessor
from tests.geometry_fixtures import (
    apply_cylindrical_warp,
    apply_rotation,
    render_synthetic_page,
    render_wide_text_page,
)
from tests.geometry_metrics import (
    baseline_curvature_px,
    registration_residual,
    ssim,
    text_block_taper,
    text_pixel_retention,
)

pytestmark = pytest.mark.slow

# Deskew clamps its correction to +/-5 degrees (preprocess_deskew), so a page
# tilted further comes out partly tilted. Measured: 8 deg leaves 3.01, 12 leaves
# 7.01.
DESKEW_CLAMP_DEG = 5.0


@pytest.fixture(scope="module")
def clean_page() -> np.ndarray:
    return render_synthetic_page()


def _process(image: np.ndarray, **config_overrides):
    """Run the real preprocessing pipeline, appearance effects off.

    The scanner effect is disabled throughout: it thickens strokes, which is
    good for OCR and pure noise for a geometric measurement.
    """
    preprocessor = ImagePreprocessor(OCRConfig(enable_scanner_effect=False, **config_overrides))
    return preprocessor.process(image), preprocessor


def _photographed(page: np.ndarray, keystone_frac: float = 0.0) -> np.ndarray:
    """A page lying on a dark surface, optionally shot at an angle.

    The perspective cascade keys on dark page borders, so text floating on
    white gives it nothing to detect -- correctly, since there is no page edge
    to find. This is the fixture that exercises it, and the one that matches
    the real phone photo the pipeline failed on.
    """
    height, width = page.shape[:2]
    canvas = np.full((int(height * 1.25), int(width * 1.25), 3), 18, np.uint8)
    offset_y = (canvas.shape[0] - height) // 2
    offset_x = (canvas.shape[1] - width) // 2
    canvas[offset_y : offset_y + height, offset_x : offset_x + width] = page
    if keystone_frac <= 0:
        return canvas

    canvas_h, canvas_w = canvas.shape[:2]
    shift = keystone_frac * canvas_w
    source = np.float32([[0, 0], [canvas_w, 0], [canvas_w, canvas_h], [0, canvas_h]])
    destination = np.float32(
        [[shift, 0], [canvas_w - shift, 0], [canvas_w, canvas_h], [0, canvas_h]]
    )
    return cv2.warpPerspective(
        canvas,
        cv2.getPerspectiveTransform(source, destination),
        (canvas_w, canvas_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(18, 18, 18),
    )


class TestIdentity:
    """A clean page must come out untouched. This is the regression guard.

    Photo perspective, margin normalisation, the contour four-point transform
    and the gentle margin path are all applied without any quality validation,
    so "correcting" an already-correct page is a real failure mode.
    """

    def test_nothing_is_applied_to_a_clean_page(self, clean_page):
        _, preprocessor = _process(clean_page)

        assert preprocessor.trace.applied_steps == []
        assert preprocessor.geometry_applied is False

    def test_an_untouched_page_is_returned_unchanged(self, clean_page):
        """If no step fired, no resampling may have happened either."""
        processed, preprocessor = _process(clean_page)

        assert preprocessor.trace.applied_steps == []
        assert np.array_equal(processed, clean_page)

    def test_a_clean_page_keeps_its_geometry(self, clean_page):
        processed, _ = _process(clean_page)

        assert baseline_curvature_px(processed) == pytest.approx(0.0, abs=1.0)
        assert text_block_taper(processed) == pytest.approx(1.0, abs=0.05)
        assert ssim(processed, clean_page) > 0.97
        assert text_pixel_retention(processed, clean_page) >= 0.98


class TestRotationRecovery:
    """Assertions read the angle the pipeline applied, from its own trace.

    Image registration is used only as a cross-check: on a page rotated by -5
    degrees ECC converges on a solution carrying a 12% scale factor, which
    inflates its rotation estimate. The trace reports the correction directly,
    so it is both exact and the thing a regression would actually change.
    """

    @pytest.mark.parametrize("degrees", [2.0, -2.0, 5.0, -5.0])
    def test_tilt_within_the_clamp_is_removed(self, clean_page, degrees):
        processed, preprocessor = _process(apply_rotation(clean_page, degrees))

        step = preprocessor.trace.step_named("deskew")
        assert step.applied is True
        # The correction opposes the applied tilt. Worst measured error is
        # 0.67 deg, at -5; the other three land within 0.11.
        assert step.params["applied_angle"] == pytest.approx(-degrees, abs=0.8)
        assert text_pixel_retention(processed, clean_page) >= 0.9

    @pytest.mark.parametrize("degrees", [8.0, 12.0])
    def test_tilt_beyond_the_clamp_is_only_partly_removed(self, clean_page, degrees):
        """Documents a real limitation rather than pretending it is absent.

        ``_apply_probmap_uniform_rotation`` clips the correction to +/-5
        degrees, so a page tilted further stays tilted by the remainder. Raising
        the clamp is a legitimate change; doing it silently is not, and this is
        what would notice.
        """
        _, preprocessor = _process(apply_rotation(clean_page, degrees))

        applied = abs(preprocessor.trace.step_named("deskew").params["applied_angle"])
        assert applied == pytest.approx(DESKEW_CLAMP_DEG, abs=0.01)

    def test_registration_agrees_for_a_well_behaved_tilt(self, clean_page):
        """One end-to-end cross-check that the trace is not self-congratulatory."""
        processed, _ = _process(apply_rotation(clean_page, 2.0))

        residual = registration_residual(processed, clean_page)
        assert residual.converged
        assert abs(residual.residual_rotation_deg) <= 0.3


class TestCurvatureRecovery:
    """What the dewarp does to a bowed page, measured rather than assumed.

    This class used to assert only that curvature fell, and passed -- but on a
    page whose text sits in a column the flattening came from *deskew*, and the
    dewarp never fired at all. Each assertion below now names the step it is
    about, and the wide fixture is what lets the dewarp be tested at all.
    """

    @pytest.fixture
    def wide_page(self):
        return render_wide_text_page()

    @pytest.mark.parametrize("curvature", [25.0, 50.0])
    def test_a_bowed_page_is_flattened_by_the_dewarp(self, wide_page, curvature):
        warped = apply_cylindrical_warp(wide_page, curvature)
        before = baseline_curvature_px(warped)

        processed, preprocessor = _process(warped)

        assert before > 2.0, "fixture did not produce measurable curvature"
        assert "dewarp" in preprocessor.trace.applied_steps
        assert baseline_curvature_px(processed) <= before * 0.75
        assert text_pixel_retention(processed, wide_page) >= 0.9

    @pytest.mark.parametrize("curvature", [25.0, 50.0])
    def test_a_bowed_column_reads_as_tilt_and_goes_to_the_deskew(self, clean_page, curvature):
        """Not a defect: over a narrow span an arc *is* a tilt.

        The default page sets its text in a column covering about 62% of the
        width. Measured there, a page-centred bow arrives at each baseline as
        36px of slope and 10px of residual bow, because the text only sees one
        flank of the arc. Removing the line's linear trend is what separates
        the two, and rotation is the deskew's job -- so the dewarp declining
        here is the design working, not failing.
        """
        warped = apply_cylindrical_warp(clean_page, curvature)

        _processed, preprocessor = _process(warped)

        assert "dewarp" not in preprocessor.trace.applied_steps
        assert "deskew" in preprocessor.trace.applied_steps

    def test_curvature_below_the_gate_is_left_alone(self, wide_page):
        """Asserts the gate, not the correction.

        ``dewarp_probmap._MIN_CURVATURE_PX`` deliberately skips a remap that
        would cost a resample for no benefit. Measured on full-width text, so
        the gate is the only reason nothing fires.
        """
        warped = apply_cylindrical_warp(wide_page, 3.0)

        processed, preprocessor = _process(warped)

        assert "dewarp" not in preprocessor.trace.applied_steps
        assert ssim(processed, warped) > 0.99


class TestPhotoPerspectiveRecovery:
    def test_a_flat_photo_is_cropped_back_to_the_page(self, clean_page):
        photo = _photographed(clean_page)

        processed, preprocessor = _process(photo)

        assert "perspective" in preprocessor.trace.applied_steps
        assert preprocessor.trace.step_named("perspective").method == "photo_borders"
        height_ratio = processed.shape[0] / clean_page.shape[0]
        width_ratio = processed.shape[1] / clean_page.shape[1]
        assert height_ratio == pytest.approx(1.0, abs=0.05)
        assert width_ratio == pytest.approx(1.0, abs=0.05)

    @pytest.mark.parametrize("keystone", [0.04, 0.08, 0.14])
    def test_a_keystoned_photo_is_squared_up(self, clean_page, keystone):
        processed, preprocessor = _process(_photographed(clean_page, keystone))

        assert preprocessor.trace.step_named("perspective").applied is True
        assert processed.shape[0] / clean_page.shape[0] == pytest.approx(1.0, abs=0.08)
        assert baseline_curvature_px(processed) <= 2.0
        assert text_pixel_retention(processed, clean_page) >= 0.9

    def test_the_dark_surround_is_gone(self, clean_page):
        """The whole point of the correction: OCR must not see the desk."""
        processed, _ = _process(_photographed(clean_page, 0.08))

        border = np.concatenate(
            [
                processed[:8, :].reshape(-1, 3),
                processed[-8:, :].reshape(-1, 3),
                processed[:, :8].reshape(-1, 3),
                processed[:, -8:].reshape(-1, 3),
            ]
        )
        assert float(border.mean()) > 128.0


class TestSsimAgreesWithOpenCVContrib:
    def test_hand_rolled_ssim_matches_the_contrib_module(self, clean_page):
        """Cross-check, where opencv-contrib happens to be installed.

        The project depends on plain opencv-python, so the implementation in
        geometry_metrics stays; this only confirms it is the same measure.
        """
        quality = pytest.importorskip("cv2.quality", reason="opencv-contrib not installed")

        blurred = cv2.GaussianBlur(clean_page, (7, 7), 2.0)
        mine = ssim(blurred, clean_page)
        theirs = float(
            np.mean(
                quality.QualitySSIM_compute(
                    cv2.cvtColor(clean_page, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY),
                )[0][:1]
            )
        )

        assert mine == pytest.approx(theirs, abs=0.02)
