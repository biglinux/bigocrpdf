"""Cascade order, and the paths where a correction declines to run.

The preprocessor picks between several dewarp and perspective strategies in a
first-hit-wins cascade whose branches catch their own exceptions. The trace
tells a correction that found nothing apart from one whose detector raised.

Detectors are mocked here, never the arithmetic: the question is which branch
runs and what it records, not whether a warp is correct. That is
test_geometry_recovery's job.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from bigocrpdf.services import perspective_correction as perspective_module
from bigocrpdf.services.ocr import preprocessor as preprocessor_module
from bigocrpdf.services.ocr.config import OCRConfig
from bigocrpdf.services.ocr.dewarp_probmap import Baselines
from bigocrpdf.services.ocr.geometry_trace import REASON_DISABLED
from bigocrpdf.services.ocr.ppocr_native import Detector
from bigocrpdf.services.ocr.preprocessor import ImagePreprocessor


@pytest.fixture
def page() -> np.ndarray:
    image = np.full((400, 300, 3), 255, np.uint8)
    image[100:130, 40:260] = 0
    image[200:230, 40:260] = 0
    return image


def _preprocessor(**overrides) -> ImagePreprocessor:
    return ImagePreprocessor(OCRConfig(enable_scanner_effect=False, **overrides), Detector())


class TestStageOrder:
    def test_geometric_stages_run_in_the_documented_order(self, page, monkeypatch):
        """Order is load-bearing: a curved page confuses deskew, so dewarp is
        first, and border trimming must follow the warps that create borders."""
        calls: list[str] = []
        preprocessor = _preprocessor()

        monkeypatch.setattr(
            preprocessor,
            "_try_probmap_dewarp",
            lambda img, step: (calls.append("dewarp"), (img, Baselines(img, 0, [], 1.0, 1.0)))[1],
        )
        monkeypatch.setattr(
            preprocessor,
            "_correct_perspective",
            lambda img, step: (calls.append("perspective"), img)[1],
        )
        monkeypatch.setattr(
            preprocessor, "_trim_dark_borders", lambda img: (calls.append("trim"), img)[1]
        )
        monkeypatch.setattr(
            preprocessor_module,
            "probmap_angle_deskew",
            lambda _detector, img, max_side, trace=None, known=None: (
                calls.append("deskew"),
                img,
            )[1],
        )

        preprocessor.process(page)

        assert calls == ["dewarp", "perspective", "trim", "deskew"]

    def test_every_stage_appears_in_the_trace(self, page):
        preprocessor = _preprocessor()

        preprocessor.process(page)

        assert [step.step for step in preprocessor.trace.steps] == [
            "dewarp",
            "perspective",
            "trim_dark_borders",
            "deskew",
        ]

    def test_the_trace_is_reset_between_pages(self, page):
        preprocessor = _preprocessor()

        preprocessor.process(page)
        preprocessor.process(page)

        assert len(preprocessor.trace.steps) == 4


class TestDisabledStages:
    @pytest.mark.parametrize(
        "flag,step",
        [
            ("enable_baseline_dewarp", "dewarp"),
            ("enable_perspective_correction", "perspective"),
            ("enable_deskew", "deskew"),
        ],
    )
    def test_a_disabled_stage_says_so(self, page, flag, step):
        """Distinguishes "switched off" from "ran and found nothing"."""
        preprocessor = _preprocessor(**{flag: False})

        preprocessor.process(page)

        assert preprocessor.trace.step_named(step).reason == REASON_DISABLED
        assert preprocessor.trace.step_named(step).applied is False

    def test_border_trimming_has_no_switch(self, page):
        """It runs on every page."""
        preprocessor = _preprocessor(
            enable_baseline_dewarp=False,
            enable_perspective_correction=False,
            enable_deskew=False,
        )

        preprocessor.process(page)

        assert preprocessor.trace.step_named("trim_dark_borders").reason != REASON_DISABLED


class TestDewarpFallback:
    def test_a_failing_probmap_records_the_exception_and_falls_back(self, page, monkeypatch):
        """A failing probability map is recorded and the fallback runs."""
        preprocessor = _preprocessor()
        fallback_ran = []

        def explode(_detector, image, max_side):
            raise RuntimeError("probmap exploded")

        monkeypatch.setattr("bigocrpdf.services.ocr.dewarp_probmap.probmap_dewarp", explode)
        monkeypatch.setattr(
            preprocessor,
            "_try_3d_dewarp",
            lambda img, step: (fallback_ran.append(True), img)[1],
        )

        preprocessor.process(page)

        assert fallback_ran == [True]
        assert preprocessor.trace.step_named("dewarp").reason == "exception:RuntimeError"

    @pytest.mark.parametrize("enable_baseline_dewarp", [True, False])
    def test_cancelled_detection_stops_the_page(self, page, monkeypatch, enable_baseline_dewarp):
        preprocessor = _preprocessor(enable_baseline_dewarp=enable_baseline_dewarp)
        fallback_ran = []

        def cancelled(*_args):
            raise InterruptedError("ppocr-native request cancelled")

        preprocessor.detector = SimpleNamespace(probability_map=cancelled, boxes=cancelled)
        monkeypatch.setattr(
            preprocessor,
            "_try_3d_dewarp",
            lambda img, step: (fallback_ran.append(True), img)[1],
        )

        with pytest.raises(InterruptedError):
            preprocessor.process(page)
        assert fallback_ran == []

    def test_a_successful_probmap_never_reaches_the_fallback(self, page, monkeypatch):
        """A probmap that succeeds owns the page, even when its result is poor."""
        preprocessor = _preprocessor()
        fallback_ran = []

        monkeypatch.setattr(
            "bigocrpdf.services.ocr.dewarp_probmap.probmap_dewarp",
            lambda _detector, image, max_side: (image, Baselines(image, max_side, [], 1.0, 1.0)),
        )
        monkeypatch.setattr(
            preprocessor,
            "_try_3d_dewarp",
            lambda img, step: (fallback_ran.append(True), img)[1],
        )

        preprocessor.process(page)

        assert fallback_ran == []
        assert preprocessor.trace.step_named("dewarp").method == "probmap"

    def test_a_flat_page_is_measured_once(self, page):
        """Deskew reuses the dewarp step's reading when the image is unchanged."""
        preprocessor = _measuring_preprocessor()

        preprocessor.process(page)

        assert len(preprocessor.detector.maps) == 1

    def test_a_trimmed_page_is_measured_once(self, page):
        """Trimming only translates the page, so its baselines still hold."""
        page[:20] = 0
        preprocessor = _measuring_preprocessor()

        preprocessor.process(page)

        assert preprocessor.trace.step_named("trim_dark_borders").applied is True
        assert len(preprocessor.detector.maps) == 1

    def test_a_warped_page_is_measured_again(self, page, monkeypatch):
        warped = np.full((320, 256, 3), 255, np.uint8)
        preprocessor = _measuring_preprocessor(enable_perspective_correction=True)
        monkeypatch.setattr(preprocessor, "_correct_perspective", lambda img, step: warped)

        preprocessor.process(page)

        assert preprocessor.detector.maps == [(384, 288, 3), (320, 256, 3)]

    def test_crop_keeps_baselines_on_text(self):
        image = np.zeros((400, 300, 3), np.uint8)
        line = (60.0, lambda x: 0.1 * x + 50.0, 10.0, 200.0)

        cropped = Baselines(image, 0, [line], 2.0, 2.0).crop(image[40:, 20:], 20, 40)

        y_centre, spline, x_start, x_end = cropped.lines[0]
        assert (y_centre, x_start, x_end) == (40.0, 0.0, 190.0)
        assert spline(0.0) == pytest.approx(31.0)
        assert spline(x_end) - spline(x_start) == pytest.approx(19.0)


def _measuring_preprocessor(enable_perspective_correction: bool = False) -> ImagePreprocessor:
    """A preprocessor whose detector records the shape of each image it maps."""
    detector = SimpleNamespace(maps=[], boxes=lambda *_args: [])

    def probability_map(image):
        detector.maps.append(image.shape)
        return np.zeros(image.shape[:2], np.float32)

    detector.probability_map = probability_map
    config = OCRConfig(
        enable_scanner_effect=False, enable_perspective_correction=enable_perspective_correction
    )
    return ImagePreprocessor(config, detector)


class TestPerspectiveCascade:
    """First hit wins, and every later detector must stay unconsulted."""

    DETECTORS = [
        ("detect_photo_document_borders", "photo_borders"),
        ("detect_perspective_distortion", "margins"),
    ]

    @pytest.mark.parametrize("winner,expected_method", DETECTORS)
    def test_the_first_detector_that_fires_owns_the_page(
        self, page, monkeypatch, winner, expected_method
    ):
        consulted: list[str] = []

        for name, _ in self.DETECTORS:
            monkeypatch.setattr(
                perspective_module,
                name,
                (lambda n: lambda image: (consulted.append(n), np.zeros((4, 2), np.float32))[1])(
                    name
                )
                if name == winner
                else (lambda n: lambda image: (consulted.append(n), None)[1])(name),
            )
        monkeypatch.setattr(
            perspective_module, "correct_photo_perspective", lambda image, corners: image.copy()
        )
        monkeypatch.setattr(
            perspective_module,
            "correct_perspective_from_margins",
            lambda image, distortion: image.copy(),
        )

        corrector = perspective_module.PerspectiveCorrector(skip_skew=True)
        corrector(page)

        assert corrector.last_method == expected_method
        index = [name for name, _ in self.DETECTORS].index(winner)
        assert consulted == [name for name, _ in self.DETECTORS][: index + 1]

    def test_nothing_detected_reports_none_and_lists_the_rejections(self, page, monkeypatch):
        for name, _ in self.DETECTORS:
            monkeypatch.setattr(perspective_module, name, lambda image: None)
        monkeypatch.setattr(
            perspective_module, "gentle_margin_perspective_correction", lambda image: None
        )

        corrector = perspective_module.PerspectiveCorrector(skip_skew=True)
        result = corrector(page)

        assert result is page
        assert corrector.last_method == "none"
        assert "photo_borders" in corrector.last_rejected
        assert "gentle_margin" in corrector.last_rejected

    def test_an_unusable_contour_falls_through_to_the_gentle_margin(self, page, monkeypatch):
        """An internal frame does not end the cascade.

        ``_try_contour_correction`` returns None for a contour that turns out
        to be an internal frame; returning the image would be terminal and
        skip the gentle-margin step.
        """
        for name, _ in self.DETECTORS:
            monkeypatch.setattr(perspective_module, name, lambda image: None)
        gentle_ran = []
        monkeypatch.setattr(
            perspective_module,
            "gentle_margin_perspective_correction",
            lambda image: (gentle_ran.append(True), None)[1],
        )
        corrector = perspective_module.PerspectiveCorrector(skip_skew=True)
        monkeypatch.setattr(corrector, "_try_contour_correction", lambda image: None)

        result = corrector(page)

        assert result is page
        assert gentle_ran == [True]
        assert "contour" in corrector.last_rejected

    def test_an_internal_frame_is_reported_as_a_rejection(self, page, monkeypatch):
        """The real function, not a stub: an internal frame must yield None."""
        import numpy as np

        # A contour inset well past _MAX_CONTOUR_INSET_RATIO on every side.
        inner = np.array([[80, 80], [220, 80], [220, 320], [80, 320]], dtype=np.float32)
        monkeypatch.setattr(perspective_module, "detect_document_contour", lambda image: inner)

        corrector = perspective_module.PerspectiveCorrector(skip_skew=True)

        assert corrector._try_contour_correction(page) is None

    def test_skew_steps_are_dead_when_the_preprocessor_deskews(self, page, monkeypatch):
        """Cascade priorities 4 and 5 never run in the default configuration.

        ``_correct_perspective`` passes ``skip_skew=config.enable_deskew``, and
        deskew is on by default, so they are conditionally dead rather than
        accidentally dead.
        """
        for name, _ in self.DETECTORS:
            monkeypatch.setattr(perspective_module, name, lambda image: None)
        monkeypatch.setattr(
            perspective_module, "gentle_margin_perspective_correction", lambda image: None
        )
        skew_ran = []
        monkeypatch.setattr(
            perspective_module,
            "detect_skew_angle",
            lambda image: (skew_ran.append(True), 0.0)[1],
        )

        perspective_module.PerspectiveCorrector(skip_skew=True)(page)
        assert skew_ran == []

        perspective_module.PerspectiveCorrector(skip_skew=False)(page)
        assert skew_ran == [True]


class TestDeskewDiagnostics:
    def test_high_angle_dispersion_is_reported_as_a_skip(self, page, monkeypatch):
        """MAD > 3 deg abandons the rotation; the reason must survive."""
        from bigocrpdf.services.ocr import preprocess_deskew

        monkeypatch.setattr(
            preprocess_deskew,
            "_probmap_measured_angles",
            lambda _detector, img, min_width, max_side, known=None: (
                [0.0, 12.0, -12.0, 9.0, -9.0, 11.0],
                [10.0, 60.0, 110.0, 160.0, 210.0, 260.0],
                [200.0] * 6,
            ),
        )
        preprocessor = _preprocessor()

        preprocessor.process(page)

        step = preprocessor.trace.step_named("deskew")
        assert step.method in {"mad_skip", "below_threshold"}
        assert step.applied is False

    def test_measured_angles_reach_the_trace(self, page, monkeypatch):
        from bigocrpdf.services.ocr import preprocess_deskew

        monkeypatch.setattr(
            preprocess_deskew,
            "_probmap_measured_angles",
            lambda _detector, img, min_width, max_side, known=None: (
                [2.0, 2.1, 1.9, 2.05],
                [10.0, 60.0, 110.0, 160.0],
                [200.0] * 4,
            ),
        )
        preprocessor = _preprocessor()

        preprocessor.process(page)

        params = preprocessor.trace.step_named("deskew").params
        assert params["n_baselines"] == 4
        assert params["angle_mean"] == pytest.approx(2.0, abs=0.2)


class TestBorderTrimming:
    def test_a_dark_frame_is_trimmed_and_recorded(self):
        image = np.full((200, 200, 3), 255, np.uint8)
        image[:12, :] = 0
        image[:, :12] = 0
        preprocessor = _preprocessor(
            enable_baseline_dewarp=False,
            enable_perspective_correction=False,
            enable_deskew=False,
        )

        processed = preprocessor.process(image)

        step = preprocessor.trace.step_named("trim_dark_borders")
        assert step.applied is True
        assert processed.shape[0] < 200
        assert step.params["offset_y"] > 0

    def test_a_uniformly_dark_page_is_not_trimmed_at_all(self):
        """A dark photograph is not a dark border.

        Trimming is the one geometric step with no on/off switch, so its own
        guard is what protects an underexposed photo, a dark-mode screenshot or
        a scan of black card from losing 5% of every side for nothing.
        """
        image = np.full((200, 200, 3), 30, np.uint8)
        preprocessor = _preprocessor(
            enable_baseline_dewarp=False,
            enable_perspective_correction=False,
            enable_deskew=False,
        )

        processed = preprocessor.process(image)

        assert processed.shape[:2] == (200, 200)
        assert preprocessor.trace.step_named("trim_dark_borders").applied is False

    def test_a_dark_border_on_a_bright_page_is_still_trimmed(self):
        """The guard is about contrast with the page, not absolute darkness."""
        image = np.full((200, 200, 3), 255, np.uint8)
        image[:14, :] = 10
        preprocessor = _preprocessor(
            enable_baseline_dewarp=False,
            enable_perspective_correction=False,
            enable_deskew=False,
        )

        processed = preprocessor.process(image)

        assert processed.shape[0] < 200
        assert preprocessor.trace.step_named("trim_dark_borders").applied is True
