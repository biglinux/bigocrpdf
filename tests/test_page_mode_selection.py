"""Standalone vs overlay: which image a page is built from.

This decides whether the processed image replaces the original. Get it wrong
and OCR coordinates measured on a corrected image are drawn over an
uncorrected one, displacing every word on the page.

Replaces ``TestGeometryChangeThreshold`` in tests/test_rotation.py, which
reimplemented the formula inside the test and so protected nothing -- it had
already drifted from production, missing the ``geometry_applied`` branch
entirely and unable to express the two-value return at all.

Rule for this file: never reimplement a production formula. Call it.
"""

from types import SimpleNamespace

import pytest

from bigocrpdf.services.rapidocr_service.backend_text_layer import BackendTextLayerMixin


class _ModeBackend(BackendTextLayerMixin):
    """Minimal host for the mixin; the pattern is used by other tests too."""

    def __init__(self, export_format: str = "original") -> None:
        self.config = SimpleNamespace(image_export_format=export_format)


def determine_mode(result: dict, proc_w: int, proc_h: int, export_format: str = "original"):
    return _ModeBackend(export_format)._determine_page_mode(result, proc_w, proc_h)


def _page(**overrides) -> dict:
    base = {
        "page_num": 1,
        "orig_w": 2814,
        "orig_h": 3864,
        "orientation_angle": 0,
        "image_prerotated": False,
        "original_pdf_rotation": 0,
        "geometry_applied": False,
    }
    base.update(overrides)
    return base


class TestDimensionThreshold:
    def test_deskew_adding_small_borders_is_not_significant(self):
        """Real numbers from contrato.pdf: 2814x3864 -> 2939x3954, ~3%."""
        use_processed, geometry_changed = determine_mode(_page(), 2939, 3954)

        assert (use_processed, geometry_changed) == (False, False)

    def test_perspective_correction_is_significant(self):
        """Real numbers from the prescription photo: 1920x2560 -> 1495x2114, ~19%."""
        result = _page(orig_w=1920, orig_h=2560)

        use_processed, geometry_changed = determine_mode(result, 1495, 2114)

        assert (use_processed, geometry_changed) == (True, True)

    def test_identical_dimensions_are_not_significant(self):
        assert determine_mode(_page(), 2814, 3864) == (False, False)

    def test_threshold_is_strictly_greater_than_five_percent(self):
        """Exactly 5% must not trip it; just above must."""
        orig_w, orig_h = 1000, 1000
        at_threshold = int(orig_w + 0.05 * (orig_w + orig_h))
        result = _page(orig_w=orig_w, orig_h=orig_h)

        assert determine_mode(result, at_threshold, orig_h)[1] is False
        assert determine_mode(result, at_threshold + 1, orig_h)[1] is True

    def test_zero_sized_page_does_not_divide_by_zero(self):
        assert determine_mode(_page(orig_w=0, orig_h=0), 0, 0) == (False, False)


class TestCoordinateSpaceChanges:
    def test_geometry_applied_forces_standalone_without_a_size_change(self):
        """The branch the old reimplemented test did not know about.

        A perspective warp can preserve dimensions while moving every pixel,
        so the flag alone must force standalone mode.
        """
        result = _page(geometry_applied=True)

        assert determine_mode(result, 2814, 3864) == (True, True)

    def test_orientation_angle_forces_standalone(self):
        assert determine_mode(_page(orientation_angle=90), 2814, 3864) == (True, True)

    def test_prerotated_image_on_a_rotated_page_forces_standalone(self):
        result = _page(image_prerotated=True, original_pdf_rotation=90)

        assert determine_mode(result, 2814, 3864) == (True, True)

    def test_prerotated_image_on_an_unrotated_page_does_not(self):
        result = _page(image_prerotated=True, original_pdf_rotation=0)

        assert determine_mode(result, 2814, 3864) == (False, False)


class TestExportFormat:
    def test_a_requested_format_replaces_the_image_without_geometry_change(self):
        """The two return values differ here, which the old test could not express."""
        use_processed, geometry_changed = determine_mode(_page(), 2814, 3864, export_format="jpeg")

        assert use_processed is True
        assert geometry_changed is False

    @pytest.mark.parametrize("export_format", ["original", ""])
    def test_the_default_formats_do_not_replace_the_image(self, export_format):
        assert determine_mode(_page(), 2814, 3864, export_format=export_format) == (False, False)


class TestDiagnosticLogging:
    def test_the_geometry_change_line_is_logged_only_when_it_changed(self, caplog):
        """That INFO line is the field-diagnosis tool for displaced text.

        The level is set on ``BigOcrPdf`` by name: another test in the suite
        raises that logger's own level, which a root-only ``at_level`` would
        not override.
        """
        import logging

        caplog.set_level(logging.INFO, logger="BigOcrPdf")

        determine_mode(_page(geometry_applied=True), 2814, 3864)
        assert "geometry/coordinate change" in caplog.text

        caplog.clear()
        determine_mode(_page(), 2814, 3864)
        assert "geometry/coordinate change" not in caplog.text
