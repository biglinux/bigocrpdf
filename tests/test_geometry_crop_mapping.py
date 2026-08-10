"""Mapping OCR boxes from a cropped image back onto the original.

``_map_crop_only_ocr_to_original_image`` is the only place in the codebase that
remaps preprocessed coordinates to the original image space, and until now only
the two helpers it calls were covered -- not the caller, and not the bail-out
that keeps it away from geometry it cannot invert.
"""

import numpy as np
import pytest

from bigocrpdf.services.rapidocr_service.config import OCRConfig
from bigocrpdf.services.rapidocr_service.ocr_postprocess import apply_ocr_box_offset
from bigocrpdf.services.rapidocr_service.pipeline_chunked_ocr import ChunkedOCRMixin
from bigocrpdf.services.rapidocr_service.preprocessor import ImagePreprocessor

remap = ChunkedOCRMixin._map_crop_only_ocr_to_original_image


def _result(**overrides) -> dict:
    base = {
        "crop_applied": True,
        "geometry_applied": False,
        "crop_offset_px": (10, 20),
        "crop_original_size_px": (100, 200),
        "ocr_raw": {
            "boxes": [[[1, 2], [3, 2], [3, 4], [1, 4]]],
            "txts": ["abc"],
            "scores": [0.9],
        },
        "ocr_img_w": 80,
        "ocr_img_h": 160,
    }
    base.update(overrides)
    return base


def test_trim_dark_borders_records_crop_offset():
    config = OCRConfig(
        enable_baseline_dewarp=False, enable_perspective_correction=False, enable_deskew=False
    )
    preprocessor = ImagePreprocessor(config)
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[:, :8] = 0
    image[:8, :] = 0

    processed = preprocessor.process(image)

    assert processed.shape[0] < image.shape[0]
    assert processed.shape[1] < image.shape[1]
    assert preprocessor.crop_applied is True
    assert preprocessor.crop_offset_px[0] > 0
    assert preprocessor.crop_offset_px[1] > 0
    assert preprocessor.crop_original_size_px == (100, 100)


def test_apply_ocr_box_offset_shifts_boxes_without_mutating_input():
    raw = {
        "boxes": [
            [[1, 2], [3, 2], [3, 4], [1, 4]],
        ],
        "txts": ["abc"],
        "scores": [0.9],
    }

    shifted = apply_ocr_box_offset(raw, (10, 20))

    assert shifted is not None
    assert shifted["boxes"][0][0] == [11, 22]
    assert raw["boxes"][0][0] == [1, 2]


class TestCropOnlyRemap:
    def test_boxes_move_by_the_crop_offset(self):
        result = _result()

        remap(result)

        assert result["ocr_raw"]["boxes"][0][0] == [11, 22]

    def test_the_original_image_size_is_restored(self):
        """Downstream placement scales by this, so the crop must be undone."""
        result = _result()

        remap(result)

        assert (result["ocr_img_w"], result["ocr_img_h"]) == (100, 200)

    def test_a_geometrically_corrected_page_is_left_alone(self):
        """A warp cannot be undone by an offset, so the remap must stand down.

        Standalone mode already guarantees the corrected image is the one drawn;
        shifting the boxes on top of that would displace them twice.
        """
        result = _result(geometry_applied=True)
        original_boxes = [list(map(list, box)) for box in result["ocr_raw"]["boxes"]]

        remap(result)

        assert result["ocr_raw"]["boxes"] == original_boxes
        assert (result["ocr_img_w"], result["ocr_img_h"]) == (80, 160)

    def test_a_page_without_a_crop_is_left_alone(self):
        result = _result(crop_applied=False)
        original_boxes = [list(map(list, box)) for box in result["ocr_raw"]["boxes"]]

        remap(result)

        assert result["ocr_raw"]["boxes"] == original_boxes

    @pytest.mark.parametrize(
        "offset",
        [
            pytest.param(None, id="missing"),
            pytest.param((5,), id="too-short"),
            pytest.param("10,20", id="string"),
            pytest.param([1, 2, 3], id="too-long"),
        ],
    )
    def test_a_malformed_offset_falls_back_to_no_shift(self, offset):
        """Bad bookkeeping must not move text; it must leave it where it was."""
        result = _result(crop_offset_px=offset)

        remap(result)

        assert result["ocr_raw"]["boxes"][0][0] == [1, 2]

    def test_a_missing_original_size_keeps_the_cropped_size(self):
        result = _result(crop_original_size_px=None)

        remap(result)

        assert (result["ocr_img_w"], result["ocr_img_h"]) == (80, 160)
