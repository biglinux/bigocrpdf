from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from bigocrpdf.services.rapidocr_service import ocr_postprocess
from bigocrpdf.services.rapidocr_service.ocr_postprocess import (
    choose_better_ocr_result,
    ocr_confidence_mean,
    should_retry_page_ocr,
)
from bigocrpdf.services.rapidocr_service.pipeline_chunked_ocr import ChunkedOCRMixin


def test_retry_decision_retries_missing_boxes():
    decision = should_retry_page_ocr({"boxes": None, "scores": []})

    assert decision.should_retry is True
    assert decision.reason == "no_boxes"


def test_retry_decision_retries_low_confidence_page():
    decision = should_retry_page_ocr({"boxes": [[[0, 0]]], "scores": [0.4]})

    assert decision.should_retry is True
    assert decision.reason == "low_confidence"


def test_retry_decision_accepts_confident_page():
    decision = should_retry_page_ocr({"boxes": [[[0, 0]]], "scores": [0.9]})

    assert decision.should_retry is False


def test_retry_decision_skips_pages_without_image_content():
    decision = should_retry_page_ocr(None, page_has_image_content=False)

    assert decision.should_retry is False


def test_choose_retry_result_with_more_text():
    original = {"boxes": [1], "txts": ["abc"], "scores": [0.9]}
    retry = {"boxes": [1], "txts": ["abcdef"], "scores": [0.7]}

    assert choose_better_ocr_result(original, retry) is retry


def test_choose_retry_result_with_same_text_and_better_confidence():
    original = {"boxes": [1], "txts": ["abc"], "scores": [0.6]}
    retry = {"boxes": [1], "txts": ["abc"], "scores": [0.8]}

    assert choose_better_ocr_result(original, retry) is retry
    assert ocr_confidence_mean(retry) == 0.8


def test_rendered_fallback_is_removed_when_ocr_is_cancelled(tmp_path: Path):
    rendered = tmp_path / "rendered.ppm"
    rendered.write_bytes(b"fallback")
    pipeline = object.__new__(ChunkedOCRMixin)
    pipeline._chunk_ocr_path = lambda *_args: (str(rendered), str(rendered))
    pipeline._ocr_subprocess = SimpleNamespace(
        recognize=lambda *_args: (_ for _ in ()).throw(InterruptedError("cancelled"))
    )
    result = {"page_num": 1, "success": True, "temp_out_path": "page.png"}

    with pytest.raises(InterruptedError, match="cancelled"):
        pipeline._ocr_chunk_result(result, object(), [], object(), input_pdf=tmp_path / "in.pdf")

    assert not rendered.exists()


def test_region_refinement_skips_empty_crop() -> None:
    ocr_fn = MagicMock()
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    outside_box = [[100, 100], [110, 100], [110, 110], [100, 110]]

    result = ocr_postprocess._reocr_region(image, outside_box, "text", 0.2, ocr_fn)

    assert result is None
    ocr_fn.assert_not_called()


def test_region_refinement_skips_failed_crop_write(monkeypatch: pytest.MonkeyPatch) -> None:
    ocr_fn = MagicMock()
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    box = [[5, 5], [40, 5], [40, 40], [5, 40]]
    monkeypatch.setattr(ocr_postprocess.cv2, "imwrite", lambda *_args, **_kwargs: False)

    result = ocr_postprocess._reocr_region(image, box, "text", 0.2, ocr_fn)

    assert result is None
    ocr_fn.assert_not_called()
