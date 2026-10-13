from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from bigocrpdf.services.ocr import ocr_postprocess
from bigocrpdf.services.ocr.backend import ProfessionalPDFOCR
from bigocrpdf.services.ocr.config import OCRConfig, ProcessingStats
from bigocrpdf.services.ocr.ocr_postprocess import (
    choose_better_ocr_result,
    ocr_confidence_mean,
    page_retry_reason,
)
from bigocrpdf.services.ocr.pipeline_chunked_ocr import ChunkedOCRMixin
from bigocrpdf.services.ocr.ppocr_native import ReadError


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        (None, "no_boxes"),
        ({"boxes": None, "scores": []}, "no_boxes"),
        ({"boxes": [[[0, 0]]], "scores": [0.4]}, "low_confidence"),
        ({"boxes": [[[0, 0]]], "scores": [0.9]}, None),
    ],
)
def test_page_retry_reason(reply, reason):
    assert page_retry_reason(reply) == reason


_PAGE_INFO = {"rotation": 0, "mediabox": [0, 0, 595, 842], "image_rect": None}
_EMPTY_REPLY = {"boxes": None, "txts": None, "scores": None}


def _read_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recognize, page_info=_PAGE_INFO, **result
):
    """A chunk page read by ``recognize``, with the retry render stubbed."""
    page = tmp_path / "page.ppm"
    page.write_bytes(b"P6 1 1 255 xxx")
    retry = tmp_path / "retry.ppm"
    retry.write_bytes(b"P6 1 1 255 xxx")
    rendered = MagicMock(return_value=str(retry))
    monkeypatch.setattr("bigocrpdf.services.ocr.pipeline_chunked_ocr.render_page_for_ocr", rendered)
    result = {
        "page_num": 1,
        "success": True,
        "temp_out_path": str(page),
        "proc_w": 1,
        "proc_h": 1,
        "orig_w": 1,
        "orig_h": 1,
        **result,
    }
    pipeline = ProfessionalPDFOCR(OCRConfig())
    pipeline._ocr_chunk_result(
        result, SimpleNamespace(recognize=recognize), [page_info], input_pdf=tmp_path / "in.pdf"
    )
    stats = ProcessingStats()
    pipeline._page_ocr_results(result, 1, stats)
    return result, stats, rendered


def test_failed_retry_keeps_read_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def recognize(_path):
        raise ReadError("ppocr-native failed: plan truncated")

    result, stats, rendered = _read_page(tmp_path, monkeypatch, recognize)

    rendered.assert_called_once()
    assert result["retry_level"] == 0
    assert "plan truncated" in stats.failed_pages[1]


def test_empty_retry_clears_read_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    replies = iter([ReadError("ppocr-native gave no answer"), _EMPTY_REPLY])

    def recognize(_path):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    result, stats, _ = _read_page(tmp_path, monkeypatch, recognize)

    assert result["retry_level"] == 1
    assert result["ocr_raw"] == _EMPTY_REPLY
    assert stats.failed_pages == {}


@pytest.mark.parametrize(
    ("page_info", "result"),
    [
        (_PAGE_INFO, {"geometry_applied": True}),
        (_PAGE_INFO, {"crop_applied": True}),
        (_PAGE_INFO, {"orientation_angle": 90}),
        ({**_PAGE_INFO, "rotation": 90}, {}),
        ({**_PAGE_INFO, "image_rect": (50, 50, 300, 400)}, {}),
    ],
)
def test_retry_skips_another_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, page_info, result
):
    recognize = MagicMock(return_value=_EMPTY_REPLY)

    done, _, rendered = _read_page(tmp_path, monkeypatch, recognize, page_info, **result)

    rendered.assert_not_called()
    recognize.assert_called_once()
    assert done["retry_level"] == 0


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
    worker = SimpleNamespace(
        recognize=lambda *_args: (_ for _ in ()).throw(InterruptedError("cancelled"))
    )
    result = {"page_num": 1, "success": True, "temp_out_path": "page.png"}

    with pytest.raises(InterruptedError, match="cancelled"):
        pipeline._ocr_chunk_result(result, worker, [], input_pdf=tmp_path / "in.pdf")

    assert not rendered.exists()


def test_region_refinement_skips_empty_crop() -> None:
    read = MagicMock()
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    outside_box = [[100, 100], [110, 100], [110, 110], [100, 110]]

    result = ocr_postprocess._reocr_region(image, outside_box, "text", 0.2, read)

    assert result is None
    read.assert_not_called()


def test_region_refinement_skips_failed_crop_write(monkeypatch: pytest.MonkeyPatch) -> None:
    read = MagicMock()
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    box = [[5, 5], [40, 5], [40, 40], [5, 40]]
    monkeypatch.setattr(ocr_postprocess.cv2, "imwrite", lambda *_args, **_kwargs: False)

    result = ocr_postprocess._reocr_region(image, box, "text", 0.2, read)

    assert result is None
    read.assert_not_called()


def test_unread_region_keeps_its_first_reading() -> None:
    read = MagicMock(side_effect=ReadError("no answer in 300 s"))
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    box = [[5, 5], [40, 5], [40, 40], [5, 40]]

    assert ocr_postprocess._reocr_region(image, box, "text", 0.2, read) is None
    read.assert_called_once()
