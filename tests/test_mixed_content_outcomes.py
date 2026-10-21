"""What the mixed-content and embedded-image pipelines record for each page."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pikepdf
import pytest

from bigocrpdf.services.ocr.backend_embedded_image_pipeline import (
    BackendEmbeddedImagePipelineMixin,
)
from bigocrpdf.services.ocr.config import OCRConfig, ProcessingStats
from bigocrpdf.services.ocr.pdf_image_analysis import ImagePosition, PdfImageInfo
from bigocrpdf.services.ocr.pipeline_mixed_content import (
    MixedContentMixin,
)
from bigocrpdf.services.ocr.pipeline_mixed_content_pages import _mixed_render_candidates
from bigocrpdf.services.ocr.ppocr_native import ReadError


def test_text_only_mixed_pdf_reports_copied_pages_as_processed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    with pikepdf.Pdf.new() as pdf:
        pdf.add_blank_page()
        pdf.add_blank_page()
        pdf.save(source)
    engine = SimpleNamespace(
        config=OCRConfig(),
        _calculate_final_stats=lambda _stats, _start: None,
    )

    with (
        patch(
            "bigocrpdf.services.ocr.pipeline_mixed_content.extract_image_positions",
            return_value={},
        ),
        patch(
            "bigocrpdf.services.ocr.pipeline_mixed_content.parse_pdfimages_list",
            return_value=({}, set()),
        ),
    ):
        stats = MixedContentMixin._process_mixed_content_pdf(
            engine,
            source,
            output,
        )

    assert stats.pages_total == 2
    assert stats.pages_processed == 2
    with pikepdf.open(output) as copied:
        assert len(copied.pages) == 2


def test_mixed_content_records_reported_tier(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    with pikepdf.Pdf.new() as pdf:
        pdf.add_blank_page()
        pdf.save(source)
    worker = SimpleNamespace(threads=7, wait_ready=MagicMock(return_value=2), close=MagicMock())
    engine = SimpleNamespace(
        config=OCRConfig(),
        _start_ocr_worker=MagicMock(return_value=worker),
        _run_mixed_ocr_pass=MagicMock(),
        _raise_if_cancelled=MagicMock(),
        _post_process_mixed=MagicMock(),
        _calculate_final_stats=MagicMock(),
    )

    with (
        patch(
            "bigocrpdf.services.ocr.pipeline_mixed_content.extract_image_positions",
            return_value={1: []},
        ),
        patch(
            "bigocrpdf.services.ocr.pipeline_mixed_content.parse_pdfimages_list",
            return_value=({}, set()),
        ),
        patch(
            "bigocrpdf.services.ocr.pipeline_mixed_content._mixed_render_candidates",
            return_value=set(),
        ),
    ):
        stats = MixedContentMixin._process_mixed_content_pdf(engine, source, output)

    runtime = stats.ocr_document.diagnostics["ocr_runtime"]
    assert (runtime["kernel_tier"], runtime["ocr_threads"]) == (2, 7)


def test_embedded_image_records_reported_tier(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    with pikepdf.Pdf.new() as pdf:
        pdf.add_blank_page()
        pdf.save(source)
    worker = SimpleNamespace(threads=4, wait_ready=MagicMock(return_value=2), close=MagicMock())
    engine = SimpleNamespace(
        config=OCRConfig(),
        _start_ocr_worker=MagicMock(return_value=worker),
        _ocr_native_text_page_image_overlays=MagicMock(),
    )
    stats = ProcessingStats()

    with patch(
        "bigocrpdf.services.ocr.backend_embedded_image_pipeline.extract_image_positions",
        return_value={1: [object()]},
    ):
        BackendEmbeddedImagePipelineMixin._ocr_native_text_page_images(
            engine,
            source,
            {1},
            stats,
        )

    runtime = stats.ocr_document.diagnostics["ocr_runtime"]
    assert (runtime["kernel_tier"], runtime["ocr_threads"]) == (2, 4)


def test_embedded_ocr_skips_worker_when_temp_write_fails() -> None:
    worker = SimpleNamespace(recognize=MagicMock())
    stats = ProcessingStats()

    with patch(
        "bigocrpdf.services.ocr.backend_embedded_image_pipeline.cv2.imwrite",
        return_value=False,
    ):
        result = BackendEmbeddedImagePipelineMixin._ocr_via_persistent(
            np.zeros((2, 2, 3), dtype=np.uint8), worker, stats, 3
        )

    assert result == []
    assert "could not write" in stats.failed_pages[3]
    worker.recognize.assert_not_called()


def test_unread_image_marks_its_page_failed() -> None:
    worker = SimpleNamespace(recognize=MagicMock(side_effect=ReadError("worker crashed")))
    stats = ProcessingStats()

    result = BackendEmbeddedImagePipelineMixin._ocr_via_persistent(
        np.zeros((8, 8, 3), dtype=np.uint8), worker, stats, 2
    )

    assert result == []
    assert stats.failed_pages == {2: "worker crashed"}


@pytest.mark.parametrize("rendered", [None, "unreadable.ppm"])
def test_unrendered_page_marks_it_failed(tmp_path: Path, rendered: str | None) -> None:
    if rendered is not None:
        rendered = str(tmp_path / rendered)
        Path(rendered).write_bytes(b"not an image")
    engine = SimpleNamespace(
        config=OCRConfig(),
        _raise_if_cancelled=lambda: None,
        _prepare_page_for_reocr=lambda *_args: True,
        _ocr_via_persistent=MagicMock(),
    )
    stats = ProcessingStats()
    with pikepdf.Pdf.new() as pdf:
        pdf.add_blank_page()
        with patch(
            "bigocrpdf.services.ocr.pipeline_mixed_content.render_page_for_ocr",
            return_value=rendered,
        ):
            MixedContentMixin._ocr_rendered_page(
                engine, tmp_path / "in.pdf", pdf, 1, 0, 1, tmp_path, stats, [], None, None, 0, 10
            )

    assert stats.failed_pages == {1: "the page could not be rendered"}
    engine._ocr_via_persistent.assert_not_called()
    assert rendered is None or not Path(rendered).exists()


def test_cancel_during_an_image_stops_the_page() -> None:
    engine = SimpleNamespace(
        config=OCRConfig(),
        _ocr_image_in_page=MagicMock(side_effect=InterruptedError("cancelled")),
    )
    pairs = [(SimpleNamespace(name="/Im0"), SimpleNamespace(idx=0))]

    with pytest.raises(InterruptedError):
        MixedContentMixin._ocr_positioned_image_pairs(
            engine,
            None,
            None,
            1,
            612,
            792,
            pairs,
            {0: Path("img-000.png")},
            ProcessingStats(),
            [],
            None,
        )


def test_oversized_embedded_jpeg_is_skipped() -> None:
    import io

    from PIL import Image

    jpeg = io.BytesIO()
    Image.new("L", (1200, 1000), 255).save(jpeg, "JPEG")
    with pikepdf.Pdf.new() as pdf:
        image = pdf.make_stream(
            jpeg.getvalue(),
            Type=pikepdf.Name.XObject,
            Subtype=pikepdf.Name.Image,
            Width=400,
            Height=400,
            ColorSpace=pikepdf.Name.DeviceGray,
            BitsPerComponent=8,
            Filter=pikepdf.Name.DCTDecode,
        )
        engine = SimpleNamespace(config=OCRConfig(max_image_megapixels=1.0))

        assert BackendEmbeddedImagePipelineMixin._extract_xobj_image(engine, image, "/Im0") is None
        engine.config.max_image_megapixels = 2.0
        pixels = BackendEmbeddedImagePipelineMixin._extract_xobj_image(engine, image, "/Im0")
        assert pixels.shape == (1000, 1200, 3)


def test_page_with_oversized_image_is_rendered() -> None:
    position = ImagePosition(name="/Im0", page_num=1, x=0, y=0, width=595, height=842)
    scan = PdfImageInfo(idx=0, img_type="image", width=10_000, height=15_000, comp_size=0)

    candidates = _mixed_render_candidates(None, {1: [position]}, {1: [scan]}, set(), 128.0)

    assert candidates == {1}
