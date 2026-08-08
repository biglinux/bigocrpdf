"""Truthful mixed-content pipeline outcomes."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pikepdf

from bigocrpdf.services.rapidocr_service.backend_embedded_image_pipeline import (
    BackendEmbeddedImagePipelineMixin,
)
from bigocrpdf.services.rapidocr_service.config import OCRConfig, ProcessingStats
from bigocrpdf.services.rapidocr_service.pipeline_mixed_content import (
    MixedContentMixin,
)


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
            "bigocrpdf.services.rapidocr_service.pipeline_mixed_content.extract_image_positions",
            return_value={},
        ),
        patch(
            "bigocrpdf.services.rapidocr_service.pipeline_mixed_content.parse_pdfimages_list",
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


def test_mixed_content_records_the_worker_runtime_that_became_ready(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    with pikepdf.Pdf.new() as pdf:
        pdf.add_blank_page()
        pdf.save(source)
    runtime = {
        "schema_version": 1,
        "engine_label": "onnxruntime_cpu",
        "engine_type": "onnxruntime",
        "gpu_backend": "off",
        "gpu_device_id": None,
    }
    process = SimpleNamespace(_bigocr_threads=7)
    subprocess_controller = SimpleNamespace(
        launch=MagicMock(return_value=process),
        wait_until_ready=MagicMock(return_value=runtime),
        stop=MagicMock(),
    )
    engine = SimpleNamespace(
        config=OCRConfig(),
        _ocr_subprocess=subprocess_controller,
        _check_openvino_available=MagicMock(return_value=False),
        _run_mixed_ocr_pass=MagicMock(),
        _post_process_mixed=MagicMock(),
        _calculate_final_stats=MagicMock(),
    )

    with (
        patch(
            "bigocrpdf.services.rapidocr_service.pipeline_mixed_content.extract_image_positions",
            return_value={1: []},
        ),
        patch(
            "bigocrpdf.services.rapidocr_service.pipeline_mixed_content.parse_pdfimages_list",
            return_value=({}, set()),
        ),
        patch(
            "bigocrpdf.services.rapidocr_service.pipeline_mixed_content._mixed_render_candidates",
            return_value=set(),
        ),
        patch(
            "bigocrpdf.services.rapidocr_service.pipeline_mixed_content.record_ocr_runtime_diagnostics"
        ) as record_runtime,
    ):
        MixedContentMixin._process_mixed_content_pdf(engine, source, output)

    record_runtime.assert_called_once()
    assert record_runtime.call_args.args[3:] == (7, 1, runtime)


def test_embedded_image_ocr_records_the_ready_worker_runtime(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    with pikepdf.Pdf.new() as pdf:
        pdf.add_blank_page()
        pdf.save(source)
    runtime = {
        "schema_version": 1,
        "engine_label": "paddle_cuda",
        "engine_type": "paddle",
        "gpu_backend": "paddle",
        "gpu_device_id": 0,
    }
    process = SimpleNamespace(_bigocr_threads=4)
    subprocess_controller = SimpleNamespace(
        launch=MagicMock(return_value=process),
        wait_until_ready=MagicMock(return_value=runtime),
        stop=MagicMock(),
    )
    engine = SimpleNamespace(
        config=OCRConfig(),
        _ocr_subprocess=subprocess_controller,
        _check_openvino_available=MagicMock(return_value=False),
        _ocr_native_text_page_image_overlays=MagicMock(),
    )
    stats = ProcessingStats()

    with (
        patch(
            "bigocrpdf.services.rapidocr_service.backend_embedded_image_pipeline.extract_image_positions",
            return_value={1: [object()]},
        ),
        patch(
            "bigocrpdf.services.rapidocr_service.backend_embedded_image_pipeline.record_ocr_runtime_diagnostics"
        ) as record_runtime,
    ):
        BackendEmbeddedImagePipelineMixin._ocr_native_text_page_images(
            engine,
            source,
            {1},
            stats,
        )

    record_runtime.assert_called_once()
    assert record_runtime.call_args.args[3:] == (4, 1, runtime)


def test_embedded_ocr_skips_worker_when_temp_write_fails() -> None:
    recognize = MagicMock()
    engine = SimpleNamespace(
        _ocr_subprocess=SimpleNamespace(recognize=recognize),
        config=SimpleNamespace(text_score_threshold=0.5),
    )

    with patch(
        "bigocrpdf.services.rapidocr_service.backend_embedded_image_pipeline.cv2.imwrite",
        return_value=False,
    ):
        result = BackendEmbeddedImagePipelineMixin._ocr_via_persistent(
            engine,
            np.zeros((2, 2, 3), dtype=np.uint8),
            object(),
        )

    assert result == []
    recognize.assert_not_called()
