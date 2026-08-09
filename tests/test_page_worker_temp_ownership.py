"""Ownership and failure tests for page-worker scratch artifacts."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bigocrpdf.services.rapidocr_service import page_worker
from bigocrpdf.services.rapidocr_service.backend_text_layer import (
    BackendTextLayerMixin,
)
from bigocrpdf.services.rapidocr_service.config import ProcessingStats
from bigocrpdf.services.rapidocr_service.pipeline_chunked_ocr import ChunkedOCRMixin


def _preprocessor() -> SimpleNamespace:
    return SimpleNamespace(
        detect_orientation=lambda _image: 0,
        process=lambda image: image,
        geometry_applied=False,
        crop_applied=False,
        crop_offset_px=(0, 0),
        crop_original_size_px=None,
    )


def test_failed_processed_image_write_removes_reserved_file(tmp_path: Path) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    with (
        patch.object(page_worker.cv2, "imwrite", return_value=False),
        pytest.raises(OSError, match="could not write"),
    ):
        page_worker._save_processed_image(image, "jpeg", 85, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_failed_ocr_image_write_removes_reserved_file(tmp_path: Path) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    with (
        patch.object(page_worker.cv2, "imwrite", return_value=False),
        pytest.raises(OSError, match="could not write"),
    ):
        page_worker._save_ocr_image_if_needed(image, "jpeg", tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_page_failure_cleans_output_created_before_later_failure(
    tmp_path: Path,
) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    args = {
        "page_num": 1,
        "img_path": "source.png",
        "config": SimpleNamespace(),
        "scratch_dir": str(tmp_path),
    }

    with (
        patch.object(page_worker, "_create_page_preprocessor", return_value=_preprocessor()),
        patch.object(page_worker, "_load_page_source_image", return_value=image),
        patch.object(page_worker, "_determine_output_format", return_value=("jpeg", 85, 0)),
        patch.object(
            page_worker,
            "_save_ocr_image_if_needed",
            side_effect=OSError("simulated OCR image failure"),
        ),
    ):
        result = page_worker.process_page(args)

    assert result["success"] is False
    assert "simulated OCR image failure" in result["error"]
    assert list(tmp_path.iterdir()) == []


def test_successful_page_transfers_only_scratch_owned_outputs(
    tmp_path: Path,
) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    args = {
        "page_num": 1,
        "img_path": "source.png",
        "config": SimpleNamespace(),
        "scratch_dir": str(tmp_path),
    }

    with (
        patch.object(page_worker, "_create_page_preprocessor", return_value=_preprocessor()),
        patch.object(page_worker, "_load_page_source_image", return_value=image),
        patch.object(page_worker, "_determine_output_format", return_value=("jpeg", 85, 0)),
    ):
        result = page_worker.process_page(args)

    assert result["success"] is True
    output_paths = {
        Path(result["temp_out_path"]),
        Path(result["temp_ocr_path"]),
    }
    assert all(path.parent == tmp_path for path in output_paths)
    assert all(path.is_file() for path in output_paths)


def test_chunk_work_items_receive_the_owned_scratch_directory(
    tmp_path: Path,
) -> None:
    mixin = ChunkedOCRMixin()
    mixin.config = SimpleNamespace(
        image_export_format="original",
        enable_deskew=False,
        enable_perspective_correction=False,
        enable_baseline_dewarp=False,
    )
    mixin.extractor = SimpleNamespace(masked_pages=set(), rendered_pages=set())
    rotation = SimpleNamespace(
        deleted=False,
        included_for_ocr=True,
        original_pdf_rotation=0,
    )
    pipe_config = SimpleNamespace(downscale_probmap=0)

    work_items = mixin._build_chunk_work_items(
        [tmp_path / "source.png"],
        0,
        [rotation],
        set(),
        {},
        pipe_config,
        scratch_dir=tmp_path,
    )

    assert work_items[0]["scratch_dir"] == str(tmp_path)


def test_preprocessing_failure_cancels_unstarted_futures() -> None:
    mixin = ChunkedOCRMixin()
    mixin.extractor = SimpleNamespace(masked_pages=set())
    futures = [MagicMock(), MagicMock()]
    executor = MagicMock()
    executor.submit.side_effect = futures

    def fail_iteration(_futures):
        raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError, match="cancelled"):
        mixin._ocr_chunk_work_items(
            executor,
            MagicMock(),
            fail_iteration,
            [{"page_num": 1}, {"page_num": 2}],
            MagicMock(),
            [],
            MagicMock(),
            Path("input.pdf"),
        )

    assert all(future.cancel.call_count == 1 for future in futures)


def test_single_preprocessing_worker_is_recycled_after_each_page(tmp_path: Path) -> None:
    mixin = ChunkedOCRMixin()
    mixin.config = SimpleNamespace()
    mixin._ocr_subprocess = MagicMock()
    mixin._ocr_subprocess.launch.return_value = object()
    mixin._ocr_subprocess.wait_until_ready.return_value = "openvino"
    mixin._check_openvino_available = MagicMock(return_value=True)
    mixin._skip_excluded_chunk = MagicMock(return_value=True)
    executor = MagicMock()
    executor.__enter__.return_value = executor
    pipe_config = SimpleNamespace(
        max_workers=1,
        chunk_size=1,
        ocr_threads=2,
    )
    ctx = {
        "total_pages": 1,
        "page_rotations": [],
        "all_rotation_dicts": [{"mediabox": [0, 0, 595, 842]}],
        "native_text_pages": set(),
        "page_encodings": {},
    }

    with (
        patch("concurrent.futures.ProcessPoolExecutor", return_value=executor) as pool,
        patch(
            "bigocrpdf.services.rapidocr_service.resource_manager.adjust_chunk_size",
            return_value=1,
        ),
        patch(
            "bigocrpdf.services.rapidocr_service.pipeline_chunked_ocr.record_ocr_runtime_diagnostics"
        ),
        patch("reportlab.pdfgen.canvas.Canvas", return_value=MagicMock()),
    ):
        mixin._run_chunked_ocr_pipeline(
            tmp_path / "input.pdf",
            tmp_path / "text.pdf",
            tmp_path / "images",
            tmp_path / "scratch",
            ctx,
            pipe_config,
            SimpleNamespace(available_ram_mb=512),
            ProcessingStats(),
            None,
        )

    pool.assert_called_once_with(
        max_workers=1,
        initializer=page_worker.worker_init,
        max_tasks_per_child=1,
    )


def test_text_layer_propagates_cancellation_from_the_last_page(
    tmp_path: Path,
) -> None:
    backend = BackendTextLayerMixin()
    process = object()
    backend._ocr_subprocess = SimpleNamespace(  # type: ignore[attr-defined]
        launch=MagicMock(return_value=process),
        stop=MagicMock(),
    )
    backend._text_layer_work_item = MagicMock(return_value={})  # type: ignore[method-assign]
    backend._run_text_layer_page = MagicMock(  # type: ignore[method-assign]
        side_effect=InterruptedError("cancelled"),
    )
    backend._release_text_layer_page_memory = MagicMock()  # type: ignore[method-assign]
    pdf_canvas = MagicMock()

    with (
        patch(
            "bigocrpdf.services.rapidocr_service.backend_text_layer.canvas.Canvas",
            return_value=pdf_canvas,
        ),
        pytest.raises(InterruptedError, match="cancelled"),
    ):
        backend._create_text_layer_pdf(
            [tmp_path / "page.png"],
            tmp_path / "text-layer.pdf",
            [{}],
            ProcessingStats(),
            None,
        )

    backend._ocr_subprocess.stop.assert_called_once_with(process)  # type: ignore[attr-defined]
    pdf_canvas.save.assert_not_called()
