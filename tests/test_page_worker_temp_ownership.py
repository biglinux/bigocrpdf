"""Ownership and failure tests for page-worker scratch artifacts."""

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bigocrpdf.services.ocr import page_worker
from bigocrpdf.services.ocr.config import ProcessingStats
from bigocrpdf.services.ocr.geometry_trace import GeometryTrace
from bigocrpdf.services.ocr.pdf_extractor import Extraction
from bigocrpdf.services.ocr.pipeline_chunked_ocr import ChunkedOCRMixin
from bigocrpdf.services.ocr.resource_manager import PipelineConfig, ResourceProfile, ResourceTier


def _preprocessor() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(enable_orientation_detection=False),
        process=lambda image: image,
        geometry_applied=False,
        crop_applied=False,
        crop_offset_px=(0, 0),
        crop_original_size_px=None,
        trace=GeometryTrace(),
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
        page_worker._save_ocr_image(image, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_page_failure_cleans_output_created_before_later_failure(
    tmp_path: Path,
) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    args = {
        "page_num": 1,
        "img_path": "source.png",
        "config": SimpleNamespace(enable_scanner_effect=False),
        "scratch_dir": str(tmp_path),
    }

    with (
        patch.object(page_worker, "_create_page_preprocessor", return_value=_preprocessor()),
        patch.object(page_worker, "_load_page_source_image", return_value=image),
        patch.object(page_worker, "_determine_output_format", return_value=("jpeg", 85, 0)),
        patch.object(
            page_worker,
            "_save_ocr_image",
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
        "config": SimpleNamespace(enable_scanner_effect=False),
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


def test_ocr_reads_the_page_without_the_scanner_effect(tmp_path: Path) -> None:
    """OCR reads the page as scanned; the effect only changes the image in the PDF."""
    rng = np.random.default_rng(0)
    image = rng.integers(150, 230, size=(64, 64, 3), dtype=np.uint8)
    args = {
        "page_num": 1,
        "img_path": "source.png",
        "config": SimpleNamespace(enable_scanner_effect=True, scanner_effect_strength=1.0),
        "scratch_dir": str(tmp_path),
    }

    with (
        patch.object(page_worker, "_create_page_preprocessor", return_value=_preprocessor()),
        patch.object(page_worker, "_load_page_source_image", return_value=image),
        patch.object(page_worker, "_determine_output_format", return_value=("png", 0, 0)),
    ):
        result = page_worker.process_page(args)

    ocr_image = page_worker.cv2.imread(result["temp_ocr_path"])
    page_image = page_worker.cv2.imread(result["temp_out_path"])
    assert np.array_equal(ocr_image, image)
    assert not np.array_equal(page_image, image)


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
    rotation = SimpleNamespace(
        deleted=False,
        included_for_ocr=True,
        original_pdf_rotation=0,
        image_turn=(False, 0),
    )
    pipe_config = SimpleNamespace(downscale_probmap=0)

    work_items = mixin._build_chunk_work_items(
        Extraction([tmp_path / "source.png"]),
        0,
        [rotation],
        set(),
        {},
        pipe_config,
        scratch_dir=tmp_path,
    )

    assert work_items[0]["scratch_dir"] == str(tmp_path)


def test_ocr_input_goes_once_read(tmp_path: Path) -> None:
    """The uncompressed OCR input leaves the disk with its page's OCR; the
    image the flush draws stays."""
    drawn, ocr_input = tmp_path / "page.jpg", tmp_path / "page.ppm"
    drawn.write_bytes(b"jpeg")
    ocr_input.write_bytes(b"ppm")
    mixin = ChunkedOCRMixin()
    mixin._retry_chunk_ocr_if_needed = MagicMock()
    result = {
        "page_num": 1,
        "success": True,
        "temp_out_path": str(drawn),
        "temp_ocr_path": str(ocr_input),
        "geometry_applied": True,
    }

    with patch("bigocrpdf.services.ocr.pipeline_chunked_ocr.read_refined", return_value=[]):
        mixin._ocr_chunk_result(result, MagicMock(), [{}], scratch_dir=tmp_path)

    assert drawn.exists()
    assert not ocr_input.exists()
    assert "temp_ocr_path" not in result


def test_preprocessing_failure_cancels_unstarted_futures() -> None:
    mixin = ChunkedOCRMixin()
    futures = [MagicMock(), MagicMock()]

    def fail_iteration(_futures):
        raise InterruptedError("cancelled")

    with (
        patch("bigocrpdf.services.ocr.pipeline_chunked_ocr.as_completed", fail_iteration),
        pytest.raises(InterruptedError, match="cancelled"),
    ):
        mixin._ocr_chunk_work_items(
            {future: idx for idx, future in enumerate(futures)},
            [{"page_num": 1}, {"page_num": 2}],
            MagicMock(),
            [],
            Path("input.pdf"),
            set(),
        )

    assert all(future.cancel.call_count == 1 for future in futures)


def test_progress_follows_pages_read() -> None:
    """A chunk reports each page as it is read, before any page is written."""
    mixin = ChunkedOCRMixin()
    mixin._ocr_chunk_result = MagicMock()
    mixin._raise_if_cancelled = MagicMock()
    futures = {MagicMock(): 0, MagicMock(): 1}
    reported = []

    with patch("bigocrpdf.services.ocr.pipeline_chunked_ocr.as_completed", iter):
        mixin._ocr_chunk_work_items(
            futures,
            [{"page_num": 3, "scratch_dir": "/x"}, {"page_num": 4, "scratch_dir": "/x"}],
            MagicMock(),
            [],
            Path("input.pdf"),
            set(),
            progress_callback=lambda *args: reported.append(args),
            pages_before=2,
            total_pages=4,
        )

    assert reported == [(61, 100, "Processing page 3/4..."), (80, 100, "Processing page 4/4...")]


def test_single_preprocessing_worker_is_recycled_after_each_page(tmp_path: Path) -> None:
    mixin = ChunkedOCRMixin()
    mixin.config = SimpleNamespace()
    mixin._start_ocr_worker = MagicMock(return_value=MagicMock(threads=2))
    executor = MagicMock()
    executor.__enter__.return_value = executor
    pipe_config = PipelineConfig(
        max_workers=1,
        chunk_size=1,
        ocr_threads=2,
        gc_after_page=True,
        downscale_probmap=1024,
    )
    ctx = {
        "total_pages": 1,
        "page_rotations": [SimpleNamespace(page_number=1, deleted=True, included_for_ocr=False)],
        "all_rotation_dicts": [{"mediabox": [0, 0, 595, 842]}],
        "native_text_pages": set(),
        "page_encodings": {},
    }

    with (
        patch(
            "bigocrpdf.services.ocr.pipeline_chunked_ocr.ProcessPoolExecutor", return_value=executor
        ) as pool,
        patch(
            "bigocrpdf.services.ocr.pipeline_chunked_ocr.adjust_chunk_size",
            return_value=1,
        ),
        patch("bigocrpdf.services.ocr.pipeline_chunked_ocr.record_ocr_runtime_diagnostics"),
        patch("reportlab.pdfgen.canvas.Canvas", return_value=MagicMock()),
    ):
        mixin._run_chunked_ocr_pipeline(
            tmp_path / "input.pdf",
            tmp_path / "text.pdf",
            tmp_path / "images",
            tmp_path / "scratch",
            ctx,
            pipe_config,
            ResourceProfile(
                available_ram_mb=512, total_ram_mb=2048, cpu_count=2, tier=ResourceTier.CONSTRAINED
            ),
            ProcessingStats(),
            None,
        )

    pool.assert_called_once()
    kwargs = pool.call_args.kwargs
    assert kwargs["max_workers"] == 1
    assert kwargs["initializer"] is page_worker.worker_init
    assert kwargs["max_tasks_per_child"] == 1


def _chunk_pipeline() -> ChunkedOCRMixin:
    """A chunked pipeline whose collaborators are mocks: one page a chunk."""
    mixin = ChunkedOCRMixin()
    mixin.config = SimpleNamespace(
        image_export_format="original",
        enable_deskew=False,
        enable_perspective_correction=False,
        enable_baseline_dewarp=False,
    )
    mixin._start_ocr_worker = MagicMock(return_value=MagicMock(threads=2))
    mixin._ocr_chunk_work_items = MagicMock(return_value={})
    mixin._flush_chunk_results = MagicMock(return_value=0.0)
    mixin._raise_if_cancelled = MagicMock()
    mixin.extractor = MagicMock()
    return mixin


def _run_chunks(
    mixin: ChunkedOCRMixin, tmp_path: Path, pages: int, executor=None, deleted=frozenset()
) -> None:
    executor = executor or MagicMock()
    executor.__enter__.return_value = executor
    rotations = [
        SimpleNamespace(
            page_number=n,
            deleted=n in deleted,
            included_for_ocr=True,
            original_pdf_rotation=0,
            image_turn=(False, 0),
        )
        for n in range(1, pages + 1)
    ]
    ctx = {
        "total_pages": pages,
        "page_rotations": rotations,
        "all_rotation_dicts": [{"mediabox": [0, 0, 595, 842]}] * pages,
        "native_text_pages": set(),
        "page_encodings": {},
    }
    with (
        patch(
            "bigocrpdf.services.ocr.pipeline_chunked_ocr.ProcessPoolExecutor", return_value=executor
        ),
        patch("bigocrpdf.services.ocr.pipeline_chunked_ocr.adjust_chunk_size", return_value=1),
        patch("bigocrpdf.services.ocr.pipeline_chunked_ocr.record_ocr_runtime_diagnostics"),
        patch("reportlab.pdfgen.canvas.Canvas", return_value=MagicMock()),
    ):
        mixin._run_chunked_ocr_pipeline(
            tmp_path / "input.pdf",
            tmp_path / "text.pdf",
            tmp_path / "images",
            tmp_path / "scratch",
            ctx,
            PipelineConfig(
                max_workers=2, chunk_size=1, ocr_threads=2, gc_after_page=False, downscale_probmap=0
            ),
            ResourceProfile(
                available_ram_mb=4096, total_ram_mb=8192, cpu_count=4, tier=ResourceTier.ABUNDANT
            ),
            ProcessingStats(),
            None,
        )


def test_prefetch_keeps_its_own_directory(tmp_path: Path) -> None:
    """The chunk prepared ahead must not wipe the images of the chunk in OCR,
    also when an excluded chunk sits between them."""
    mixin = _chunk_pipeline()
    used = []

    def extract(_pdf, *, output_dir, page_range, skip_pages):
        used.append((page_range, output_dir))
        return Extraction([output_dir / f"page_{page_range[0]}.png"])

    mixin.extractor.extract.side_effect = extract
    _run_chunks(mixin, tmp_path, 3, deleted={2})

    assert [page_range for page_range, _ in used] == [(1, 1), (3, 3)]
    assert used[0][1] != used[1][1]


def test_chunk_images_go_before_flush(tmp_path: Path) -> None:
    mixin = _chunk_pipeline()
    present_at_flush = []

    def extract(_pdf, *, output_dir, page_range, skip_pages):
        output_dir.mkdir(parents=True, exist_ok=True)
        image = output_dir / f"page_{page_range[0]}.png"
        image.write_bytes(b"png")
        return Extraction([image])

    mixin.extractor.extract.side_effect = extract
    mixin._flush_chunk_results.side_effect = lambda *_args: (
        present_at_flush.append(sorted(p.name for p in (tmp_path / "images").rglob("page_1.png")))
        or 0.0
    )
    _run_chunks(mixin, tmp_path, 1)

    assert present_at_flush == [[]]


def test_interrupt_queues_no_prefetched_pages(tmp_path: Path) -> None:
    """An error while the next chunk is still extracting must not queue its
    preprocessing once the extraction returns."""
    mixin = _chunk_pipeline()
    release = threading.Event()

    def extract(_pdf, *, output_dir, page_range, skip_pages):
        if page_range == (2, 2):
            release.wait(5)
        return Extraction([output_dir / f"page_{page_range[0]}.png"])

    def interrupted(*_args, **_kwargs):
        threading.Timer(0.2, release.set).start()
        raise KeyboardInterrupt

    mixin.extractor.extract.side_effect = extract
    mixin._ocr_chunk_work_items.side_effect = interrupted
    executor = MagicMock()

    with pytest.raises(KeyboardInterrupt):
        _run_chunks(mixin, tmp_path, 2, executor)

    queued = [call.args[1]["page_num"] for call in executor.submit.call_args_list]
    assert queued == [1]
