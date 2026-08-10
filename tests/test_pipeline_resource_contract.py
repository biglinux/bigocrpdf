"""The pipeline must run within the budget it computed for the machine.

``compute_pipeline_config`` decides how many workers to start, how many pages
to hold in memory at once, and whether to collect garbage per page. Those
decisions are what keep the process inside a 2 GB container -- and nothing
checked that the pipeline actually applies them. A change that ignored the
tier would look identical to one that honoured it, until a small machine died
on a large PDF.

The OCR engine and the page workers are stubbed: the question here is which
configuration the pipeline commits to, not what the OCR returns.
"""

import gc
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from bigocrpdf.services.rapidocr_service import page_worker
from bigocrpdf.services.rapidocr_service.config import ProcessingStats
from bigocrpdf.services.rapidocr_service.pipeline_chunked_ocr import ChunkedOCRMixin
from bigocrpdf.services.rapidocr_service.resource_manager import (
    ResourceProfile,
    ResourceTier,
    compute_pipeline_config,
)

A4 = [0, 0, 595, 842]


def _mixin() -> ChunkedOCRMixin:
    mixin = ChunkedOCRMixin()
    mixin.config = SimpleNamespace()
    mixin._ocr_subprocess = MagicMock()
    mixin._ocr_subprocess.launch.return_value = object()
    mixin._ocr_subprocess.wait_until_ready.return_value = "openvino"
    mixin._check_openvino_available = MagicMock(return_value=True)
    # Every chunk is skipped: the pipeline still sets up its executor, chunking
    # and subprocess, which is all this file asserts on.
    mixin._skip_excluded_chunk = MagicMock(return_value=True)
    return mixin


def _run(mixin: ChunkedOCRMixin, tmp_path: Path, pipe_cfg, profile, pages: int):
    executor = MagicMock()
    executor.__enter__.return_value = executor
    ctx = {
        "total_pages": pages,
        "page_rotations": [],
        "all_rotation_dicts": [{"mediabox": A4} for _ in range(pages)],
        "native_text_pages": set(),
        "page_encodings": {},
    }

    with (
        patch("concurrent.futures.ProcessPoolExecutor", return_value=executor) as pool,
        patch(
            "bigocrpdf.services.rapidocr_service.pipeline_chunked_ocr.record_ocr_runtime_diagnostics"
        ) as diagnostics,
        patch("reportlab.pdfgen.canvas.Canvas", return_value=MagicMock()),
        patch.object(gc, "collect") as collect,
    ):
        mixin._run_chunked_ocr_pipeline(
            tmp_path / "input.pdf",
            tmp_path / "text.pdf",
            tmp_path / "images",
            tmp_path / "scratch",
            ctx,
            pipe_cfg,
            profile,
            ProcessingStats(),
            None,
        )
    return SimpleNamespace(pool=pool, collect=collect, diagnostics=diagnostics)


def _profile(available_mb: int, tier: ResourceTier, cpu_count: int = 8) -> ResourceProfile:
    return ResourceProfile(
        available_ram_mb=available_mb,
        total_ram_mb=max(available_mb, 2048),
        cpu_count=cpu_count,
        tier=tier,
    )


class TestConstrainedTier:
    """A small machine must be serialised, chunked small, and swept per page."""

    @pytest.fixture
    def outcome(self, tmp_path: Path):
        profile = _profile(1500, ResourceTier.CONSTRAINED)
        return _run(_mixin(), tmp_path, compute_pipeline_config(profile), profile, pages=12)

    def test_only_one_preprocessing_worker_starts(self, outcome):
        assert outcome.pool.call_args.kwargs["max_workers"] == 1

    def test_the_single_worker_is_recycled_after_each_page(self, outcome):
        """Otherwise one leaky page keeps its memory for the whole document."""
        assert outcome.pool.call_args.kwargs["max_tasks_per_child"] == 1

    def test_the_worker_initialiser_is_the_real_one(self, outcome):
        assert outcome.pool.call_args.kwargs["initializer"] is page_worker.worker_init

    def test_garbage_is_collected_after_every_page(self, tmp_path: Path):
        """gc_after_page is the tier's main lever against peak memory.

        Asserted on ``_flush_chunk_results``, which is where the flag is read;
        the per-chunk collect above it runs on every tier and so says nothing
        about this one.
        """
        mixin = _mixin()
        mixin._flush_chunk_page = MagicMock(return_value=0.0)
        pipe_cfg = compute_pipeline_config(_profile(1500, ResourceTier.CONSTRAINED))
        assert pipe_cfg.gc_after_page is True

        work_items = [{"page_num": n} for n in (1, 2, 3)]
        ocr_done = {index: ({}, item) for index, item in enumerate(work_items)}

        with patch.object(gc, "collect") as collect:
            mixin._flush_chunk_results(
                MagicMock(),
                ocr_done,
                work_items,
                [],
                3,
                ProcessingStats(),
                [],
                {},
                pipe_cfg,
                None,
            )

        assert collect.call_count == len(work_items)

    def test_no_per_page_collection_on_a_large_machine(self):
        """The sweep costs time; only the constrained tier pays for it."""
        mixin = _mixin()
        mixin._flush_chunk_page = MagicMock(return_value=0.0)
        pipe_cfg = compute_pipeline_config(_profile(64000, ResourceTier.ABUNDANT, cpu_count=32))
        assert pipe_cfg.gc_after_page is False

        work_items = [{"page_num": n} for n in (1, 2, 3)]
        ocr_done = {index: ({}, item) for index, item in enumerate(work_items)}

        with patch.object(gc, "collect") as collect:
            mixin._flush_chunk_results(
                MagicMock(), ocr_done, work_items, [], 3, ProcessingStats(), [], {}, pipe_cfg, None
            )

        assert collect.call_count == 0

    def test_the_probmap_is_downscaled(self, tmp_path: Path):
        """A smaller DBNet input is the other lever: about 30% less peak memory."""
        config = compute_pipeline_config(_profile(1500, ResourceTier.CONSTRAINED))

        assert config.downscale_probmap == 1024

    def test_exactly_one_ocr_subprocess_serves_the_run(self, tmp_path: Path):
        """Model loading dominates start-up, so it must not repeat per chunk."""
        mixin = _mixin()
        profile = _profile(1500, ResourceTier.CONSTRAINED)
        _run(mixin, tmp_path, compute_pipeline_config(profile), profile, pages=12)

        assert mixin._ocr_subprocess.launch.call_count == 1
        assert mixin._ocr_subprocess.stop.call_count == 1


class TestAbundantTier:
    @pytest.fixture
    def outcome(self, tmp_path: Path):
        profile = _profile(64000, ResourceTier.ABUNDANT, cpu_count=32)
        return _run(_mixin(), tmp_path, compute_pipeline_config(profile), profile, pages=40)

    def test_workers_are_capped(self, outcome):
        """Beyond a dozen the OCR subprocess, not preprocessing, is the limit."""
        assert outcome.pool.call_args.kwargs["max_workers"] == 12

    def test_workers_are_not_recycled(self, outcome):
        """Recycling costs an interpreter start per page; only the small tier pays it."""
        assert outcome.pool.call_args.kwargs["max_tasks_per_child"] is None


class TestDiagnosticsReportTheEffectiveConfiguration:
    def test_the_worker_count_is_reported_not_assumed(self, tmp_path: Path):
        """``ocr_workers`` was hard-coded to 1, so every record misreported it."""
        profile = _profile(64000, ResourceTier.ABUNDANT, cpu_count=32)
        pipe_cfg = compute_pipeline_config(profile)

        outcome = _run(_mixin(), tmp_path, pipe_cfg, profile, pages=8)

        assert outcome.diagnostics.call_args.kwargs["ocr_workers"] == pipe_cfg.max_workers

    def test_the_tier_and_budget_are_recorded(self, tmp_path: Path):
        """A slow run on a small machine must be distinguishable from a slow one."""
        profile = _profile(1500, ResourceTier.CONSTRAINED)

        outcome = _run(_mixin(), tmp_path, compute_pipeline_config(profile), profile, pages=8)

        resource = outcome.diagnostics.call_args.kwargs["resource"]
        assert resource["tier"] == "CONSTRAINED"
        assert resource["available_ram_mb"] == 1500
        assert resource["gc_after_page"] is True

    def test_a_partial_profile_does_not_abort_the_run(self, tmp_path: Path):
        """Diagnostics are a report, never a reason to fail a page of OCR."""
        profile = SimpleNamespace(available_ram_mb=4096)

        outcome = _run(
            _mixin(),
            tmp_path,
            compute_pipeline_config(_profile(4096, ResourceTier.MODERATE)),
            profile,
            pages=4,
        )

        assert outcome.diagnostics.call_args.kwargs["resource"]["tier"] is None
