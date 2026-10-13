"""The OCR configuration a job ran with, recorded for benchmarks and structured output."""

import json

from bigocrpdf.services.ocr import ppocr_native
from bigocrpdf.services.ocr.config import OCRConfig, ProcessingStats
from bigocrpdf.utils.logger import logger


def record_ocr_runtime_diagnostics(
    stats: ProcessingStats,
    config: OCRConfig,
    kernel_tier: int,
    ocr_threads: int,
    chunk_size: int,
    ocr_workers: int = 1,
    resource: dict[str, object] | None = None,
) -> None:
    """Store and log the configuration that ran.

    Args:
        kernel_tier: What the worker reported in its ready line.
        ocr_workers: Preprocessing workers the pipeline actually started, so a
            benchmark record reports the parallelism that ran.
        resource: The tier and budget the pipeline computed, when the caller
            knows it. Without it a slow run cannot be told apart from a run
            that was correctly throttled for a small machine.
    """
    diagnostics = {
        "model": ppocr_native.MODEL,
        "dpi": config.dpi,
        "kernel_tier": kernel_tier,
        "detection_limit_side_len": config.detection_limit_side_len,
        "detection_full_resolution": config.detection_full_resolution,
        "box_thresh": config.box_thresh,
        "unclip_ratio": config.unclip_ratio,
        "text_score_threshold": config.text_score_threshold,
        "ocr_threads": ocr_threads,
        "ocr_workers": ocr_workers,
        "chunk_size": chunk_size,
        "worker_path": ppocr_native.find_worker(),
        "plans_dir": str(ppocr_native.plans_dir()),
        "font_path": str(config.font_path),
        **({"resource": resource} if resource else {}),
    }
    stats.ocr_document.diagnostics["ocr_runtime"] = diagnostics
    logger.info("OCR runtime configuration: %s", json.dumps(diagnostics, sort_keys=True))
