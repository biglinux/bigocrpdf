"""Structured OCR runtime diagnostics for sidecars and benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from bigocrpdf.services.rapidocr_service.config import OCRConfig, ProcessingStats
from bigocrpdf.utils.logger import logger


class OpenVinoProbe(Protocol):
    def __call__(self) -> bool: ...


def record_ocr_runtime_diagnostics(
    stats: ProcessingStats,
    config: OCRConfig,
    openvino_probe: OpenVinoProbe,
    ocr_threads: int,
    chunk_size: int,
    worker_runtime: dict[str, object] | None = None,
) -> None:
    """Store and log effective OCR runtime configuration."""
    diagnostics = build_ocr_runtime_diagnostics(
        config,
        _probe_openvino(openvino_probe),
        ocr_threads,
        chunk_size,
        worker_runtime,
    )
    stats.ocr_document.diagnostics["ocr_runtime"] = diagnostics
    logger.info("OCR runtime configuration: %s", json.dumps(diagnostics, sort_keys=True))


def build_ocr_runtime_diagnostics(
    config: OCRConfig,
    openvino_available: bool,
    ocr_threads: int,
    chunk_size: int,
    worker_runtime: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return the effective OCR runtime configuration."""
    fallback_engine_type = config.engine_type
    if fallback_engine_type != "onnxruntime" and not openvino_available:
        fallback_engine_type = "onnxruntime"
    runtime = worker_runtime if isinstance(worker_runtime, dict) else {}
    engine_type = runtime.get("engine_type", fallback_engine_type)
    gpu_backend = runtime.get("gpu_backend", config.gpu_backend)
    gpu_device_id = runtime.get("gpu_device_id", config.gpu_device_id)
    return {
        "language": config.language,
        "dpi": config.dpi,
        "engine_type": engine_type,
        "requested_engine_type": config.engine_type,
        "openvino_available": openvino_available,
        "ocr_version": "PPOCRV6",
        "model_type": config.model_type,
        "rec_batch_num": config.rec_batch_num,
        "use_textline_cls": config.use_textline_cls,
        "engine_label": runtime.get("engine_label"),
        "gpu_backend": gpu_backend,
        "requested_gpu_backend": config.gpu_backend,
        "gpu_device_id": gpu_device_id,
        "requested_gpu_device_id": config.gpu_device_id,
        "gpu_fp16": config.gpu_fp16,
        "gpu_fallback_to_cpu": config.gpu_fallback_to_cpu,
        "detection_limit_side_len": config.detection_limit_side_len,
        "detection_full_resolution": config.detection_full_resolution,
        "box_thresh": config.box_thresh,
        "unclip_ratio": config.unclip_ratio,
        "text_score_threshold": config.text_score_threshold,
        "score_mode": config.score_mode,
        "ocr_threads": ocr_threads,
        "ocr_workers": 1,
        "chunk_size": chunk_size,
        "rec_model_path": _path_diagnostic(config.get_rec_model_path()),
        "rec_keys_path": _path_diagnostic(config.get_rec_keys_path()),
        "det_model_path": _path_diagnostic(config.get_det_model_path()),
        "font_path": _path_diagnostic(config.get_font_path()),
    }


def _probe_openvino(openvino_probe: OpenVinoProbe) -> bool:
    try:
        return openvino_probe()
    except (ImportError, OSError, AttributeError):
        return False


def _path_diagnostic(path: Path | None) -> str | None:
    return str(path) if path else None
