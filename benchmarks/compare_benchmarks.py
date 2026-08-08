#!/usr/bin/env python3
"""Compare BigOCRPDF benchmark JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

RecordIdentity = tuple[str, str, str, int, int]
COMPARABILITY_FIELDS = (
    "manifest_sha256",
    "ground_truth_language",
    "effective_engine_type",
    "effective_ocr_version",
    "effective_model_type",
    "effective_rec_batch_num",
    "effective_gpu_backend",
    "effective_rec_model_sha256",
    "effective_det_model_sha256",
    "effective_dictionary_sha256",
    "effective_font_sha256",
    "effective_language_hint",
    "effective_ocr_config",
    "dpi",
    "warmup_runs",
    "benchmark_environment",
    "peak_rss_method",
    "peak_rss_sample_interval_seconds",
    "text_extractor",
    "text_extractor_version",
)
REQUIRED_METRICS = (
    "char_error_rate",
    "word_error_rate",
    "end_to_end_seconds",
    "peak_rss_mb",
    "text_layer_ok",
)
MAX_ERROR_RATE_REGRESSION = 0.01
MAX_TIME_REGRESSION = 0.20
MAX_PEAK_RSS_REGRESSION = 0.20
SHA256_FIELDS = {
    "manifest_sha256",
    "effective_rec_model_sha256",
    "effective_det_model_sha256",
    "effective_dictionary_sha256",
    "effective_font_sha256",
}
REQUIRED_OCR_CONFIG_FIELDS = {
    "language",
    "dpi",
    "engine_type",
    "requested_engine_type",
    "openvino_available",
    "ocr_version",
    "model_type",
    "rec_batch_num",
    "use_textline_cls",
    "gpu_backend",
    "requested_gpu_backend",
    "gpu_device_id",
    "requested_gpu_device_id",
    "gpu_fp16",
    "gpu_fallback_to_cpu",
    "detection_limit_side_len",
    "detection_full_resolution",
    "box_thresh",
    "unclip_ratio",
    "text_score_threshold",
    "score_mode",
    "ocr_threads",
    "ocr_workers",
    "chunk_size",
}
REQUIRED_ENVIRONMENT_FIELDS = {
    "schema_version",
    "platform",
    "machine",
    "python",
    "cpu_model",
    "cpu_count",
    "dependencies",
    "gpu_devices",
    "gpu_runtimes",
}
REQUIRED_CORE_DEPENDENCIES = {
    "bigocrpdf",
    "numpy",
    "Pillow",
    "pikepdf",
    "rapidocr",
}
REQUIRED_GPU_DEVICE_FIELDS = {
    "card",
    "pci_address",
    "vendor_id",
    "device_id",
    "driver",
    "driver_version",
}
CPU_GPU_BACKENDS = {"off", "none", "cpu"}
GPU_BACKEND_RUNTIME_DISTRIBUTIONS = {
    "paddle": frozenset({"paddlepaddle", "paddlepaddle-gpu"}),
    "torch": frozenset({"torch"}),
    "tensorrt": frozenset({"tensorrt"}),
    "onnxruntime_cuda_experimental": frozenset({"onnxruntime-gpu"}),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows, _sha256 = _load_jsonl_snapshot(path)
    return rows


def _load_jsonl_snapshot(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line in payload.decode("utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows, hashlib.sha256(payload).hexdigest()


def mean_number(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return statistics.fmean(values)


def percentile_number(
    rows: list[dict[str, Any]],
    key: str,
    percentile: float,
) -> float | None:
    values = sorted(float(row[key]) for row in rows if row.get(key) is not None)
    if not values:
        return None
    rank = max(1, math.ceil(percentile * len(values)))
    return values[min(rank, len(values)) - 1]


def ok_ratio(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get(key)) / len(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    return {
        "pages": len(rows),
        "mean_cer": mean_number(rows, "char_error_rate"),
        "p95_cer": percentile_number(rows, "char_error_rate", 0.95),
        "mean_wer": mean_number(rows, "word_error_rate"),
        "p95_wer": percentile_number(rows, "word_error_rate", 0.95),
        "mean_seconds_page": mean_number(rows, "end_to_end_seconds"),
        "median_seconds_page": (
            statistics.median(
                float(row["end_to_end_seconds"])
                for row in rows
                if row.get("end_to_end_seconds") is not None
            )
            if any(row.get("end_to_end_seconds") is not None for row in rows)
            else None
        ),
        "p95_seconds_page": percentile_number(rows, "end_to_end_seconds", 0.95),
        "peak_rss_mb": max(
            (float(value) for row in rows if (value := row.get("peak_rss_mb")) is not None),
            default=None,
        ),
        "text_layer_ok": ok_ratio(rows, "text_layer_ok"),
    }


def _record_identity(row: dict[str, Any], row_number: int) -> RecordIdentity:
    values: list[str] = []
    for field in ("benchmark_profile", "dataset", "id"):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"benchmark row {row_number} requires a non-empty {field!r} field")
        values.append(value.strip())

    page_index = row.get("page_index")
    if isinstance(page_index, bool) or not isinstance(page_index, int) or page_index < 0:
        raise ValueError(
            f"benchmark row {row_number} requires a non-negative integer 'page_index' field"
        )
    run_index = row.get("run_index", 0)
    if isinstance(run_index, bool) or not isinstance(run_index, int) or run_index < 0:
        raise ValueError(
            f"benchmark row {row_number} requires a non-negative integer 'run_index' field"
        )
    return values[0], values[1], values[2], page_index, run_index


def _index_record_identities(
    rows: list[dict[str, Any]],
    source_name: str,
) -> dict[RecordIdentity, dict[str, Any]]:
    identities: dict[RecordIdentity, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        identity = _record_identity(row, row_number)
        if identity in identities:
            raise ValueError(
                f"duplicate benchmark record in {source_name}: {_format_record_identity(identity)}"
            )
        identities[identity] = row
    return identities


def _format_record_identity(identity: RecordIdentity) -> str:
    profile, dataset, sample_id, page_index, run_index = identity
    return f"{profile} / {dataset} / {sample_id} / page {page_index} / run {run_index}"


def _comparison_value(row: dict[str, Any], field: str) -> Any:
    if field == "benchmark_environment":
        return _normalized_benchmark_environment(row, row.get(field))
    return row.get(field)


def _effective_gpu_backend(row: dict[str, Any]) -> str:
    return str(row.get("effective_gpu_backend") or "").strip().lower()


def _relevant_gpu_runtimes(
    backend: str,
    runtimes: dict[str, Any],
) -> dict[str, Any]:
    expected_names = GPU_BACKEND_RUNTIME_DISTRIBUTIONS.get(backend, frozenset())
    return {name: version for name, version in runtimes.items() if name in expected_names}


def _normalized_benchmark_environment(
    row: dict[str, Any],
    value: Any,
) -> Any:
    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    backend = _effective_gpu_backend(row)
    if backend in CPU_GPU_BACKENDS:
        normalized["gpu_devices"] = []
        normalized["gpu_runtimes"] = {}
    elif backend in GPU_BACKEND_RUNTIME_DISTRIBUTIONS:
        runtimes = value.get("gpu_runtimes")
        normalized["gpu_runtimes"] = (
            _relevant_gpu_runtimes(backend, runtimes) if isinstance(runtimes, dict) else runtimes
        )
    return normalized


def _has_complete_contract_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        return bool(normalized) and normalized not in {"n/a", "unknown", "unavailable"}
    if isinstance(value, dict):
        return bool(value) and all(
            isinstance(key, str)
            and bool(key.strip())
            and _has_complete_contract_value(nested_value)
            for key, nested_value in value.items()
        )
    if isinstance(value, list | tuple):
        return bool(value) and all(_has_complete_contract_value(item) for item in value)
    return True


def _comparison_field_complete(row: dict[str, Any], field: str) -> bool:
    value = _comparison_value(row, field)
    if field in SHA256_FIELDS:
        return _is_sha256(value)
    if field == "effective_ocr_config":
        return _ocr_config_complete(row, value)
    if field == "benchmark_environment":
        return _benchmark_environment_complete(row, value)
    if field in {"effective_rec_batch_num", "dpi"}:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0
    if field == "warmup_runs":
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    if field == "peak_rss_sample_interval_seconds":
        return (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0
        )
    return _has_complete_contract_value(value)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _ocr_config_complete(row: dict[str, Any], value: Any) -> bool:
    if not isinstance(value, dict) or not REQUIRED_OCR_CONFIG_FIELDS.issubset(value):
        return False
    nullable_fields = {"gpu_device_id"}
    if not all(
        field in nullable_fields or _has_complete_contract_value(value[field])
        for field in REQUIRED_OCR_CONFIG_FIELDS
    ):
        return False
    gpu_backend = _effective_gpu_backend(row)
    gpu_device_id = value.get("gpu_device_id")
    requested_gpu_device_id = value.get("requested_gpu_device_id")
    if (
        isinstance(requested_gpu_device_id, bool)
        or not isinstance(requested_gpu_device_id, int)
        or requested_gpu_device_id < 0
    ):
        return False
    if gpu_backend in CPU_GPU_BACKENDS:
        if gpu_device_id is not None:
            return False
    elif (
        gpu_backend not in GPU_BACKEND_RUNTIME_DISTRIBUTIONS
        or isinstance(gpu_device_id, bool)
        or not isinstance(gpu_device_id, int)
        or gpu_device_id < 0
    ):
        return False
    return (
        value.get("engine_type") == row.get("effective_engine_type")
        and value.get("ocr_version") == row.get("effective_ocr_version")
        and value.get("model_type") == row.get("effective_model_type")
        and value.get("rec_batch_num") == row.get("effective_rec_batch_num")
        and value.get("gpu_backend") == row.get("effective_gpu_backend")
        and value.get("language") == row.get("effective_language_hint")
        and value.get("dpi") == row.get("dpi")
    )


def _benchmark_environment_complete(row: dict[str, Any], value: Any) -> bool:
    if not isinstance(value, dict) or not REQUIRED_ENVIRONMENT_FIELDS.issubset(value):
        return False
    if value.get("schema_version") != 1:
        return False
    for field in {"platform", "machine", "python", "cpu_model"}:
        if not _has_complete_contract_value(value.get(field)):
            return False
    cpu_count = value.get("cpu_count")
    if isinstance(cpu_count, bool) or not isinstance(cpu_count, int) or cpu_count <= 0:
        return False

    dependencies = value.get("dependencies")
    if not isinstance(dependencies, dict) or not REQUIRED_CORE_DEPENDENCIES.issubset(dependencies):
        return False
    if not all(
        isinstance(name, str) and bool(name.strip()) and _has_complete_contract_value(version)
        for name, version in dependencies.items()
    ):
        return False
    effective_engine = row.get("effective_engine_type")
    if effective_engine in {"openvino", "onnxruntime"} and effective_engine not in dependencies:
        return False

    gpu_devices = value.get("gpu_devices")
    gpu_runtimes = value.get("gpu_runtimes")
    if not isinstance(gpu_devices, list) or not isinstance(gpu_runtimes, dict):
        return False
    gpu_backend = _effective_gpu_backend(row)
    if gpu_backend in CPU_GPU_BACKENDS:
        return True
    if gpu_backend not in GPU_BACKEND_RUNTIME_DISTRIBUTIONS:
        return False
    relevant_runtimes = _relevant_gpu_runtimes(gpu_backend, gpu_runtimes)
    if not gpu_devices or not relevant_runtimes:
        return False
    if not all(
        isinstance(device, dict)
        and REQUIRED_GPU_DEVICE_FIELDS.issubset(device)
        and all(_has_complete_contract_value(device[field]) for field in REQUIRED_GPU_DEVICE_FIELDS)
        for device in gpu_devices
    ):
        return False
    return all(
        isinstance(name, str) and bool(name.strip()) and _has_complete_contract_value(version)
        for name, version in relevant_runtimes.items()
    )


def _metric_number(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    if key in {"end_to_end_seconds", "peak_rss_mb"} and numeric <= 0:
        return None
    if key in {"char_error_rate", "word_error_rate"} and numeric < 0:
        return None
    return numeric


def _missing_metrics(row: dict[str, Any]) -> list[str]:
    missing = [
        key
        for key in REQUIRED_METRICS
        if (
            not isinstance(row.get(key), bool)
            if key == "text_layer_ok"
            else _metric_number(row, key) is None
        )
    ]
    return missing


def _record_regression_fields(
    baseline_row: dict[str, Any],
    candidate_row: dict[str, Any],
) -> list[str]:
    regressions: list[str] = []
    thresholds = {
        "char_error_rate": MAX_ERROR_RATE_REGRESSION,
        "word_error_rate": MAX_ERROR_RATE_REGRESSION,
        "end_to_end_seconds": MAX_TIME_REGRESSION,
        "peak_rss_mb": MAX_PEAK_RSS_REGRESSION,
    }
    for key, threshold in thresholds.items():
        baseline_value = _metric_number(baseline_row, key)
        candidate_value = _metric_number(candidate_row, key)
        if baseline_value is None or candidate_value is None:
            continue
        if relative_worse(candidate_value, baseline_value) > threshold:
            regressions.append(key)
    if baseline_row.get("text_layer_ok") is True and candidate_row.get("text_layer_ok") is False:
        regressions.append("text_layer_ok")
    return regressions


def compare_record_coverage(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Verify that both benchmark files contain the exact same sample/profile rows."""
    baseline_identities = _index_record_identities(baseline_rows, "baseline")
    candidate_identities = _index_record_identities(candidate_rows, "candidate")
    baseline_keys = set(baseline_identities)
    candidate_keys = set(candidate_identities)
    baseline_only = sorted(baseline_keys - candidate_keys)
    candidate_only = sorted(candidate_keys - baseline_keys)
    configuration_mismatches: list[str] = []
    configuration_missing: list[str] = []
    metric_missing: list[str] = []
    record_regressions: list[str] = []
    for identity in sorted(baseline_keys & candidate_keys):
        baseline_row = baseline_identities[identity]
        candidate_row = candidate_identities[identity]
        for source_name, row in (("baseline", baseline_row), ("candidate", candidate_row)):
            missing_fields = [
                field
                for field in COMPARABILITY_FIELDS
                if not _comparison_field_complete(row, field)
            ]
            if missing_fields:
                configuration_missing.append(
                    f"{_format_record_identity(identity)} ({source_name}): "
                    f"{', '.join(missing_fields)}"
                )
            missing_metric_fields = _missing_metrics(row)
            if missing_metric_fields:
                metric_missing.append(
                    f"{_format_record_identity(identity)} ({source_name}): "
                    f"{', '.join(missing_metric_fields)}"
                )
        mismatched_fields = [
            field
            for field in COMPARABILITY_FIELDS
            if _comparison_field_complete(baseline_row, field)
            and _comparison_field_complete(candidate_row, field)
            if _comparison_value(baseline_row, field) != _comparison_value(candidate_row, field)
        ]
        if mismatched_fields:
            configuration_mismatches.append(
                f"{_format_record_identity(identity)}: {', '.join(mismatched_fields)}"
            )
        regression_fields = _record_regression_fields(baseline_row, candidate_row)
        if regression_fields:
            record_regressions.append(
                f"{_format_record_identity(identity)}: {', '.join(regression_fields)}"
            )
    return {
        "comparable": (
            not baseline_only
            and not candidate_only
            and not configuration_mismatches
            and not configuration_missing
            and not metric_missing
        ),
        "baseline_record_count": len(baseline_identities),
        "candidate_record_count": len(candidate_identities),
        "baseline_only_count": len(baseline_only),
        "candidate_only_count": len(candidate_only),
        "configuration_mismatch_count": len(configuration_mismatches),
        "configuration_missing_count": len(configuration_missing),
        "metric_missing_count": len(metric_missing),
        "record_regression_count": len(record_regressions),
        "baseline_only": [_format_record_identity(identity) for identity in baseline_only],
        "candidate_only": [_format_record_identity(identity) for identity in candidate_only],
        "configuration_mismatches": configuration_mismatches,
        "configuration_missing": configuration_missing,
        "metric_missing": metric_missing,
        "record_regressions": record_regressions,
    }


def relative_worse(candidate: float | None, baseline: float | None) -> float:
    if candidate is None or baseline is None:
        return math.inf
    return (candidate - baseline) / max(abs(baseline), 1e-9)


def render_markdown(
    baseline_path: Path,
    candidate_path: Path,
    baseline: dict[str, float | int | None],
    candidate: dict[str, float | int | None],
    coverage: dict[str, Any] | None = None,
    *,
    baseline_sha256: str | None = None,
    candidate_sha256: str | None = None,
) -> tuple[str, bool]:
    coverage = coverage or {
        "comparable": True,
        "baseline_record_count": baseline["pages"],
        "candidate_record_count": candidate["pages"],
        "baseline_only_count": 0,
        "candidate_only_count": 0,
        "configuration_mismatch_count": 0,
        "configuration_missing_count": 0,
        "metric_missing_count": 0,
        "record_regression_count": 0,
        "baseline_only": [],
        "candidate_only": [],
        "configuration_mismatches": [],
        "configuration_missing": [],
        "metric_missing": [],
        "record_regressions": [],
    }
    cer_regression = (
        relative_worse(candidate["mean_cer"], baseline["mean_cer"]) > MAX_ERROR_RATE_REGRESSION
    )
    wer_regression = (
        relative_worse(candidate["mean_wer"], baseline["mean_wer"]) > MAX_ERROR_RATE_REGRESSION
    )
    time_regression = (
        relative_worse(candidate["mean_seconds_page"], baseline["mean_seconds_page"])
        > MAX_TIME_REGRESSION
    )
    p95_time_regression = (
        relative_worse(candidate["p95_seconds_page"], baseline["p95_seconds_page"])
        > MAX_TIME_REGRESSION
    )
    text_layer_regression = float(candidate["text_layer_ok"] or 0) < float(
        baseline["text_layer_ok"] or 0
    )
    baseline_rss = baseline["peak_rss_mb"]
    candidate_rss = candidate["peak_rss_mb"]
    if (
        isinstance(baseline_rss, int | float)
        and isinstance(candidate_rss, int | float)
        and baseline_rss > 0
        and candidate_rss > 0
    ):
        rss_metric_missing = False
        rss_regression = (
            relative_worse(float(candidate_rss), float(baseline_rss)) > MAX_PEAK_RSS_REGRESSION
        )
    else:
        rss_metric_missing = True
        rss_regression = False
    coverage_mismatch = not bool(coverage["comparable"])
    metric_missing = bool(coverage.get("metric_missing_count"))
    record_regression = bool(coverage.get("record_regression_count"))
    failed = (
        coverage_mismatch
        or cer_regression
        or wer_regression
        or time_regression
        or p95_time_regression
        or text_layer_regression
        or metric_missing
        or record_regression
        or rss_metric_missing
        or rss_regression
    )

    lines = [
        "# BigOCRPDF benchmark comparison",
        "",
        f"baseline: `{baseline_path}`",
        f"candidate: `{candidate_path}`",
    ]
    if baseline_sha256:
        lines.append(f"baseline SHA-256: `{baseline_sha256}`")
    if candidate_sha256:
        lines.append(f"candidate SHA-256: `{candidate_sha256}`")
    lines.extend(
        [
            "",
            "| metric | baseline | candidate |",
            "|---|---:|---:|",
        ]
    )
    for key in [
        "pages",
        "mean_cer",
        "mean_wer",
        "p95_wer",
        "text_layer_ok",
        "mean_seconds_page",
        "median_seconds_page",
        "p95_seconds_page",
        "peak_rss_mb",
    ]:
        lines.append(f"| {key} | {baseline[key]} | {candidate[key]} |")

    lines.extend(
        [
            "",
            "## Record coverage",
            "",
            f"baseline records: {coverage['baseline_record_count']}",
            f"candidate records: {coverage['candidate_record_count']}",
            f"baseline-only records: {coverage['baseline_only_count']}",
            f"candidate-only records: {coverage['candidate_only_count']}",
            f"configuration mismatches: {coverage['configuration_mismatch_count']}",
            f"missing configuration fields: {coverage.get('configuration_missing_count', 0)}",
            f"missing metrics: {coverage.get('metric_missing_count', 0)}",
            f"record regressions: {coverage.get('record_regression_count', 0)}",
        ]
    )
    for label in (
        "baseline_only",
        "candidate_only",
        "configuration_mismatches",
        "configuration_missing",
        "metric_missing",
        "record_regressions",
    ):
        entries = coverage[label]
        if entries:
            lines.append(f"{label.replace('_', '-')} examples:")
            lines.extend(f"- {entry}" for entry in entries[:10])

    lines.extend(
        [
            "",
            f"result: {'FAIL' if failed else 'PASS'}",
            f"coverage_mismatch: {coverage_mismatch}",
            f"cer_regression: {cer_regression}",
            f"wer_regression: {wer_regression}",
            f"time_regression: {time_regression}",
            f"p95_time_regression: {p95_time_regression}",
            f"text_layer_regression: {text_layer_regression}",
            f"metric_missing: {metric_missing}",
            f"record_regression: {record_regression}",
            f"rss_metric_missing: {rss_metric_missing}",
            f"rss_regression: {rss_regression}",
        ]
    )
    return "\n".join(lines) + "\n", failed


def _write_text_atomically(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, path)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.close(temp_fd)
        except OSError:
            pass
        temp_path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    baseline_rows, baseline_sha256 = _load_jsonl_snapshot(args.baseline)
    candidate_rows, candidate_sha256 = _load_jsonl_snapshot(args.candidate)
    baseline_summary = summarize(baseline_rows)
    candidate_summary = summarize(candidate_rows)
    try:
        coverage = compare_record_coverage(baseline_rows, candidate_rows)
    except ValueError as error:
        markdown = "\n".join(
            [
                "# BigOCRPDF benchmark comparison",
                "",
                f"baseline: `{args.baseline}`",
                f"candidate: `{args.candidate}`",
                f"baseline SHA-256: `{baseline_sha256}`",
                f"candidate SHA-256: `{candidate_sha256}`",
                "",
                "result: FAIL",
                f"coverage_error: {error}",
                "",
            ]
        )
        if args.out:
            _write_text_atomically(args.out, markdown)
        else:
            print(markdown, end="")
        return 1
    markdown, failed = render_markdown(
        args.baseline,
        args.candidate,
        baseline_summary,
        candidate_summary,
        coverage,
        baseline_sha256=baseline_sha256,
        candidate_sha256=candidate_sha256,
    )
    if args.out:
        _write_text_atomically(args.out, markdown)
    else:
        print(markdown, end="")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
