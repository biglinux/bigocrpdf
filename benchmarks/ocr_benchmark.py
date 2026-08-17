#!/usr/bin/env python3
"""Run a small BigOCRPDF OCR benchmark from a manifest JSONL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import platform
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from itertools import product
from pathlib import Path
from statistics import fmean, median
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks.ocr_metrics import (
    aggregate_confidence,
    char_error_rate,
    levenshtein_ratio,
    word_error_rate,
)
from benchmarks.validate_text_layer import build_report
from bigocrpdf.services.rapidocr_service.ocr_document_io import load_ocr_document_json

PROFILES: dict[str, dict[str, str]] = {
    "fast_cpu": {
        "model_type": "small",
        "dpi": "250",
        "engine": "openvino",
        "rec_batch_num": "1",
    },
    "balanced_cpu": {
        "model_type": "small",
        "dpi": "300",
        "engine": "openvino",
        "rec_batch_num": "1",
    },
    "quality_cpu": {
        "model_type": "medium",
        "dpi": "350",
        "engine": "openvino",
        "rec_batch_num": "1",
    },
}
MATRIX_KEYS = {"engine", "model_type", "rec_batch_num", "dpi", "gpu_backend"}
REC_BATCH_SWEEP_VALUES = ["1", "2", "4", "8", "16"]
PROCESS_RSS_SAMPLE_INTERVAL_SECONDS = 0.05
SAFE_SAMPLE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
BENCHMARK_DEPENDENCIES = (
    "bigocrpdf",
    "numpy",
    "onnxruntime",
    "openvino",
    "opencv-python",
    "Pillow",
    "pikepdf",
    "pycairo",
    "PyGObject",
    "rapidocr",
    "reportlab",
    "scipy",
)
GPU_RUNTIME_DISTRIBUTIONS = (
    "onnxruntime-gpu",
    "paddlepaddle",
    "paddlepaddle-gpu",
    "tensorrt",
    "torch",
)
DEPENDENCY_MODULES = {
    "numpy": "numpy",
    "onnxruntime": "onnxruntime",
    "openvino": "openvino",
    "opencv-python": "cv2",
    "Pillow": "PIL",
    "pikepdf": "pikepdf",
    "pycairo": "cairo",
    "PyGObject": "gi",
    "rapidocr": "rapidocr",
    "reportlab": "reportlab",
    "scipy": "scipy",
}
PPOCRV6_LANGUAGES = {
    "ch",
    "chinese_cht",
    "en",
    "japan",
    "af",
    "az",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "es",
    "et",
    "eu",
    "fi",
    "fr",
    "ga",
    "gl",
    "hr",
    "hu",
    "id",
    "is",
    "it",
    "ku",
    "la",
    "lb",
    "lt",
    "lv",
    "mi",
    "ms",
    "mt",
    "nl",
    "no",
    "oc",
    "pl",
    "pt",
    "qu",
    "rm",
    "ro",
    "rs_latin",
    "sk",
    "sl",
    "sq",
    "sv",
    "sw",
    "tl",
    "tr",
    "uz",
    "vi",
}
PPOCRV6_LANGUAGE_ALIASES = {
    "zh": "ch",
    "zh_cn": "ch",
    "zh_hans": "ch",
    "zh_tw": "chinese_cht",
    "zh_hant": "chinese_cht",
    "ja": "japan",
    "jp": "japan",
    "english": "en",
    "french": "fr",
    "german": "de",
    "pt_br": "pt",
    "pt_pt": "pt",
    "latin": "en",
}
SUMMARY_COLUMNS = [
    "source_jsonl_sha256",
    "benchmark_profile",
    "summary_group",
    "pages",
    "successful_pages",
    "failed_pages",
    "text_layer_ok_pages",
    "text_layer_ok_percent",
    "mean_cer",
    "micro_cer",
    "mean_wer",
    "micro_wer",
    "mean_levenshtein_ratio",
    "mean_seconds_page",
    "median_seconds_page",
    "p95_seconds_page",
    "peak_rss_mb",
    "mean_pdf_size_bytes",
]


@contextmanager
def load_manifest(
    path: Path,
    limit: int | None,
) -> Iterator[list[dict[str, Any]]]:
    """Yield manifest rows backed by private, hash-verified material snapshots."""
    rows: list[dict[str, Any]] = []
    base = path.parent.resolve(strict=True)
    manifest_snapshot = path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_snapshot).hexdigest()
    sample_ids: set[str] = set()
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open(base, directory_flags)
    with tempfile.TemporaryDirectory(prefix="bigocrpdf_manifest_") as snapshot_name:
        snapshot_root = Path(snapshot_name)
        try:
            for line_number, line in enumerate(
                manifest_snapshot.decode("utf-8").splitlines(),
                start=1,
            ):
                if limit is not None and len(rows) >= limit:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("manifest_version") != 1:
                    raise ValueError(
                        f"Manifest row {line_number} has an unsupported or missing manifest_version"
                    )
                sample_id = row.get("id")
                if not isinstance(sample_id, str) or not sample_id.strip():
                    raise ValueError(f"Manifest row {line_number} has no stable sample id")
                if not SAFE_SAMPLE_ID_PATTERN.fullmatch(sample_id):
                    raise ValueError(
                        f"Manifest row {line_number} has an unsafe sample id: {sample_id!r}"
                    )
                if sample_id in sample_ids:
                    raise ValueError(f"Duplicate benchmark sample id: {sample_id}")
                sample_ids.add(sample_id)
                expected_hashes = row.get("file_sha256")
                expected_sizes = row.get("file_bytes")
                if not isinstance(expected_hashes, dict):
                    raise ValueError(f"Manifest sample {sample_id} has no file_sha256 map")
                if not isinstance(expected_sizes, dict):
                    raise ValueError(f"Manifest sample {sample_id} has no file_bytes map")

                sample_snapshot = snapshot_root / f"{len(rows):04d}"
                sample_snapshot.mkdir(mode=0o700)
                source_paths: dict[str, str] = {}
                snapshot_names = {
                    "image": "source_image",
                    "pdf": "input.pdf",
                    "gt_text": "ground_truth.txt",
                }
                for key in ["image", "pdf", "gt_text"]:
                    if row.get(key):
                        raw_path = str(row[key])
                        source_paths[key] = raw_path
                        expected_hash = expected_hashes.get(key)
                        expected_size = expected_sizes.get(key)
                        if not isinstance(expected_hash, str) or not expected_hash:
                            raise ValueError(
                                f"Manifest sample {sample_id} has no SHA-256 for {key}"
                            )
                        if (
                            isinstance(expected_size, bool)
                            or not isinstance(expected_size, int)
                            or expected_size < 0
                        ):
                            raise ValueError(
                                f"Manifest sample {sample_id} has no byte size for {key}"
                            )
                        source_fd = _open_regular_beneath(
                            root_fd,
                            raw_path,
                            sample_id=sample_id,
                        )
                        destination = sample_snapshot / snapshot_names[key]
                        try:
                            _copy_verified_manifest_fd(
                                source_fd,
                                destination,
                                expected_hash,
                                expected_size,
                                sample_id=sample_id,
                                material_key=key,
                            )
                        finally:
                            os.close(source_fd)
                        row[key] = str(destination)
                row["_manifest_source_paths"] = source_paths
                row["_manifest_sha256"] = manifest_sha256
                rows.append(row)
            yield rows
        finally:
            os.close(root_fd)


def _open_regular_beneath(
    root_fd: int,
    raw_path: str,
    *,
    sample_id: str,
) -> int:
    """Open one manifest material without following links or traversal."""
    relative_path = Path(raw_path)
    if relative_path.is_absolute() or not relative_path.parts or ".." in relative_path.parts:
        raise ValueError(f"Manifest sample {sample_id} has an unsafe material path: {raw_path}")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    file_flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    directory_fd = os.dup(root_fd)
    try:
        for component in relative_path.parts[:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        source_fd = os.open(relative_path.parts[-1], file_flags, dir_fd=directory_fd)
    except OSError as error:
        raise ValueError(
            f"Manifest sample {sample_id} material is not a safe regular file: {raw_path}"
        ) from error
    finally:
        os.close(directory_fd)
    if not stat.S_ISREG(os.fstat(source_fd).st_mode):
        os.close(source_fd)
        raise ValueError(
            f"Manifest sample {sample_id} material is not a safe regular file: {raw_path}"
        )
    return source_fd


def _copy_verified_manifest_fd(
    source_fd: int,
    destination: Path,
    expected_sha256: str,
    expected_bytes: int,
    *,
    sample_id: str,
    material_key: str,
) -> None:
    """Copy and authenticate exactly the bytes read from one opened material."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    destination_fd = os.open(destination, flags, 0o600)
    digest = hashlib.sha256()
    copied_bytes = 0
    try:
        with os.fdopen(destination_fd, "wb") as output:
            while chunk := os.read(source_fd, 1024 * 1024):
                copied_bytes += len(chunk)
                digest.update(chunk)
                output.write(chunk)
        if copied_bytes != expected_bytes or digest.hexdigest() != expected_sha256:
            raise ValueError(
                f"Manifest sample {sample_id} failed SHA-256 verification for {material_key}"
            )
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _resolve_manifest_input(base: Path, raw_path: str, sample_id: str) -> Path:
    relative_path = Path(raw_path)
    if relative_path.is_absolute():
        raise ValueError(f"Manifest sample {sample_id} uses an absolute material path")
    try:
        resolved = (base / relative_path).resolve(strict=True)
    except OSError as error:
        raise ValueError(
            f"Manifest sample {sample_id} material does not exist: {raw_path}"
        ) from error
    if not resolved.is_relative_to(base) or not resolved.is_file():
        raise ValueError(f"Manifest sample {sample_id} has an unsafe material path: {raw_path}")
    return resolved


def run_bigocrpdf(
    row: dict[str, Any],
    profile: dict[str, str],
    output_pdf: Path,
    sidecar_json: Path,
    gpu_backend: str,
) -> tuple[subprocess.CompletedProcess[str], float | None]:
    input_pdf = row.get("pdf")
    if not input_pdf:
        raise ValueError(f"Manifest row {row.get('id')} has no pdf path")

    cmd = [
        sys.executable,
        "-m",
        "bigocrpdf.cli",
        "ocr",
        str(input_pdf),
        "-o",
        str(output_pdf),
        "--dpi",
        profile["dpi"],
        "--engine-type",
        profile["engine"],
        "--model-type",
        profile["model_type"],
        "--rec-batch-num",
        profile["rec_batch_num"],
        "--gpu-backend",
        gpu_backend,
        # Structured OCR is opt-in and never lands beside a user's file; the
        # benchmark asks for it inside its own temporary directory.
        "--sidecar-json",
        str(sidecar_json),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    process = subprocess.Popen(  # noqa: S603 - argv is fixed and paths are separate arguments
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    stop_sampling = threading.Event()
    peak_rss_kib = [0.0]

    def sample_process_tree() -> None:
        while not stop_sampling.is_set():
            current_rss_kib = _sample_process_tree_rss_kib(process.pid)
            if current_rss_kib is not None:
                peak_rss_kib[0] = max(peak_rss_kib[0], current_rss_kib)
            stop_sampling.wait(PROCESS_RSS_SAMPLE_INTERVAL_SECONDS)

    sampler = threading.Thread(
        target=sample_process_tree,
        name="bigocrpdf-benchmark-rss",
        daemon=True,
    )
    sampler.start()
    try:
        stdout, stderr = process.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        stdout, stderr = _terminate_and_reap_process_group(process)
        stderr = f"{stderr}\nBenchmark timed out after 600 seconds".strip()
    except BaseException:
        try:
            _terminate_and_reap_process_group(process)
        except BaseException:
            pass
        raise
    finally:
        stop_sampling.set()
        sampler.join(timeout=1)

    result = subprocess.CompletedProcess(
        cmd,
        process.returncode if process.returncode is not None else -1,
        stdout,
        stderr,
    )
    peak_rss_mb = peak_rss_kib[0] / 1024.0 if peak_rss_kib[0] > 0 else None
    return result, peak_rss_mb


def _terminate_process_group(process: subprocess.Popen[str], signal_number: signal.Signals) -> None:
    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        return


def _terminate_and_reap_process_group(
    process: subprocess.Popen[str],
) -> tuple[str, str]:
    """Stop a benchmark process session and always reap its leader."""
    _terminate_process_group(process, signal.SIGTERM)
    try:
        return process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process, signal.SIGKILL)
        return process.communicate()


def _sample_process_tree_rss_kib(
    root_pid: int,
    proc_root: Path = Path("/proc"),
) -> float | None:
    """Return the current summed RSS for a Linux process and all its descendants."""
    rss_pages: dict[int, int] = {}
    children: dict[int, list[int]] = defaultdict(list)
    try:
        process_dirs = list(proc_root.iterdir())
    except OSError:
        return None

    for process_dir in process_dirs:
        if not process_dir.name.isdigit():
            continue
        try:
            stat_line = process_dir.joinpath("stat").read_text(encoding="ascii")
            command_end = stat_line.rfind(")")
            fields = stat_line[command_end + 2 :].split()
            pid = int(process_dir.name)
            parent_pid = int(fields[1])
            rss_pages[pid] = max(0, int(fields[21]))
            children[parent_pid].append(pid)
        except (IndexError, OSError, UnicodeError, ValueError):
            continue

    if root_pid not in rss_pages:
        return None
    process_ids: list[int] = [root_pid]
    seen: set[int] = set()
    total_pages = 0
    while process_ids:
        pid = process_ids.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total_pages += rss_pages.get(pid, 0)
        process_ids.extend(children.get(pid, ()))
    page_size_kib = os.sysconf("SC_PAGE_SIZE") / 1024.0
    return total_pages * page_size_kib


def benchmark_row(
    row: dict[str, Any],
    profile_name: str,
    profile: dict[str, str],
    gpu_backend: str,
    *,
    run_index: int = 0,
    warmup_runs: int = 0,
) -> dict[str, Any]:
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="bigocrpdf_bench_") as temp_dir:
        output_pdf = Path(temp_dir) / "result.pdf"
        sidecar_json = Path(temp_dir) / "result.bigocr.json"
        result, peak_rss_mb = run_bigocrpdf(row, profile, output_pdf, sidecar_json, gpu_backend)
        elapsed = time.perf_counter() - started
        ground_truth_language = row.get("language")
        source_paths = row.get("_manifest_source_paths")
        logical_source = (
            source_paths.get("pdf") or source_paths.get("image")
            if isinstance(source_paths, dict)
            else row.get("pdf") or row.get("image")
        )

        record: dict[str, Any] = {
            "id": row.get("id"),
            "benchmark_profile": profile_name,
            "dataset": row.get("dataset") or "unknown",
            "manifest_sha256": row.get("_manifest_sha256"),
            "arquivo_origem": logical_source,
            "page_index": 0,
            "run_index": run_index,
            "warmup_runs": warmup_runs,
            "ground_truth_language": ground_truth_language,
            "language": ground_truth_language,
            "requested_language_hint": "latin",
            "tags": _record_tags(row),
            "engine_type": profile["engine"],
            "gpu_backend": gpu_backend,
            "ocr_version": "PPOCRV6",
            "model_type": profile["model_type"],
            "rec_batch_num": int(profile["rec_batch_num"]),
            "dpi": int(profile["dpi"]),
            "retry_level": 0,
            "retry_pages": None,
            "ocr_seconds": None,
            "end_to_end_seconds": elapsed,
            "peak_rss_mb": peak_rss_mb,
            "peak_rss_method": "linux_proc_process_tree",
            "peak_rss_sample_interval_seconds": PROCESS_RSS_SAMPLE_INTERVAL_SECONDS,
            "gpu_mem_peak_mb": None,
            "pdf_output_size_bytes": output_pdf.stat().st_size if output_pdf.exists() else 0,
            "ocr_text_chars": None,
            "failure_reason": "",
            "benchmark_environment": _benchmark_environment(),
        }

        if result.returncode != 0 or not output_pdf.exists():
            record["text_layer_ok"] = False
            record["failure_reason"] = (result.stderr or result.stdout).strip()[:500]
            return record

        record.update(read_ocr_sidecar_metadata(sidecar_json, output_pdf))
        effective_model_type = str(record.get("effective_model_type") or profile["model_type"])
        record["model_supports_ground_truth_language"] = _ppocrv6_supports_language(
            str(ground_truth_language or ""),
            effective_model_type,
        )

        expected_path = Path(row["gt_text"]) if row.get("gt_text") else None
        try:
            validation = build_report(output_pdf, expected_path)
            record.update(validation)
        except Exception as exc:
            record["text_layer_ok"] = False
            record["failure_reason"] = str(exc)

        if expected_path and output_pdf.exists():
            expected = expected_path.read_text(encoding="utf-8")
            extracted_chars = int(record.get("extracted_pdf_text_chars") or 0)
            record.setdefault("char_error_rate", char_error_rate("", expected))
            record.setdefault("word_error_rate", word_error_rate("", expected))
            record.setdefault("levenshtein_ratio", levenshtein_ratio("", expected))
            record["ocr_text_chars"] = extracted_chars
        return record


@lru_cache(maxsize=1)
def _benchmark_environment() -> dict[str, Any]:
    dependencies = {
        distribution: version
        for distribution in BENCHMARK_DEPENDENCIES
        if (version := _distribution_version(distribution)) is not None
    }
    gpu_runtimes = {
        distribution: version
        for distribution in GPU_RUNTIME_DISTRIBUTIONS
        if (version := _distribution_version(distribution)) is not None
    }
    return {
        "schema_version": 1,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count() or 1,
        "dependencies": dependencies,
        "gpu_devices": _gpu_devices(),
        "gpu_runtimes": gpu_runtimes,
    }


def _gpu_devices() -> list[dict[str, str]]:
    """Return stable Linux GPU/driver identifiers without optional Python APIs."""
    devices: list[dict[str, str]] = []
    drm_root = Path("/sys/class/drm")
    try:
        card_paths = sorted(drm_root.glob("card[0-9]*"))
    except OSError:
        return devices

    seen_devices: set[Path] = set()
    for card_path in card_paths:
        device_path = card_path / "device"
        try:
            resolved_device = device_path.resolve(strict=True)
            if resolved_device in seen_devices:
                continue
            seen_devices.add(resolved_device)
            pci_class = _read_sysfs_value(device_path / "class")
            if pci_class and not pci_class.lower().startswith("0x03"):
                continue
            driver_path = device_path / "driver"
            driver = driver_path.resolve(strict=True).name
            driver_version = _read_sysfs_value(Path("/sys/module") / driver / "version")
            devices.append(
                {
                    "card": card_path.name,
                    "pci_address": resolved_device.name,
                    "vendor_id": _read_sysfs_value(device_path / "vendor") or "unknown",
                    "device_id": _read_sysfs_value(device_path / "device") or "unknown",
                    "driver": driver,
                    "driver_version": driver_version or platform.release(),
                }
            )
        except OSError:
            continue
    return devices


def _read_sysfs_value(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    return value or None


def _cpu_model() -> str:
    processor = platform.processor().strip()
    if processor:
        return processor
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() in {"model name", "hardware"}:
                model = value.strip()
                if model:
                    return model
    except OSError:
        pass
    return platform.machine() or "unknown"


def _distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        module_name = DEPENDENCY_MODULES.get(distribution)
        if module_name is None:
            return None
        try:
            module = importlib.import_module(module_name)
        except (ImportError, OSError):
            return None
        version = getattr(module, "__version__", None)
        return str(version) if version else None


def _ppocrv6_supports_language(language: str, model_type: str) -> bool | None:
    normalized = language.strip().lower().replace("-", "_")
    if not normalized:
        return None
    normalized = PPOCRV6_LANGUAGE_ALIASES.get(normalized, normalized)
    if normalized not in PPOCRV6_LANGUAGES:
        return False
    return not (model_type.lower() == "tiny" and normalized == "japan")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=64)
def _sha256_optional_path(raw_path: str | None) -> str | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    try:
        if not path.is_file():
            return None
        return _sha256_file(path)
    except OSError:
        return None


def read_ocr_sidecar_metadata(sidecar_json: Path, output_pdf: Path) -> dict[str, Any]:
    """Read structured OCR JSON diagnostics into benchmark fields."""
    try:
        document = load_ocr_document_json(sidecar_json, output_pdf)
    except ValueError as exc:
        return {"ocr_sidecar_error": str(exc)}
    if document is None:
        return {"ocr_sidecar_error": "missing OCR JSON"}

    runtime = document.diagnostics.get("ocr_runtime") or {}
    if not isinstance(runtime, dict):
        runtime = {}
    rec_model_path = cast(str | None, runtime.get("rec_model_path"))
    det_model_path = cast(str | None, runtime.get("det_model_path"))
    font_path = cast(str | None, runtime.get("font_path"))
    rec_keys_path = cast(str | None, runtime.get("rec_keys_path"))
    layout_blocks = [block for page in document.pages for block in page.layout_blocks]
    # Per-region data has always been in the sidecar; it was simply discarded
    # here, leaving ocr_box_count and the confidence fields permanently null.
    confidences = [
        float(result.confidence) for page in document.pages for result in page.text_results
    ]
    confidence_summary = aggregate_confidence(confidences)
    pages_with_zero_boxes = sum(1 for page in document.pages if not page.text_results)
    auto_verified_pages = [
        cast(dict[str, Any], page_data)
        for page in document.pages
        if isinstance(page_data := page.diagnostics.get("auto_verified"), dict)
    ]
    return {
        "effective_ocr_runtime": runtime,
        "effective_ocr_config": {
            key: value
            for key, value in runtime.items()
            if key
            not in {
                "rec_model_path",
                "rec_keys_path",
                "det_model_path",
                "font_path",
            }
        },
        "ocr_box_count": len(confidences),
        "ocr_confidence_mean": confidence_summary["mean"],
        "ocr_confidence_median": confidence_summary["median"],
        "ocr_confidence_p10": confidence_summary["p10"],
        "ocr_confidence_min": confidence_summary["min"],
        # The invariant that would have caught the real-world zero-OCR failure:
        # a CER budget cannot notice a page that produced nothing, because the
        # document average absorbs it.
        "pages_with_zero_boxes": pages_with_zero_boxes,
        "preprocess_traces": [
            page.diagnostics.get("preprocess")
            for page in document.pages
            if page.diagnostics.get("preprocess")
        ],
        "effective_engine_type": runtime.get("engine_type"),
        "effective_ocr_version": runtime.get("ocr_version"),
        "effective_language_hint": runtime.get("language"),
        "effective_model_type": runtime.get("model_type"),
        "effective_rec_batch_num": runtime.get("rec_batch_num"),
        "effective_gpu_backend": runtime.get("gpu_backend"),
        "effective_rec_model_family": runtime.get("ocr_version"),
        "effective_rec_model_path": rec_model_path,
        "effective_rec_model_sha256": _sha256_optional_path(rec_model_path),
        "effective_det_model_path": det_model_path,
        "effective_det_model_sha256": _sha256_optional_path(det_model_path),
        "effective_dictionary_path": rec_keys_path,
        "effective_dictionary_sha256": _sha256_optional_path(rec_keys_path),
        "effective_font_path": font_path,
        "effective_font_sha256": _sha256_optional_path(font_path),
        "sidecar_page_count": len(document.pages),
        "sidecar_layout_block_count": len(layout_blocks),
        "sidecar_table_block_count": sum(1 for block in layout_blocks if block.kind == "table"),
        "auto_verified_accepted_lines": sum(
            int(page_data.get("accepted_lines") or 0) for page_data in auto_verified_pages
        ),
        "auto_verified_rejected_lines": sum(
            int(page_data.get("rejected_lines") or 0) for page_data in auto_verified_pages
        ),
    }


def parse_matrix_arg(raw: str) -> tuple[str, list[str]]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("matrix entries must use key=value1,value2")
    key, values_raw = raw.split("=", 1)
    key = key.strip()
    if key not in MATRIX_KEYS:
        raise argparse.ArgumentTypeError(f"unsupported matrix key: {key}")
    values = [value.strip() for value in values_raw.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"matrix key {key} has no values")
    return key, values


def expand_profiles(
    base_profile_name: str,
    matrix: list[tuple[str, list[str]]],
) -> list[tuple[str, dict[str, str]]]:
    base = dict(PROFILES[base_profile_name])
    if not matrix:
        return [(base_profile_name, base)]

    keys = [key for key, _values in matrix]
    value_lists = [values for _key, values in matrix]
    expanded: list[tuple[str, dict[str, str]]] = []
    for values in product(*value_lists):
        profile = dict(base)
        profile_name_parts = [base_profile_name]
        for key, value in zip(keys, values, strict=True):
            profile[key] = value
            profile_name_parts.append(f"{key}-{value}")
        expanded.append(("_".join(profile_name_parts), profile))
    return expanded


def benchmark_matrix(
    matrix: list[tuple[str, list[str]]],
    rec_batch_sweep: bool,
) -> list[tuple[str, list[str]]]:
    """Return the effective benchmark matrix requested by CLI flags."""
    effective = list(matrix)
    if rec_batch_sweep and not any(key == "rec_batch_num" for key, _values in effective):
        effective.append(("rec_batch_num", REC_BATCH_SWEEP_VALUES))
    return effective


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate benchmark records by benchmark profile."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("benchmark_profile") or "unknown")].append(record)

    return [
        _summary_for_records(profile_name, "profile", profile_records)
        for profile_name, profile_records in sorted(grouped.items())
    ]


def summarize_records_by_dataset_and_tag(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate benchmark records by dataset and tag within each profile."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        profile_name = str(record.get("benchmark_profile") or "unknown")
        dataset = str(record.get("dataset") or "unknown")
        grouped[(profile_name, f"dataset:{dataset}")].append(record)
        for tag in _record_tags(record):
            grouped[(profile_name, f"tag:{tag}")].append(record)
    return [
        _summary_for_records(profile_name, summary_group, group_records)
        for (profile_name, summary_group), group_records in sorted(grouped.items())
    ]


def _summary_for_records(
    profile_name: str,
    summary_group: str,
    profile_records: list[dict[str, Any]],
) -> dict[str, Any]:
    text_layer_ok_pages = sum(1 for record in profile_records if record.get("text_layer_ok"))
    failed_pages = sum(1 for record in profile_records if record.get("failure_reason"))
    successful_pages = len(profile_records) - failed_pages
    return {
        "benchmark_profile": profile_name,
        "summary_group": summary_group,
        "pages": len(profile_records),
        "successful_pages": successful_pages,
        "failed_pages": failed_pages,
        "text_layer_ok_pages": text_layer_ok_pages,
        "text_layer_ok_percent": _percent(text_layer_ok_pages, len(profile_records)),
        "mean_cer": _mean_metric(profile_records, "char_error_rate"),
        "micro_cer": _micro_error_rate(
            profile_records,
            "char_edit_distance",
            "expected_char_count",
        ),
        "mean_wer": _mean_metric(profile_records, "word_error_rate"),
        "micro_wer": _micro_error_rate(
            profile_records,
            "word_edit_distance",
            "expected_word_count",
        ),
        "mean_levenshtein_ratio": _mean_metric(profile_records, "levenshtein_ratio"),
        "mean_seconds_page": _mean_metric(profile_records, "end_to_end_seconds"),
        "median_seconds_page": _median_metric(profile_records, "end_to_end_seconds"),
        "p95_seconds_page": _percentile_metric(profile_records, "end_to_end_seconds", 0.95),
        "peak_rss_mb": _max_metric(profile_records, "peak_rss_mb"),
        "mean_pdf_size_bytes": _mean_metric(profile_records, "pdf_output_size_bytes"),
    }


def _record_tags(record: dict[str, Any]) -> list[str]:
    raw_tags = record.get("tags") or []
    if not isinstance(raw_tags, list):
        return []
    return [str(tag) for tag in raw_tags if str(tag)]


def _summary_group(summary: dict[str, Any]) -> str:
    return str(summary.get("summary_group") or "profile")


def _summary_table_lines(summaries: list[dict[str, Any]], include_group: bool) -> list[str]:
    group_header = " | group" if include_group else ""
    group_align = " |---" if include_group else ""
    return [
        f"| profile{group_header} | pages | ok text layer | macro CER | micro CER | macro WER | micro WER | median seconds/page | p95 seconds/page | peak RSS MB |",
        f"|---{group_align}|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]


def write_summary_csv(
    path: Path,
    summaries: list[dict[str, Any]],
    *,
    source_sha256: str | None = None,
) -> None:
    """Write benchmark summaries as CSV."""
    csv_output = io.StringIO(newline="")
    writer = csv.DictWriter(csv_output, fieldnames=SUMMARY_COLUMNS)
    writer.writeheader()
    for summary in summaries:
        output_row = {column: summary.get(column) for column in SUMMARY_COLUMNS}
        output_row["source_jsonl_sha256"] = source_sha256 or ""
        writer.writerow(output_row)
    _write_text_atomically(path, csv_output.getvalue())


def write_summary_markdown(
    path: Path,
    summaries: list[dict[str, Any]],
    *,
    source_sha256: str | None = None,
) -> None:
    """Write benchmark summaries as Markdown."""
    lines = [
        "# BigOCRPDF benchmark summary",
        "",
    ]
    if source_sha256:
        lines.extend([f"Source JSONL SHA-256: `{source_sha256}`", ""])
    lines.extend(_summary_table_lines(summaries, include_group=False))
    for summary in summaries:
        lines.append(_summary_markdown_row(summary, include_group=False))
    lines.extend(
        [
            "",
            "CSV companion: same basename with `.summary.csv` unless overridden.",
            "",
        ]
    )
    _write_text_atomically(path, "\n".join(lines))


def write_group_summary_markdown(
    path: Path,
    summaries: list[dict[str, Any]],
    *,
    source_sha256: str | None = None,
) -> None:
    """Write dataset/tag benchmark summaries as Markdown."""
    lines = [
        "# BigOCRPDF grouped benchmark summary",
        "",
        "Rows are grouped by dataset and tag within each benchmark profile.",
        "",
    ]
    if source_sha256:
        lines.extend([f"Source JSONL SHA-256: `{source_sha256}`", ""])
    lines.extend(_summary_table_lines(summaries, include_group=True))
    for summary in summaries:
        lines.append(_summary_markdown_row(summary, include_group=True))
    lines.append("")
    _write_text_atomically(path, "\n".join(lines))


def write_group_summary_csv(
    path: Path,
    summaries: list[dict[str, Any]],
    *,
    source_sha256: str | None = None,
) -> None:
    """Write dataset/tag benchmark summaries as CSV."""
    write_summary_csv(path, summaries, source_sha256=source_sha256)


def _summary_markdown_row(summary: dict[str, Any], include_group: bool) -> str:
    group_cell = f" | {_summary_group(summary)}" if include_group else ""
    return (
        "| {profile}{group} | {pages} | {ok_pages}/{pages} ({ok_percent}) | {cer} | "
        "{micro_cer} | {wer} | {micro_wer} | {median_seconds} | {p95_seconds} | "
        "{peak_rss} |"
    ).format(
        profile=summary["benchmark_profile"],
        group=group_cell,
        pages=summary["pages"],
        ok_pages=summary["text_layer_ok_pages"],
        ok_percent=_format_metric(summary["text_layer_ok_percent"]),
        cer=_format_metric(summary["mean_cer"]),
        micro_cer=_format_metric(summary.get("micro_cer")),
        wer=_format_metric(summary["mean_wer"]),
        micro_wer=_format_metric(summary.get("micro_wer")),
        median_seconds=_format_metric(
            summary.get("median_seconds_page", summary["mean_seconds_page"])
        ),
        p95_seconds=_format_metric(summary["p95_seconds_page"]),
        peak_rss=_format_metric(summary["peak_rss_mb"]),
    )


def _mean_metric(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return None
    return fmean(values)


def _median_metric(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return median(values) if values else None


def _micro_error_rate(
    records: list[dict[str, Any]],
    distance_key: str,
    expected_count_key: str,
) -> float | None:
    pairs = [
        (float(record[distance_key]), float(record[expected_count_key]))
        for record in records
        if record.get(distance_key) is not None and record.get(expected_count_key) is not None
    ]
    if not pairs:
        return None
    expected_total = sum(expected_count for _distance, expected_count in pairs)
    if expected_total <= 0:
        return None
    return sum(distance for distance, _expected_count in pairs) / expected_total


def _percentile_metric(records: list[dict[str, Any]], key: str, percentile: float) -> float | None:
    values = sorted(float(record[key]) for record in records if record.get(key) is not None)
    if not values:
        return None
    index = min(len(values) - 1, max(0, round((len(values) - 1) * percentile)))
    return values[index]


def _max_metric(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return max(values) if values else None


def _percent(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return part / total * 100.0


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(directory, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_text_atomically(path: Path, payload: str) -> None:
    """Replace a generated report without following the destination symlink."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8", newline="") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            os.close(temp_fd)
        except OSError:
            pass
        temp_path.unlink(missing_ok=True)
        raise


def _write_jsonl_atomically(path: Path, records: list[dict[str, Any]]) -> str:
    """Publish a complete benchmark result without truncating an older baseline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(temp_name)
    digest = hashlib.sha256()
    try:
        with os.fdopen(temp_fd, "wb") as output:
            for record in records:
                payload = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
                digest.update(payload)
                output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
        return digest.hexdigest()
    except Exception:
        try:
            os.close(temp_fd)
        except OSError:
            pass
        temp_path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="balanced_cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Measured runs per sample/profile (default: 3).",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Unrecorded warm-up runs per sample/profile (default: 0).",
    )
    parser.add_argument("--gpu-backend", default="off")
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--group-summary-md", type=Path)
    parser.add_argument("--group-summary-csv", type=Path)
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--no-group-summary", action="store_true")
    parser.add_argument(
        "--rec-batch-sweep",
        action="store_true",
        help="Add the recommended rec_batch_num=1,2,4,8,16 benchmark matrix.",
    )
    parser.add_argument(
        "--matrix",
        action="append",
        type=parse_matrix_arg,
        default=[],
        help="Sweep a benchmark dimension, e.g. --matrix engine=openvino,onnxruntime",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")

    matrix = benchmark_matrix(args.matrix, args.rec_batch_sweep)
    profiles = expand_profiles(args.profile, matrix)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with load_manifest(args.manifest, args.limit) as rows:
        for profile_name, profile in profiles:
            gpu_backend = str(profile.get("gpu_backend") or args.gpu_backend or "off")
            for row in rows:
                for warmup_index in range(args.warmup_runs):
                    benchmark_row(
                        row,
                        profile_name,
                        profile,
                        gpu_backend,
                        run_index=-(warmup_index + 1),
                        warmup_runs=args.warmup_runs,
                    )
                for run_index in range(args.repeats):
                    record = benchmark_row(
                        row,
                        profile_name,
                        profile,
                        gpu_backend,
                        run_index=run_index,
                        warmup_runs=args.warmup_runs,
                    )
                    records.append(record)
    source_sha256 = _write_jsonl_atomically(args.out, records)
    if not args.no_summary:
        summaries = summarize_records(records)
        summary_md = args.summary_md or args.out.with_suffix(".summary.md")
        summary_csv = args.summary_csv or args.out.with_suffix(".summary.csv")
        write_summary_markdown(summary_md, summaries, source_sha256=source_sha256)
        write_summary_csv(summary_csv, summaries, source_sha256=source_sha256)
    if not args.no_group_summary:
        group_summaries = summarize_records_by_dataset_and_tag(records)
        group_summary_md = args.group_summary_md or args.out.with_suffix(".groups.summary.md")
        group_summary_csv = args.group_summary_csv or args.out.with_suffix(".groups.summary.csv")
        write_group_summary_markdown(
            group_summary_md,
            group_summaries,
            source_sha256=source_sha256,
        )
        write_group_summary_csv(
            group_summary_csv,
            group_summaries,
            source_sha256=source_sha256,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
