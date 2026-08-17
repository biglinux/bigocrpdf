import argparse
import csv
import hashlib
import os
import signal
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from benchmarks import ocr_benchmark, validate_text_layer
from benchmarks.ocr_benchmark import (
    _benchmark_environment,
    _ppocrv6_supports_language,
    _sample_process_tree_rss_kib,
    _write_jsonl_atomically,
    benchmark_matrix,
    benchmark_row,
    expand_profiles,
    load_manifest,
    parse_matrix_arg,
    read_ocr_sidecar_metadata,
    run_bigocrpdf,
    summarize_records,
    summarize_records_by_dataset_and_tag,
    write_group_summary_csv,
    write_group_summary_markdown,
    write_summary_csv,
    write_summary_markdown,
)
from benchmarks.prepare_benchmark_datasets import write_manifest
from bigocrpdf.services.rapidocr_service.config import OcrDocument, OcrLayoutBlock, OcrPage
from bigocrpdf.services.rapidocr_service.ocr_document_io import write_ocr_document_json


def _write_proc_stat(proc_root: Path, pid: int, ppid: int, rss_pages: int) -> None:
    process_dir = proc_root / str(pid)
    process_dir.mkdir()
    fields_after_command = ["S", str(ppid), *(["0"] * 19), str(rss_pages)]
    process_dir.joinpath("stat").write_text(
        f"{pid} (benchmark worker) {' '.join(fields_after_command)}\n",
        encoding="ascii",
    )


def test_peak_rss_samples_complete_process_tree(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _write_proc_stat(proc_root, 100, 1, 2)
    _write_proc_stat(proc_root, 101, 100, 3)
    _write_proc_stat(proc_root, 102, 1, 100)

    expected_kib = 5 * os.sysconf("SC_PAGE_SIZE") / 1024
    assert _sample_process_tree_rss_kib(100, proc_root) == expected_kib


@pytest.mark.parametrize(
    ("language", "model_type", "expected"),
    [
        ("pt_BR", "small", True),
        ("zh", "small", True),
        ("ja", "small", True),
        ("ja", "tiny", False),
        ("ar", "medium", False),
        ("el", "small", False),
    ],
)
def test_ppocrv6_language_support_is_reported_truthfully(
    language: str, model_type: str, expected: bool
) -> None:
    assert _ppocrv6_supports_language(language, model_type) is expected


def test_load_manifest_rejects_material_changed_after_publication(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    ground_truth_path = tmp_path / "sample.txt"
    pdf_path.write_bytes(b"%PDF-1.7\n")
    ground_truth_path.write_text("expected", encoding="utf-8")
    manifest_path = write_manifest(
        tmp_path,
        [
            {
                "id": "sample",
                "dataset": "synthetic",
                "pdf": pdf_path.name,
                "gt_text": ground_truth_path.name,
                "language": "en",
                "tags": [],
            }
        ],
    )
    ground_truth_path.write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        with load_manifest(manifest_path, None):
            pass


def test_load_manifest_hashes_and_parses_one_file_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "sample.pdf"
    ground_truth_path = tmp_path / "sample.txt"
    pdf_path.write_bytes(b"%PDF-1.7\n")
    ground_truth_path.write_text("expected", encoding="utf-8")
    manifest_path = write_manifest(
        tmp_path,
        [
            {
                "id": "original",
                "dataset": "synthetic",
                "pdf": pdf_path.name,
                "gt_text": ground_truth_path.name,
                "language": "en",
                "tags": [],
            }
        ],
    )
    original_manifest = manifest_path.read_bytes()
    original_sha256 = hashlib.sha256(original_manifest).hexdigest()
    replacement = original_manifest.replace(b'"original"', b'"replaced"')
    real_sha256_file = ocr_benchmark._sha256_file

    def replace_manifest_after_hash(path: Path) -> str:
        digest = real_sha256_file(path)
        if path == manifest_path:
            manifest_path.write_bytes(replacement)
        return digest

    monkeypatch.setattr(ocr_benchmark, "_sha256_file", replace_manifest_after_hash)

    with load_manifest(manifest_path, None) as rows:
        assert rows[0]["id"] == "original"
        assert rows[0]["_manifest_sha256"] == original_sha256


@pytest.mark.parametrize("sample_id", ("../escape", "/absolute", "nested/sample", "bad\nid"))
def test_load_manifest_rejects_unsafe_sample_id(tmp_path: Path, sample_id: str) -> None:
    pdf_path = tmp_path / "sample.pdf"
    ground_truth_path = tmp_path / "sample.txt"
    pdf_path.write_bytes(b"%PDF-1.7\n")
    ground_truth_path.write_text("expected", encoding="utf-8")
    with pytest.raises(ValueError, match="sample id"):
        manifest_path = write_manifest(
            tmp_path,
            [
                {
                    "id": sample_id,
                    "dataset": "synthetic",
                    "pdf": pdf_path.name,
                    "gt_text": ground_truth_path.name,
                    "language": "en",
                    "tags": [],
                }
            ],
        )
        with load_manifest(manifest_path, None):
            pass


def test_manifest_snapshot_pins_material_bytes_after_source_replacement(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "sample.pdf"
    ground_truth_path = tmp_path / "sample.txt"
    pdf_path.write_bytes(b"%PDF-1.7 original\n")
    ground_truth_path.write_text("expected", encoding="utf-8")
    manifest_path = write_manifest(
        tmp_path,
        [
            {
                "id": "sample",
                "dataset": "synthetic",
                "pdf": pdf_path.name,
                "gt_text": ground_truth_path.name,
                "language": "en",
                "tags": [],
            }
        ],
    )

    with load_manifest(manifest_path, None) as rows:
        snapshot_pdf = Path(rows[0]["pdf"])
        snapshot_text = Path(rows[0]["gt_text"])
        pdf_path.write_bytes(b"%PDF-1.7 replacement\n")
        ground_truth_path.write_text("changed", encoding="utf-8")

        assert snapshot_pdf.read_bytes() == b"%PDF-1.7 original\n"
        assert snapshot_text.read_text(encoding="utf-8") == "expected"
        assert rows[0]["_manifest_source_paths"]["pdf"] == "sample.pdf"

    assert not snapshot_pdf.exists()
    assert not snapshot_text.exists()


def test_manifest_snapshot_rejects_symlinked_material(tmp_path: Path) -> None:
    target = tmp_path / "target.pdf"
    target.write_bytes(b"%PDF-1.7\n")
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.symlink_to(target)
    ground_truth_path = tmp_path / "sample.txt"
    ground_truth_path.write_text("expected", encoding="utf-8")
    manifest_path = write_manifest(
        tmp_path,
        [
            {
                "id": "sample",
                "dataset": "synthetic",
                "pdf": pdf_path.name,
                "gt_text": ground_truth_path.name,
                "language": "en",
                "tags": [],
            }
        ],
    )

    with pytest.raises(ValueError, match="safe regular file"):
        with load_manifest(manifest_path, None):
            pass


def test_benchmark_output_path_never_uses_untrusted_sample_id(monkeypatch) -> None:
    captured_paths: list[Path] = []

    def fake_run(_row, _profile, output_pdf, _sidecar_json, _gpu_backend):
        captured_paths.append(output_pdf)
        return subprocess.CompletedProcess([], 1, "", "failed"), 10.0

    monkeypatch.setattr("benchmarks.ocr_benchmark.run_bigocrpdf", fake_run)

    benchmark_row(
        {
            "id": "../../escape",
            "dataset": "synthetic",
            "pdf": "/input.pdf",
            "language": "en",
        },
        "balanced_cpu",
        {
            "engine": "openvino",
            "model_type": "small",
            "rec_batch_num": "1",
            "dpi": "300",
        },
        "off",
    )

    assert captured_paths[0].name == "result.pdf"


def test_atomic_jsonl_publication_preserves_previous_result_on_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "baseline.jsonl"
    destination.write_text('{"old": true}\n', encoding="utf-8")
    monkeypatch.setattr(
        "benchmarks.ocr_benchmark.os.replace",
        MagicMock(side_effect=OSError("interrupted")),
    )

    with pytest.raises(OSError, match="interrupted"):
        _write_jsonl_atomically(destination, [{"new": True}])

    assert destination.read_text(encoding="utf-8") == '{"old": true}\n'
    assert list(tmp_path.glob(".baseline.jsonl.*.tmp")) == []


def test_atomic_jsonl_returns_hash_of_the_bytes_it_published(tmp_path: Path) -> None:
    destination = tmp_path / "baseline.jsonl"
    records = [{"text": "ação"}, {"value": 2}]
    expected = ('{"text": "ação"}\n{"value": 2}\n').encode()

    digest = _write_jsonl_atomically(destination, records)

    assert destination.read_bytes() == expected
    assert digest == hashlib.sha256(expected).hexdigest()


def test_atomic_jsonl_hash_cannot_be_relabelled_by_concurrent_replacement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "baseline.jsonl"
    expected = b'{"owner": "first"}\n'
    replacement = b'{"owner": "second"}\n'
    real_replace = os.replace

    def replace_then_race(source, target):
        real_replace(source, target)
        Path(target).write_bytes(replacement)

    monkeypatch.setattr(ocr_benchmark.os, "replace", replace_then_race)

    digest = _write_jsonl_atomically(destination, [{"owner": "first"}])

    assert destination.read_bytes() == replacement
    assert digest == hashlib.sha256(expected).hexdigest()


def test_benchmark_environment_records_hardware_and_dependencies() -> None:
    _benchmark_environment.cache_clear()
    environment = _benchmark_environment()

    assert environment["schema_version"] == 1
    assert environment["cpu_count"]
    assert environment["cpu_model"]
    assert environment["dependencies"]["Pillow"]
    assert isinstance(environment["gpu_devices"], list)
    assert isinstance(environment["gpu_runtimes"], dict)


def test_run_bigocrpdf_reaps_process_group_when_interrupted(monkeypatch) -> None:
    class InterruptedProcess:
        pid = 4242
        returncode = None

        def __init__(self) -> None:
            self.communicate_calls = 0

        def communicate(self, timeout=None):
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                raise KeyboardInterrupt
            if self.communicate_calls == 2:
                raise subprocess.TimeoutExpired(["ocr"], timeout)
            self.returncode = -int(signal.SIGKILL)
            return "", ""

    process = InterruptedProcess()
    termination_signals: list[signal.Signals] = []
    monkeypatch.setattr(ocr_benchmark.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(ocr_benchmark, "_sample_process_tree_rss_kib", lambda _pid: None)
    monkeypatch.setattr(
        ocr_benchmark,
        "_terminate_process_group",
        lambda _process, signal_number: termination_signals.append(signal_number),
    )

    with pytest.raises(KeyboardInterrupt):
        run_bigocrpdf(
            {"id": "sample", "pdf": "/input.pdf"},
            {
                "dpi": "300",
                "engine": "openvino",
                "model_type": "small",
                "rec_batch_num": "1",
            },
            Path("/output.pdf"),
            Path("/output.bigocr.json"),
            "off",
        )

    assert termination_signals == [signal.SIGTERM, signal.SIGKILL]
    assert process.communicate_calls == 3


def test_parse_matrix_arg_accepts_known_key_values() -> None:
    key, values = parse_matrix_arg("engine=openvino,onnxruntime")

    assert key == "engine"
    assert values == ["openvino", "onnxruntime"]


def test_parse_matrix_arg_rejects_unknown_key() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_matrix_arg("backend=rapidocr")


def test_expand_profiles_builds_cartesian_product() -> None:
    profiles = expand_profiles(
        "balanced_cpu",
        [
            ("engine", ["openvino", "onnxruntime"]),
            ("rec_batch_num", ["1", "4"]),
        ],
    )

    assert len(profiles) == 4
    assert profiles[0][1]["engine"] == "openvino"
    assert profiles[-1][1]["engine"] == "onnxruntime"
    assert profiles[-1][1]["rec_batch_num"] == "4"


def test_expand_profiles_can_set_gpu_backend() -> None:
    profiles = expand_profiles("balanced_cpu", [("gpu_backend", ["paddle"])])

    assert profiles[0][1]["gpu_backend"] == "paddle"


def test_benchmark_matrix_adds_rec_batch_sweep_once() -> None:
    assert benchmark_matrix([], rec_batch_sweep=True) == [
        ("rec_batch_num", ["1", "2", "4", "8", "16"])
    ]
    assert benchmark_matrix(
        [("rec_batch_num", ["2"])],
        rec_batch_sweep=True,
    ) == [("rec_batch_num", ["2"])]


def test_read_ocr_sidecar_metadata_exposes_runtime_and_layout_counts(tmp_path: Path) -> None:
    output_pdf = tmp_path / "out.pdf"
    output_pdf.write_bytes(b"%PDF-1.7\nbenchmark")
    document = OcrDocument(
        diagnostics={
            "ocr_runtime": {
                "engine_type": "openvino",
                "ocr_version": "PPOCRV6",
                "model_type": "small",
                "rec_batch_num": 4,
                "gpu_backend": "off",
                "language": "latin",
                "rec_model_path": "/models/latin_rec.onnx",
                "det_model_path": "/models/ch_det.onnx",
            }
        },
        pages=[
            OcrPage(
                page_index=1,
                width_px=800,
                height_px=1000,
                dpi=300,
                diagnostics={"auto_verified": {"accepted_lines": 2, "rejected_lines": 1}},
                layout_blocks=[
                    OcrLayoutBlock(kind="table", rows=[["A", "B"], ["1", "2"]]),
                    OcrLayoutBlock(kind="paragraph", text="Body"),
                ],
            )
        ],
    )
    sidecar_json = tmp_path / "result.bigocr.json"
    write_ocr_document_json(document, output_pdf, sidecar_json)

    metadata = read_ocr_sidecar_metadata(sidecar_json, output_pdf)

    assert metadata["effective_engine_type"] == "openvino"
    assert metadata["effective_rec_batch_num"] == 4
    assert metadata["effective_language_hint"] == "latin"
    assert metadata["sidecar_page_count"] == 1
    assert metadata["sidecar_layout_block_count"] == 2
    assert metadata["sidecar_table_block_count"] == 1
    assert metadata["auto_verified_accepted_lines"] == 2


def test_summarize_records_by_dataset_and_tag_tracks_peak_rss() -> None:
    summaries = summarize_records_by_dataset_and_tag(
        [
            {
                "benchmark_profile": "balanced_cpu",
                "dataset": "DharmaOCR",
                "tags": ["form", "english"],
                "text_layer_ok": True,
                "char_error_rate": 0.1,
                "word_error_rate": 0.2,
                "levenshtein_ratio": 0.9,
                "end_to_end_seconds": 1.0,
                "peak_rss_mb": 80.0,
                "pdf_output_size_bytes": 1000,
                "failure_reason": "",
            },
            {
                "benchmark_profile": "balanced_cpu",
                "dataset": "DharmaOCR",
                "tags": ["form"],
                "text_layer_ok": False,
                "char_error_rate": 0.3,
                "word_error_rate": 0.4,
                "levenshtein_ratio": 0.7,
                "end_to_end_seconds": 3.0,
                "peak_rss_mb": 140.0,
                "pdf_output_size_bytes": 2000,
                "failure_reason": "empty extracted text",
            },
        ]
    )

    by_group = {summary["summary_group"]: summary for summary in summaries}
    assert by_group["dataset:DharmaOCR"]["pages"] == 2
    assert by_group["dataset:DharmaOCR"]["peak_rss_mb"] == 140.0
    assert by_group["tag:english"]["pages"] == 1
    assert by_group["tag:form"]["text_layer_ok_percent"] == 50.0


def test_summarize_records_groups_profiles_and_metrics() -> None:
    summaries = summarize_records(
        [
            {
                "benchmark_profile": "balanced_cpu",
                "text_layer_ok": True,
                "char_error_rate": 0.1,
                "word_error_rate": 0.2,
                "levenshtein_ratio": 0.9,
                "end_to_end_seconds": 1.0,
                "peak_rss_mb": 80.0,
                "pdf_output_size_bytes": 1000,
                "failure_reason": "",
            },
            {
                "benchmark_profile": "balanced_cpu",
                "text_layer_ok": False,
                "char_error_rate": 0.3,
                "word_error_rate": 0.4,
                "levenshtein_ratio": 0.7,
                "end_to_end_seconds": 3.0,
                "peak_rss_mb": 120.0,
                "pdf_output_size_bytes": 2000,
                "failure_reason": "empty extracted text",
            },
        ]
    )

    assert summaries[0]["benchmark_profile"] == "balanced_cpu"
    assert summaries[0]["pages"] == 2
    assert summaries[0]["successful_pages"] == 1
    assert summaries[0]["failed_pages"] == 1
    assert summaries[0]["text_layer_ok_pages"] == 1
    assert summaries[0]["text_layer_ok_percent"] == 50.0
    assert summaries[0]["mean_cer"] == pytest.approx(0.2)
    assert summaries[0]["peak_rss_mb"] == 120.0


def test_summarize_records_reports_micro_error_rate_and_median_latency() -> None:
    summaries = summarize_records(
        [
            {
                "benchmark_profile": "balanced_cpu",
                "text_layer_ok": True,
                "char_error_rate": 1.0,
                "char_edit_distance": 1,
                "expected_char_count": 1,
                "word_error_rate": 1.0,
                "word_edit_distance": 1,
                "expected_word_count": 1,
                "end_to_end_seconds": 1.0,
                "failure_reason": "",
            },
            {
                "benchmark_profile": "balanced_cpu",
                "text_layer_ok": True,
                "char_error_rate": 0.0,
                "char_edit_distance": 0,
                "expected_char_count": 99,
                "word_error_rate": 0.0,
                "word_edit_distance": 0,
                "expected_word_count": 9,
                "end_to_end_seconds": 10.0,
                "failure_reason": "",
            },
            {
                "benchmark_profile": "balanced_cpu",
                "text_layer_ok": True,
                "char_error_rate": 0.0,
                "char_edit_distance": 0,
                "expected_char_count": 100,
                "word_error_rate": 0.0,
                "word_edit_distance": 0,
                "expected_word_count": 10,
                "end_to_end_seconds": 2.0,
                "failure_reason": "",
            },
        ]
    )

    assert summaries[0]["mean_cer"] == pytest.approx(1 / 3)
    assert summaries[0]["micro_cer"] == pytest.approx(1 / 200)
    assert summaries[0]["micro_wer"] == pytest.approx(1 / 20)
    assert summaries[0]["median_seconds_page"] == 2.0


def test_text_layer_question_mark_is_not_counted_as_unicode_loss(
    monkeypatch, tmp_path: Path
) -> None:
    expected_path = tmp_path / "expected.txt"
    expected_path.write_text("What? �□", encoding="utf-8")
    monkeypatch.setattr(
        validate_text_layer,
        "extract_pdf_text_with_method",
        lambda _pdf_path: ("What? �□", "pdftotext", "24.02"),
    )

    report = validate_text_layer.build_report(tmp_path / "result.pdf", expected_path)

    assert report["unicode_loss_count"] == 2


def test_text_layer_report_identifies_the_extractor(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        validate_text_layer,
        "extract_pdf_text_with_method",
        lambda _pdf_path: ("text", "pdftotext", "24.02"),
    )

    report = validate_text_layer.build_report(tmp_path / "result.pdf", None)

    assert report["text_extractor"] == "pdftotext"
    assert report["text_extractor_version"] == "24.02"


def test_summary_writers_create_readable_markdown_and_csv(tmp_path: Path) -> None:
    summaries = [
        {
            "benchmark_profile": "balanced_cpu",
            "summary_group": "profile",
            "pages": 2,
            "successful_pages": 2,
            "failed_pages": 0,
            "text_layer_ok_pages": 2,
            "text_layer_ok_percent": 100.0,
            "mean_cer": 0.0,
            "mean_wer": 0.0,
            "mean_levenshtein_ratio": 1.0,
            "mean_seconds_page": 1.2,
            "p95_seconds_page": 1.4,
            "peak_rss_mb": 90.0,
            "mean_pdf_size_bytes": 1234.0,
        }
    ]
    markdown_path = tmp_path / "summary.md"
    csv_path = tmp_path / "summary.csv"

    source_sha256 = "a" * 64
    write_summary_markdown(markdown_path, summaries, source_sha256=source_sha256)
    write_summary_csv(csv_path, summaries, source_sha256=source_sha256)

    assert "| balanced_cpu | 2 | 2/2" in markdown_path.read_text(encoding="utf-8")
    assert source_sha256 in markdown_path.read_text(encoding="utf-8")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert rows[0]["benchmark_profile"] == "balanced_cpu"
    assert rows[0]["text_layer_ok_percent"] == "100.0"
    assert rows[0]["source_jsonl_sha256"] == source_sha256


def test_group_summary_writers_include_group_column(tmp_path: Path) -> None:
    summaries = [
        {
            "benchmark_profile": "balanced_cpu",
            "summary_group": "dataset:DharmaOCR",
            "pages": 2,
            "successful_pages": 2,
            "failed_pages": 0,
            "text_layer_ok_pages": 2,
            "text_layer_ok_percent": 100.0,
            "mean_cer": 0.0,
            "mean_wer": 0.0,
            "mean_levenshtein_ratio": 1.0,
            "mean_seconds_page": 1.2,
            "p95_seconds_page": 1.4,
            "peak_rss_mb": 90.0,
            "mean_pdf_size_bytes": 1234.0,
        }
    ]
    markdown_path = tmp_path / "groups.md"
    csv_path = tmp_path / "groups.csv"

    source_sha256 = "b" * 64
    write_group_summary_markdown(
        markdown_path,
        summaries,
        source_sha256=source_sha256,
    )
    write_group_summary_csv(
        csv_path,
        summaries,
        source_sha256=source_sha256,
    )

    assert "| balanced_cpu | dataset:DharmaOCR | 2 | 2/2" in markdown_path.read_text(
        encoding="utf-8"
    )
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert rows[0]["source_jsonl_sha256"] == source_sha256
    assert rows[0]["summary_group"] == "dataset:DharmaOCR"


def test_summary_writer_replaces_symlink_without_overwriting_target(tmp_path: Path) -> None:
    protected = tmp_path / "protected.md"
    protected.write_text("keep", encoding="utf-8")
    report = tmp_path / "summary.md"
    report.symlink_to(protected)

    write_summary_markdown(report, [], source_sha256="c" * 64)

    assert protected.read_text(encoding="utf-8") == "keep"
    assert not report.is_symlink()
    assert "BigOCRPDF benchmark summary" in report.read_text(encoding="utf-8")


def test_profiles_are_pairwise_distinct() -> None:
    """Two profiles with identical settings measure the same thing twice.

    ``gpu_experimental`` was byte-identical to ``balanced_cpu`` and set no GPU
    backend, so its results were indistinguishable while its name promised
    otherwise. A GPU profile can come back when it configures a GPU.
    """
    from benchmarks.ocr_benchmark import PROFILES

    settings = [tuple(sorted(profile.items())) for profile in PROFILES.values()]

    assert len(set(settings)) == len(settings), (
        f"profiles must differ in at least one setting: {sorted(PROFILES)} -> {settings}"
    )


def test_a_profile_named_for_a_backend_configures_it() -> None:
    """Guards against reintroducing a name that does not match its settings."""
    from benchmarks.ocr_benchmark import PROFILES

    for name, profile in PROFILES.items():
        if "gpu" in name:
            assert profile.get("gpu_backend") not in (None, "", "off"), (
                f"profile {name!r} claims a GPU but does not select one"
            )
