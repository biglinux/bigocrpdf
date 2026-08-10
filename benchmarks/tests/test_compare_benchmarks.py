import json
from pathlib import Path

import pytest

from benchmarks.compare_benchmarks import (
    _record_regression_fields,
    compare_record_coverage,
    main,
    render_markdown,
    summarize,
)


def _record(
    sample_id: str,
    *,
    profile: str = "balanced_cpu",
    dataset: str = "DharmaOCR",
    page_index: int = 0,
    cer: float = 0.1,
    **overrides: object,
) -> dict[str, object]:
    record = {
        "id": sample_id,
        "benchmark_profile": profile,
        "dataset": dataset,
        "page_index": page_index,
        "char_error_rate": cer,
        "word_error_rate": cer,
        "end_to_end_seconds": 1.0,
        "peak_rss_mb": 100.0,
        "text_layer_ok": True,
        "manifest_sha256": "1" * 64,
        "ground_truth_language": "en",
        "effective_engine_type": "openvino",
        "effective_ocr_version": "PPOCRV6",
        "effective_model_type": "small",
        "effective_rec_batch_num": 1,
        "effective_gpu_backend": "off",
        "effective_rec_model_sha256": "2" * 64,
        "effective_det_model_sha256": "3" * 64,
        "effective_dictionary_sha256": "4" * 64,
        "effective_font_sha256": "5" * 64,
        "effective_language_hint": "latin",
        "effective_ocr_config": {
            "language": "latin",
            "dpi": 300,
            "engine_type": "openvino",
            "requested_engine_type": "openvino",
            "openvino_available": True,
            "ocr_version": "PPOCRV6",
            "model_type": "small",
            "rec_batch_num": 1,
            "use_textline_cls": False,
            "gpu_backend": "off",
            "requested_gpu_backend": "off",
            "gpu_device_id": None,
            "requested_gpu_device_id": 0,
            "gpu_fp16": False,
            "gpu_fallback_to_cpu": True,
            "detection_limit_side_len": 4000,
            "detection_full_resolution": False,
            "box_thresh": 0.5,
            "unclip_ratio": 1.2,
            "text_score_threshold": 0.3,
            "score_mode": "slow",
            "ocr_threads": 4,
            "ocr_workers": 1,
            "chunk_size": 8,
        },
        "dpi": 300,
        "warmup_runs": 0,
        "benchmark_environment": {
            "schema_version": 1,
            "platform": "Linux",
            "machine": "x86_64",
            "python": "3.14",
            "cpu_model": "Example CPU",
            "cpu_count": 8,
            "dependencies": {
                "bigocrpdf": "3.0.0",
                "numpy": "2.5.1",
                "Pillow": "12.3.0",
                "pikepdf": "10.9.1",
                "rapidocr": "3.9.0",
                "openvino": "2026.2.0",
            },
            "gpu_devices": [],
            "gpu_runtimes": {},
        },
        "peak_rss_method": "linux_proc_process_tree",
        "peak_rss_sample_interval_seconds": 0.05,
        "text_extractor": "pdftotext",
        "text_extractor_version": "24.02",
    }
    record.update(overrides)
    return record


def test_comparison_rejects_candidate_that_drops_a_hard_sample() -> None:
    baseline_rows = [_record("easy", cer=0.0), _record("hard", cer=0.8)]
    candidate_rows = [_record("easy", cer=0.0)]

    coverage = compare_record_coverage(baseline_rows, candidate_rows)
    markdown, failed = render_markdown(
        Path("baseline.jsonl"),
        Path("candidate.jsonl"),
        summarize(baseline_rows),
        summarize(candidate_rows),
        coverage,
    )

    assert failed is True
    assert coverage["baseline_only_count"] == 1
    assert "coverage_mismatch: True" in markdown
    assert "balanced_cpu / DharmaOCR / hard / page 0" in markdown


def test_comparison_accepts_identical_record_identity_sets() -> None:
    baseline_rows = [_record("one"), _record("two")]
    candidate_rows = [_record("two", cer=0.09), _record("one", cer=0.09)]

    coverage = compare_record_coverage(baseline_rows, candidate_rows)

    assert coverage["comparable"] is True
    assert coverage["baseline_only_count"] == 0
    assert coverage["candidate_only_count"] == 0


def test_comparison_rejects_different_effective_model_hash() -> None:
    baseline = _record("same")
    baseline["effective_rec_model_sha256"] = "a" * 64
    candidate = _record("same")
    candidate["effective_rec_model_sha256"] = "b" * 64

    coverage = compare_record_coverage([baseline], [candidate])

    assert coverage["comparable"] is False
    assert coverage["configuration_mismatch_count"] == 1
    assert "effective_rec_model_sha256" in coverage["configuration_mismatches"][0]


@pytest.mark.parametrize(
    ("field", "baseline_value", "candidate_value"),
    [
        ("effective_det_model_sha256", "a" * 64, "b" * 64),
        ("effective_font_sha256", "a" * 64, "b" * 64),
        ("effective_language_hint", "latin", "ch"),
        ("warmup_runs", 0, 1),
        ("peak_rss_sample_interval_seconds", 0.05, 0.1),
    ],
)
def test_comparison_rejects_different_measurement_contract(
    field: str,
    baseline_value: object,
    candidate_value: object,
) -> None:
    baseline = _record("same")
    baseline[field] = baseline_value
    candidate = _record("same")
    candidate[field] = candidate_value

    coverage = compare_record_coverage([baseline], [candidate])

    assert coverage["comparable"] is False
    assert field in coverage["configuration_mismatches"][0]


def test_comparison_fails_on_large_peak_rss_regression() -> None:
    baseline_rows = [_record("same")]
    candidate_rows = [_record("same")]
    candidate_rows[0]["peak_rss_mb"] = 1000.0
    coverage = compare_record_coverage(baseline_rows, candidate_rows)

    markdown, failed = render_markdown(
        Path("baseline.jsonl"),
        Path("candidate.jsonl"),
        summarize(baseline_rows),
        summarize(candidate_rows),
        coverage,
    )

    assert failed is True
    assert "rss_regression: True" in markdown


def test_comparison_fails_closed_when_peak_rss_is_missing() -> None:
    baseline_rows = [_record("same")]
    candidate_rows = [_record("same")]
    candidate_rows[0]["peak_rss_mb"] = None
    coverage = compare_record_coverage(baseline_rows, candidate_rows)

    markdown, failed = render_markdown(
        Path("baseline.jsonl"),
        Path("candidate.jsonl"),
        summarize(baseline_rows),
        summarize(candidate_rows),
        coverage,
    )

    assert failed is True
    assert "rss_metric_missing: True" in markdown


def test_comparison_rejects_duplicate_record_identity() -> None:
    duplicate_rows = [_record("same"), _record("same")]

    with pytest.raises(ValueError, match="duplicate benchmark record"):
        compare_record_coverage(duplicate_rows, [_record("same")])


@pytest.mark.parametrize("missing_field", ["id", "benchmark_profile", "dataset", "page_index"])
def test_comparison_requires_stable_record_identity(missing_field: str) -> None:
    incomplete = _record("sample")
    del incomplete[missing_field]

    with pytest.raises(ValueError, match=missing_field):
        compare_record_coverage([incomplete], [_record("sample")])


@pytest.mark.parametrize(
    "missing_metric",
    [
        "char_error_rate",
        "word_error_rate",
        "end_to_end_seconds",
        "peak_rss_mb",
        "text_layer_ok",
    ],
)
def test_comparison_fails_closed_when_candidate_metric_is_missing(
    missing_metric: str,
) -> None:
    baseline_rows = [_record("same")]
    candidate_rows = [_record("same")]
    candidate_rows[0][missing_metric] = None
    coverage = compare_record_coverage(baseline_rows, candidate_rows)

    markdown, failed = render_markdown(
        Path("baseline.jsonl"),
        Path("candidate.jsonl"),
        summarize(baseline_rows),
        summarize(candidate_rows),
        coverage,
    )

    assert failed is True
    assert coverage["metric_missing_count"] == 1
    assert "metric_missing: True" in markdown


def test_comparison_gates_word_error_rate() -> None:
    baseline_rows = [_record("same")]
    candidate_rows = [_record("same")]
    candidate_rows[0]["word_error_rate"] = 0.8
    coverage = compare_record_coverage(baseline_rows, candidate_rows)

    markdown, failed = render_markdown(
        Path("baseline.jsonl"),
        Path("candidate.jsonl"),
        summarize(baseline_rows),
        summarize(candidate_rows),
        coverage,
    )

    assert failed is True
    assert "wer_regression: True" in markdown


def test_comparison_detects_regression_hidden_by_global_average() -> None:
    baseline_rows = [_record("a", cer=0.1), _record("b", cer=0.9)]
    candidate_rows = [_record("a", cer=0.9), _record("b", cer=0.1)]
    coverage = compare_record_coverage(baseline_rows, candidate_rows)

    markdown, failed = render_markdown(
        Path("baseline.jsonl"),
        Path("candidate.jsonl"),
        summarize(baseline_rows),
        summarize(candidate_rows),
        coverage,
    )

    assert summarize(baseline_rows)["mean_cer"] == summarize(candidate_rows)["mean_cer"]
    assert failed is True
    assert coverage["record_regression_count"] >= 1
    assert "record_regression: True" in markdown


def test_comparison_rejects_missing_comparability_metadata_on_both_sides() -> None:
    baseline = _record("same")
    candidate = _record("same")
    del baseline["text_extractor"]
    del candidate["text_extractor"]

    coverage = compare_record_coverage([baseline], [candidate])

    assert coverage["comparable"] is False
    assert coverage["configuration_missing_count"] == 2
    assert "text_extractor" in coverage["configuration_missing"][0]


@pytest.mark.parametrize(
    ("field", "partial_value"),
    [
        ("benchmark_environment", {"platform": "Linux"}),
        ("effective_ocr_config", {"engine_type": "openvino"}),
    ],
)
def test_comparison_rejects_partial_nested_contracts(
    field: str,
    partial_value: object,
) -> None:
    baseline = _record("same")
    candidate = _record("same")
    baseline[field] = partial_value
    candidate[field] = partial_value

    coverage = compare_record_coverage([baseline], [candidate])

    assert coverage["comparable"] is False
    assert coverage["configuration_missing_count"] == 2
    assert field in coverage["configuration_missing"][0]


def test_gpu_comparison_requires_device_and_runtime_identity() -> None:
    baseline = _record("same")
    candidate = _record("same")
    baseline["effective_gpu_backend"] = "paddle"
    candidate["effective_gpu_backend"] = "paddle"
    baseline_config = baseline["effective_ocr_config"]
    candidate_config = candidate["effective_ocr_config"]
    assert isinstance(baseline_config, dict)
    assert isinstance(candidate_config, dict)
    baseline_config["gpu_backend"] = "paddle"
    candidate_config["gpu_backend"] = "paddle"

    coverage = compare_record_coverage([baseline], [candidate])

    assert coverage["comparable"] is False
    assert "benchmark_environment" in coverage["configuration_missing"][0]


def _configure_gpu_record(
    row: dict[str, object],
    backend: str,
    runtimes: dict[str, str],
) -> None:
    row["effective_gpu_backend"] = backend
    config = row["effective_ocr_config"]
    environment = row["benchmark_environment"]
    assert isinstance(config, dict)
    assert isinstance(environment, dict)
    config["gpu_backend"] = backend
    config["requested_gpu_backend"] = backend
    config["gpu_device_id"] = 0
    environment["gpu_devices"] = [
        {
            "card": "card0",
            "pci_address": "0000:01:00.0",
            "vendor_id": "0x10de",
            "device_id": "0x1234",
            "driver": "nvidia",
            "driver_version": "999.1",
        }
    ]
    environment["gpu_runtimes"] = runtimes


def test_gpu_contract_requires_runtime_for_the_effective_backend() -> None:
    baseline = _record("same")
    candidate = _record("same")
    _configure_gpu_record(baseline, "paddle", {"torch": "2.0"})
    _configure_gpu_record(candidate, "paddle", {"torch": "2.0"})

    coverage = compare_record_coverage([baseline], [candidate])

    assert coverage["comparable"] is False
    assert "benchmark_environment" in coverage["configuration_missing"][0]


def test_cpu_comparison_ignores_unrelated_gpu_inventory() -> None:
    baseline = _record("same")
    candidate = _record("same")
    candidate_environment = candidate["benchmark_environment"]
    assert isinstance(candidate_environment, dict)
    candidate_environment["gpu_devices"] = [
        {
            "card": "card0",
            "pci_address": "0000:01:00.0",
            "vendor_id": "0x10de",
            "device_id": "0x1234",
            "driver": "nvidia",
            "driver_version": "999.1",
        }
    ]
    candidate_environment["gpu_runtimes"] = {"torch": "2.0"}

    coverage = compare_record_coverage([baseline], [candidate])

    assert coverage["comparable"] is True


def test_gpu_comparison_uses_only_the_effective_backend_runtime() -> None:
    baseline = _record("same")
    candidate = _record("same")
    _configure_gpu_record(
        baseline,
        "paddle",
        {"paddlepaddle-gpu": "3.2", "torch": "2.0"},
    )
    _configure_gpu_record(
        candidate,
        "paddle",
        {"paddlepaddle-gpu": "3.2", "torch": "9.0"},
    )

    assert compare_record_coverage([baseline], [candidate])["comparable"] is True

    candidate_environment = candidate["benchmark_environment"]
    assert isinstance(candidate_environment, dict)
    candidate_environment["gpu_runtimes"] = {"paddlepaddle-gpu": "4.0"}
    coverage = compare_record_coverage([baseline], [candidate])

    assert coverage["comparable"] is False
    assert "benchmark_environment" in coverage["configuration_mismatches"][0]


def test_auto_is_not_a_valid_effective_gpu_backend() -> None:
    baseline = _record("same")
    candidate = _record("same")
    _configure_gpu_record(baseline, "auto", {"torch": "2.0"})
    _configure_gpu_record(candidate, "auto", {"torch": "2.0"})

    coverage = compare_record_coverage([baseline], [candidate])

    assert coverage["comparable"] is False
    assert "benchmark_environment" in coverage["configuration_missing"][0]


def test_summary_reports_p95_latency() -> None:
    rows = [_record(str(index)) for index in range(20)]
    for index, row in enumerate(rows, start=1):
        row["end_to_end_seconds"] = float(index)

    summary = summarize(rows)

    assert summary["p95_seconds_page"] == 19.0


def test_comparison_report_replaces_symlink_without_overwriting_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    payload = json.dumps(_record("same")) + "\n"
    baseline_path.write_text(payload, encoding="utf-8")
    candidate_path.write_text(payload, encoding="utf-8")
    protected = tmp_path / "protected.md"
    protected.write_text("keep", encoding="utf-8")
    report = tmp_path / "comparison.md"
    report.symlink_to(protected)
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_benchmarks.py",
            str(baseline_path),
            str(candidate_path),
            "--out",
            str(report),
        ],
    )

    assert main() == 0
    assert protected.read_text(encoding="utf-8") == "keep"
    assert not report.is_symlink()
    assert "baseline SHA-256" in report.read_text(encoding="utf-8")


def test_a_sample_that_stops_producing_boxes_fails_the_gate() -> None:
    """The shape of the real-world failure: text one day, silence the next.

    The error rate stays inside its budget here, so only the zero-output
    invariant can catch it.
    """
    baseline = _record("photo", cer=0.02)
    baseline["ocr_box_count"] = 42
    candidate = _record("photo", cer=0.02)
    candidate["ocr_box_count"] = 0

    assert "ocr_box_count_zero" in _record_regression_fields(baseline, candidate)


def test_pages_going_blank_fails_even_when_the_document_average_holds() -> None:
    """An eighteen-page contract with seventeen blank pages still reads well
    on average; per-page counting is what refuses it."""
    baseline = _record("contrato", cer=0.02)
    baseline["pages_with_zero_boxes"] = 0
    candidate = _record("contrato", cer=0.02)
    candidate["pages_with_zero_boxes"] = 17

    assert "pages_with_zero_boxes" in _record_regression_fields(baseline, candidate)


def test_a_sample_that_never_produced_boxes_is_not_a_new_regression() -> None:
    """A blank page in the baseline is a known state, not a fresh failure."""
    baseline = _record("blank")
    baseline["ocr_box_count"] = 0
    candidate = _record("blank")
    candidate["ocr_box_count"] = 0

    assert _record_regression_fields(baseline, candidate) == []


def test_recovering_blank_pages_is_not_a_regression() -> None:
    baseline = _record("contrato")
    baseline["pages_with_zero_boxes"] = 5
    candidate = _record("contrato")
    candidate["pages_with_zero_boxes"] = 0

    assert _record_regression_fields(baseline, candidate) == []


def test_records_without_box_counts_are_ignored_by_the_invariant() -> None:
    """Older baselines predate these fields and must stay comparable."""
    assert _record_regression_fields(_record("legacy"), _record("legacy")) == []


class TestOptionalDigestsDoNotBlockComparison:
    """A recogniser with no separate dictionary file still has a comparable run.

    PP-OCRv6 carries its dictionary inside the model, so the producer records
    ``effective_dictionary_sha256`` as an explicit null. Reading that as "the
    producer never told us" made every comparison fail on coverage -- a file
    compared against itself included, which is what exposed it.
    """

    def test_a_null_digest_on_both_sides_is_comparable(self):
        rows = [_record("amostra", effective_dictionary_sha256=None)]

        coverage = compare_record_coverage(rows, rows)

        assert coverage["configuration_missing_count"] == 0

    def test_a_digest_on_one_side_only_is_still_a_mismatch(self):
        """The property that makes the exemption safe: disagreement still fails."""
        baseline = [_record("amostra", effective_dictionary_sha256=None)]
        candidate = [_record("amostra", effective_dictionary_sha256="b" * 64)]

        coverage = compare_record_coverage(baseline, candidate)

        assert (
            coverage["configuration_mismatch_count"] + coverage["configuration_missing_count"] > 0
        )

    def test_an_omitted_key_still_fails(self):
        """An older producer that never wrote the field is a real gap."""
        row = _record("amostra")
        del row["effective_dictionary_sha256"]

        coverage = compare_record_coverage([row], [row])

        assert coverage["configuration_missing_count"] > 0

    def test_a_required_digest_may_not_be_null(self):
        """Only the dictionary is optional; the model digests are not."""
        rows = [_record("amostra", effective_rec_model_sha256=None)]

        coverage = compare_record_coverage(rows, rows)

        assert coverage["configuration_missing_count"] > 0
