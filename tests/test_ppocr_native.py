"""The ppocr-native workers as bigocrpdf starts, drives and stops them."""

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from bigocrpdf.services.ocr import backend, ppocr_native, preprocess_deskew
from bigocrpdf.services.ocr.config import OCRConfig
from bigocrpdf.services.screen_capture import ImageOcrStatus, ScreenCaptureService

_PAGE = {"boxes": [[[0, 0], [1, 0], [1, 1], [0, 1]]], "txts": ["ok"], "scores": [0.9]}


def test_worker_command_carries_the_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ppocr_native, "find_worker", lambda: "/usr/bin/ppocr-native")
    monkeypatch.setattr(ppocr_native, "physical_cores", lambda: 8)
    config = OCRConfig(detection_full_resolution=True)

    assert ppocr_native.worker_command(config, threads=6) == [
        "/usr/bin/ppocr-native",
        "--threads",
        "6",
        "--box-thresh",
        "0.5",
        "--unclip-ratio",
        "1.5",
        "--text-score",
        "0.3",
        "--limit-side-len",
        "4096",
        "--full-resolution",
        "--persistent",
    ]
    assert ppocr_native.worker_command(config, threads=16)[1:3] == ["--threads", "8"]
    assert ppocr_native.worker_command(config)[1:3] == ["--threads", "8"]


def test_missing_worker_is_reported() -> None:
    from bigocrpdf import _check_ocr_dependencies

    state = _check_ocr_dependencies()

    assert not state.is_available
    assert ppocr_native.WORKER in state.error
    with pytest.raises(FileNotFoundError):
        ppocr_native.worker_command(OCRConfig(), threads=1)


def test_worker_needs_its_plans(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.undo()
    worker = tmp_path / "bin" / ppocr_native.WORKER
    worker.parent.mkdir()
    worker.write_text("#!/bin/sh\n")
    worker.chmod(0o755)
    plans = tmp_path / "plans"
    plans.mkdir()
    monkeypatch.setenv("PATH", str(worker.parent))
    monkeypatch.setenv(ppocr_native.PLANS_ENV, str(plans))
    ppocr_native.find_worker.cache_clear()
    try:
        assert ppocr_native.find_worker() is None
        for name in ppocr_native.PLAN_FILES:
            (plans / name).touch()
        ppocr_native.find_worker.cache_clear()
        assert ppocr_native.find_worker() == str(worker)
    finally:
        ppocr_native.find_worker.cache_clear()


@pytest.mark.parametrize(
    ("idle_tools", "prefix"),
    [
        ({"nice", "ionice"}, ["/usr/bin/nice", "-n", "19", "/usr/bin/ionice", "-c", "3", "-t"]),
        ({"ionice"}, ["/usr/bin/ionice", "-c", "3", "-t"]),
        (set(), []),
    ],
)
def test_ocr_worker_runs_at_idle_priority(
    monkeypatch: pytest.MonkeyPatch, idle_tools: set[str], prefix: list[str]
) -> None:
    started: list[list[str]] = []
    monkeypatch.setattr(ppocr_native, "find_worker", lambda: "/usr/bin/ppocr-native")
    monkeypatch.setattr(ppocr_native, "physical_cores", lambda: 4)
    monkeypatch.setattr(
        ppocr_native.shutil,
        "which",
        lambda tool: f"/usr/bin/{tool}" if tool in idle_tools else None,
    )
    monkeypatch.setattr(
        ppocr_native.subprocess, "Popen", lambda command, **_kwargs: started.append(command)
    )

    worker = ppocr_native.start_ocr_worker(OCRConfig(), threading.Event(), threads=16)

    assert started[0][: len(prefix) + 1] == [*prefix, "/usr/bin/ppocr-native"]
    assert worker.threads == 4


def _fake_worker(tmp_path: Path, body: str, ready: str = '{"ready": true, "tier": 3}') -> str:
    """A worker that sends ``ready``, then runs ``body``."""
    worker = tmp_path / ppocr_native.WORKER
    worker.write_text(f"#!/bin/sh\necho '{ready}'\n{body}\n", encoding="utf-8")
    worker.chmod(0o755)
    return str(worker)


def _echo_worker(tmp_path: Path, reply: dict) -> str:
    return _fake_worker(tmp_path, f"while read -r _; do echo '{json.dumps(reply)}'; done")


def test_worker_answers_after_its_ready_line(tmp_path: Path) -> None:
    worker = ppocr_native.Worker([_echo_worker(tmp_path, _PAGE)])
    try:
        assert worker.recognize("/tmp/page.ppm") == _PAGE
        assert worker.wait_ready() == 3
    finally:
        worker.close()
    assert worker.exited()


@pytest.mark.parametrize(
    "ready",
    ['{"ready": true}', '{"ready": true, "tier": 4}', '{"ready": true, "tier": "3"}'],
)
def test_ready_without_tier_is_refused(tmp_path: Path, ready: str) -> None:
    worker = ppocr_native.Worker([_fake_worker(tmp_path, "cat", ready)])

    with pytest.raises(ppocr_native.ProtocolError, match="kernel tier"):
        worker.wait_ready()
    assert worker.exited()


def test_fatal_start_is_reported(tmp_path: Path) -> None:
    worker = ppocr_native.Worker([_fake_worker(tmp_path, "", '{"fatal": "plan truncated"}')])

    with pytest.raises(RuntimeError, match="plan truncated"):
        worker.wait_ready()


def test_page_error_keeps_the_worker(tmp_path: Path) -> None:
    worker = ppocr_native.Worker([_echo_worker(tmp_path, {"error": "not a PPM"})])
    try:
        with pytest.raises(ppocr_native.ReadError, match="not a PPM"):
            worker.recognize("/tmp/page.png")
        assert not worker.exited()
    finally:
        worker.close()


def test_dead_worker_is_started_again(tmp_path: Path) -> None:
    died = tmp_path / "died"
    body = f"while read -r _; do [ -e '{died}' ] || {{ : > '{died}'; exit 1; }}; "
    body += f"echo '{json.dumps(_PAGE)}'; done"
    worker = ppocr_native.Worker([_fake_worker(tmp_path, body)])
    try:
        assert worker.recognize("/tmp/page.ppm") == _PAGE
        assert worker.recognize("/tmp/page.ppm") == _PAGE
    finally:
        worker.close()


def test_worker_dying_twice_fails_the_page(tmp_path: Path) -> None:
    worker = ppocr_native.Worker([_fake_worker(tmp_path, "read -r _")])
    try:
        with pytest.raises(ppocr_native.ReadError, match="exited"):
            worker.recognize("/tmp/page.ppm")
    finally:
        worker.close()


def test_line_break_in_path_is_refused(tmp_path: Path) -> None:
    worker = ppocr_native.Worker([_echo_worker(tmp_path, _PAGE)])
    try:
        with pytest.raises(ValueError, match="line break"):
            worker.recognize("/tmp/bad\npage.ppm")
    finally:
        worker.close()


def test_silent_worker_is_killed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ppocr_native, "REQUEST_TIMEOUT_SECONDS", 0.2)
    worker = ppocr_native.Worker([_fake_worker(tmp_path, "exec sleep 60")])

    with pytest.raises(ppocr_native.ReadError, match="no answer"):
        worker.recognize("/tmp/page.ppm")
    started = time.monotonic()
    worker.close()
    # A live worker would hold close() for its 5 s grace.
    assert time.monotonic() - started < 2


def _request_signal(tmp_path: Path, then: str) -> tuple[str, Path]:
    """A worker that marks a file once it has read a request, then runs ``then``."""
    arrived = tmp_path / "arrived"
    return _fake_worker(tmp_path, f"read -r _; : > '{arrived}'; {then}"), arrived


def _wait_for(path: Path) -> None:
    deadline = time.monotonic() + 10
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.01)


def test_cancel_frees_a_waiting_request(tmp_path: Path) -> None:
    command, arrived = _request_signal(tmp_path, "exec sleep 60")
    cancel = threading.Event()
    worker = ppocr_native.Worker([command], cancel)
    failure: list[BaseException] = []

    def request() -> None:
        try:
            worker.recognize("/tmp/page.ppm")
        except BaseException as exc:
            failure.append(exc)

    waiting = threading.Thread(target=request)
    waiting.start()
    try:
        _wait_for(arrived)
        cancel.set()
        waiting.join(timeout=2)
        assert not waiting.is_alive()
        assert isinstance(failure[0], InterruptedError)
    finally:
        worker.close()


def test_close_lets_a_request_finish(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    command, arrived = _request_signal(tmp_path, "sleep 0.3; echo '{\"boxes\": []}'; cat")
    monkeypatch.setattr(ppocr_native, "find_worker", lambda: command)
    detector = ppocr_native.Detector()
    boxes: list[list] = []
    waiting = threading.Thread(
        target=lambda: boxes.append(detector.boxes(np.zeros((8, 8, 3), np.uint8), 64))
    )
    waiting.start()
    _wait_for(arrived)

    detector.close()
    waiting.join(timeout=5)

    assert boxes == [[]]


def test_jobs_share_their_worker_with_preprocessing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    command = _echo_worker(tmp_path, {"boxes": []})
    monkeypatch.setattr(ppocr_native, "worker_command", lambda *_args, **_kwargs: [command])
    monkeypatch.setattr(ppocr_native.shutil, "which", lambda _tool: None)
    image = np.zeros((8, 8, 3), np.uint8)
    first = backend.ProfessionalPDFOCR(OCRConfig())
    second = backend.ProfessionalPDFOCR(OCRConfig())
    first_worker = first._start_ocr_worker()
    second_worker = second._start_ocr_worker()
    try:
        assert first.preprocessor.detector.boxes(image, 64) == []

        first.cancel()
        with pytest.raises(InterruptedError):
            first.preprocessor.detector.boxes(image, 64)

        assert second.preprocessor.detector.boxes(image, 64) == []
        second.preprocessor.detector.close()
        assert not second_worker.exited()
    finally:
        first_worker.close()
        second_worker.close()


def test_failed_detection_leaves_no_boxes(tmp_path: Path) -> None:
    """Every caller of detect_text_boxes has a geometric fallback for an empty list."""
    detector = ppocr_native.Detector()

    assert preprocess_deskew.detect_text_boxes(detector, np.zeros((8, 8, 3), np.uint8)) == []


class TestOcrResults:
    """The page reply contract, shared by every caller."""

    def test_lines_keep_their_order(self) -> None:
        reply = {
            "boxes": [
                [[0, 0], [100, 0], [100, 20], [0, 20]],
                [[0, 30], [100, 30], [100, 50], [0, 50]],
            ],
            "txts": ["Line 1", "Line 2"],
            "scores": [0.9, 0.85],
        }

        results = ppocr_native.ocr_results(reply)

        assert [(r.text, r.confidence) for r in results] == [("Line 1", 0.9), ("Line 2", 0.85)]

    @pytest.mark.parametrize("reply", [{"boxes": None}, {"boxes": [], "txts": [], "scores": []}])
    def test_blank_page_is_empty(self, reply: dict) -> None:
        assert ppocr_native.ocr_results(reply) == []

    @pytest.mark.parametrize(
        "reply",
        [
            {"error": "model not found"},
            {"result": "none"},
            {"boxes": None, "txts": ["orphan"]},
            {"boxes": [[[0, 0]]], "txts": ["a", "b"], "scores": [0.9]},
            {"boxes": [[[0, 0]]], "txts": "a", "scores": [0.9]},
        ],
    )
    def test_broken_reply_is_refused(self, reply: dict) -> None:
        with pytest.raises(ppocr_native.ProtocolError):
            ppocr_native.ocr_results(reply)

    def test_garbage_is_refused(self) -> None:
        with pytest.raises(ppocr_native.ProtocolError):
            ppocr_native.parse_reply("not json{")


@pytest.mark.real_ocr
@pytest.mark.skipif(
    ppocr_native.find_worker() is None, reason="ppocr-native and its plans are not installed"
)
def test_installed_worker_answers_the_protocol(tmp_path: Path) -> None:
    """The installed ppocr-native, driven the way the pipeline drives a worker."""
    page = np.full((120, 560, 3), 255, dtype=np.uint8)
    cv2.putText(page, "OCR 2026", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 5)
    page_path = tmp_path / f"page{ppocr_native.OCR_IMAGE_SUFFIX}"
    assert cv2.imwrite(str(page_path), page)
    png_path = tmp_path / "page.png"
    assert cv2.imwrite(str(png_path), page)

    worker = ppocr_native.start_ocr_worker(OCRConfig(), threading.Event(), threads=2)
    try:
        assert worker.wait_ready() in range(4)
        result = worker.recognize(str(page_path))
        assert "OCR 2026" in " ".join(line.text for line in ppocr_native.ocr_results(result))
        assert all(len(box) == 4 for box in result["boxes"])
        # Formats other than PPM and PGM are refused per request, not fatally.
        with pytest.raises(ppocr_native.ReadError):
            worker.recognize(str(png_path))
        assert not worker.exited()
    finally:
        worker.close()

    # The image window reads a pasted texture, written as RGBA TIFF, through its own worker.
    pasted = tmp_path / "pasted.tiff"
    Image.fromarray(cv2.cvtColor(page, cv2.COLOR_BGR2RGBA)).save(pasted)
    outcome = ScreenCaptureService()._extract_text_result(str(pasted), OCRConfig())
    assert outcome.status == ImageOcrStatus.SUCCESS
    assert "OCR 2026" in (outcome.text or "")

    detector = ppocr_native.Detector()
    try:
        prob = detector.probability_map(cv2.resize(page, (544, 96)))
        assert prob.shape == (96, 544)
        assert 0.0 <= float(prob.min()) and float(prob.max()) <= 1.0 and float(prob.max()) > 0.5
        assert len(detector.boxes(page, 1536)) >= 1
    finally:
        detector.close()
