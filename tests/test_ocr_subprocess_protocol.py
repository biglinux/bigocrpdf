"""Contract tests for the persistent OCR subprocess protocol."""

import json
import os
import queue
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from bigocrpdf.services.rapidocr_service.ocr_subprocess_controller import (
    OCRSubprocessController,
)
from bigocrpdf.services.rapidocr_service.resource_manager import ResourceTier

_EOF = object()


class _QueuedStdout:
    def __init__(self, *lines: str | object) -> None:
        self._lines: queue.Queue[str | object] = queue.Queue()
        for line in lines:
            self._lines.put(line)
        self.closed = False
        self.read_started = threading.Event()

    def readline(self, _size: int = -1) -> str:
        self.read_started.set()
        line = self._lines.get()
        return "" if line is _EOF else str(line)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._lines.put(_EOF)


class _FakeStdin:
    def __init__(self, proc: "_FakeProcess") -> None:
        self._proc = proc
        self.writes: list[str] = []
        self.closed = False

    def write(self, value: str) -> int:
        if self.closed:
            raise BrokenPipeError
        self.writes.append(value)
        return len(value)

    def flush(self) -> None:
        if self.closed:
            raise BrokenPipeError

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._proc.returncode is None:
            self._proc.returncode = 0
        self._proc.stdout.close()


class _FakeProcess:
    def __init__(self, *stdout_lines: str | object) -> None:
        self.pid = 4242
        self.returncode: int | None = None
        self.stdout = _QueuedStdout(*stdout_lines)
        self.stdin = _FakeStdin(self)
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_timeouts: list[float] = []

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float) -> int:
        self.wait_timeouts.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired("ocr-worker", timeout)
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -15
        self.stdout.close()

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9
        self.stdout.close()


class _StubbornProcess(_FakeProcess):
    def terminate(self) -> None:
        self.terminate_calls += 1


def _controller(cancel_event: threading.Event | None = None) -> OCRSubprocessController:
    config = SimpleNamespace(
        ocr_ready_timeout_seconds=0.5,
        ocr_response_timeout_seconds=0.5,
    )
    return OCRSubprocessController(config, cancel_event or threading.Event(), lambda: False)


def _line(payload: dict) -> str:
    return json.dumps(payload) + "\n"


_CPU_RUNTIME = {
    "schema_version": 1,
    "engine_label": "openvino_cpu",
    "engine_type": "openvino",
    "gpu_backend": "off",
    "gpu_device_id": None,
}


def _ready_line(runtime: dict | None = None) -> str:
    return _line({"ready": True, "runtime": runtime or _CPU_RUNTIME})


@pytest.mark.parametrize(
    ("tier", "low_memory"),
    (
        (ResourceTier.CONSTRAINED, True),
        (ResourceTier.MODERATE, False),
        (ResourceTier.ABUNDANT, False),
    ),
)
def test_worker_command_uses_low_memory_mode_only_when_constrained(
    tier: ResourceTier,
    low_memory: bool,
) -> None:
    controller = _controller()

    with (
        patch(
            "bigocrpdf.services.rapidocr_service.ocr_subprocess_controller.detect_resources",
            return_value=SimpleNamespace(tier=tier),
        ),
        patch(
            "bigocrpdf.services.rapidocr_service.ocr_subprocess_controller.build_ocr_worker_command",
            return_value=["worker"],
        ) as build_command,
    ):
        assert controller._build_command(ocr_threads=2) == ["worker"]

    assert build_command.call_args.kwargs["low_memory_openvino"] is low_memory


def test_launch_wraps_priority_as_argv_without_preexec() -> None:
    controller = _controller()
    worker_command = ["/usr/bin/python", "-m", "worker"]
    launched_process = MagicMock()

    with (
        patch.object(controller, "_build_command", return_value=worker_command),
        patch(
            "bigocrpdf.services.rapidocr_service.ocr_subprocess_controller.shutil.which",
            side_effect=lambda command: f"/usr/bin/{command}",
        ),
        patch(
            "bigocrpdf.services.rapidocr_service.ocr_subprocess_controller.subprocess.Popen",
            return_value=launched_process,
        ) as popen,
    ):
        result = controller.launch(ocr_threads=2)

    assert result is launched_process
    assert result._bigocr_threads == 2
    launch_args, launch_kwargs = popen.call_args
    assert launch_args[0] == [
        "/usr/bin/nice",
        "-n",
        "19",
        "/usr/bin/ionice",
        "-c",
        "3",
        "-t",
        *worker_command,
    ]
    assert "preexec_fn" not in launch_kwargs


@pytest.mark.parametrize(
    ("available_tools", "expected_prefix"),
    (
        (set(), []),
        ({"nice"}, ["/usr/bin/nice", "-n", "19"]),
        ({"ionice"}, ["/usr/bin/ionice", "-c", "3", "-t"]),
    ),
)
def test_launch_supports_each_optional_priority_tool(
    available_tools: set[str],
    expected_prefix: list[str],
) -> None:
    controller = _controller()
    worker_command = ["/opt/bigocr/worker", "--persistent"]

    with (
        patch.object(controller, "_build_command", return_value=worker_command),
        patch(
            "bigocrpdf.services.rapidocr_service.ocr_subprocess_controller.shutil.which",
            side_effect=lambda command: (
                f"/usr/bin/{command}" if command in available_tools else None
            ),
        ),
        patch(
            "bigocrpdf.services.rapidocr_service.ocr_subprocess_controller.subprocess.Popen",
        ) as popen,
    ):
        controller.launch()

    launch_args, launch_kwargs = popen.call_args
    assert launch_args[0] == [*expected_prefix, *worker_command]
    assert "preexec_fn" not in launch_kwargs


def test_real_worker_launch_and_reap_is_bounded_from_thread() -> None:
    worker_script = (
        "import json, sys\n"
        f"print(json.dumps({{'ready': True, 'runtime': {_CPU_RUNTIME!r}}}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    print(json.dumps({'boxes': None, 'path': line.strip()}), flush=True)\n"
    )
    failures: queue.Queue[BaseException] = queue.Queue()

    def exercise_lifecycle() -> None:
        try:
            for _iteration in range(3):
                controller = _controller()
                command = [sys.executable, "-u", "-c", worker_script]
                with patch.object(controller, "_build_command", return_value=command):
                    proc = controller.launch(ocr_threads=1)
                controller.wait_until_ready(proc, timeout_seconds=2)
                assert controller.recognize(proc, "/tmp/page.png", timeout_seconds=2) == {
                    "boxes": None,
                    "path": "/tmp/page.png",
                }
                controller.stop(proc)
                assert proc.poll() is not None
                assert proc.stdin is not None and proc.stdin.closed
                assert proc.stdout is not None and proc.stdout.closed
                assert getattr(proc, "_bigocr_protocol_reader", None) is None
                with pytest.raises(ChildProcessError):
                    os.waitpid(proc.pid, os.WNOHANG)
        except BaseException as exc:
            failures.put(exc)

    lifecycle_thread = threading.Thread(target=exercise_lifecycle, daemon=True)
    lifecycle_thread.start()
    lifecycle_thread.join(timeout=10)

    assert not lifecycle_thread.is_alive()
    if not failures.empty():
        raise failures.get_nowait()


def test_ready_and_normal_response_follow_protocol() -> None:
    controller = _controller()
    proc = _FakeProcess(
        _ready_line(),
        _line({"boxes": [[[0, 0], [1, 0], [1, 1], [0, 1]]], "txts": ["ok"], "scores": [0.9]}),
    )

    runtime = controller.wait_until_ready(proc, timeout_seconds=0.5)
    result = controller.recognize(proc, "/tmp/page.png", timeout_seconds=0.5)

    assert result is not None
    assert runtime == _CPU_RUNTIME
    assert result["txts"] == ["ok"]
    assert proc.stdin.writes == ["/tmp/page.png\n"]
    controller.stop(proc)


def test_first_request_performs_ready_handshake_automatically() -> None:
    controller = _controller()
    proc = _FakeProcess(_ready_line(), _line({"boxes": None}))

    assert controller.recognize(proc, "/tmp/page.png") == {"boxes": None}
    assert proc.stdin.writes == ["/tmp/page.png\n"]
    controller.stop(proc)


def test_ready_timeout_terminates_and_cleans_worker() -> None:
    controller = _controller()
    proc = _FakeProcess()

    with pytest.raises(RuntimeError, match="ready signal timed out"):
        controller.wait_until_ready(proc, timeout_seconds=0.01)

    assert proc.terminate_calls == 1
    assert proc.poll() == -15
    assert proc.stdin.closed
    assert proc.stdout.closed


def test_timeout_escalates_to_kill_when_worker_ignores_terminate() -> None:
    controller = _controller()
    proc = _StubbornProcess()

    with pytest.raises(RuntimeError, match="ready signal timed out"):
        controller.wait_until_ready(proc, timeout_seconds=0.01)

    assert proc.terminate_calls == 1
    assert proc.kill_calls == 1
    assert proc.poll() == -9
    assert proc.stdin.closed
    assert proc.stdout.closed


def test_response_timeout_terminates_worker() -> None:
    controller = _controller()
    proc = _FakeProcess(_ready_line())

    assert controller.recognize(proc, "/tmp/page.png", timeout_seconds=0.01) is None
    assert proc.terminate_calls == 1
    assert proc.poll() == -15


def test_response_eof_terminates_worker() -> None:
    controller = _controller()
    proc = _FakeProcess(_ready_line(), _EOF)

    assert controller.recognize(proc, "/tmp/page.png") is None
    assert proc.terminate_calls == 1
    assert proc.stdin.closed
    assert proc.stdout.closed


def test_ready_eof_terminates_worker() -> None:
    controller = _controller()
    proc = _FakeProcess(_EOF)

    with pytest.raises(RuntimeError, match="closed before ready"):
        controller.wait_until_ready(proc)

    assert proc.terminate_calls == 1
    assert proc.stdin.closed
    assert proc.stdout.closed


def test_unexpected_ready_payload_is_rejected_and_cleaned() -> None:
    controller = _controller()
    proc = _FakeProcess(_line({"status": "loading"}))

    with pytest.raises(RuntimeError, match="required ready signal"):
        controller.wait_until_ready(proc)

    assert proc.terminate_calls == 1
    assert proc.stdin.closed
    assert proc.stdout.closed


@pytest.mark.parametrize(
    "payload",
    (
        {"ready": True},
        {"ready": True, "runtime": {"schema_version": 2}},
        {
            "ready": True,
            "runtime": {
                **_CPU_RUNTIME,
                "gpu_backend": "auto",
            },
        },
        {
            "ready": True,
            "runtime": {
                **_CPU_RUNTIME,
                "gpu_device_id": 0,
            },
        },
    ),
)
def test_ready_runtime_contract_fails_closed(payload: dict) -> None:
    controller = _controller()
    proc = _FakeProcess(_line(payload))

    with pytest.raises(RuntimeError, match="runtime"):
        controller.wait_until_ready(proc)

    assert proc.terminate_calls == 1


def test_repeated_ready_wait_returns_stored_runtime_without_reading_again() -> None:
    controller = _controller()
    proc = _FakeProcess(_ready_line())

    assert controller.wait_until_ready(proc) == _CPU_RUNTIME
    assert controller.wait_until_ready(proc) == _CPU_RUNTIME

    controller.stop(proc)


def test_dead_worker_is_cleaned_without_writing_request() -> None:
    controller = _controller()
    proc = _FakeProcess(_EOF)
    proc.returncode = 7

    assert controller.recognize(proc, "/tmp/page.png") is None
    assert proc.stdin.writes == []
    assert proc.stdin.closed
    assert proc.stdout.closed


def test_cancellation_interrupts_wait_and_terminates_worker() -> None:
    cancel_event = threading.Event()
    controller = _controller(cancel_event)
    proc = _FakeProcess()
    outcome: queue.Queue[Exception | None] = queue.Queue()

    def _wait() -> None:
        try:
            controller.wait_until_ready(proc)
        except Exception as exc:
            outcome.put(exc)
        else:
            outcome.put(None)

    waiter = threading.Thread(target=_wait, daemon=True)
    waiter.start()
    assert proc.stdout.read_started.wait(timeout=1)

    cancel_event.set()
    waiter.join(timeout=1)

    assert not waiter.is_alive()
    error = outcome.get_nowait()
    assert isinstance(error, InterruptedError)
    assert "cancelled" in str(error)
    assert proc.terminate_calls == 1
    assert proc.poll() == -15


def test_stop_is_idempotent_and_closes_protocol_pipes() -> None:
    controller = _controller()
    proc = _FakeProcess(_ready_line())

    controller.stop(proc)
    controller.stop(proc)

    assert proc.stdin.closed
    assert proc.stdout.closed
    assert proc.kill_calls == 0


@pytest.mark.parametrize("image_path", ["bad\npath.png", "bad\rpath.png"])
def test_request_rejects_line_break_in_image_path(image_path: str) -> None:
    controller = _controller()
    proc = _FakeProcess(_ready_line())

    with pytest.raises(ValueError, match="line break"):
        controller.recognize(proc, image_path)

    assert proc.stdin.writes == []
    controller.stop(proc)


def test_request_rejects_invalid_timeout_before_writing() -> None:
    controller = _controller()
    proc = _FakeProcess(_ready_line())

    with pytest.raises(ValueError, match="Invalid ocr_response_timeout_seconds"):
        controller.recognize(proc, "/tmp/page.png", timeout_seconds=0)

    assert proc.stdin.writes == []
    controller.stop(proc)
