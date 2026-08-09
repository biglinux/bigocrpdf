"""Persistent OCR subprocess lifecycle and line protocol."""

# ``subprocess.Popen`` receives private protocol state at runtime.
# pyright: reportAttributeAccessIssue=false

import json
import math
import queue
import shutil
import subprocess
import threading
import time
from collections.abc import Callable

from bigocrpdf.services.rapidocr_service.config import OCRConfig
from bigocrpdf.services.rapidocr_service.ocr_worker_engine import build_ocr_worker_command
from bigocrpdf.services.rapidocr_service.resource_manager import ResourceTier, detect_resources
from bigocrpdf.utils.logger import logger

_DEFAULT_READY_TIMEOUT_SECONDS = 300.0
_DEFAULT_RESPONSE_TIMEOUT_SECONDS = 300.0
_PROTOCOL_POLL_SECONDS = 0.1
_MAX_PROTOCOL_LINE_CHARS = 64 * 1024 * 1024
_READER_STOP = object()
_RUNTIME_BY_ENGINE_LABEL = {
    "openvino_cpu": ("openvino", "off"),
    "onnxruntime_cpu": ("onnxruntime", "off"),
    "paddle_cuda": ("paddle", "paddle"),
    "torch_cuda": ("torch", "torch"),
    "tensorrt": ("tensorrt", "tensorrt"),
    "onnxruntime_cuda_experimental": (
        "onnxruntime",
        "onnxruntime_cuda_experimental",
    ),
}


class _ProtocolLineReader:
    """Read one line per request without blocking the caller indefinitely."""

    def __init__(self, stream) -> None:
        self._stream = stream
        self._requests: queue.Queue[object] = queue.Queue()
        self._responses: queue.Queue[str | Exception] = queue.Queue()
        self._thread = threading.Thread(
            target=self._serve,
            name="bigocr-stdout-reader",
            daemon=True,
        )
        self._thread.start()

    def _serve(self) -> None:
        while self._requests.get() is not _READER_STOP:
            try:
                line = self._stream.readline(_MAX_PROTOCOL_LINE_CHARS + 1)
                if len(line) > _MAX_PROTOCOL_LINE_CHARS:
                    raise ValueError("OCR subprocess response exceeds the protocol size limit")
                if line and not line.endswith("\n"):
                    raise ValueError("OCR subprocess returned a truncated protocol line")
                self._responses.put(line)
            except Exception as exc:
                self._responses.put(exc)

    def readline(self, timeout_seconds: float, *, cancel_event: threading.Event) -> str:
        if cancel_event.is_set():
            raise InterruptedError("OCR subprocess wait cancelled")

        self._requests.put(object())
        deadline = time.monotonic() + timeout_seconds
        while True:
            if cancel_event.is_set():
                raise InterruptedError("OCR subprocess wait cancelled")

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("OCR subprocess response timed out")
            try:
                response = self._responses.get(timeout=min(_PROTOCOL_POLL_SECONDS, remaining))
            except queue.Empty:
                continue

            if isinstance(response, Exception):
                raise response
            return response

    def close(self) -> None:
        self._requests.put(_READER_STOP)
        self._thread.join(timeout=1)


class OCRSubprocessController:
    """Own persistent OCR subprocess creation, requests, and cleanup."""

    def __init__(
        self,
        config: OCRConfig,
        cancel_event: threading.Event,
        openvino_checker: Callable[[], bool],
    ) -> None:
        self._config = config
        self._cancel_event = cancel_event
        self._openvino_checker = openvino_checker

    def _build_command(self, ocr_threads: int = 0) -> list[str]:
        """Build command-line args for a persistent OCR subprocess."""
        ocr_threads = self._resolve_ocr_threads(ocr_threads)

        try:
            openvino_available = (
                self._config.engine_type != "onnxruntime" and self._openvino_checker()
            )
        except (ImportError, OSError, AttributeError):
            openvino_available = False

        return build_ocr_worker_command(
            self._config,
            persistent=True,
            threads=ocr_threads,
            openvino_available=openvino_available,
            low_memory_openvino=detect_resources().tier == ResourceTier.CONSTRAINED,
        )

    @staticmethod
    def _resolve_ocr_threads(ocr_threads: int) -> int:
        """Resolve the concrete inference thread count sent to the worker."""
        import multiprocessing

        if ocr_threads <= 0:
            return max(2, multiprocessing.cpu_count())
        return ocr_threads

    @staticmethod
    def _background_command(command: list[str]) -> list[str]:
        """Apply best-effort Linux scheduling priority through executable argv."""
        background_command = list(command)
        if ionice := shutil.which("ionice"):
            background_command = [ionice, "-c", "3", "-t", *background_command]
        if nice := shutil.which("nice"):
            background_command = [nice, "-n", "19", *background_command]
        return background_command

    def launch(self, ocr_threads: int = 0) -> subprocess.Popen:
        """Start an OCR subprocess without waiting for model readiness."""
        resolved_threads = self._resolve_ocr_threads(ocr_threads)
        cmd = self._background_command(self._build_command(ocr_threads=resolved_threads))
        logger.debug(f"Launching OCR subprocess (background): {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        process._bigocr_threads = resolved_threads
        return process

    def _timeout(
        self,
        config_name: str,
        default_seconds: float,
        override_seconds: float | None,
    ) -> float:
        raw_timeout = (
            override_seconds
            if override_seconds is not None
            else getattr(self._config, config_name, default_seconds)
        )
        try:
            timeout_seconds = float(raw_timeout)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid {config_name}: {raw_timeout!r}") from exc
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError(f"Invalid {config_name}: {raw_timeout!r}")
        return timeout_seconds

    @staticmethod
    def _reader(proc: subprocess.Popen) -> _ProtocolLineReader:
        reader = getattr(proc, "_bigocr_protocol_reader", None)
        if isinstance(reader, _ProtocolLineReader):
            return reader
        if proc.stdout is None:
            raise RuntimeError("OCR subprocess stdout pipe is unavailable")
        reader = _ProtocolLineReader(proc.stdout)
        proc._bigocr_protocol_reader = reader
        return reader

    def _read_line(self, proc: subprocess.Popen, timeout_seconds: float) -> str:
        return self._reader(proc).readline(
            timeout_seconds,
            cancel_event=self._cancel_event,
        )

    @staticmethod
    def _close_protocol(proc: subprocess.Popen) -> None:
        proc._bigocr_ready = False
        proc._bigocr_runtime = None
        reader = getattr(proc, "_bigocr_protocol_reader", None)
        if isinstance(reader, _ProtocolLineReader):
            reader.close()
            proc._bigocr_protocol_reader = None
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream is not None and not stream.closed:
                    stream.close()
            except (OSError, ValueError):
                pass

    def _abort(self, proc: subprocess.Popen, reason: str) -> None:
        logger.warning(f"Terminating OCR subprocess PID {proc.pid}: {reason}")
        try:
            self._terminate(proc)
        finally:
            self._close_protocol(proc)

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> None:
        try:
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=2)
        except (ProcessLookupError, OSError):
            pass
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except (ProcessLookupError, subprocess.TimeoutExpired, OSError):
                logger.warning(f"Could not confirm OCR subprocess PID {proc.pid} termination")

    def wait_until_ready(  # noqa: C901 - protocol failure states stay explicit
        self,
        proc: subprocess.Popen,
        timeout_seconds: float | None = None,
    ) -> dict[str, object]:
        """Wait a bounded time for the OCR subprocess to load its model."""
        if proc is None:
            raise RuntimeError("OCR subprocess is unavailable")
        if getattr(proc, "_bigocr_ready", False) is True:
            runtime = getattr(proc, "_bigocr_runtime", None)
            if isinstance(runtime, dict):
                return dict(runtime)
            raise RuntimeError("OCR subprocess has no stored runtime diagnostics")

        timeout_seconds = self._timeout(
            "ocr_ready_timeout_seconds",
            _DEFAULT_READY_TIMEOUT_SECONDS,
            timeout_seconds,
        )
        if proc.poll() is not None:
            self.stop(proc)
            raise RuntimeError("OCR subprocess exited before ready signal")

        logger.debug("Waiting for OCR subprocess ready signal...")
        try:
            ready_line = self._read_line(proc, timeout_seconds)
        except InterruptedError:
            self._abort(proc, "startup cancelled")
            raise
        except TimeoutError as exc:
            self._abort(proc, "ready timeout")
            raise RuntimeError(
                f"OCR subprocess ready signal timed out after {timeout_seconds:.1f}s"
            ) from exc
        except (EOFError, OSError, RuntimeError, ValueError) as exc:
            self._abort(proc, "invalid ready response")
            raise RuntimeError("OCR subprocess stdout closed before ready signal") from exc

        if not ready_line:
            self._abort(proc, "stdout closed before ready")
            raise RuntimeError("OCR subprocess stdout closed before ready signal")

        try:
            ready = json.loads(ready_line.strip())
        except json.JSONDecodeError as exc:
            self._abort(proc, "invalid ready JSON")
            raise RuntimeError("OCR subprocess returned invalid ready JSON") from exc
        if not isinstance(ready, dict):
            self._abort(proc, "invalid ready payload")
            raise RuntimeError("OCR subprocess returned invalid ready payload")
        if ready.get("fatal"):
            self._abort(proc, "fatal startup response")
            raise RuntimeError(f"OCR subprocess fatal error: {ready['fatal']}")
        if ready.get("ready") is not True:
            self._abort(proc, "unexpected startup response")
            raise RuntimeError("OCR subprocess did not send the required ready signal")

        try:
            runtime = self._validate_runtime_diagnostics(ready.get("runtime"))
        except ValueError as exc:
            self._abort(proc, "invalid runtime diagnostics")
            raise RuntimeError("OCR subprocess returned invalid runtime diagnostics") from exc
        proc._bigocr_ready = True
        proc._bigocr_runtime = runtime
        logger.info("OCR subprocess ready (model loaded)")
        return dict(runtime)

    @staticmethod
    def _validate_runtime_diagnostics(value: object) -> dict[str, object]:
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ValueError("runtime diagnostics require schema_version=1")
        engine_label = value.get("engine_label")
        if not isinstance(engine_label, str) or engine_label not in _RUNTIME_BY_ENGINE_LABEL:
            raise ValueError("runtime diagnostics contain an unsupported engine label")
        expected_engine, expected_backend = _RUNTIME_BY_ENGINE_LABEL[engine_label]
        if value.get("engine_type") != expected_engine:
            raise ValueError("runtime diagnostics engine type does not match its label")
        if value.get("gpu_backend") != expected_backend:
            raise ValueError("runtime diagnostics GPU backend does not match its label")
        device_id = value.get("gpu_device_id")
        if expected_backend == "off":
            if device_id is not None:
                raise ValueError("CPU runtime diagnostics must not name a GPU device")
        elif isinstance(device_id, bool) or not isinstance(device_id, int) or device_id < 0:
            raise ValueError("GPU runtime diagnostics require a non-negative device ID")
        return {
            "schema_version": 1,
            "engine_label": engine_label,
            "engine_type": expected_engine,
            "gpu_backend": expected_backend,
            "gpu_device_id": device_id,
        }

    def recognize(  # noqa: C901 - protocol failure states stay explicit
        self,
        proc: subprocess.Popen,
        image_path: str,
        timeout_seconds: float | None = None,
    ) -> dict | None:
        """Send one image path to a persistent OCR subprocess."""
        if "\n" in image_path or "\r" in image_path:
            raise ValueError("OCR image path contains a line break")
        if proc is None or proc.poll() is not None:
            logger.error("OCR subprocess not running")
            if proc is not None:
                self.stop(proc)
            return None

        response_timeout = self._timeout(
            "ocr_response_timeout_seconds",
            _DEFAULT_RESPONSE_TIMEOUT_SECONDS,
            timeout_seconds,
        )
        try:
            self.wait_until_ready(proc)
            if proc.stdin is None:
                logger.error("OCR subprocess stdin pipe is unavailable")
                self._abort(proc, "stdin pipe unavailable")
                return None

            proc.stdin.write(f"{image_path}\n")
            proc.stdin.flush()
            line = self._read_line(proc, response_timeout)

            if not line:
                logger.error("OCR subprocess stdout closed")
                self._abort(proc, "stdout closed during OCR response")
                return None

            result = json.loads(line.strip())
            if not isinstance(result, dict):
                raise ValueError("OCR subprocess response must be a JSON object")
            if result.get("error"):
                logger.error(f"OCR error for {image_path}: {result['error']}")
                return None
            return result
        except InterruptedError:
            self._abort(proc, "request cancelled")
            raise
        except TimeoutError:
            logger.error("OCR subprocess response timed out")
            self._abort(proc, "response timeout")
            return None
        except RuntimeError as exc:
            logger.error(f"OCR subprocess startup failed: {exc}")
            return None
        except (BrokenPipeError, OSError) as exc:
            logger.error(f"OCR subprocess pipe error: {exc}")
            self._abort(proc, "pipe error")
            return None
        except (EOFError, json.JSONDecodeError, ValueError) as exc:
            logger.error(f"OCR subprocess invalid response: {exc}")
            self._abort(proc, "invalid response")
            return None

    def stop(self, proc: subprocess.Popen) -> None:
        """Gracefully stop a persistent OCR subprocess."""
        if proc is None:
            return

        logger.debug(f"Stopping OCR subprocess PID {proc.pid}...")
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
            logger.debug("Waiting for subprocess to exit...")
            proc.wait(timeout=10)
            logger.debug("Subprocess exited gracefully")
        except (ProcessLookupError, subprocess.TimeoutExpired, OSError) as exc:
            logger.debug(f"Error stopping subprocess: {exc}, terminating...")
            self._terminate(proc)
        finally:
            self._close_protocol(proc)
