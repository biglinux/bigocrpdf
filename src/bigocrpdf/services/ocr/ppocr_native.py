"""The OCR engine, ppocr-native: finding it, its command line, and its workers.

ppocr-native runs PP-OCRv6 detection and recognition and answers in JSON
lines. It reads binary PPM and PGM only, which is why OCR inputs are written
as PPM.
"""

import contextlib
import functools
import json
import logging
import os
import select
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bigocrpdf.services.ocr.config import OCRConfig, OCRResult

logger = logging.getLogger(__name__)

WORKER = "ppocr-native"
# The recognition model the plans hold: one model for every script it reads.
MODEL = "PP-OCRv6 small"
# The worker reads the same variable; the default is where its package installs.
PLANS_ENV = "PPOCR_NATIVE_PLANS"
DEFAULT_PLANS = Path("/usr/share/ppocr-native")
PLAN_FILES = (
    "PP-OCRv6_det_small.plan",
    "PP-OCRv6_rec_small_argmax.plan",
    "PP-OCRv6_rec_small.characters.txt",
)
OCR_IMAGE_SUFFIX = ".ppm"
# Loading the models or reading one dense page takes seconds even on the
# slowest kernel tier; past this the worker is hung.
REQUEST_TIMEOUT_SECONDS = 300.0
# How often a waiting request looks at its job's cancel event.
_CANCEL_POLL_MS = 100
_MAX_REPLY_BYTES = 64 * 1024 * 1024


class ProtocolError(RuntimeError):
    """A worker reply that breaks the JSON-lines contract."""


class ReadError(RuntimeError):
    """The worker could not read an image; the message says why."""


def plans_dir() -> Path:
    return Path(os.environ.get(PLANS_ENV) or DEFAULT_PLANS)


@functools.cache
def find_worker() -> str | None:
    """The ppocr-native executable, when it and its plans are installed.

    The worker validates the plans when it loads them; this only confirms
    they are there, so a broken install is reported before a job starts.
    """
    worker = shutil.which(WORKER)
    if worker is None:
        return None
    missing = [name for name in PLAN_FILES if not (plans_dir() / name).is_file()]
    if missing:
        logger.warning(f"{worker} found, but {plans_dir()} lacks {', '.join(missing)}")
        return None
    return worker


def physical_cores() -> int:
    """The cores this process may run on, counting SMT siblings once."""
    cpus = os.sched_getaffinity(0)
    cores = set()
    for cpu in cpus:
        try:
            siblings = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
            cores.add(siblings.read_text(encoding="ascii").strip())
        except OSError:
            return len(cpus)
    return len(cores)


def recognisers(requested: int | None = None) -> int:
    """Recogniser threads for a worker: one per physical core unless fewer
    are asked for, since SMT siblings add memory without speed."""
    cores = physical_cores()
    return cores if requested is None else max(1, min(requested, cores))


def worker_command(config: OCRConfig, *, threads: int | None = None) -> list[str]:
    """The command of a persistent ppocr-native worker.

    Each thread is a recogniser reading a page's lines; see ``recognisers``.
    """
    worker = find_worker()
    if worker is None:
        raise FileNotFoundError(f"{WORKER} is not installed")
    command = [worker, "--threads", str(recognisers(threads))]
    for flag, value in (
        ("--box-thresh", config.box_thresh),
        ("--unclip-ratio", config.unclip_ratio),
        ("--text-score", config.text_score_threshold),
        ("--limit-side-len", config.detection_limit_side_len),
    ):
        command.extend([flag, str(value)])
    if config.detection_full_resolution:
        command.append("--full-resolution")
    command.append("--persistent")
    return command


def parse_reply(text: str | bytes) -> dict:
    """One reply line as a JSON object."""
    try:
        reply = json.loads(text)
    except ValueError as exc:
        raise ProtocolError(f"{WORKER} answered {text[:200]!r}") from exc
    if not isinstance(reply, dict):
        raise ProtocolError(f"{WORKER} answered {text[:200]!r}")
    return reply


def ocr_results(reply: dict) -> list[OCRResult]:
    """The text lines of a page reply.

    The worker already dropped the lines scoring below ``--text-score``.
    """
    if reply.get("error"):
        raise ProtocolError(f"{WORKER}: {reply['error']}")
    if "boxes" not in reply:
        raise ProtocolError(f"{WORKER} reply has no boxes")
    boxes, txts, scores = reply["boxes"], reply.get("txts"), reply.get("scores")
    if boxes is None:
        if txts in (None, []) and scores in (None, []):
            return []
        raise ProtocolError(f"{WORKER} reply has text but no boxes")
    if not (isinstance(boxes, list) and isinstance(txts, list) and isinstance(scores, list)):
        raise ProtocolError(f"{WORKER} reply fields must be arrays")
    if not len(boxes) == len(txts) == len(scores):
        raise ProtocolError(f"{WORKER} reply arrays differ in length")
    return [
        OCRResult(text, box, score) for text, box, score in zip(txts, boxes, scores, strict=True)
    ]


class Worker:
    """A persistent ppocr-native process answering one JSON line per request.

    The process loads its models while the caller does other work; the first
    request, or ``wait_ready``, waits for its ready line. A request waits at
    most ``REQUEST_TIMEOUT_SECONDS``, and returns at once when ``cancel_event``
    is set; either way the worker is killed. A request to a worker that has
    exited starts a new one.
    """

    def __init__(
        self, command: list[str], cancel_event: threading.Event | None = None, threads: int = 0
    ) -> None:
        self.threads = threads
        self.tier: int | None = None
        self._command = command
        self._cancel_event = cancel_event or threading.Event()
        self._proc = self._start()

    def _start(self) -> subprocess.Popen:
        logger.debug(f"Starting {' '.join(self._command)}")
        return subprocess.Popen(self._command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def wait_ready(self) -> int:
        """Wait for the models to load and return the kernel tier the worker
        runs: 3 is AVX2 with FMA, 2 AVX, 1 SSE4.1, 0 plain code."""
        if self.tier is None:
            ready = self._read()
            tier = ready.get("tier")
            if ready.get("ready") is not True:
                self.close()
                raise RuntimeError(f"{WORKER} did not start: {ready.get('fatal', ready)}")
            if isinstance(tier, bool) or not isinstance(tier, int) or not 0 <= tier <= 3:
                self.close()
                raise ProtocolError(f"{WORKER} reported no valid kernel tier: {ready}")
            self.tier = tier
        return self.tier

    def request(self, line: str) -> dict:
        """Send one request line and return the reply."""
        if "\n" in line or "\r" in line:
            raise ValueError(f"{WORKER} request cannot hold a line break")
        if self._cancel_event.is_set():
            raise InterruptedError(f"{WORKER} request cancelled")
        if self.exited():
            self._release_pipes()
            self._proc = self._start()
            self.tier = None
        self.wait_ready()
        assert self._proc.stdin is not None
        self._proc.stdin.write(os.fsencode(line) + b"\n")
        self._proc.stdin.flush()
        return self._read()

    def recognize(self, image_path: str) -> dict:
        """The page reply for an image.

        A worker that died or broke the protocol is killed and the image is
        sent once more to a new one; a timeout is not retried. Raises
        ``ReadError`` when the image could not be read.
        """
        try:
            try:
                reply = self.request(image_path)
            except (OSError, RuntimeError) as exc:
                if isinstance(exc, (InterruptedError, TimeoutError)):
                    raise
                logger.warning(f"{WORKER} failed on {image_path}, starting it again: {exc}")
                self.kill()
                reply = self.request(image_path)
        except InterruptedError:
            raise
        except (OSError, RuntimeError) as exc:
            self.kill()
            raise ReadError(f"{WORKER} failed on {image_path}: {exc}") from exc
        if reply.get("error"):
            raise ReadError(f"{WORKER} could not read {image_path}: {reply['error']}")
        return reply

    def _read(self) -> dict:
        """The next reply line, read straight from the pipe so the deadline holds."""
        assert self._proc.stdout is not None
        fd = self._proc.stdout.fileno()
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        deadline = time.monotonic() + REQUEST_TIMEOUT_SECONDS
        line = bytearray()
        while not line.endswith(b"\n"):
            if self._cancel_event.is_set():
                self.kill()
                raise InterruptedError(f"{WORKER} request cancelled")
            if time.monotonic() > deadline:
                self.kill()
                raise TimeoutError(f"{WORKER} gave no answer in {REQUEST_TIMEOUT_SECONDS:g} s")
            if not poller.poll(_CANCEL_POLL_MS):
                continue
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                raise RuntimeError(f"{WORKER} exited")
            line += chunk
            if len(line) > _MAX_REPLY_BYTES:
                self.kill()
                raise ProtocolError(f"{WORKER} reply exceeds {_MAX_REPLY_BYTES} bytes")
        return parse_reply(bytes(line))

    def exited(self) -> bool:
        return self._proc.poll() is not None

    def kill(self) -> None:
        """Stop the worker now; the next request starts a new one."""
        with contextlib.suppress(ProcessLookupError):
            self._proc.kill()
        self._proc.wait()

    def close(self) -> None:
        """Let the worker exit at the end of its input, killing it after 5 s."""
        self._release_pipes()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.kill()

    def _release_pipes(self) -> None:
        for pipe in (self._proc.stdin, self._proc.stdout):
            # A dead worker leaves an unsent request that closing would flush.
            if pipe is not None:
                with contextlib.suppress(BrokenPipeError):
                    pipe.close()


def start_ocr_worker(
    config: OCRConfig, cancel_event: threading.Event, threads: int | None = None
) -> Worker:
    """A persistent OCR worker at idle priority, loading its models."""
    threads = recognisers(threads)
    command = worker_command(config, threads=threads)
    for tool, args in (("ionice", ["-c", "3", "-t"]), ("nice", ["-n", "19"])):
        if path := shutil.which(tool):
            command = [path, *args, *command]
    return Worker(command, cancel_event, threads)


class Detector:
    """The ppocr-native detector, for the deskew and dewarp steps.

    Each job owns one. It asks ``worker`` when given, a full worker the
    caller owns and closes; otherwise it starts a detect-only worker on the
    first request. Requests carry temporary file paths, and the lock keeps
    them, and ``close``, one at a time.
    """

    def __init__(
        self, cancel_event: threading.Event | None = None, worker: Worker | None = None
    ) -> None:
        self._cancel_event = cancel_event or threading.Event()
        self._lock = threading.Lock()
        self._worker = worker
        self._owns_worker = worker is None

    def _request(self, *fields: str) -> dict:
        if any("\t" in field for field in fields):
            raise ValueError("detector request fields cannot hold tabs")
        with self._lock:
            if self._worker is None:
                worker = find_worker()
                if worker is None:
                    raise RuntimeError(f"{WORKER} is not installed")
                self._worker = Worker([worker, "--persistent", "--detect-only"], self._cancel_event)
            reply = self._worker.request("\t".join(fields))
        if "error" in reply:
            raise RuntimeError(f"{WORKER}: {reply['error']}")
        return reply

    def probability_map(self, image_bgr: Any) -> Any:
        """The detector's probability map of an image already at its input size.

        The worker normalizes the pixels in R, G, B order with the ImageNet
        mean and deviation.
        """
        import cv2
        import numpy as np

        height, width = image_bgr.shape[:2]
        with temp_file(".ppm") as image_path, temp_file(".f32") as map_path:
            if not cv2.imwrite(image_path, image_bgr):
                raise OSError(f"OpenCV could not write {image_path}")
            reply = self._request("probmap", image_path, map_path)
            if (reply.get("height"), reply.get("width")) != (height, width):
                raise RuntimeError(f"{WORKER} returned a map of the wrong size: {reply}")
            return np.fromfile(map_path, dtype="<f4", count=height * width).reshape(height, width)

    def boxes(self, image_bgr: Any, max_side: int) -> list:
        """Text boxes on the image as given, detected with the page fitted within ``max_side``."""
        import cv2

        with temp_file(".ppm") as image_path:
            if not cv2.imwrite(image_path, image_bgr):
                raise OSError(f"OpenCV could not write {image_path}")
            boxes = self._request("detect", image_path, str(max_side)).get("boxes")
        if not isinstance(boxes, list):
            raise RuntimeError(f"{WORKER} returned no box list")
        return boxes

    def close(self) -> None:
        """Stop its own worker once any request in flight has its answer."""
        with self._lock:
            if self._worker is not None and self._owns_worker:
                self._worker.close()
                self._worker = None


@contextlib.contextmanager
def temp_file(suffix: str) -> Iterator[str]:
    """A private temporary path, removed on exit."""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="bigocrpdf-ocr-")
    os.close(fd)
    try:
        yield path
    finally:
        Path(path).unlink(missing_ok=True)
