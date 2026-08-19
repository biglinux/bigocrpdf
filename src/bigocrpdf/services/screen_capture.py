"""
BigOcrPdf - Screen Capture OCR Service

This module provides screen capture and OCR functionality using external tools
(Spectacle, GNOME Screenshot, Flameshot) and RapidOCR for text extraction.
"""

import json
import math
import os
import secrets
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
from gi.repository import Gio, GLib
from PIL import Image, ImageOps, UnidentifiedImageError

from bigocrpdf.services.rapidocr_service.config import (
    DEFAULT_MAX_IMAGE_MEGAPIXELS,
    OCRConfig,
    OCRResult,
)
from bigocrpdf.services.rapidocr_service.ocr_worker_engine import build_ocr_worker_command
from bigocrpdf.services.rapidocr_service.preprocessor import ImagePreprocessor
from bigocrpdf.services.rapidocr_service.text_formatting_controller import (
    TextFormattingController,
)
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.temp_manager import mkstemp as tm_mkstemp
from bigocrpdf.utils.temp_manager import remove_file as tm_remove


class PortalCaptureStatus(StrEnum):
    """Terminal state of a screenshot portal attempt."""

    SUCCESS = "success"
    CANCELLED = "cancelled"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class CliCaptureStatus(StrEnum):
    """Terminal state of one command-line screenshot attempt."""

    SUCCESS = "success"
    CANCELLED = "cancelled"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class PortalCaptureResult:
    """Portal state plus the borrowed path returned on success."""

    status: PortalCaptureStatus
    path: str | None = None


class ImageOcrStatus(StrEnum):
    """Terminal state of one image OCR request."""

    SUCCESS = "success"
    EMPTY = "empty"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass(frozen=True)
class ImageOcrOutcome:
    """Text or actionable failure state returned to the request owner."""

    status: ImageOcrStatus
    text: str | None = None
    message: str | None = None


class OcrWorkerProtocolError(ValueError):
    """Raised when the OCR worker does not return its documented JSON contract."""


class ImageInputError(ValueError):
    """Raised when an untrusted image cannot be decoded within safety limits."""


class ImageOcrCancelled(Exception):
    """Raised inside a worker when its request owner cancels the operation."""


class ImageOcrRequest:
    """Thread-safe cancellation and subprocess ownership for one image OCR job."""

    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._portal_connection = None
        self._portal_handle: str | None = None
        self._done = False
        self.cancellable = Gio.Cancellable()

    @property
    def is_cancelled(self) -> bool:
        """Whether the owner requested cancellation."""
        return self._cancelled.is_set()

    def raise_if_cancelled(self) -> None:
        """Stop worker code at the next cooperative cancellation point."""
        if self.is_cancelled:
            raise ImageOcrCancelled

    def bind_process(self, process: subprocess.Popen) -> None:
        """Attach a child process so cancellation can terminate it."""
        with self._lock:
            self._process = process
            should_terminate = self._cancelled.is_set() or self._done
        if should_terminate:
            self._terminate_process(process)

    def unbind_process(self, process: subprocess.Popen) -> None:
        """Detach a child process if it is still the active one."""
        with self._lock:
            if self._process is process:
                self._process = None

    def bind_portal(self, connection, handle: str) -> None:
        """Attach the XDG request object so cancellation can close it."""
        with self._lock:
            self._portal_connection = connection
            self._portal_handle = handle
            should_close = self._cancelled.is_set() or self._done
        if should_close:
            self._close_portal_request(connection, handle)

    def unbind_portal(self, handle: str) -> None:
        """Detach a portal request after its terminal response."""
        with self._lock:
            if self._portal_handle == handle:
                self._portal_connection = None
                self._portal_handle = None

    def close_portal(self) -> None:
        """Close the currently attached portal request without cancelling the job."""
        with self._lock:
            connection = self._portal_connection
            handle = self._portal_handle
        if connection is not None and handle:
            self._close_portal_request(connection, handle)

    def cancel(self) -> None:
        """Cancel once and promptly signal every currently owned operation."""
        with self._lock:
            if self._done or self._cancelled.is_set():
                return
            self._cancelled.set()
            process = self._process
            connection = self._portal_connection
            handle = self._portal_handle

        self.cancellable.cancel()
        if process is not None:
            self._terminate_process(process)
        if connection is not None and handle:
            self._close_portal_request(connection, handle)

    def complete(self) -> bool:
        """Mark terminal, release references, and report whether cancel won."""
        with self._lock:
            was_cancelled = self._cancelled.is_set()
            self._done = True
            self._process = None
            self._portal_connection = None
            self._portal_handle = None
            return was_cancelled

    @staticmethod
    def _terminate_process(process: subprocess.Popen) -> None:
        try:
            if process.poll() is None:
                process.terminate()
        except OSError as e:
            logger.debug(f"Could not terminate cancelled image OCR worker: {e}")

    @staticmethod
    def _close_portal_request(connection, handle: str) -> None:
        try:
            connection.call(
                "org.freedesktop.portal.Desktop",
                handle,
                "org.freedesktop.portal.Request",
                "Close",
                None,
                None,
                Gio.DBusCallFlags.NONE,
                -1,
                None,
                None,
                None,
            )
        except (GLib.Error, OSError, TypeError) as e:
            logger.debug(f"Could not close image capture portal request: {e}")


MAX_IMAGE_FILE_BYTES = 192 * 1024 * 1024
DIRECT_IMAGE_HARD_MAX_MEGAPIXELS = 24.0

_SAFE_IMAGE_FORMATS = frozenset(
    (
        "AVIF",
        "BMP",
        "GIF",
        "JPEG",
        "PNG",
        "PPM",
        "TIFF",
        "WEBP",
    )
)


def _safe_runtime_image_formats() -> tuple[str, ...]:
    """Return conservative in-process Pillow decoders available at runtime."""
    Image.init()
    return tuple(
        image_format for image_format in sorted(_SAFE_IMAGE_FORMATS) if image_format in Image.OPEN
    )


def get_supported_image_extensions() -> frozenset[str]:
    """Return extensions backed by the safe runtime decoder allowlist."""
    formats = frozenset(_safe_runtime_image_formats())
    return frozenset(
        extension
        for extension, image_format in Image.registered_extensions().items()
        if image_format in formats
    )


@dataclass(frozen=True)
class _ImageProcessJob:
    """One request owned by the bounded latest-image worker lane."""

    image_path: str | None
    config: OCRConfig
    callback: Callable[[ImageOcrOutcome], None]
    on_processing: Callable[[], None] | None
    request: ImageOcrRequest


class ScreenCaptureService:
    """Service to capture screen regions and extract text using RapidOCR."""

    def __init__(self) -> None:
        """Create one bounded, latest-only lane for direct image OCR."""
        self._image_lane_lock = threading.Lock()
        self._image_lane_pending: _ImageProcessJob | None = None
        self._image_lane_active: _ImageProcessJob | None = None
        self._image_lane_thread: threading.Thread | None = None
        self._image_lane_stopping = False

    def process_image_file(
        self,
        image_path: str,
        callback: Callable[[ImageOcrOutcome], None],
        on_processing: Callable[[], None] | None = None,
        language: str = "latin",
        config: OCRConfig | None = None,
    ) -> ImageOcrRequest:
        """Process an existing image file and extract text using RapidOCR.

        Args:
            image_path: Path to the image file
            callback: Callback function to receive the result (text, error)
            on_processing: Optional callback invoked when processing starts
            language: Language/script code for OCR (default: "latin")
        """
        return self._queue_image_job(
            image_path=image_path,
            config=config or OCRConfig(language=language),
            callback=callback,
            on_processing=on_processing,
        )

    def _queue_image_job(
        self,
        *,
        image_path: str | None,
        config: OCRConfig,
        callback: Callable[[ImageOcrOutcome], None],
        on_processing: Callable[[], None] | None,
    ) -> ImageOcrRequest:
        """Queue one direct-image or capture request on the shared bounded lane."""
        request = ImageOcrRequest()
        job = _ImageProcessJob(
            image_path=image_path,
            config=config,
            callback=callback,
            on_processing=on_processing,
            request=request,
        )
        superseded: _ImageProcessJob | None = None
        active: _ImageProcessJob | None = None
        thread_to_start: threading.Thread | None = None
        rejected = False
        with self._image_lane_lock:
            if self._image_lane_stopping:
                rejected = True
            else:
                superseded = self._image_lane_pending
                active = self._image_lane_active
                self._image_lane_pending = job
                if self._image_lane_thread is None:
                    thread_to_start = threading.Thread(
                        target=self._run_image_lane,
                        daemon=True,
                    )
                    self._image_lane_thread = thread_to_start

        if active is not None:
            active.request.cancel()
        if superseded is not None:
            self._cancel_queued_image_job(superseded)
        if rejected:
            self._cancel_queued_image_job(job)
        elif thread_to_start is not None:
            try:
                thread_to_start.start()
            except BaseException:
                with self._image_lane_lock:
                    if self._image_lane_thread is thread_to_start:
                        self._image_lane_thread = None
                    if self._image_lane_pending is job:
                        self._image_lane_pending = None
                request.complete()
                raise
        return request

    def _run_image_lane(self) -> None:
        """Run at most one decoder while retaining only the newest queued job."""
        while True:
            with self._image_lane_lock:
                job = self._image_lane_pending
                self._image_lane_pending = None
                if job is None:
                    self._image_lane_thread = None
                    return
                self._image_lane_active = job

            try:
                if job.image_path is None:
                    self._run_capture_thread(
                        job.config,
                        job.callback,
                        job.on_processing,
                        job.request,
                    )
                else:
                    self._run_image_process(
                        job.image_path,
                        job.config,
                        job.callback,
                        job.on_processing,
                        job.request,
                    )
            finally:
                with self._image_lane_lock:
                    if self._image_lane_active is job:
                        self._image_lane_active = None

    def _cancel_queued_image_job(self, job: _ImageProcessJob) -> None:
        """Complete a job that was superseded before its worker could start."""
        job.request.cancel()
        job.request.complete()
        self._invoke_callback(
            job.callback,
            ImageOcrOutcome(ImageOcrStatus.CANCELLED),
        )

    def shutdown(self, *, wait: bool = False) -> None:
        """Reject new image jobs and cancel active and queued work."""
        with self._image_lane_lock:
            self._image_lane_stopping = True
            active = self._image_lane_active
            pending = self._image_lane_pending
            self._image_lane_pending = None
            worker = self._image_lane_thread

        if active is not None:
            active.request.cancel()
        if pending is not None:
            self._cancel_queued_image_job(pending)
        if wait and worker is not None and worker is not threading.current_thread():
            worker.join()

    def _run_image_process(
        self,
        image_path: str,
        config: OCRConfig,
        callback: Callable[[ImageOcrOutcome], None],
        on_processing: Callable[[], None] | None,
        request: ImageOcrRequest | None = None,
    ) -> None:
        """Execute the image processing in a thread."""
        request = request or ImageOcrRequest()
        try:
            request.raise_if_cancelled()
            self._invoke_processing_callback(on_processing)
            text, error = self._extract_text_result(
                image_path,
                config,
                request=request,
            )
            outcome = self._outcome_from_text_result(text, error)
        except ImageOcrCancelled:
            outcome = ImageOcrOutcome(ImageOcrStatus.CANCELLED)
        except Exception as e:
            logger.error(f"Image OCR worker error: {e}")
            outcome = ImageOcrOutcome(
                ImageOcrStatus.ERROR,
                message=_("OCR processing failed."),
            )
        if request.complete():
            outcome = ImageOcrOutcome(ImageOcrStatus.CANCELLED)
        self._invoke_callback(callback, outcome)

    def capture_screen_region(
        self,
        callback: Callable[[ImageOcrOutcome], None],
        on_processing: Callable[[], None] | None = None,
        language: str = "latin",
        config: OCRConfig | None = None,
    ) -> ImageOcrRequest:
        """Capture a region of the screen and extract text from it.

        Args:
            callback: Callback function to receive the result (text, error)
            on_processing: Optional callback invoked when processing starts
            language: Language/script code for OCR (default: "latin")
        """
        return self._queue_image_job(
            image_path=None,
            config=config or OCRConfig(language=language),
            callback=callback,
            on_processing=on_processing,
        )

    def _run_capture_thread(
        self,
        config: OCRConfig,
        callback: Callable[[ImageOcrOutcome], None],
        on_processing: Callable[[], None] | None,
        request: ImageOcrRequest | None = None,
    ) -> None:
        """Execute the capture and OCR process in a thread."""
        request = request or ImageOcrRequest()
        temp_path = None
        outcome: ImageOcrOutcome | None = None
        try:
            # Generate a tracked temporary file path
            fd, temp_path = tm_mkstemp(suffix=".png", prefix="bigocrpdf_capture_")
            os.close(fd)

            request.raise_if_cancelled()
            outcome = self._capture_into_owned_file(temp_path, request)
            request.raise_if_cancelled()

            # Check if file has content (screenshot was taken, not cancelled)
            if outcome is None and os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                request.raise_if_cancelled()
                self._invoke_processing_callback(on_processing)

                text, error = self._extract_text_result(
                    temp_path,
                    config,
                    request=request,
                )
                outcome = self._outcome_from_text_result(text, error)
            elif outcome is None:
                outcome = ImageOcrOutcome(
                    ImageOcrStatus.CANCELLED,
                )

        except ImageOcrCancelled:
            outcome = ImageOcrOutcome(ImageOcrStatus.CANCELLED)
        except ImageInputError as e:
            logger.warning(f"Rejected captured image: {e}")
            outcome = ImageOcrOutcome(ImageOcrStatus.ERROR, message=str(e))
        except Exception as e:
            logger.error(f"Screenshot capture error: {e}")
            outcome = ImageOcrOutcome(
                ImageOcrStatus.ERROR,
                message=_("Screen capture failed. Please try again or open an image file."),
            )
        finally:
            if temp_path:
                self._cleanup_temp_file(temp_path)

        if request.complete():
            outcome = ImageOcrOutcome(ImageOcrStatus.CANCELLED)
        self._invoke_callback(
            callback,
            outcome
            or ImageOcrOutcome(
                ImageOcrStatus.ERROR,
                message=_("Screen capture failed."),
            ),
        )

    # ── Screenshot Capture ──────────────────────────────────────────────

    @staticmethod
    def _prefers_native_capture_tool() -> bool:
        """Whether this desktop's own screenshot tool beats the portal for a region."""
        return "kde" in os.environ.get("XDG_CURRENT_DESKTOP", "").lower()

    def _capture_into_owned_file(
        self,
        temp_path: str,
        request: ImageOcrRequest,
    ) -> ImageOcrOutcome | None:
        """Capture a screen region into an owned file, reporting None on success.

        On KDE, `spectacle --region --background` opens the region selector directly
        and writes exactly the file we ask for. The portal's interactive mode there
        opens a dialog that defaults to a full-screen grab with the pointer drawn in,
        which costs several clicks and burns the cursor into the OCR input, so the
        portal runs second. Everywhere else the portal leads, because it is the only
        backend that works from inside a sandbox.
        """
        backends = [self._capture_with_cli_tools, self._capture_via_portal_into]
        if not self._prefers_native_capture_tool():
            backends.reverse()

        status = CliCaptureStatus.UNAVAILABLE
        for backend in backends:
            request.raise_if_cancelled()
            status = backend(temp_path, request)
            if status != CliCaptureStatus.UNAVAILABLE:
                break
        return self._outcome_for_capture_status(status)

    def _capture_via_portal_into(
        self,
        temp_path: str,
        request: ImageOcrRequest,
    ) -> CliCaptureStatus:
        """Run the portal backend and land its borrowed image in our owned file."""
        result = self._capture_via_portal(request)
        if result.status != PortalCaptureStatus.SUCCESS:
            # Both enums are StrEnums over the same terminal states.
            return CliCaptureStatus(result.status)
        if not result.path:
            return CliCaptureStatus.FAILED
        self._copy_owned_image_file(result.path, temp_path)
        return CliCaptureStatus.SUCCESS

    @staticmethod
    def _outcome_for_capture_status(status: CliCaptureStatus) -> ImageOcrOutcome | None:
        """Map a terminal capture status to the outcome to report, None when captured."""
        if status == CliCaptureStatus.SUCCESS:
            return None
        if status == CliCaptureStatus.CANCELLED:
            return ImageOcrOutcome(ImageOcrStatus.CANCELLED)
        if status == CliCaptureStatus.UNAVAILABLE:
            return ImageOcrOutcome(
                ImageOcrStatus.ERROR,
                message=_(
                    "No screenshot tool available. Please install spectacle, "
                    "gnome-screenshot, or flameshot."
                ),
            )
        return ImageOcrOutcome(
            ImageOcrStatus.ERROR,
            message=_("Screen capture failed. Please try again or open an image file."),
        )

    @staticmethod
    def _is_portal_unavailable_error(error: Exception) -> bool:
        """Return whether D-Bus reports that the screenshot portal is absent."""
        matches = getattr(error, "matches", None)
        if not callable(matches):
            return False

        unavailable_dbus_codes = (
            "SERVICE_UNKNOWN",
            "NAME_HAS_NO_OWNER",
            "SPAWN_SERVICE_NOT_FOUND",
            "UNKNOWN_INTERFACE",
            "UNKNOWN_METHOD",
            "UNKNOWN_OBJECT",
            "NO_SERVER",
        )
        try:
            dbus_domain = Gio.DBusError.quark()
            for code_name in unavailable_dbus_codes:
                code = getattr(Gio.DBusError, code_name, None)
                if code is not None and matches(dbus_domain, code):
                    return True
        except (AttributeError, TypeError, ValueError):
            pass

        unavailable_io_codes = (
            "NOT_FOUND",
            "CONNECTION_REFUSED",
            "HOST_UNREACHABLE",
        )
        try:
            io_domain = Gio.io_error_quark()
            for code_name in unavailable_io_codes:
                code = getattr(Gio.IOErrorEnum, code_name, None)
                if code is not None and matches(io_domain, code):
                    return True
        except (AttributeError, TypeError, ValueError):
            pass
        return False

    def _capture_via_portal(
        self,
        request: ImageOcrRequest | None = None,
    ) -> PortalCaptureResult:
        """Attempt screenshot via XDG Desktop Portal (Flatpak-safe).

        Returns:
            Terminal portal state and a borrowed path on success.
        """
        connection = None
        subscription_id = 0
        request = request or ImageOcrRequest()
        active_handle: str | None = None
        try:
            request.raise_if_cancelled()
            connection = Gio.bus_get_sync(
                Gio.BusType.SESSION,
                request.cancellable,
            )
            unique_name = connection.get_unique_name()
            if not unique_name:
                return PortalCaptureResult(PortalCaptureStatus.UNAVAILABLE)

            sender = unique_name.removeprefix(":").replace(".", "_")
            handle_token = f"bigocrpdf_{secrets.token_hex(8)}"
            expected_handle = f"/org/freedesktop/portal/desktop/request/{sender}/{handle_token}"
            active_handle = expected_handle
            request.bind_portal(connection, expected_handle)
            response_event = threading.Event()
            response_payload: dict[str, object] = {}

            def on_response(
                _connection,
                _sender_name,
                _object_path,
                _interface_name,
                _signal_name,
                parameters,
            ) -> None:
                response, results = parameters.unpack()
                response_payload["response"] = int(response)
                response_payload["results"] = results
                response_event.set()

            subscription_id = connection.signal_subscribe(
                "org.freedesktop.portal.Desktop",
                "org.freedesktop.portal.Request",
                "Response",
                expected_handle,
                None,
                Gio.DBusSignalFlags.NONE,
                on_response,
            )
            proxy = Gio.DBusProxy.new_sync(
                connection,
                Gio.DBusProxyFlags.NONE,
                None,
                "org.freedesktop.portal.Desktop",
                "/org/freedesktop/portal/desktop",
                "org.freedesktop.portal.Screenshot",
                request.cancellable,
            )
            result = proxy.call_sync(
                "Screenshot",
                GLib.Variant(
                    "(sa{sv})",
                    (
                        "",
                        {
                            "handle_token": GLib.Variant("s", handle_token),
                            "interactive": GLib.Variant("b", True),
                        },
                    ),
                ),
                Gio.DBusCallFlags.NONE,
                60000,
                request.cancellable,
            )
            if not result:
                return PortalCaptureResult(PortalCaptureStatus.FAILED)

            returned_handle = result.unpack()[0]
            if returned_handle != expected_handle and not response_event.is_set():
                request.unbind_portal(expected_handle)
                active_handle = returned_handle
                request.bind_portal(connection, returned_handle)
                connection.signal_unsubscribe(subscription_id)
                subscription_id = connection.signal_subscribe(
                    "org.freedesktop.portal.Desktop",
                    "org.freedesktop.portal.Request",
                    "Response",
                    returned_handle,
                    None,
                    Gio.DBusSignalFlags.NONE,
                    on_response,
                )

            deadline = time.monotonic() + 60
            while not response_event.wait(0.1):
                request.raise_if_cancelled()
                if time.monotonic() >= deadline:
                    logger.warning("XDG Portal screenshot request timed out")
                    request.close_portal()
                    return PortalCaptureResult(PortalCaptureStatus.FAILED)
            request.raise_if_cancelled()
            response = response_payload.get("response")
            if response == 1:
                return PortalCaptureResult(PortalCaptureStatus.CANCELLED)
            if response != 0:
                return PortalCaptureResult(PortalCaptureStatus.FAILED)

            results = response_payload.get("results")
            if not isinstance(results, dict):
                return PortalCaptureResult(PortalCaptureStatus.FAILED)
            uri_value = results.get("uri")
            if uri_value is None:
                return PortalCaptureResult(PortalCaptureStatus.FAILED)
            uri = uri_value.unpack() if hasattr(uri_value, "unpack") else uri_value
            if not isinstance(uri, str):
                return PortalCaptureResult(PortalCaptureStatus.FAILED)
            parsed = urlparse(uri)
            if parsed.scheme != "file" or parsed.hostname not in (None, "", "localhost"):
                return PortalCaptureResult(PortalCaptureStatus.FAILED)
            return PortalCaptureResult(
                PortalCaptureStatus.SUCCESS,
                unquote(parsed.path),
            )
        except ImageOcrCancelled:
            return PortalCaptureResult(PortalCaptureStatus.CANCELLED)
        except Exception as e:
            if request.is_cancelled:
                return PortalCaptureResult(PortalCaptureStatus.CANCELLED)
            if self._is_portal_unavailable_error(e):
                logger.debug(f"XDG Portal screenshot service unavailable: {e}")
                return PortalCaptureResult(PortalCaptureStatus.UNAVAILABLE)
            logger.warning(f"XDG Portal screenshot request failed: {e}")
            return PortalCaptureResult(PortalCaptureStatus.FAILED)
        finally:
            if connection is not None and subscription_id:
                connection.signal_unsubscribe(subscription_id)
            if active_handle:
                request.unbind_portal(active_handle)

    def _capture_with_cli_tools(
        self,
        temp_path: str,
        request: ImageOcrRequest | None = None,
    ) -> CliCaptureStatus:
        """Capture screen using CLI tools (spectacle, gnome-screenshot, flameshot).

        Args:
            temp_path: The path where the screenshot should be saved.

        Returns:
            Explicit success, cancellation, failure, or unavailability.
        """
        try:
            commands = self._get_screenshot_commands(temp_path)
            return self._try_screenshot_tools(
                commands,
                temp_path,
                request or ImageOcrRequest(),
            )
        except (OSError, subprocess.SubprocessError) as e:
            logger.error(f"Screenshot capture error: {e}")
            return CliCaptureStatus.FAILED

    def _get_screenshot_commands(self, temp_path: str) -> list[list[str]]:
        """Get ordered list of screenshot commands based on desktop environment.

        Args:
            temp_path: The path where the screenshot should be saved.

        Returns:
            List of command arrays to try in order.
        """
        desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()

        if "kde" in desktop:
            return [
                ["spectacle", "-r", "-b", "-n", "-o", temp_path],
                ["flameshot", "gui", "--raw"],
                ["gnome-screenshot", "-a", "-f", temp_path],
            ]
        return [
            ["gnome-screenshot", "-a", "-f", temp_path],
            ["flameshot", "gui", "--raw"],
            ["spectacle", "-r", "-b", "-n", "-o", temp_path],
        ]

    def _try_screenshot_tools(
        self,
        commands: list[list[str]],
        temp_path: str,
        request: ImageOcrRequest | None = None,
    ) -> CliCaptureStatus:
        """Try each screenshot tool until one succeeds or is cancelled.

        Args:
            commands: List of command arrays to try.
            temp_path: The path where the screenshot should be saved.

        Returns:
            Aggregate terminal status for the ordered fallback chain.
        """
        request = request or ImageOcrRequest()
        attempted_failure = False
        for cmd in commands:
            request.raise_if_cancelled()
            result = self._try_single_tool(cmd, temp_path, request)
            if result in (CliCaptureStatus.SUCCESS, CliCaptureStatus.CANCELLED):
                return result
            if result == CliCaptureStatus.FAILED:
                attempted_failure = True
        if attempted_failure:
            return CliCaptureStatus.FAILED
        return CliCaptureStatus.UNAVAILABLE

    def _try_single_tool(
        self,
        cmd: list[str],
        temp_path: str,
        request: ImageOcrRequest | None = None,
    ) -> CliCaptureStatus:
        """Try to execute a single screenshot tool.

        Args:
            cmd: Command array to execute.
            temp_path: The path where the screenshot should be saved.

        Returns:
            Explicit terminal state for this tool attempt.
        """
        request = request or ImageOcrRequest()
        try:
            self._truncate_capture_destination(temp_path)
            if cmd[0] == "flameshot":
                return self._run_flameshot(cmd, temp_path, request)
            return self._run_standard_tool(cmd, temp_path, request)
        except FileNotFoundError:
            return CliCaptureStatus.UNAVAILABLE
        except subprocess.TimeoutExpired:
            logger.warning(f"Screenshot tool {cmd[0]} timed out.")
            return CliCaptureStatus.FAILED
        except (OSError, subprocess.SubprocessError) as e:
            logger.warning(f"Error running screenshot tool {cmd[0]}: {e}")
            return CliCaptureStatus.FAILED

    @staticmethod
    def _truncate_capture_destination(temp_path: str) -> None:
        """Clear partial output before another screenshot tool owns the file."""
        flags = os.O_WRONLY | os.O_TRUNC | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(temp_path, flags)
        os.close(fd)

    def _run_flameshot(
        self,
        cmd: list[str],
        temp_path: str,
        request: ImageOcrRequest | None = None,
    ) -> CliCaptureStatus:
        """Run flameshot and save output to file.

        Args:
            cmd: Flameshot command array.
            temp_path: The path where the screenshot should be saved.

        Returns:
            Explicit result based on exit status and emitted PNG bytes.
        """
        request = request or ImageOcrRequest()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        request.bind_process(proc)
        try:
            stdout, stderr = self._communicate_process(proc, request, timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return CliCaptureStatus.FAILED
        finally:
            request.unbind_process(proc)
        if proc.returncode == 0 and stdout:
            with open(temp_path, "wb") as f:
                f.write(stdout)
            return CliCaptureStatus.SUCCESS
        if proc.returncode in {0, 2}:
            return CliCaptureStatus.CANCELLED
        logger.debug(f"Flameshot exited with code {proc.returncode}: {stderr.decode().strip()}")
        return CliCaptureStatus.FAILED

    def _run_standard_tool(
        self,
        cmd: list[str],
        temp_path: str,
        request: ImageOcrRequest | None = None,
    ) -> CliCaptureStatus:
        """Run a standard screenshot tool (spectacle, gnome-screenshot).

        Args:
            cmd: Command array to execute.

        Returns:
            Explicit result based on availability, exit status, and output.
        """
        request = request or ImageOcrRequest()
        tool_name = cmd[0]
        if shutil.which(tool_name) is None:
            return CliCaptureStatus.UNAVAILABLE

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        request.bind_process(proc)
        try:
            _stdout, stderr = self._communicate_process(proc, request, timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return CliCaptureStatus.FAILED
        finally:
            request.unbind_process(proc)
        # Spectacle has been observed to crash on exit after writing a complete PNG,
        # so the bytes on disk decide the outcome before the exit status does.
        if os.path.getsize(temp_path) > 0:
            return CliCaptureStatus.SUCCESS
        if proc.returncode != 0:
            logger.debug(
                f"{tool_name} exited with code {proc.returncode}: {stderr.decode().strip()}"
            )
            return CliCaptureStatus.FAILED
        return CliCaptureStatus.CANCELLED

    # ── RapidOCR Image Processing ───────────────────────────────────────

    def extract_text_from_image(
        self,
        image_path: str,
        language: str = "latin",
    ) -> str | None:
        """Extract text from an image using the configured RapidOCR model.

        Applies geometric corrections (deskew, orientation detection, border trimming)
        before running OCR for optimal accuracy with photographed documents.

        Args:
            image_path: Path to the image file
            language: Language/script code for OCR (default: "latin")

        Returns:
            Extracted text or None on error
        """
        text, _error = self._extract_text_result(
            image_path,
            OCRConfig(language=language),
        )
        return text

    def _extract_text_result(
        self,
        image_path: str,
        config: OCRConfig,
        *,
        request: ImageOcrRequest | None = None,
    ) -> tuple[str | None, str | None]:
        """Extract text and preserve a user-facing error for the request owner."""
        request = request or ImageOcrRequest()
        try:
            request.raise_if_cancelled()
            img = self._load_image_for_ocr(image_path, config, request=request)
            request.raise_if_cancelled()

            # Apply geometric corrections (deskew, orientation, border trim)
            # Color enhancements stay off to preserve the source for OCR.
            preprocessor = ImagePreprocessor(config)
            img = preprocessor.process(
                img,
                cancel_check=request.raise_if_cancelled,
            )
            request.raise_if_cancelled()

            # Write preprocessed image to temp file for OCR worker subprocess
            fd, temp_img_path = tm_mkstemp(suffix=".png")
            os.close(fd)

            try:
                if not cv2.imwrite(temp_img_path, img):
                    logger.error("Failed to write the preprocessed image")
                    return None, _("Could not prepare the image for OCR.")

                # Run OCR via subprocess (avoids GTK/ONNX Runtime conflicts)
                cmd = self._build_ocr_command(temp_img_path, config)
                logger.info(f"Running image OCR: language={config.language}")

                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                request.bind_process(proc)
                try:
                    stdout, stderr = self._communicate_process(
                        proc,
                        request,
                        timeout=120,
                    )
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                    logger.error("OCR processing timed out")
                    return None, _("OCR processing timed out.")
                finally:
                    request.unbind_process(proc)

                request.raise_if_cancelled()
                if proc.returncode != 0:
                    logger.error(f"OCR subprocess failed: {stderr}")
                    return None, _("OCR processing failed.")

                # Parse OCR results
                try:
                    results = self._parse_ocr_payload(stdout)
                except OcrWorkerProtocolError as e:
                    logger.error(f"Invalid OCR worker response: {e}")
                    return None, _("OCR processing failed.")
                if not results:
                    return None, None

                # Reuse the pipeline formatter so a capture and a page of the same
                # layout produce the same lines, columns, and paragraphs.
                text = TextFormattingController(config).format(results, float(img.shape[1]))
                return (text if text.strip() else None), None

            finally:
                tm_remove(temp_img_path)

        except ImageOcrCancelled:
            raise
        except FileNotFoundError:
            logger.error("RapidOCR worker not found")
            return (
                None,
                _("OCR engine not available. Please check your installation."),
            )
        except ImageInputError as e:
            logger.warning(f"Rejected image input: {e}")
            return None, str(e)
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            logger.error(f"OCR processing error: {e}")
            return None, _("OCR processing failed.")

    @staticmethod
    def _communicate_process(
        process: subprocess.Popen,
        request: ImageOcrRequest,
        *,
        timeout: float,
    ) -> tuple:
        """Communicate in short waits so cancellation interrupts child work."""
        deadline = time.monotonic() + timeout
        while True:
            if request.is_cancelled:
                ScreenCaptureService._reap_cancelled_process(process)
                raise ImageOcrCancelled
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, timeout)
            try:
                return process.communicate(timeout=min(0.2, remaining))
            except subprocess.TimeoutExpired:
                continue

    @staticmethod
    def _reap_cancelled_process(process: subprocess.Popen) -> None:
        """Drain and reap a process after the owner already sent SIGTERM."""
        try:
            process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
        except OSError as e:
            logger.debug(f"Could not reap cancelled image OCR worker: {e}")

    @staticmethod
    def _load_image_for_ocr(
        image_path: str,
        config: OCRConfig,
        *,
        request: ImageOcrRequest | None = None,
    ) -> np.ndarray:
        """Snapshot and decode one bounded still image into OpenCV BGR order."""
        request = request or ImageOcrRequest()
        request.raise_if_cancelled()
        canonical_path = os.path.realpath(image_path)
        open_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        open_flags |= getattr(os, "O_NOFOLLOW", 0)
        open_flags |= getattr(os, "O_NONBLOCK", 0)

        try:
            fd = os.open(canonical_path, open_flags)
        except (OSError, ValueError) as e:
            raise ImageInputError(_("Could not load image file.")) from e

        try:
            source_stream = os.fdopen(fd, "rb")
        except OSError as e:
            os.close(fd)
            raise ImageInputError(_("Could not load image file.")) from e

        try:
            with source_stream, tempfile.TemporaryFile(mode="w+b") as image_snapshot:
                initial_stat = os.fstat(source_stream.fileno())
                if not stat.S_ISREG(initial_stat.st_mode):
                    raise ImageInputError(_("Could not load image file."))
                if initial_stat.st_size <= 0:
                    raise ImageInputError(_("Unsupported or corrupted image file."))
                if initial_stat.st_size > MAX_IMAGE_FILE_BYTES:
                    raise ImageInputError(_("Image file is too large to process safely."))

                ScreenCaptureService._copy_image_snapshot(
                    source_stream,
                    image_snapshot,
                    request=request,
                )
                request.raise_if_cancelled()
                ScreenCaptureService._ensure_unchanged_file(
                    initial_stat,
                    os.fstat(source_stream.fileno()),
                )
                image_snapshot.seek(0)

                safe_formats = _safe_runtime_image_formats()
                if not safe_formats:
                    raise ImageInputError(_("No supported image decoder is available."))

                max_megapixels = float(config.max_image_megapixels)
                if not math.isfinite(max_megapixels) or max_megapixels <= 0:
                    max_megapixels = DEFAULT_MAX_IMAGE_MEGAPIXELS
                max_megapixels = min(
                    max_megapixels,
                    DIRECT_IMAGE_HARD_MAX_MEGAPIXELS,
                )
                max_pixels = int(max_megapixels * 1_000_000)

                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    try:
                        with Image.open(image_snapshot, formats=safe_formats) as probe:
                            ScreenCaptureService._validate_image_header(
                                probe,
                                max_pixels,
                                safe_formats,
                            )
                            probe.verify()

                        request.raise_if_cancelled()
                        image_snapshot.seek(0)

                        with Image.open(image_snapshot, formats=safe_formats) as source:
                            ScreenCaptureService._validate_image_header(
                                source,
                                max_pixels,
                                safe_formats,
                            )
                            request.raise_if_cancelled()
                            oriented = ImageOps.exif_transpose(source)
                            try:
                                pixels = ScreenCaptureService._image_to_rgb_array(
                                    oriented,
                                )
                            finally:
                                oriented.close()
                    except (
                        Image.DecompressionBombError,
                        Image.DecompressionBombWarning,
                        UnidentifiedImageError,
                        OSError,
                        SyntaxError,
                        ValueError,
                    ) as e:
                        if isinstance(e, ImageInputError):
                            raise
                        raise ImageInputError(_("Unsupported or corrupted image file.")) from e

        except ImageInputError:
            raise
        except OSError as e:
            raise ImageInputError(_("Could not load image file.")) from e

        return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _copy_owned_image_file(source_path: str, destination_path: str) -> None:
        """Copy a stable regular image into an already-owned bounded file."""
        canonical_source = os.path.realpath(source_path)
        source_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        source_flags |= getattr(os, "O_NOFOLLOW", 0)
        source_flags |= getattr(os, "O_NONBLOCK", 0)
        destination_flags = os.O_WRONLY | os.O_TRUNC | getattr(os, "O_CLOEXEC", 0)
        destination_flags |= getattr(os, "O_NOFOLLOW", 0)

        source_fd: int | None = None
        destination_fd: int | None = None
        try:
            source_fd = os.open(canonical_source, source_flags)
            destination_fd = os.open(destination_path, destination_flags)
        except OSError as e:
            for opened_fd in (source_fd, destination_fd):
                if opened_fd is not None:
                    os.close(opened_fd)
            raise ImageInputError(_("Could not load image file.")) from e

        assert source_fd is not None
        assert destination_fd is not None
        with os.fdopen(source_fd, "rb") as source_stream:
            with os.fdopen(destination_fd, "wb") as destination_stream:
                initial_stat = os.fstat(source_stream.fileno())
                if not stat.S_ISREG(initial_stat.st_mode) or initial_stat.st_size <= 0:
                    raise ImageInputError(_("Could not load image file."))
                if initial_stat.st_size > MAX_IMAGE_FILE_BYTES:
                    raise ImageInputError(_("Image file is too large to process safely."))
                ScreenCaptureService._copy_image_snapshot(
                    source_stream,
                    destination_stream,
                )
                ScreenCaptureService._ensure_unchanged_file(
                    initial_stat,
                    os.fstat(source_stream.fileno()),
                )

    @staticmethod
    def _copy_image_snapshot(
        source_stream,
        image_snapshot,
        *,
        request: ImageOcrRequest | None = None,
    ) -> None:
        """Copy a bounded input into a private, immutable decoding snapshot."""
        request = request or ImageOcrRequest()
        copied = 0
        while True:
            request.raise_if_cancelled()
            chunk = source_stream.read(min(1024 * 1024, MAX_IMAGE_FILE_BYTES + 1 - copied))
            if not chunk:
                break
            copied += len(chunk)
            if copied > MAX_IMAGE_FILE_BYTES:
                raise ImageInputError(_("Image file is too large to process safely."))
            image_snapshot.write(chunk)
        request.raise_if_cancelled()

    @staticmethod
    def _image_to_rgb_array(image: Image.Image) -> np.ndarray:
        """Composite transparency on white and return owned RGB pixels."""
        has_alpha = "A" in image.getbands() or "transparency" in image.info
        if not has_alpha:
            rgb_image = image if image.mode == "RGB" else image.convert("RGB")
            try:
                rgb_image.load()
                return np.asarray(rgb_image, dtype=np.uint8).copy()
            finally:
                if rgb_image is not image:
                    rgb_image.close()

        foreground = image.convert("RGBA")
        alpha = foreground.getchannel("A")
        foreground_rgb = foreground.convert("RGB")
        background = Image.new("RGB", foreground.size, (255, 255, 255))
        try:
            background.paste(foreground_rgb, (0, 0), alpha)
            background.load()
            return np.asarray(background, dtype=np.uint8).copy()
        finally:
            background.close()
            foreground_rgb.close()
            alpha.close()
            foreground.close()

    @staticmethod
    def _validate_image_header(
        image: Image.Image,
        max_pixels: int,
        safe_formats: tuple[str, ...],
    ) -> None:
        """Validate image metadata before allocating its pixel buffer."""
        if image.format not in safe_formats:
            raise ImageInputError(_("Unsupported image format."))
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ImageInputError(_("Unsupported or corrupted image file."))
        if width * height > max_pixels:
            raise ImageInputError(_("Image dimensions exceed the configured safety limit."))
        try:
            image.seek(1)
        except EOFError:
            image.seek(0)
        else:
            raise ImageInputError(_("Animated or multi-page images are not supported."))

    @staticmethod
    def _ensure_unchanged_file(initial_stat: os.stat_result, current_stat: os.stat_result) -> None:
        """Reject in-place mutation while an untrusted image is decoded."""
        initial_identity = (
            initial_stat.st_dev,
            initial_stat.st_ino,
            initial_stat.st_size,
            initial_stat.st_mtime_ns,
            initial_stat.st_ctime_ns,
        )
        current_identity = (
            current_stat.st_dev,
            current_stat.st_ino,
            current_stat.st_size,
            current_stat.st_mtime_ns,
            current_stat.st_ctime_ns,
        )
        if current_identity != initial_identity:
            raise ImageInputError(
                _("Image file changed while it was being read. Please try again.")
            )

    def _build_ocr_command(self, image_path: str, config: OCRConfig) -> list[str]:
        """Build the OCR subprocess command.

        Args:
            image_path: Path to the image file to process
            config: OCR configuration

        Returns:
            Command list for subprocess.run()
        """
        cpu_count = os.cpu_count() or 4
        return build_ocr_worker_command(
            config,
            image_path=image_path,
            threads=max(2, cpu_count),
            openvino_available=config.engine_type != "onnxruntime",
        )

    @staticmethod
    def _parse_ocr_payload(stdout: str) -> list[OCRResult]:
        """Parse and validate the OCR worker's JSON contract."""
        try:
            raw = json.loads(stdout.strip())
        except json.JSONDecodeError as e:
            raise OcrWorkerProtocolError("worker output is not valid JSON") from e

        if not isinstance(raw, dict):
            raise OcrWorkerProtocolError("worker output must be a JSON object")
        if raw.get("error"):
            raise OcrWorkerProtocolError(f"worker reported an error: {raw['error']}")

        if "boxes" not in raw:
            raise OcrWorkerProtocolError("worker output is missing boxes")
        boxes = raw["boxes"]
        txts = raw.get("txts")
        scores = raw.get("scores")
        if boxes is None:
            if txts in (None, []) and scores in (None, []):
                return []
            raise OcrWorkerProtocolError("worker empty result contains unexpected values")
        if boxes == []:
            if txts == [] and scores == []:
                return []
            raise OcrWorkerProtocolError("worker output is missing OCR result arrays")
        if (
            not isinstance(boxes, list)
            or not isinstance(txts, list)
            or not isinstance(scores, list)
        ):
            raise OcrWorkerProtocolError("worker OCR result fields must be arrays")
        if len(boxes) != len(txts) or len(boxes) != len(scores):
            raise OcrWorkerProtocolError("worker OCR result arrays have different lengths")

        logger.info(f"RapidOCR found {len(boxes)} text regions")
        return [
            OCRResult(
                text=txts[index],
                box=boxes[index],
                confidence=scores[index],
            )
            for index in range(len(boxes))
        ]

    @classmethod
    def _parse_ocr_results(cls, stdout: str) -> list[OCRResult]:
        """Compatibility parser that logs invalid worker payloads and returns no rows.

        Args:
            stdout: Raw stdout from OCR worker

        Returns:
            List of OCRResult objects
        """
        try:
            return cls._parse_ocr_payload(stdout)
        except OcrWorkerProtocolError as e:
            logger.error(f"Failed to parse OCR result: {e}")
            return []

    # ── Callback Helpers ────────────────────────────────────────────────

    def _cleanup_temp_file(self, path: str) -> None:
        """Clean up a tracked temporary file.

        Args:
            path: Path to the file to delete
        """
        if path:
            tm_remove(path)

    @staticmethod
    def _outcome_from_text_result(
        text: str | None,
        error: str | None,
    ) -> ImageOcrOutcome:
        if error:
            return ImageOcrOutcome(ImageOcrStatus.ERROR, message=error)
        if text:
            return ImageOcrOutcome(ImageOcrStatus.SUCCESS, text=text)
        return ImageOcrOutcome(ImageOcrStatus.EMPTY)

    @staticmethod
    def _invoke_callback(
        callback: Callable[[ImageOcrOutcome], None],
        outcome: ImageOcrOutcome,
    ) -> None:
        """Schedule a request-owned callback with the result.

        Args:
            outcome: Explicit terminal request state
        """

        def callback_wrapper():
            callback(outcome)
            return False

        GLib.idle_add(callback_wrapper)

    @staticmethod
    def _invoke_processing_callback(callback: Callable[[], None] | None) -> None:
        """Schedule the request-owned processing callback."""
        if callback:

            def callback_wrapper():
                callback()
                return False

            GLib.idle_add(callback_wrapper)
