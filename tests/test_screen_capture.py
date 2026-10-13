"""Screen capture, the image OCR lane, cancellation and image loading."""

import os
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from bigocrpdf.services import screen_capture as screen_capture_module
from bigocrpdf.services.ocr import ppocr_native
from bigocrpdf.services.ocr.config import OCRConfig, OCRResult
from bigocrpdf.services.ocr.text_formatting_controller import (
    TextFormattingController,
)
from bigocrpdf.services.screen_capture import (
    ImageOcrCancelled,
    ImageOcrRequest,
    ScreenCaptureService,
)


def test_portal_capture_keeps_borrowed_image_and_processes_owned_copy(
    monkeypatch,
    tmp_path,
):
    borrowed_path = tmp_path / "portal capture.png"
    borrowed_path.write_bytes(b"portal image")
    owned_path = tmp_path / "owned.png"
    processed_paths = []
    completed = []

    def fake_mkstemp(*, suffix, prefix):
        assert suffix == ".png"
        assert prefix == "bigocrpdf_capture_"
        fd = os.open(owned_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        return fd, str(owned_path)

    service = ScreenCaptureService()
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
    monkeypatch.setattr("bigocrpdf.services.screen_capture.tm_mkstemp", fake_mkstemp)
    monkeypatch.setattr(
        service,
        "_capture_via_portal",
        lambda _request: SimpleNamespace(status="success", path=str(borrowed_path)),
    )
    monkeypatch.setattr(
        service,
        "_extract_text_result",
        lambda path, _config, *, request: (
            processed_paths.append(path)
            or screen_capture_module.ImageOcrOutcome(
                screen_capture_module.ImageOcrStatus.SUCCESS, text="text"
            )
        ),
    )
    monkeypatch.setattr(
        service,
        "_invoke_processing_callback",
        lambda callback: callback() if callback is not None else None,
    )
    monkeypatch.setattr(
        service,
        "_invoke_callback",
        lambda callback, outcome: callback(outcome),
    )

    service._run_capture_thread(
        OCRConfig(),
        completed.append,
        None,
    )

    assert processed_paths == [str(owned_path)]
    assert len(completed) == 1
    assert completed[0].status == "success"
    assert completed[0].text == "text"
    assert borrowed_path.read_bytes() == b"portal image"
    assert not owned_path.exists()


def test_portal_capture_waits_for_response_before_returning_uri(
    monkeypatch,
    tmp_path,
):
    screenshot_path = tmp_path / "portal capture.png"
    screenshot_path.write_bytes(b"png")
    subscription = {}

    class PackedVariant:
        def __init__(self, _signature, value=None):
            self.value = _signature if value is None else value

        def unpack(self):
            return self.value

    class FakeConnection:
        def get_unique_name(self):
            return ":1.42"

        def signal_subscribe(
            self,
            sender,
            interface,
            member,
            object_path,
            arg0,
            flags,
            callback,
        ):
            subscription.update(
                {
                    "sender": sender,
                    "interface": interface,
                    "member": member,
                    "object_path": object_path,
                    "callback": callback,
                }
            )
            return 17

        def signal_unsubscribe(self, subscription_id):
            subscription["unsubscribed"] = subscription_id

    connection = FakeConnection()

    class FakeProxy:
        def call_sync(self, method, parameters, _flags, _timeout, _cancellable):
            assert method == "Screenshot"
            assert subscription["member"] == "Response"
            _parent, options = parameters.unpack()
            token = options["handle_token"].unpack()
            expected_handle = f"/org/freedesktop/portal/desktop/request/1_42/{token}"
            assert subscription["object_path"] == expected_handle
            subscription["callback"](
                connection,
                "org.freedesktop.portal.Desktop",
                expected_handle,
                "org.freedesktop.portal.Request",
                "Response",
                PackedVariant((0, {"uri": screenshot_path.as_uri()})),
            )
            return PackedVariant((expected_handle,))

    monkeypatch.setattr(
        screen_capture_module.Gio,
        "bus_get_sync",
        lambda _bus_type, _cancellable: connection,
    )
    monkeypatch.setattr(
        screen_capture_module.Gio.DBusProxy,
        "new_sync",
        lambda *_args: FakeProxy(),
    )
    monkeypatch.setattr(screen_capture_module.GLib, "Error", RuntimeError)
    monkeypatch.setattr(screen_capture_module.GLib, "Variant", PackedVariant)

    result = ScreenCaptureService()._capture_via_portal()

    assert result.status == "success"
    assert result.path == str(screenshot_path)
    assert subscription["unsubscribed"] == 17


def test_portal_cancellation_is_terminal_and_does_not_launch_cli_fallback(
    monkeypatch,
    tmp_path,
):
    owned_path = tmp_path / "owned.png"
    completed = []

    def fake_mkstemp(*, suffix, prefix):
        fd = os.open(owned_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        return fd, str(owned_path)

    service = ScreenCaptureService()
    cli_capture = MagicMock()
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
    monkeypatch.setattr("bigocrpdf.services.screen_capture._", lambda message: message)
    monkeypatch.setattr("bigocrpdf.services.screen_capture.tm_mkstemp", fake_mkstemp)
    monkeypatch.setattr(
        service,
        "_capture_via_portal",
        lambda _request: SimpleNamespace(status="cancelled", path=None),
    )
    monkeypatch.setattr(service, "_capture_with_cli_tools", cli_capture)
    monkeypatch.setattr(
        service,
        "_invoke_callback",
        lambda callback, outcome: callback(outcome),
    )

    service._run_capture_thread(
        OCRConfig(),
        completed.append,
        None,
    )

    cli_capture.assert_not_called()
    assert len(completed) == 1
    assert completed[0].status == "cancelled"
    assert not owned_path.exists()


def test_kde_captures_with_spectacle_before_asking_the_portal(monkeypatch, tmp_path):
    # The portal's interactive mode on KDE opens a dialog defaulting to a full-screen
    # grab with the pointer drawn in; Spectacle's region mode is one drag.
    service = ScreenCaptureService()
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    portal = MagicMock()
    monkeypatch.setattr(service, "_capture_via_portal_into", portal)
    monkeypatch.setattr(
        service,
        "_capture_with_cli_tools",
        lambda _path, _request: screen_capture_module.CliCaptureStatus.SUCCESS,
    )

    outcome = service._capture_into_owned_file(
        str(tmp_path / "owned.png"),
        screen_capture_module.ImageOcrRequest(),
    )

    assert outcome is None
    portal.assert_not_called()


def test_capture_falls_back_to_the_portal_when_no_native_tool_answers(monkeypatch, tmp_path):
    service = ScreenCaptureService()
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    monkeypatch.setattr(
        service,
        "_capture_with_cli_tools",
        lambda _path, _request: screen_capture_module.CliCaptureStatus.UNAVAILABLE,
    )
    monkeypatch.setattr(
        service,
        "_capture_via_portal_into",
        lambda _path, _request: screen_capture_module.CliCaptureStatus.SUCCESS,
    )

    outcome = service._capture_into_owned_file(
        str(tmp_path / "owned.png"),
        screen_capture_module.ImageOcrRequest(),
    )

    assert outcome is None


@pytest.mark.parametrize(
    "status",
    [
        screen_capture_module.CliCaptureStatus.UNAVAILABLE,
        screen_capture_module.CliCaptureStatus.FAILED,
    ],
)
def test_capture_failure_names_the_capture(monkeypatch, tmp_path, status):
    service = ScreenCaptureService()
    monkeypatch.setattr("bigocrpdf.services.screen_capture._", lambda message: message)
    monkeypatch.setattr(service, "_capture_with_cli_tools", lambda _path, _request: status)
    monkeypatch.setattr(service, "_capture_via_portal_into", lambda _path, _request: status)

    outcome = service._capture_into_owned_file(str(tmp_path / "owned.png"), ImageOcrRequest())

    assert outcome is not None
    assert outcome.status == "error"
    assert outcome.heading == "Could not capture the screen"


def test_screenshot_tool_that_crashes_after_writing_still_yields_the_capture(
    monkeypatch,
    tmp_path,
):
    # Observed on Plasma: spectacle wrote a complete PNG and then exited on SIGSEGV.
    capture_path = tmp_path / "shot.png"
    capture_path.write_bytes(b"complete png bytes")
    service = ScreenCaptureService()
    monkeypatch.setattr(screen_capture_module.shutil, "which", lambda _tool: "/usr/bin/spectacle")
    monkeypatch.setattr(
        screen_capture_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: SimpleNamespace(
            args=["spectacle"],
            returncode=-11,
            poll=lambda: -11,
            communicate=lambda timeout=None: (b"", b""),
        ),
    )

    status = service._run_standard_tool(
        ["spectacle", "-r", "-b", "-n", "-o", str(capture_path)],
        str(capture_path),
        screen_capture_module.ImageOcrRequest(),
    )

    assert status == screen_capture_module.CliCaptureStatus.SUCCESS


@pytest.mark.parametrize(
    ("is_unavailable", "expected_status"),
    ((True, "unavailable"), (False, "failed")),
)
def test_portal_exception_falls_back_only_when_service_is_absent(
    monkeypatch,
    is_unavailable,
    expected_status,
):
    service = ScreenCaptureService()
    portal_error = RuntimeError("portal rejected request")
    monkeypatch.setattr(
        screen_capture_module.Gio,
        "bus_get_sync",
        MagicMock(side_effect=portal_error),
    )
    monkeypatch.setattr(
        service,
        "_is_portal_unavailable_error",
        MagicMock(return_value=is_unavailable),
    )

    result = service._capture_via_portal()

    assert result.status == expected_status


def test_portal_error_classifier_accepts_only_absence_errors(monkeypatch):
    class FakeError(Exception):
        def __init__(self, matching_code):
            super().__init__("portal error")
            self.matching_code = matching_code

        def matches(self, domain, code):
            return domain == 71 and code == self.matching_code

    dbus_error = SimpleNamespace(
        quark=lambda: 71,
        SERVICE_UNKNOWN=1,
        NAME_HAS_NO_OWNER=2,
        SPAWN_SERVICE_NOT_FOUND=3,
        UNKNOWN_INTERFACE=4,
        UNKNOWN_METHOD=5,
        UNKNOWN_OBJECT=6,
        NO_SERVER=7,
        ACCESS_DENIED=8,
        TIMED_OUT=9,
    )
    monkeypatch.setattr(screen_capture_module.Gio, "DBusError", dbus_error)

    assert ScreenCaptureService._is_portal_unavailable_error(FakeError(1)) is True
    assert ScreenCaptureService._is_portal_unavailable_error(FakeError(4)) is True
    assert ScreenCaptureService._is_portal_unavailable_error(FakeError(8)) is False
    assert ScreenCaptureService._is_portal_unavailable_error(FakeError(9)) is False


def test_image_lane_cancels_active_and_superseded_jobs_but_runs_latest(monkeypatch):
    first_started = threading.Event()
    release_first = threading.Event()
    latest_finished = threading.Event()
    processed_paths = []
    callbacks = {}

    def extract(path, _config, *, request):
        processed_paths.append(path)
        if path == "first":
            first_started.set()
            assert release_first.wait(2)
        request.raise_if_cancelled()
        return screen_capture_module.ImageOcrOutcome(
            screen_capture_module.ImageOcrStatus.SUCCESS, text=path
        )

    def callback_for(path):
        def on_complete(outcome):
            callbacks[path] = outcome
            if path == "latest":
                latest_finished.set()

        return on_complete

    service = ScreenCaptureService()
    monkeypatch.setattr(service, "_extract_text_result", extract)
    monkeypatch.setattr(
        "bigocrpdf.services.screen_capture.GLib.idle_add",
        lambda callback: callback(),
    )

    first = service.process_image_file("first", callback_for("first"))
    assert first_started.wait(2)
    superseded = service.process_image_file("superseded", callback_for("superseded"))
    latest = service.process_image_file("latest", callback_for("latest"))
    release_first.set()
    assert latest_finished.wait(2)
    service.shutdown(wait=True)

    assert processed_paths == ["first", "latest"]
    assert first.is_cancelled
    assert superseded.is_cancelled
    assert not latest.is_cancelled
    assert callbacks["first"].status == "cancelled"
    assert callbacks["superseded"].status == "cancelled"
    assert callbacks["latest"].text == "latest"


def test_image_processing_returns_a_request_owned_by_the_caller(monkeypatch):
    pending_threads = []

    class DeferredThread:
        daemon = False

        def __init__(self, *, target, daemon):
            assert daemon is True
            self.target = target
            pending_threads.append(self)

        def start(self):
            return None

    monkeypatch.setattr("bigocrpdf.services.screen_capture.threading.Thread", DeferredThread)

    service = ScreenCaptureService()
    request = service.process_image_file("image.png", MagicMock())

    assert isinstance(request, screen_capture_module.ImageOcrRequest)
    assert service._image_lane_pending.request is request


def test_image_lane_start_failure_rolls_back_and_allows_retry(monkeypatch):
    starts = 0
    completed = []

    class FlakyThread:
        def __init__(self, *, target, daemon):
            assert daemon is True
            self.target = target

        def start(self):
            nonlocal starts
            starts += 1
            if starts == 1:
                raise RuntimeError("thread unavailable")
            self.target()

        def join(self):
            raise AssertionError("an unstarted thread must not remain registered")

    service = ScreenCaptureService()
    monkeypatch.setattr("bigocrpdf.services.screen_capture.threading.Thread", FlakyThread)
    monkeypatch.setattr(
        service,
        "_extract_text_result",
        lambda path, _config, *, request: screen_capture_module.ImageOcrOutcome(
            screen_capture_module.ImageOcrStatus.SUCCESS, text=path
        ),
    )
    monkeypatch.setattr(
        "bigocrpdf.services.screen_capture.GLib.idle_add",
        lambda callback: callback(),
    )

    with pytest.raises(RuntimeError, match="thread unavailable"):
        service.process_image_file("first.png", completed.append)

    assert service._image_lane_thread is None
    assert service._image_lane_pending is None

    service.process_image_file("second.png", completed.append)
    service.shutdown(wait=True)

    assert starts == 2
    assert completed[-1].text == "second.png"


def test_capture_requests_share_the_bounded_latest_image_lane(monkeypatch):
    pending_threads = []
    completed = []

    class DeferredThread:
        def __init__(self, *, target, daemon):
            assert daemon is True
            self.target = target
            pending_threads.append(self)

        def start(self):
            return None

    service = ScreenCaptureService()
    monkeypatch.setattr("bigocrpdf.services.screen_capture.threading.Thread", DeferredThread)
    monkeypatch.setattr(
        "bigocrpdf.services.screen_capture.GLib.idle_add",
        lambda callback: callback(),
    )

    first = service.capture_screen_region(completed.append)
    second = service.capture_screen_region(completed.append)

    assert len(pending_threads) == 1
    assert first.is_cancelled
    assert not second.is_cancelled
    assert service._image_lane_pending.request is second
    assert completed[0].status == screen_capture_module.ImageOcrStatus.CANCELLED


def test_failed_cli_capture_tool_falls_back_to_the_next_tool(monkeypatch, tmp_path):
    service = ScreenCaptureService()
    destination = tmp_path / "capture.png"
    destination.write_bytes(b"")
    commands = [["first-tool"], ["second-tool"]]
    attempts = MagicMock(
        side_effect=[
            screen_capture_module.CliCaptureStatus.FAILED,
            screen_capture_module.CliCaptureStatus.SUCCESS,
        ]
    )
    monkeypatch.setattr(service, "_try_single_tool", attempts)

    status = service._try_screenshot_tools(
        commands,
        str(destination),
        screen_capture_module.ImageOcrRequest(),
    )

    assert status == screen_capture_module.CliCaptureStatus.SUCCESS
    assert attempts.call_count == 2


def test_flameshot_exit_two_is_user_cancellation(monkeypatch, tmp_path):
    process = SimpleNamespace(returncode=2)
    service = ScreenCaptureService()
    monkeypatch.setattr(
        screen_capture_module.subprocess, "Popen", lambda *_args, **_kwargs: process
    )
    monkeypatch.setattr(
        service,
        "_communicate_process",
        lambda _process, _request, timeout: (b"", b"aborted"),
    )

    status = service._run_flameshot(
        ["flameshot", "gui", "--raw"],
        str(tmp_path / "capture.png"),
    )

    assert status == screen_capture_module.CliCaptureStatus.CANCELLED


def test_cancelled_cli_capture_is_terminal(monkeypatch, tmp_path):
    service = ScreenCaptureService()
    destination = tmp_path / "capture.png"
    destination.write_bytes(b"")
    attempts = MagicMock(
        side_effect=[
            screen_capture_module.CliCaptureStatus.CANCELLED,
            AssertionError("fallback must not run after cancellation"),
        ]
    )
    monkeypatch.setattr(service, "_try_single_tool", attempts)

    status = service._try_screenshot_tools(
        [["first-tool"], ["second-tool"]],
        str(destination),
        screen_capture_module.ImageOcrRequest(),
    )

    assert status == screen_capture_module.CliCaptureStatus.CANCELLED
    attempts.assert_called_once()


def test_cancelling_request_terminates_its_bound_worker_once():
    request = screen_capture_module.ImageOcrRequest()
    process = MagicMock()
    process.poll.return_value = None

    request.bind_process(process)
    request.cancel()
    request.cancel()

    assert request.is_cancelled
    process.terminate.assert_called_once_with()


def test_cancelled_image_request_skips_ocr_and_completes_as_cancelled(monkeypatch):
    service = ScreenCaptureService()
    request = screen_capture_module.ImageOcrRequest()
    request.cancel()
    extract = MagicMock()
    outcomes = []
    pending_idle = []
    monkeypatch.setattr(service, "_extract_text_result", extract)
    monkeypatch.setattr(
        "bigocrpdf.services.screen_capture.GLib.idle_add",
        lambda idle_callback: pending_idle.append(idle_callback),
    )

    service._run_image_process(
        "image.png",
        OCRConfig(),
        outcomes.append,
        None,
        request,
    )
    for idle_callback in pending_idle:
        idle_callback()

    extract.assert_not_called()
    assert [outcome.status for outcome in outcomes] == ["cancelled"]


def test_cancelled_worker_is_reaped_and_killed_when_sigterm_does_not_finish():
    request = screen_capture_module.ImageOcrRequest()
    process = MagicMock()
    process.poll.return_value = None
    process.args = ["ocr-worker"]
    process.communicate.side_effect = [
        screen_capture_module.subprocess.TimeoutExpired(process.args, 1),
        ("", ""),
    ]
    request.bind_process(process)
    request.cancel()

    with pytest.raises(screen_capture_module.ImageOcrCancelled):
        ScreenCaptureService._communicate_process(
            process,
            request,
            timeout=120,
        )

    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.communicate.call_count == 2


def test_unexpected_image_worker_error_still_completes_request(monkeypatch):
    service = ScreenCaptureService()
    outcomes = []
    pending_idle = []
    monkeypatch.setattr("bigocrpdf.services.screen_capture._", lambda message: message)
    monkeypatch.setattr(
        "bigocrpdf.services.screen_capture.GLib.idle_add",
        lambda idle_callback: pending_idle.append(idle_callback),
    )
    monkeypatch.setattr(
        service,
        "_extract_text_result",
        MagicMock(side_effect=RuntimeError("worker failed")),
    )

    service._run_image_process(
        "image.png",
        OCRConfig(),
        outcomes.append,
        None,
    )
    for idle_callback in pending_idle:
        idle_callback()

    assert len(outcomes) == 1
    assert outcomes[0].status == "error"
    assert outcomes[0].message == "OCR processing failed."


class _FakeWorker:
    """Stands in for ppocr_native.Worker, answering ``recognize`` with ``reply``."""

    def __init__(self, reply):
        self.reply = reply
        self.cancel_event = None
        self.read = []
        self.closed = False

    def __call__(self, _command, cancel_event):
        self.cancel_event = cancel_event
        return self

    def recognize(self, path):
        self.read.append(path)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    def close(self):
        self.closed = True


def _read_image(monkeypatch, reply, *, config=None, request=None, process=None, imwrite=True):
    """Image OCR through a fake worker: the outcome, the worker, the preprocessor's arguments."""
    worker = _FakeWorker(reply)
    preprocessor_args = []

    def preprocessor(config, detector):
        preprocessor_args.append((config, detector))
        return SimpleNamespace(
            process=process or (lambda image, *, cancel_check: cancel_check() or image)
        )

    service = ScreenCaptureService()
    monkeypatch.setattr("bigocrpdf.services.screen_capture._", lambda message: message)
    image = SimpleNamespace(shape=(4, 4, 3))
    monkeypatch.setattr(service, "_load_image_for_ocr", lambda *_args, **_kwargs: image)
    monkeypatch.setattr(screen_capture_module, "ImagePreprocessor", preprocessor)
    monkeypatch.setattr(screen_capture_module.cv2, "imwrite", lambda _path, _image: imwrite)
    monkeypatch.setattr(ppocr_native, "worker_command", lambda _config: ["ocr-worker"])
    monkeypatch.setattr(ppocr_native, "Worker", worker)
    outcome = service._extract_text_result("image.png", config or OCRConfig(), request=request)
    return outcome, worker, preprocessor_args


def test_image_ocr_shares_one_worker(monkeypatch):
    config = OCRConfig(detection_full_resolution=True, max_image_megapixels=7)
    request = ImageOcrRequest()

    outcome, worker, preprocessor_args = _read_image(
        monkeypatch, {"boxes": [], "txts": [], "scores": []}, config=config, request=request
    )

    assert outcome.status == "empty"
    [(seen_config, detector)] = preprocessor_args
    assert seen_config is config
    assert detector._worker is worker
    assert worker.cancel_event is request.cancel_event
    assert worker.read[0].endswith(ppocr_native.OCR_IMAGE_SUFFIX)
    assert worker.closed


def test_cancelled_detection_cancels_request(monkeypatch):
    request = ImageOcrRequest()

    def process(_image, *, cancel_check):
        request.cancel()
        raise InterruptedError("ppocr-native request cancelled")

    with pytest.raises(ImageOcrCancelled):
        _read_image(monkeypatch, {"boxes": []}, request=request, process=process)


@pytest.mark.parametrize(
    ("reply", "detail"),
    [
        (ppocr_native.ReadError("could not read: /plans/rec.plan: truncated"), "truncated"),
        ({"txts": ["text"]}, "reply has no boxes"),
    ],
)
def test_worker_failure_cause_is_shown(monkeypatch, reply, detail):
    outcome, worker, _ = _read_image(monkeypatch, reply)

    assert outcome.status == "error"
    assert detail in (outcome.detail or "")
    assert ppocr_native.WORKER in (outcome.message or "")
    assert worker.closed


def test_unwritten_image_is_not_read(monkeypatch):
    outcome, worker, _ = _read_image(monkeypatch, {"boxes": []}, imwrite=False)

    assert outcome.message == "Could not prepare the image for OCR."
    assert worker.read == []
    assert worker.closed


def test_image_loader_rejects_pixel_budget_before_array_conversion(
    monkeypatch,
    tmp_path,
):
    image_path = tmp_path / "oversized.png"
    Image.new("RGB", (2000, 1000), "white").save(image_path)
    service = ScreenCaptureService()
    array_conversion = MagicMock(side_effect=AssertionError("pixels were decoded"))
    monkeypatch.setattr(screen_capture_module.np, "asarray", array_conversion)

    with pytest.raises(
        screen_capture_module.ImageInputError,
        match="dimensions exceed",
    ):
        service._load_image_for_ocr(
            str(image_path),
            OCRConfig(max_image_megapixels=1),
        )

    array_conversion.assert_not_called()


def test_image_loader_rejects_animated_or_multipage_images(tmp_path):
    image_path = tmp_path / "animated.gif"
    first = Image.new("RGB", (2, 2), "black")
    second = Image.new("RGB", (2, 2), "white")
    first.save(image_path, save_all=True, append_images=[second], format="GIF")

    with pytest.raises(
        screen_capture_module.ImageInputError,
        match="Animated or multi-page",
    ):
        ScreenCaptureService()._load_image_for_ocr(
            str(image_path),
            OCRConfig(),
        )


def test_image_loader_applies_exif_orientation(tmp_path):
    image_path = tmp_path / "rotated.jpg"
    source = Image.new("RGB", (2, 3), "white")
    exif = source.getexif()
    exif[274] = 6
    source.save(image_path, exif=exif)

    loaded = ScreenCaptureService()._load_image_for_ocr(
        str(image_path),
        OCRConfig(),
    )

    assert loaded.shape == (2, 3, 3)


def test_image_loader_composites_transparency_onto_white(tmp_path):
    image_path = tmp_path / "transparent.png"
    Image.new("RGBA", (1, 1), (255, 0, 0, 0)).save(image_path)

    loaded = ScreenCaptureService()._load_image_for_ocr(
        str(image_path),
        OCRConfig(),
    )

    assert loaded.tolist() == [[[255, 255, 255]]]


def test_image_loader_rejects_corrupted_images(tmp_path):
    image_path = tmp_path / "broken.png"
    image_path.write_bytes(b"not an image")

    with pytest.raises(
        screen_capture_module.ImageInputError,
        match="Unsupported or corrupted",
    ):
        ScreenCaptureService()._load_image_for_ocr(
            str(image_path),
            OCRConfig(),
        )


def test_image_loader_rejects_fifo_without_blocking(tmp_path):
    image_path = tmp_path / "image.pipe"
    os.mkfifo(image_path)

    with pytest.raises(screen_capture_module.ImageInputError):
        ScreenCaptureService()._load_image_for_ocr(
            str(image_path),
            OCRConfig(),
        )


def test_image_loader_rejects_encoded_size_before_pillow_open(
    monkeypatch,
    tmp_path,
):
    image_path = tmp_path / "huge.png"
    with image_path.open("wb") as stream:
        stream.truncate(screen_capture_module.MAX_IMAGE_FILE_BYTES + 1)
    pillow_open = MagicMock(side_effect=AssertionError("decoder was reached"))
    monkeypatch.setattr(screen_capture_module.Image, "open", pillow_open)

    with pytest.raises(
        screen_capture_module.ImageInputError,
        match="too large",
    ):
        ScreenCaptureService()._load_image_for_ocr(
            str(image_path),
            OCRConfig(),
        )

    pillow_open.assert_not_called()


def test_image_loader_accepts_content_independent_of_filename_extension(tmp_path):
    image_path = tmp_path / "scan.unknown"
    Image.new("RGB", (2, 2), "white").save(image_path, format="PNG")

    loaded = ScreenCaptureService()._load_image_for_ocr(
        str(image_path),
        OCRConfig(),
    )

    assert loaded.shape == (2, 2, 3)


def test_image_loader_rejects_disallowed_content_with_image_extension(tmp_path):
    image_path = tmp_path / "document.png"
    image_path.write_text("%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 1 1\n")

    with pytest.raises(
        screen_capture_module.ImageInputError,
        match="Unsupported or corrupted",
    ):
        ScreenCaptureService()._load_image_for_ocr(
            str(image_path),
            OCRConfig(),
        )


def test_supported_image_extensions_follow_safe_runtime_decoders():
    extensions = screen_capture_module.get_supported_image_extensions()

    assert ".png" in extensions
    assert ".jpg" in extensions
    assert ".jfif" in extensions
    assert ".ico" not in extensions
    assert ".eps" not in extensions
    assert ".psd" not in extensions
    assert ".jxl" not in extensions


class TestFormatText:
    """Image captures are formatted by the shared pipeline formatter."""

    @staticmethod
    def _format(results):
        return TextFormattingController(OCRConfig()).format(results, 400.0)

    def test_empty_results(self):
        assert self._format([]) == ""

    def test_single_line(self):
        results = [OCRResult(text="Hello World", box=[[0, 10], [200, 10], [200, 30], [0, 30]])]
        assert "Hello World" in self._format(results)

    def test_reading_order(self):
        # Second box is above first box — should appear first in output
        results = [
            OCRResult(text="Line 2", box=[[0, 50], [200, 50], [200, 70], [0, 70]]),
            OCRResult(text="Line 1", box=[[0, 10], [200, 10], [200, 30], [0, 30]]),
        ]
        text = self._format(results)
        assert text.find("Line 1") < text.find("Line 2")

    def test_paragraph_break(self):
        results = [
            OCRResult(text="Para 1", box=[[0, 10], [200, 10], [200, 30], [0, 30]]),
            OCRResult(text="Para 2", box=[[0, 60], [200, 60], [200, 80], [0, 80]]),
        ]
        assert "\n\n" in self._format(results)

    @pytest.mark.parametrize(
        ("gaps", "paragraph_sizes"),
        [
            ([5, 5, 25, 5, 5], [3, 3]),  # single-spaced, one paragraph gap
            ([24, 23, 25, 24, 24], [6]),  # double-spaced, no paragraph gap
            ([24, 23, 68, 24, 24], [3, 3]),  # double-spaced, one paragraph gap
        ],
    )
    def test_paragraph_gap_relative_to_line_spacing(self, gaps, paragraph_sizes):
        lines = [f"line {index} of flowing body text" for index in range(len(gaps) + 1)]
        results, top = [], 10
        for index, text in enumerate(lines):
            results.append(
                OCRResult(text=text, box=[[0, top], [390, top], [390, top + 20], [0, top + 20]])
            )
            if index < len(gaps):
                top += 20 + gaps[index]

        expected, start = [], 0
        for size in paragraph_sizes:
            expected.append(" ".join(lines[start : start + size]))
            start += size
        assert self._format(results).split("\n\n") == expected

    def test_boxes_on_one_visual_line_keep_left_to_right_order(self):
        # Real detections jitter vertically by a few pixels on the same line. Sorting
        # by box top alone reorders them, which scrambled every multi-box capture.
        results = [
            OCRResult(text="Nome:", box=[[10, 100], [90, 100], [90, 130], [10, 130]]),
            OCRResult(text="Joao", box=[[100, 98], [200, 98], [200, 128], [100, 128]]),
            OCRResult(text="Silva", box=[[210, 102], [320, 102], [320, 132], [210, 132]]),
        ]
        text = self._format(results)
        assert text.split() == ["Nome:", "Joao", "Silva"]
        assert "\n" not in text.strip()
