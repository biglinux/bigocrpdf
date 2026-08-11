import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from bigocrpdf.services.rapidocr_service import dewarp_detection, ocr_worker, ocr_worker_engine
from bigocrpdf.services.rapidocr_service.config import OCRConfig
from bigocrpdf.services.rapidocr_service.discovery import ModelDiscovery
from bigocrpdf.services.rapidocr_service.ocr_worker_engine import build_ocr_worker_command


class FakeEngineType:
    OPENVINO = "openvino"
    ONNXRUNTIME = "onnxruntime"
    PADDLE = "paddle"
    TORCH = "torch"
    TENSORRT = "tensorrt"


class FakeLangRec:
    LATIN = "latin"


class FakeOCRVersion:
    PPOCRV6 = "v6"


class FakeModelType:
    MOBILE = "mobile"
    SMALL = "small"
    MEDIUM = "medium"


def test_dewarp_detector_uses_the_shared_worker_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = object()

    def rapidocr(*, params):
        return detector

    monkeypatch.setattr(dewarp_detection, "_cached_detector", None)
    monkeypatch.setattr(dewarp_detection, "_cached_detector_key", None)
    monkeypatch.setattr(
        ocr_worker_engine,
        "_import_rapidocr_api",
        lambda: (FakeEngineType, FakeLangRec, FakeModelType, FakeOCRVersion, rapidocr),
    )
    monkeypatch.setattr(ocr_worker_engine, "_rec_lang", lambda _lang: "latin")
    monkeypatch.setattr(ocr_worker_engine, "_set_model_version_params", lambda *_args: None)

    assert dewarp_detection._get_inprocess_detector("latin", 1536) is detector


def test_worker_command_modes_share_every_configured_flag(tmp_path: Path) -> None:
    model_files = "PP-OCRv6_rec_medium.onnx PP-OCRv6_det_medium.onnx latin.ttf"
    for filename in model_files.split():
        (tmp_path / filename).touch()
    config = OCRConfig(
        language="pt",
        model_type="medium",
        rec_batch_num=8,
        model_base_path=tmp_path,
        font_base_path=tmp_path,
        gpu_backend="auto",
        gpu_device_id=2,
        use_textline_cls=True,
        gpu_fallback_to_cpu=False,
        detection_full_resolution=True,
    )
    one_shot = build_ocr_worker_command(
        config, image_path="/tmp/page.png", threads=12, openvino_available=False
    )
    persistent = build_ocr_worker_command(
        config, persistent=True, threads=3, openvino_available=False
    )

    def normalize(command: list[str]) -> list[str]:
        common = command[4:].copy()
        common[common.index("--threads") + 1] = "<threads>"
        return common

    assert normalize(one_shot) == normalize(persistent)
    assert one_shot[:3] == [
        sys.executable,
        "-m",
        "bigocrpdf.services.rapidocr_service.ocr_worker",
    ]
    expected_flags = "--gpu-fp16 --use-textline-cls --no-gpu-fallback --full-resolution --no-openvino --rec-model-path --det-model-path --font-path"
    assert set(expected_flags.split()) <= set(one_shot)
    assert "--low-memory-openvino" not in persistent
    assert "--low-memory-openvino" in build_ocr_worker_command(
        config,
        persistent=True,
        threads=3,
        low_memory_openvino=True,
    )
    with pytest.raises(ValueError, match="exactly one"):
        build_ocr_worker_command(config, threads=2)


def test_v6_models_do_not_mix_with_legacy_files(tmp_path: Path) -> None:
    v6_rec = tmp_path / "PP-OCRv6_rec_small.onnx"
    v6_det = tmp_path / "PP-OCRv6_det_small.onnx"
    v5_rec = tmp_path / "en_PP-OCRv5_rec_mobile_infer.onnx"
    v5_det = tmp_path / "ch_PP-OCRv5_mobile_det.onnx"
    v5_keys = tmp_path / "ppocrv5_en_dict.txt"
    for path in (v6_rec, v6_det, v5_rec, v5_det, v5_keys):
        path.touch()
    config = OCRConfig(
        language="en",
        model_type="small",
        model_base_path=tmp_path,
    )

    assert config.get_rec_model_path() == v6_rec
    assert config.get_det_model_path() == v6_det
    assert config.get_rec_keys_path() is None


def test_v5_files_do_not_make_models_available(tmp_path: Path) -> None:
    (tmp_path / "latin_PP-OCRv5_rec_mobile_infer.onnx").touch()
    assert ModelDiscovery(model_path=tmp_path).get_available_languages() == []


def test_v6_pair_exposes_only_unified_recognition(tmp_path: Path) -> None:
    for filename in ModelDiscovery.V6_MODEL_FILES:
        (tmp_path / filename).touch()
    assert ModelDiscovery(model_path=tmp_path).get_available_languages() == [
        ("latin", "Automatic multilingual recognition (PP-OCRv6)")
    ]


def test_worker_rejects_missing_v6_models(tmp_path: Path) -> None:
    config = OCRConfig(model_base_path=tmp_path, font_base_path=tmp_path)
    with pytest.raises(FileNotFoundError, match="Required PP-OCRv6 small"):
        build_ocr_worker_command(config, image_path="/tmp/page.png", threads=2)


def test_one_shot_ocr_uses_the_shared_engine_owner(monkeypatch) -> None:
    import cv2

    captured_options = {}

    class Result:
        boxes = [[[1, 2], [3, 4]]]
        txts = ["text"]
        scores = [0.875]

    def create_engine(**options):
        captured_options.update(options)
        return lambda *_args, **_kwargs: Result()

    monkeypatch.setattr(ocr_worker, "_create_ocr_engine", create_engine)
    monkeypatch.setattr(cv2, "imread", lambda _path: object())

    result = ocr_worker.run_ocr_full(
        "/tmp/page.png", language="pt", threads=7, full_resolution=True, gpu_fp16=False
    )

    assert len(captured_options) == 21
    assert captured_options["retry_with_cpu"] is False
    assert captured_options["language"] == "pt"
    assert captured_options["threads"] == 7
    assert captured_options["full_resolution"] is True
    assert captured_options["gpu_fp16"] is False
    assert result == {
        "boxes": [[[1, 2], [3, 4]]],
        "txts": ["text"],
        "scores": [0.875],
    }


def test_recognition_language_is_the_unified_one():
    """PP-OCRv6 ships one recogniser for every script, so there is one answer.

    The fifteen-entry language map this replaces only ever reached the
    per-language PP-OCRv5 recognisers, and asking for a script v6 does not
    cover returned nothing rather than falling back.
    """
    assert ocr_worker_engine._rec_lang(FakeLangRec) == FakeLangRec.LATIN


def test_model_version_is_always_v6():
    params = {}
    ocr_worker_engine._set_model_version_params(
        params,
        FakeOCRVersion,
        FakeModelType,
        "medium",
    )
    assert params["Det.ocr_version"] == "v6"
    assert params["Rec.ocr_version"] == "v6"
    assert params["Det.model_type"] == "medium"
    assert params["Rec.model_type"] == "medium"


def test_batch_and_textline_classifier_params_are_configurable():
    params = ocr_worker._build_ocr_engine_params(
        "openvino",
        "latin",
        FakeOCRVersion,
        FakeModelType,
        True,
        4096,
        0.5,
        1.2,
        0.3,
        "slow",
        "",
        "",
        "",
        "",
        4,
        False,
        "small",
        8,
        True,
        "openvino_cpu",
    )
    assert params["Rec.rec_batch_num"] == 8
    assert params["Global.use_cls"] is True


def test_gpu_off_keeps_cpu_engine():
    config = SimpleNamespace(engine_type="openvino", gpu_backend="off")
    engine, extra, label = ocr_worker_engine.resolve_rapidocr_engine_params(config, FakeEngineType)
    assert engine == "openvino"
    assert extra == {}
    assert label == "openvino_cpu"


def test_gpu_auto_without_capability_falls_back_to_cpu(monkeypatch):
    capability = SimpleNamespace(
        nvidia_smi=False,
        paddle_cuda=False,
        torch_cuda=False,
        tensorrt=False,
        onnxruntime_cuda=False,
    )
    monkeypatch.setattr(
        "bigocrpdf.services.rapidocr_service.gpu_detection.detect_gpu_capabilities",
        lambda: capability,
    )
    config = SimpleNamespace(
        engine_type="onnxruntime",
        gpu_backend="auto",
        gpu_fallback_to_cpu=True,
    )
    engine, extra, label = ocr_worker_engine.resolve_rapidocr_engine_params(config, FakeEngineType)
    assert engine == "onnxruntime"
    assert extra == {}
    assert label == "onnxruntime_cpu"


@pytest.mark.parametrize(
    ("label", "device_id", "expected_engine", "expected_backend", "expected_device"),
    (
        ("openvino_cpu", 3, "openvino", "off", None),
        ("onnxruntime_cpu", 3, "onnxruntime", "off", None),
        ("paddle_cuda", 3, "paddle", "paddle", 3),
        ("torch_cuda", 3, "torch", "torch", 3),
        ("tensorrt", 3, "tensorrt", "tensorrt", 3),
        (
            "onnxruntime_cuda_experimental",
            3,
            "onnxruntime",
            "onnxruntime_cuda_experimental",
            3,
        ),
    ),
)
def test_runtime_diagnostics_report_the_initialized_engine(
    label,
    device_id,
    expected_engine,
    expected_backend,
    expected_device,
):
    runtime = ocr_worker_engine._runtime_from_engine_label(label, device_id)

    assert runtime == {
        "schema_version": 1,
        "engine_label": label,
        "engine_type": expected_engine,
        "gpu_backend": expected_backend,
        "gpu_device_id": expected_device,
    }


def test_engine_initialization_reports_cpu_after_gpu_runtime_fallback(monkeypatch):
    calls = []

    def rapidocr(*, params):
        calls.append(params)
        if params["Det.engine_type"] == FakeEngineType.PADDLE:
            raise RuntimeError("CUDA provider unavailable")
        return object()

    capability = SimpleNamespace(
        nvidia_smi=True,
        paddle_cuda=True,
        torch_cuda=False,
        tensorrt=False,
        onnxruntime_cuda=False,
    )
    monkeypatch.setattr(
        ocr_worker_engine,
        "_import_rapidocr_api",
        lambda: (FakeEngineType, FakeLangRec, FakeModelType, FakeOCRVersion, rapidocr),
    )
    monkeypatch.setattr(
        "bigocrpdf.services.rapidocr_service.gpu_detection.detect_gpu_capabilities",
        lambda: capability,
    )

    _engine, runtime = ocr_worker_engine._create_ocr_engine_with_runtime(
        engine_type="openvino",
        use_openvino=True,
        gpu_backend="paddle",
        gpu_device_id=2,
    )

    assert len(calls) == 2
    assert runtime["engine_label"] == "openvino_cpu"
    assert runtime["gpu_backend"] == "off"
    assert runtime["gpu_device_id"] is None


def test_gpu_backend_without_fallback_raises(monkeypatch):
    capability = SimpleNamespace(
        nvidia_smi=False,
        paddle_cuda=False,
        torch_cuda=False,
        tensorrt=False,
        onnxruntime_cuda=False,
    )
    monkeypatch.setattr(
        "bigocrpdf.services.rapidocr_service.gpu_detection.detect_gpu_capabilities",
        lambda: capability,
    )
    config = SimpleNamespace(
        engine_type="openvino",
        gpu_backend="paddle",
        gpu_fallback_to_cpu=False,
    )
    with pytest.raises(RuntimeError, match="not available"):
        ocr_worker_engine.resolve_rapidocr_engine_params(config, FakeEngineType)


def test_dewarp_subprocess_skips_worker_when_temp_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "dewarp.jpg"

    def make_temp(*_args, **_kwargs):
        return os.open(staged, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600), str(staged)

    run = MagicMock()
    monkeypatch.setattr(dewarp_detection.tempfile, "mkstemp", make_temp)
    monkeypatch.setattr(dewarp_detection.cv2, "imwrite", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(dewarp_detection.subprocess, "run", run)

    result = dewarp_detection._detect_subprocess(np.zeros((2, 2, 3), dtype=np.uint8))

    assert result == []
    assert not staged.exists()
    run.assert_not_called()


def test_openvino_worker_releases_detector_request_before_recognition(monkeypatch) -> None:
    active_request = object()
    rebuilt_request = object()
    model = object()
    detector_session = SimpleNamespace(model=model, session=active_request)
    recognized_with = []

    class Engine:
        text_det = SimpleNamespace(session=detector_session)

        def recognize_txt(self, images):
            recognized_with.append(detector_session.session)
            return images

        def __call__(self, _image, **_options):
            return self.recognize_txt(["result"])

    compiled = MagicMock()
    compiled.create_infer_request.return_value = rebuilt_request
    core = MagicMock()
    core.compile_model.return_value = compiled
    monkeypatch.setattr("openvino.Core", lambda: core)

    engine = Engine()
    original_recognize = engine.recognize_txt

    assert ocr_worker._run_ocr_engine(engine, object(), 0.3, 0.5, True, 2) == ["result"]
    assert detector_session.session is None
    assert engine.recognize_txt == original_recognize
    assert recognized_with == [None]
    core.compile_model.assert_not_called()

    assert ocr_worker._run_ocr_engine(engine, object(), 0.3, 0.5, True, 2) == ["result"]
    core.compile_model.assert_called_once_with(
        model=model,
        device_name="CPU",
        config={"INFERENCE_NUM_THREADS": "2"},
    )
    compiled.create_infer_request.assert_called_once_with()
    assert detector_session.session is None
    assert engine.recognize_txt == original_recognize
    assert recognized_with == [None, None]
