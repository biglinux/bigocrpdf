import subprocess
from unittest.mock import patch

from bigocrpdf.services.rapidocr_service import gpu_detection


def test_has_nvidia_smi_false_when_command_missing(monkeypatch):
    def raise_missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", raise_missing)
    assert gpu_detection.has_nvidia_smi() is False


def test_detect_gpu_capabilities_without_optional_modules(monkeypatch):
    monkeypatch.setattr(gpu_detection, "has_nvidia_smi", lambda: False)
    monkeypatch.setattr(gpu_detection, "_has_module", lambda name: False)

    capability = gpu_detection.detect_gpu_capabilities()

    assert capability.nvidia_smi is False
    assert capability.onnxruntime_cuda is False
    assert capability.torch_cuda is False
    assert capability.paddle_cuda is False
    assert capability.tensorrt is False
    assert capability.reason == ""


def test_detect_gpu_capabilities_ignores_broken_optional_module_specs(monkeypatch):
    monkeypatch.setattr(gpu_detection, "has_nvidia_smi", lambda: False)

    with patch("importlib.util.find_spec", side_effect=ValueError("broken spec")):
        capability = gpu_detection.detect_gpu_capabilities()

    assert capability == gpu_detection.GpuCapability()
