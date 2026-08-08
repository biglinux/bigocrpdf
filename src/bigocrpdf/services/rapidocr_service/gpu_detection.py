"""Optional GPU capability detection for RapidOCR backends."""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass
class GpuCapability:
    """Detected optional GPU runtime capabilities."""

    nvidia_smi: bool = False
    onnxruntime_cuda: bool = False
    torch_cuda: bool = False
    paddle_cuda: bool = False
    tensorrt: bool = False
    reason: str = ""


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _load_optional_module(name: str) -> Any:
    return importlib.import_module(name)


def has_nvidia_smi() -> bool:
    """Return True when nvidia-smi is available and reports a working driver."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return False
    return result.returncode == 0


def detect_gpu_capabilities() -> GpuCapability:
    """Detect optional GPU providers without requiring any of them."""
    capability = GpuCapability(nvidia_smi=has_nvidia_smi())

    if _has_module("onnxruntime"):
        try:
            ort = _load_optional_module("onnxruntime")

            capability.onnxruntime_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception as exc:
            capability.reason += f" onnxruntime_check_failed={exc!r};"

    if _has_module("torch"):
        try:
            torch = _load_optional_module("torch")

            capability.torch_cuda = bool(torch.cuda.is_available())
        except Exception as exc:
            capability.reason += f" torch_check_failed={exc!r};"

    if _has_module("paddle"):
        try:
            paddle = _load_optional_module("paddle")

            capability.paddle_cuda = bool(paddle.device.is_compiled_with_cuda())
        except Exception as exc:
            capability.reason += f" paddle_check_failed={exc!r};"

    capability.tensorrt = _has_module("tensorrt")
    return capability
