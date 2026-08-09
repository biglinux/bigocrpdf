"""RapidOCR engine construction for the OCR worker subprocess.

allow-noisy-log: worker diagnostics are written to stderr for the parent process.
"""

import os
import sys
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from bigocrpdf.services.rapidocr_service.config import OCRConfig

_OCR_ENGINE_DEFAULTS = {
    "language": "latin",
    "limit_side_len": 4000,
    "use_openvino": True,
    "box_thresh": 0.5,
    "unclip_ratio": 1.2,
    "text_score": 0.3,
    "score_mode": "slow",
    "rec_model_path": "",
    "rec_keys_path": "",
    "det_model_path": "",
    "font_path": "",
    "threads": 4,
    "full_resolution": False,
    "model_type": "small",
    "rec_batch_num": 1,
    "use_textline_cls": False,
    "gpu_backend": "off",
    "gpu_device_id": 0,
    "gpu_fp16": True,
    "gpu_fallback_to_cpu": True,
}


def build_ocr_worker_command(  # noqa: C901 - command flags intentionally mirror OCRConfig
    config: OCRConfig,
    *,
    image_path: str | None = None,
    persistent: bool = False,
    threads: int,
    openvino_available: bool = True,
    low_memory_openvino: bool = False,
) -> list[str]:
    """Build the shared one-shot or persistent OCR worker command."""
    if persistent == (image_path is not None):
        raise ValueError("Select exactly one OCR worker mode: image_path or persistent")

    command = [sys.executable, "-m", "bigocrpdf.services.rapidocr_service.ocr_worker"]
    if persistent:
        command.append("--persistent")
    else:
        assert image_path is not None
        command.append(image_path)
    for flag, value in (
        ("--language", config.language),
        ("--limit_side_len", config.detection_limit_side_len),
        ("--box-thresh", config.box_thresh),
        ("--unclip-ratio", config.unclip_ratio),
        ("--text-score", config.text_score_threshold),
        ("--score-mode", config.score_mode),
        ("--model-type", config.model_type),
        ("--rec-batch-num", config.rec_batch_num),
        ("--gpu-backend", config.gpu_backend),
        ("--gpu-device-id", config.gpu_device_id),
        ("--threads", threads),
    ):
        command.extend([flag, str(value)])

    if config.gpu_fp16:
        command.append("--gpu-fp16")
    if config.use_textline_cls:
        command.append("--use-textline-cls")
    if not config.gpu_fallback_to_cpu:
        command.append("--no-gpu-fallback")
    if config.detection_full_resolution:
        command.append("--full-resolution")
    if config.engine_type == "onnxruntime" or not openvino_available:
        command.append("--no-openvino")
    elif low_memory_openvino:
        command.append("--low-memory-openvino")

    rec_model_path = config.get_rec_model_path()
    det_model_path = config.get_det_model_path()
    if rec_model_path is None or det_model_path is None:
        raise FileNotFoundError(
            f"Required PP-OCRv6 {config.model_type} detection and recognition models are missing"
        )

    for flag, model_path in (
        ("--rec-model-path", rec_model_path),
        ("--det-model-path", det_model_path),
        ("--font-path", config.get_font_path()),
    ):
        if model_path:
            command.extend([flag, str(model_path)])

    return command


def _create_ocr_engine(
    *,
    retry_with_cpu: bool = True,
    **overrides,
) -> Callable[..., Any]:
    engine, _runtime = _create_ocr_engine_with_runtime(
        retry_with_cpu=retry_with_cpu,
        **overrides,
    )
    return engine


def _create_ocr_engine_with_runtime(
    *,
    retry_with_cpu: bool = True,
    **overrides,
) -> tuple[Callable[..., Any], dict[str, object]]:
    """Create RapidOCR and report the engine that actually initialized."""
    values = dict(_OCR_ENGINE_DEFAULTS)
    values.update(overrides)
    EngineType, LangRec, ModelType, OCRVersion, RapidOCR = _import_rapidocr_api()
    lang_rec = _lang_rec_from_code(LangRec, values["language"])
    options = _ocr_engine_options(values)
    engine_request = SimpleNamespace(
        engine_type="openvino" if values["use_openvino"] else "onnxruntime",
        gpu_backend=values["gpu_backend"],
        gpu_device_id=values["gpu_device_id"],
        gpu_fp16=values["gpu_fp16"],
        gpu_fallback_to_cpu=values["gpu_fallback_to_cpu"],
    )
    primary_engine, engine_extra_params, engine_label = resolve_rapidocr_engine_params(
        engine_request,
        EngineType,
    )
    fallback_engine, _, fallback_label = _cpu_engine_params(engine_request, EngineType)

    try:
        params = _build_ocr_engine_params(
            primary_engine,
            lang_rec,
            OCRVersion,
            ModelType,
            values["use_openvino"],
            options,
            engine_label,
        )
        params.update(engine_extra_params)
        return (
            RapidOCR(params=params),
            _runtime_from_engine_label(engine_label, int(values["gpu_device_id"])),
        )
    except Exception as primary_err:
        if not retry_with_cpu:
            raise
        if not _rapidocr_missing_backend_error(primary_err) and values["gpu_backend"] in {
            "off",
            "none",
            "cpu",
        }:
            raise
        fallback = _create_fallback_ocr_engine(
            RapidOCR,
            fallback_engine,
            lang_rec,
            OCRVersion,
            ModelType,
            fallback_label == "openvino_cpu",
            options,
            engine_label,
            fallback_label,
            primary_err,
        )
        return (
            fallback,
            _runtime_from_engine_label(fallback_label, int(values["gpu_device_id"])),
        )


def _runtime_from_engine_label(
    engine_label: str,
    requested_device_id: int,
) -> dict[str, object]:
    """Map the successful RapidOCR engine label to the ready protocol."""
    engine_type, gpu_backend = {
        "openvino_cpu": ("openvino", "off"),
        "onnxruntime_cpu": ("onnxruntime", "off"),
        "paddle_cuda": ("paddle", "paddle"),
        "torch_cuda": ("torch", "torch"),
        "tensorrt": ("tensorrt", "tensorrt"),
        "onnxruntime_cuda_experimental": (
            "onnxruntime",
            "onnxruntime_cuda_experimental",
        ),
    }[engine_label]
    return {
        "schema_version": 1,
        "engine_label": engine_label,
        "engine_type": engine_type,
        "gpu_backend": gpu_backend,
        "gpu_device_id": (None if gpu_backend == "off" else requested_device_id),
    }


def _ocr_engine_options(values: dict) -> SimpleNamespace:
    return SimpleNamespace(
        limit_side_len=values["limit_side_len"],
        box_thresh=values["box_thresh"],
        unclip_ratio=values["unclip_ratio"],
        text_score=values["text_score"],
        score_mode=values["score_mode"],
        rec_model_path=values["rec_model_path"],
        rec_keys_path=values["rec_keys_path"],
        det_model_path=values["det_model_path"],
        font_path=values["font_path"],
        threads=values["threads"],
        full_resolution=values["full_resolution"],
        model_type=values["model_type"],
        rec_batch_num=values["rec_batch_num"],
        use_textline_cls=values["use_textline_cls"],
    )


def _import_rapidocr_api():
    from rapidocr import EngineType, LangRec, OCRVersion, RapidOCR

    try:
        from rapidocr import ModelType
    except (ImportError, ModuleNotFoundError):
        ModelType = None
    return EngineType, LangRec, ModelType, OCRVersion, RapidOCR


def _get_lang_enum(LangRec, *names: str):
    for name in names:
        if hasattr(LangRec, name):
            return getattr(LangRec, name)
    return LangRec.LATIN


def _lang_rec_from_code(LangRec, language: str):
    lang = (language or "latin").lower().replace("-", "_")
    aliases = {
        "pt": "pt",
        "pt_br": "pt",
        "por": "pt",
        "portuguese": "pt",
        "english": "en",
        "zh": "ch",
        "zh_cn": "ch",
        "zh_tw": "chinese_cht",
        "ja": "japan",
        "jp": "japan",
        "ko": "korean",
        "ru": "cyrillic",
        "el": "greek",
        "hi": "devanagari",
        "ta": "tamil",
        "te": "telugu",
        "th": "thai",
        "ar": "arabic",
    }
    lang = aliases.get(lang, lang)
    lang_map = {
        "pt": _get_lang_enum(LangRec, "PT", "LATIN"),
        "latin": _get_lang_enum(LangRec, "LATIN"),
        "en": _get_lang_enum(LangRec, "EN", "LATIN"),
        "ch": _get_lang_enum(LangRec, "CH"),
        "chinese_cht": _get_lang_enum(LangRec, "CHINESE_CHT", "CH"),
        "japan": _get_lang_enum(LangRec, "JAPAN", "CH"),
        "korean": _get_lang_enum(LangRec, "KOREAN"),
        "arabic": _get_lang_enum(LangRec, "ARABIC"),
        "cyrillic": _get_lang_enum(LangRec, "CYRILLIC", "ESLAV", "LATIN"),
        "devanagari": _get_lang_enum(LangRec, "DEVANAGARI", "LATIN"),
        "greek": _get_lang_enum(LangRec, "EL", "GREEK", "LATIN"),
        "tamil": _get_lang_enum(LangRec, "TA", "TAMIL", "LATIN"),
        "telugu": _get_lang_enum(LangRec, "TE", "TELUGU", "LATIN"),
        "thai": _get_lang_enum(LangRec, "TH", "THAI", "LATIN"),
    }
    selected = lang_map.get(lang, _get_lang_enum(LangRec, "LATIN"))
    if selected == _get_lang_enum(LangRec, "LATIN") and lang not in {"latin", "en", "pt"}:
        print(f"[OCR Worker] Language {language!r} fell back to LATIN", file=sys.stderr)
    return selected


def _enum_by_name(enum_cls, name: str | None, fallback: str):
    key = (name or fallback).upper().replace("-", "_")
    if hasattr(enum_cls, key):
        return getattr(enum_cls, key)
    if hasattr(enum_cls, fallback):
        return getattr(enum_cls, fallback)
    if hasattr(enum_cls, "MOBILE"):
        print(
            f"[OCR Worker] Enum value {key!r} unavailable; falling back to MOBILE",
            file=sys.stderr,
        )
        return enum_cls.MOBILE
    raise AttributeError(f"{enum_cls!r} has neither {key!r} nor {fallback!r}")


def _set_model_version_params(
    params: dict,
    OCRVersion,
    ModelType,
    model_type: str,
) -> None:
    selected_version = _enum_by_name(OCRVersion, "PPOCRV6", "PPOCRV6")
    params["Det.ocr_version"] = selected_version
    params["Rec.ocr_version"] = selected_version
    if ModelType is None:
        print(
            "[OCR Worker] RapidOCR ModelType enum unavailable; using default model type",
            file=sys.stderr,
        )
        return
    selected_model_type = _enum_by_name(ModelType, model_type, "SMALL")
    params["Det.model_type"] = selected_model_type
    params["Rec.model_type"] = selected_model_type


def _cpu_engine_params(config, EngineType):
    if getattr(config, "engine_type", "openvino") == "openvino":
        return EngineType.OPENVINO, {}, "openvino_cpu"
    return EngineType.ONNXRUNTIME, {}, "onnxruntime_cpu"


def _gpu_backend_available(backend: str, capability) -> bool:
    return {
        "paddle": capability.paddle_cuda,
        "torch": capability.torch_cuda,
        "tensorrt": capability.nvidia_smi and capability.tensorrt,
        "onnxruntime_cuda_experimental": capability.onnxruntime_cuda,
    }.get(backend, False)


def _auto_gpu_backend(capability) -> str | None:
    if capability.nvidia_smi and capability.tensorrt:
        return "tensorrt"
    if capability.paddle_cuda:
        return "paddle"
    if capability.torch_cuda:
        return "torch"
    return None


def resolve_rapidocr_engine_params(config, EngineType) -> tuple[object, dict, str]:
    backend = (getattr(config, "gpu_backend", "off") or "off").lower()
    if backend in {"off", "none", "cpu"}:
        return _cpu_engine_params(config, EngineType)

    from bigocrpdf.services.rapidocr_service.gpu_detection import detect_gpu_capabilities

    capability = detect_gpu_capabilities()
    if backend == "auto":
        backend = _auto_gpu_backend(capability) or "off"
        if backend == "off":
            print(
                "[OCR Worker] gpu_backend=auto found no usable GPU backend; using CPU",
                file=sys.stderr,
            )
            return _cpu_engine_params(config, EngineType)

    if not _gpu_backend_available(backend, capability):
        reason = f"GPU backend {backend!r} is not available"
        if getattr(config, "gpu_fallback_to_cpu", True):
            print(f"[OCR Worker] {reason}; using CPU fallback", file=sys.stderr)
            return _cpu_engine_params(config, EngineType)
        raise RuntimeError(reason)

    return _gpu_engine_params(backend, config, EngineType)


def _gpu_engine_params(backend: str, config, EngineType) -> tuple[object, dict, str]:
    device_id = getattr(config, "gpu_device_id", 0)
    if backend == "paddle":
        return (
            EngineType.PADDLE,
            {
                "EngineConfig.paddle.use_cuda": True,
                "EngineConfig.paddle.cuda_ep_cfg.device_id": device_id,
            },
            "paddle_cuda",
        )
    if backend == "torch":
        return (
            EngineType.TORCH,
            {
                "EngineConfig.torch.use_cuda": True,
                "EngineConfig.torch.cuda_ep_cfg.device_id": device_id,
            },
            "torch_cuda",
        )
    if backend == "tensorrt":
        return (
            EngineType.TENSORRT,
            {
                "EngineConfig.tensorrt.use_fp16": bool(getattr(config, "gpu_fp16", True)),
                "EngineConfig.tensorrt.device_id": device_id,
            },
            "tensorrt",
        )
    if backend == "onnxruntime_cuda_experimental":
        return (
            EngineType.ONNXRUNTIME,
            {"EngineConfig.onnxruntime.use_cuda": True},
            "onnxruntime_cuda_experimental",
        )
    raise ValueError(f"Unsupported gpu_backend={backend!r}")


def _build_ocr_engine_params(
    engine_type,
    lang_rec,
    OCRVersion,
    ModelType,
    use_openvino_threads: bool,
    options,
    engine_label: str = "",
    *legacy_args,
) -> dict:
    if legacy_args:
        options, engine_label = _legacy_ocr_engine_options(options, engine_label, legacy_args)
    params = {
        "Det.engine_type": engine_type,
        "Det.box_thresh": options.box_thresh,
        "Det.unclip_ratio": options.unclip_ratio,
        "Det.score_mode": options.score_mode,
        "Det.limit_side_len": options.limit_side_len,
        "Det.limit_type": "min" if options.full_resolution else "max",
        "Rec.engine_type": engine_type,
        "Rec.lang_type": lang_rec,
        "Rec.rec_batch_num": options.rec_batch_num,
        "Cls.engine_type": engine_type,
        "Global.use_cls": options.use_textline_cls,
        "Global.text_score": options.text_score,
        "Global.max_side_len": options.limit_side_len,
    }
    _set_model_version_params(params, OCRVersion, ModelType, options.model_type)
    _set_ocr_thread_params(params, use_openvino_threads, options.threads)
    _set_optional_ocr_model_paths(
        params,
        options.rec_model_path,
        options.rec_keys_path,
        options.det_model_path,
        options.font_path,
    )
    _remove_unrelated_thread_params(params, engine_label)
    return params


def _legacy_ocr_engine_options(limit_side_len, box_thresh, args) -> tuple[SimpleNamespace, str]:
    (
        unclip_ratio,
        text_score,
        score_mode,
        rec_model_path,
        rec_keys_path,
        det_model_path,
        font_path,
        threads,
        full_resolution,
        model_type,
        rec_batch_num,
        use_textline_cls,
        engine_label,
    ) = args
    return (
        SimpleNamespace(
            limit_side_len=limit_side_len,
            box_thresh=box_thresh,
            unclip_ratio=unclip_ratio,
            text_score=text_score,
            score_mode=score_mode,
            rec_model_path=rec_model_path,
            rec_keys_path=rec_keys_path,
            det_model_path=det_model_path,
            font_path=font_path,
            threads=threads,
            full_resolution=full_resolution,
            model_type=model_type,
            rec_batch_num=rec_batch_num,
            use_textline_cls=use_textline_cls,
        ),
        engine_label,
    )


def _remove_unrelated_thread_params(params: dict, engine_label: str) -> None:
    if engine_label == "openvino_cpu":
        params.pop("EngineConfig.onnxruntime.intra_op_num_threads", None)
        params.pop("EngineConfig.onnxruntime.inter_op_num_threads", None)
    elif engine_label in {"onnxruntime_cpu", "onnxruntime_cuda_experimental"}:
        params.pop("EngineConfig.openvino.inference_num_threads", None)
    else:
        params.pop("EngineConfig.openvino.inference_num_threads", None)
        params.pop("EngineConfig.onnxruntime.intra_op_num_threads", None)
        params.pop("EngineConfig.onnxruntime.inter_op_num_threads", None)


def _set_ocr_thread_params(params: dict, use_openvino_threads: bool, threads: int) -> None:
    if use_openvino_threads:
        params["EngineConfig.openvino.inference_num_threads"] = threads
        return
    params["EngineConfig.onnxruntime.intra_op_num_threads"] = threads
    params["EngineConfig.onnxruntime.inter_op_num_threads"] = 2


def _set_optional_ocr_model_paths(
    params: dict,
    rec_model_path: str,
    rec_keys_path: str,
    det_model_path: str,
    font_path: str,
) -> None:
    if rec_model_path:
        params["Rec.model_path"] = rec_model_path
    if rec_keys_path:
        params["Rec.rec_keys_path"] = rec_keys_path
    if det_model_path:
        params["Det.model_path"] = det_model_path
    if font_path and os.path.exists(font_path):
        params["Global.font_path"] = font_path


def _rapidocr_missing_backend_error(error: Exception) -> bool:
    error_msg = str(error).lower()
    return "not installed" in error_msg or "no module" in error_msg or "import" in error_msg


def _create_fallback_ocr_engine(
    RapidOCR,
    fallback_engine,
    lang_rec,
    OCRVersion,
    ModelType,
    use_openvino_threads: bool,
    options,
    primary_name: str,
    fallback_name: str,
    primary_err: Exception,
) -> Callable[..., Any]:
    print(f"[OCR Worker] {primary_name} failed: {primary_err}", file=sys.stderr)
    print(f"[OCR Worker] Trying fallback to {fallback_name}...", file=sys.stderr)

    try:
        params = _build_ocr_engine_params(
            fallback_engine,
            lang_rec,
            OCRVersion,
            ModelType,
            use_openvino_threads,
            options,
            fallback_name,
        )
        return RapidOCR(params=params)
    except Exception as fallback_err:
        raise RuntimeError(
            f"Both OCR engines failed.\n"
            f"{primary_name}: {primary_err}\n"
            f"{fallback_name}: {fallback_err}"
        ) from fallback_err
