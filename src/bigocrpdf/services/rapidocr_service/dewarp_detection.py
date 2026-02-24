"""
Dewarp Detection — OCR-based text box detection for dewarping.

Provides in-process (cached RapidOCR) and subprocess-based text box
detection used as input for displacement field computation.

Functions:
    detect_text_boxes    — primary entry point (tries in-process, falls back)
    _get_inprocess_detector — cached RapidOCR instance management
    _detect_inprocess    — in-process detection path
    _detect_subprocess   — subprocess fallback for GTK contexts
    _parse_boxes         — raw OCR output → list of box dicts
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Module-level cached detector ────────────────────────────────────
# Loaded once, shared by all threads. Protected by _detector_lock to
# prevent race conditions during first initialization.
_cached_detector = None
_cached_detector_key: tuple | None = None
_detector_lock = threading.Lock()

# Detection resolution for dewarp. Lower than full OCR resolution because
# we only need approximate box positions for surface fitting.
# 1536 balances speed (~2x faster than 2048) with sufficient box detection
# accuracy for displacement field computation.
_DEWARP_DETECT_SIDE_LEN = 1536


def _get_inprocess_detector(language: str, limit_side_len: int):
    """Get or create a cached RapidOCR instance for in-process detection.

    Loads ONLY the detection model (skips recognition and classification)
    to minimize memory usage (~100 MB vs ~400 MB with all models).
    Falls back to loading all models if detection-only init fails.

    The instance is cached at module level so it is shared by all threads.
    A threading.Lock ensures only one thread loads the model; others wait
    and then reuse it.
    """
    global _cached_detector, _cached_detector_key

    key = (language, limit_side_len)

    # Fast path: already cached (no lock needed for read)
    if _cached_detector is not None and _cached_detector_key == key:
        return _cached_detector

    with _detector_lock:
        # Double-check after acquiring lock (another thread may have loaded it)
        if _cached_detector is not None and _cached_detector_key == key:
            return _cached_detector

        try:
            from rapidocr import EngineType, LangRec, OCRVersion, RapidOCR
        except ModuleNotFoundError:
            from bigocrpdf.utils.python_compat import get_module_from_attribute

            EngineType, LangRec, RapidOCR = get_module_from_attribute(
                "rapidocr", "EngineType", "LangRec", "RapidOCR"
            )
            OCRVersion = get_module_from_attribute("rapidocr", "OCRVersion")[0]

        lang_map = {
            "latin": LangRec.LATIN,
            "en": LangRec.EN,
            "ch": LangRec.CH,
            "chinese_cht": LangRec.CHINESE_CHT,
            "japan": LangRec.JAPAN,
            "korean": LangRec.KOREAN,
            "arabic": LangRec.ARABIC,
        }

        # Try detection-only params first (much less memory: ~100 MB vs ~400 MB)
        det_only_params = {
            "Det.engine_type": EngineType.OPENVINO,
            "Det.engine_cfg.inference_num_threads": 2,
            "Det.limit_side_len": limit_side_len,
            "Det.limit_type": "max",
            "Global.use_cls": False,
            "Global.text_score": 0.3,
        }

        # Full params as fallback (loads all models)
        full_params = {
            **det_only_params,
            "Rec.engine_type": EngineType.OPENVINO,
            "Rec.engine_cfg.inference_num_threads": 2,
            "Rec.lang_type": lang_map.get(language, LangRec.LATIN),
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.rec_batch_num": 16,
            "Cls.engine_type": EngineType.OPENVINO,
        }

        detector = None
        det_mode = "detection-only"

        # Attempt 1: Detection-only (minimal memory)
        try:
            detector = RapidOCR(params=det_only_params)
        except Exception as e:
            logger.debug(f"Detection-only init failed ({e}), trying full model")

        # Attempt 2: Full model with OpenVINO
        if detector is None:
            det_mode = "full (OpenVINO)"
            try:
                detector = RapidOCR(params=full_params)
            except Exception:
                pass

        # Attempt 3: Full model with ONNX Runtime fallback
        if detector is None:
            det_mode = "full (ONNX)"
            full_params["Det.engine_type"] = EngineType.ONNXRUNTIME
            full_params["Rec.engine_type"] = EngineType.ONNXRUNTIME
            full_params["Cls.engine_type"] = EngineType.ONNXRUNTIME
            det_only_params["Det.engine_type"] = EngineType.ONNXRUNTIME
            detector = RapidOCR(params=full_params)

        _cached_detector = detector
        _cached_detector_key = key
        logger.info(
            f"Cached in-process OCR detector [{det_mode}] "
            f"(lang={language}, limit_side_len={limit_side_len})"
        )
        return detector


def _parse_boxes(boxes_raw, txts_raw, scores_raw) -> list[dict]:
    """Convert raw OCR output arrays to list of box dicts."""
    boxes_data = []
    txts = txts_raw or []
    scores = scores_raw or []
    for i, box in enumerate(boxes_raw):
        boxes_data.append(
            {
                "box": np.asarray(box, dtype=np.float64),
                "text": txts[i] if i < len(txts) else "",
                "score": float(scores[i]) if i < len(scores) else 0.0,
            }
        )
    return boxes_data


# Lock for serializing OCR inference calls (RapidOCR may not be thread-safe)
_inference_lock = threading.Lock()


def _detect_inprocess(
    image: np.ndarray,
    language: str = "latin",
    limit_side_len: int = _DEWARP_DETECT_SIDE_LEN,
) -> list[dict] | None:
    """Run OCR detection in-process using cached RapidOCR instance.

    Returns list of box dicts, or None if in-process detection is not
    available (e.g., GTK process, import failure).

    Inference is serialized with _inference_lock because the RapidOCR
    instance is shared across threads.
    """
    # Safety: avoid ONNX/GTK threading conflicts in GUI processes
    if "gi.repository.Gtk" in sys.modules:
        return None

    try:
        detector = _get_inprocess_detector(language, limit_side_len)
        with _inference_lock:
            # Detection-only: skip recognition since dewarp only needs box
            # positions for surface fitting. ~4.5x faster.
            result = detector(
                image,
                use_cls=False,
                use_rec=False,
                text_score=0.3,
                box_thresh=0.5,
            )

        if result.boxes is None:
            return []

        # TextDetOutput from use_rec=False has boxes/scores but no txts
        return _parse_boxes(
            result.boxes,
            None,
            getattr(result, "scores", None),
        )
    except Exception as e:
        logger.debug(f"In-process detection failed: {e}")
        return None


def _detect_subprocess(
    image: np.ndarray,
    language: str = "latin",
    limit_side_len: int = _DEWARP_DETECT_SIDE_LEN,
) -> list[dict]:
    """Run OCR detection via subprocess (fallback for GTK contexts).

    Saves the image to a temp JPEG (quality 50 — sufficient for box
    detection, much faster I/O than quality 95) and spawns an OCR worker.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    cv2.imwrite(tmp_path, image, [cv2.IMWRITE_JPEG_QUALITY, 50])

    try:
        worker = Path(__file__).parent / "ocr_worker.py"
        result = subprocess.run(
            [
                sys.executable,
                str(worker),
                tmp_path,
                "--language",
                language,
                "--limit_side_len",
                str(limit_side_len),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        logger.warning("OCR detection subprocess timed out")
        return []
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if result.returncode != 0:
        logger.warning(f"OCR detection subprocess failed: {result.stderr[:300]}")
        return []

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        logger.warning("OCR detection subprocess returned invalid JSON")
        return []

    if not data.get("boxes"):
        return []

    return _parse_boxes(
        data["boxes"],
        data.get("txts", []),
        data.get("scores", []),
    )


def detect_text_boxes(
    image: np.ndarray,
    language: str = "latin",
    limit_side_len: int = _DEWARP_DETECT_SIDE_LEN,
) -> list[dict]:
    """Detect text boxes using the fastest available method.

    Tries in-process detection first (cached model, no subprocess overhead),
    falls back to subprocess for GTK process contexts.

    Args:
        image: Input BGR image (OpenCV format)
        language: OCR language model to use
        limit_side_len: Maximum side length for detection scaling

    Returns:
        List of dicts with 'box' (4x2 ndarray), 'text', 'score' keys.
        Empty list if detection fails.
    """
    # Try in-process detection (3-5x faster, cached model)
    boxes = _detect_inprocess(image, language, limit_side_len)
    if boxes is not None:
        return boxes

    # Subprocess fallback (GTK process or import failure)
    return _detect_subprocess(image, language, limit_side_len)
