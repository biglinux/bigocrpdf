#!/usr/bin/env python3
"""
Standalone OCR worker script.

allow-noisy-log: this subprocess worker writes JSON to stdout and worker
diagnostics to stderr as part of its process contract.

This script is called via subprocess to run OCR in an isolated environment,
avoiding GTK/GLib interference with ONNX Runtime.

Usage:
    python ocr_worker.py <image_path> [--language latin] [--limit_side_len 4000]
    python ocr_worker.py --batch <img1.png> <img2.png> ... [--language latin]

Output:
    JSON on stdout with OCR results
"""

import argparse
import gc
import json
import os
import sys
from typing import Any

from bigocrpdf.services.rapidocr_service.ocr_worker_engine import (
    _build_ocr_engine_params,
    _create_ocr_engine,
    _create_ocr_engine_with_runtime,
    _lang_rec_from_code,
)

__all__ = ["_build_ocr_engine_params"]

# Setup Python version compatibility moved to main() to avoid output pollution


def run_ocr_batch(
    image_paths: list, language: str, limit_side_len: int, use_openvino: bool = True
) -> list:
    """Run OCR on multiple images with shared RapidOCR instance."""
    import cv2
    from rapidocr import EngineType, LangRec, RapidOCR

    lang_rec = _lang_rec_from_code(LangRec, language)

    # Create single RapidOCR instance for all images
    params = {
        "Det.limit_side_len": limit_side_len,
        "Global.text_score": 0.3,
        "Rec.lang_type": lang_rec,
    }
    if use_openvino:
        params["Det.engine_type"] = EngineType.OPENVINO
        params["Rec.engine_type"] = EngineType.OPENVINO
        params["Cls.engine_type"] = EngineType.OPENVINO

    rapid = RapidOCR(params=params)

    results = []
    for image_path in image_paths:
        try:
            img = cv2.imread(image_path)
            if img is None:
                results.append({"success": False, "error": f"Failed to load: {image_path}"})
                continue

            serialized = _reorient_vertical_regions(
                rapid, img, _serialize_ocr_result(rapid(img), empty_when_no_boxes=True)
            )

            results.append(
                {
                    "success": True,
                    **serialized,
                    "count": len(serialized["txts"]),
                }
            )
        except Exception as e:
            results.append({"success": False, "error": str(e)})

    return results


def run_ocr_full(
    image_path: str,
    language: str = "latin",
    limit_side_len: int = 4000,
    use_openvino: bool = False,
    box_thresh: float = 0.5,
    unclip_ratio: float = 1.2,
    text_score: float = 0.3,
    score_mode: str = "slow",
    rec_model_path: str = "",
    rec_keys_path: str = "",
    det_model_path: str = "",
    font_path: str = "",
    threads: int = 4,
    full_resolution: bool = False,
    model_type: str = "small",
    rec_batch_num: int = 1,
    use_textline_cls: bool = False,
    gpu_backend: str = "off",
    gpu_device_id: int = 0,
    gpu_fp16: bool = True,
    gpu_fallback_to_cpu: bool = True,
) -> dict:
    """Run OCR on a single image with full parameter control.

    This is the primary entry point for subprocess OCR calls from the backend.
    All parameters match the reference implementation exactly.

    Args:
        image_path: Path to the image file
        language: Language code (latin, en, ch, etc.)
        limit_side_len: Maximum side length for detection
        use_openvino: Whether to use OpenVINO backend
        box_thresh: Box detection threshold
        unclip_ratio: Unclip ratio for text detection
        text_score: Minimum text score threshold
        score_mode: Score mode (fast/slow)
        rec_model_path: Path to recognition model
        rec_keys_path: Path to recognition keys file
        det_model_path: Path to detection model
        font_path: Path to font file
        threads: Number of threads for ONNX inference

    Returns:
        Dictionary with boxes, txts, scores keys or error info
    """
    try:
        import cv2

        img = cv2.imread(image_path)
        if img is None:
            return {"error": f"Could not load image: {image_path}"}

        ocr = _create_ocr_engine(
            retry_with_cpu=False,
            language=language,
            limit_side_len=limit_side_len,
            use_openvino=use_openvino,
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
            gpu_backend=gpu_backend,
            gpu_device_id=gpu_device_id,
            gpu_fp16=gpu_fp16,
            gpu_fallback_to_cpu=gpu_fallback_to_cpu,
        )
        # Pass text_score and box_thresh both at init AND per-call to ensure
        # they are definitely applied (per-call overrides take precedence)
        result = ocr(
            img,
            use_cls=use_textline_cls,
            text_score=text_score,
            box_thresh=box_thresh,
        )
        serialized = _serialize_ocr_result(result)
        return _reorient_vertical_regions(ocr, img, serialized)

    except Exception as e:
        return {"error": str(e)}


def _reorient_vertical_regions(ocr: Any, img: Any, ocr_raw: dict) -> dict:
    """Re-read tall regions the other way up, keeping whichever reads better.

    See ``vertical_text``: RapidOCR rotates every tall crop the same way, so
    the half of vertical captions that run top-to-bottom come back upside-down
    and unreadable. Only recognition runs again -- the region is already known.
    """
    import cv2
    import numpy as np
    from rapidocr.utils.process_img import get_rotate_crop_image

    from bigocrpdf.services.rapidocr_service.vertical_text import (
        choose_better_reading,
        vertical_candidates,
    )

    candidates = vertical_candidates(ocr_raw)
    if not candidates:
        return ocr_raw

    boxes, txts, scores = ocr_raw["boxes"], list(ocr_raw["txts"]), list(ocr_raw["scores"])
    replaced = 0
    for index in candidates:
        try:
            points = np.array(boxes[index], dtype=np.float32)
            crop = get_rotate_crop_image(img, points)
            flipped = cv2.rotate(crop, cv2.ROTATE_180)
            rec = ocr(flipped, use_det=False, use_cls=False, use_rec=True)
            if not rec or not rec.txts:
                continue
            text, score, changed = choose_better_reading(
                txts[index], scores[index], rec.txts[0], rec.scores[0]
            )
            txts[index], scores[index] = text, score
            replaced += int(changed)
        except Exception:
            # A region that cannot be re-read keeps its first reading; this is
            # an improvement pass, never a reason to lose a page. Not logged:
            # stdout here is the worker's JSON protocol.
            continue

    if replaced:
        ocr_raw["txts"], ocr_raw["scores"] = txts, scores
        ocr_raw["vertical_reoriented"] = replaced
    return ocr_raw


def _serialize_ocr_result(
    result: Any,
    *,
    empty_when_no_boxes: bool = False,
) -> dict[str, Any]:
    boxes = getattr(result, "boxes", None) if empty_when_no_boxes else result.boxes
    if boxes is None:
        return {"boxes": [], "txts": [], "scores": []} if empty_when_no_boxes else {"boxes": None}
    return {
        "boxes": [box.tolist() if hasattr(box, "tolist") else list(box) for box in boxes],
        "txts": list(result.txts) if result.txts else [],
        "scores": [float(score) for score in result.scores] if result.scores else [],
    }


def _set_openvino_request(session_owner: Any, enabled: bool, threads: int) -> None:
    if enabled:
        if session_owner.session is not None:
            return

        from openvino import Core

        # Retain the model, but rebuild the large compiled request only when needed.
        compiled = Core().compile_model(
            model=session_owner.model,
            device_name="CPU",
            config={"INFERENCE_NUM_THREADS": str(threads)},
        )
        session_owner.session = compiled.create_infer_request()
        return

    if session_owner.session is None:
        return
    session_owner.session = None
    gc.collect()


def _run_ocr_engine(
    engine: Any,
    image: Any,
    text_score: float,
    box_thresh: float,
    low_memory_openvino: bool,
    threads: int,
) -> Any:
    if not low_memory_openvino:
        return engine(image, use_cls=False, text_score=text_score, box_thresh=box_thresh)

    detector_session = engine.text_det.session
    _set_openvino_request(detector_session, True, threads)
    recognize_txt = engine.recognize_txt

    def recognize_without_detector(images):
        _set_openvino_request(detector_session, False, threads)
        return recognize_txt(images)

    engine.recognize_txt = recognize_without_detector
    try:
        return engine(image, use_cls=False, text_score=text_score, box_thresh=box_thresh)
    finally:
        engine.recognize_txt = recognize_txt
        _set_openvino_request(detector_session, False, threads)


def _ocr_single_image(
    engine: Any,
    image_path: str,
    text_score: float = 0.3,
    box_thresh: float = 0.5,
    low_memory_openvino: bool = False,
    threads: int = 2,
) -> dict:
    """Run OCR on a single image using a pre-created engine.

    Returns:
        Dict with boxes/txts/scores keys, or error info.
    """
    import cv2

    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"error": f"Could not load image: {image_path}"}

        result = _run_ocr_engine(
            engine,
            img,
            text_score,
            box_thresh,
            low_memory_openvino,
            threads,
        )

        serialized = _reorient_vertical_regions(engine, img, _serialize_ocr_result(result))

        # Release image memory immediately
        del img

        return serialized
    except Exception as e:
        return {"error": str(e)}


def _engine_options_from_args(args: argparse.Namespace, threads: int) -> dict[str, Any]:
    return {
        "language": args.language,
        "limit_side_len": args.limit_side_len,
        "use_openvino": not args.no_openvino,
        "box_thresh": args.box_thresh,
        "unclip_ratio": args.unclip_ratio,
        "text_score": args.text_score,
        "score_mode": args.score_mode,
        "rec_model_path": args.rec_model_path,
        "rec_keys_path": args.rec_keys_path,
        "det_model_path": args.det_model_path,
        "font_path": args.font_path,
        "threads": threads,
        "full_resolution": args.full_resolution,
        "model_type": args.model_type,
        "rec_batch_num": args.rec_batch_num,
        "use_textline_cls": args.use_textline_cls,
        "gpu_backend": args.gpu_backend,
        "gpu_device_id": args.gpu_device_id,
        "gpu_fp16": args.gpu_fp16,
        "gpu_fallback_to_cpu": not args.no_gpu_fallback,
    }


def run_persistent(args: argparse.Namespace) -> None:
    """Persistent OCR mode: reads image paths from stdin, writes results to stdout.

    The model is loaded ONCE at startup. Each line on stdin is an image path;
    the corresponding JSON result is written to stdout (one JSON per line).
    This eliminates model loading overhead for multi-page PDFs.

    Memory usage: ~400 MB (single model instance) vs ~2+ GB (subprocess per page).
    """

    # Redirect any stray library output away from our JSON protocol
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    threads = args.threads if args.threads > 0 else max(2, os.cpu_count() or 4)

    try:
        engine, runtime = _create_ocr_engine_with_runtime(
            **_engine_options_from_args(args, threads)
        )
    except Exception as e:
        real_stdout.write(json.dumps({"fatal": str(e)}) + "\n")
        real_stdout.flush()
        return

    low_memory_openvino = (
        args.low_memory_openvino
        and runtime["engine_label"] == "openvino_cpu"
        and not args.use_textline_cls
    )
    if low_memory_openvino:
        engine.text_cls = None
        gc.collect()

    # Signal readiness
    real_stdout.write(json.dumps({"ready": True, "runtime": runtime}) + "\n")
    real_stdout.flush()

    # Process images from stdin (one path per line)
    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue

        result = _ocr_single_image(
            engine,
            path,
            args.text_score,
            args.box_thresh,
            low_memory_openvino,
            threads,
        )
        real_stdout.write(json.dumps(result) + "\n")
        real_stdout.flush()

        # Prevent memory accumulation across pages
        gc.collect()


def main():
    # Set reduced CPU priority so interactive applications stay responsive
    try:
        os.nice(10)
    except OSError:
        pass

    parser = argparse.ArgumentParser(description="Standalone OCR worker")
    parser.add_argument("images", nargs="*", help="Paths to image files")
    parser.add_argument("--batch", action="store_true", help="Batch mode (multiple images)")
    parser.add_argument("--persistent", action="store_true", help="Persistent mode (stdin/stdout)")
    parser.add_argument("--language", default="latin", help="Language code")
    parser.add_argument("--limit_side_len", type=int, default=4000, help="Max side length")
    parser.add_argument("--no-openvino", action="store_true", help="Disable OpenVINO")
    parser.add_argument(
        "--low-memory-openvino",
        action="store_true",
        help="Release OpenVINO detector buffers before recognition",
    )
    parser.add_argument("--box-thresh", type=float, default=0.5, help="Box threshold")
    parser.add_argument("--unclip-ratio", type=float, default=1.2, help="Unclip ratio")
    parser.add_argument("--text-score", type=float, default=0.3, help="Text score threshold")
    parser.add_argument("--score-mode", default="slow", help="Score mode (fast/slow)")
    parser.add_argument("--rec-model-path", default="", help="Recognition model path")
    parser.add_argument("--rec-keys-path", default="", help="Recognition keys path")
    parser.add_argument("--det-model-path", default="", help="Detection model path")
    parser.add_argument("--font-path", default="", help="Font path")
    parser.add_argument("--model-type", default="small", help="RapidOCR ModelType name")
    parser.add_argument(
        "--rec-batch-num",
        type=int,
        default=1,
        help="RapidOCR recognition batch size",
    )
    parser.add_argument(
        "--use-textline-cls",
        action="store_true",
        help="Enable RapidOCR text-line orientation classifier",
    )
    parser.add_argument(
        "--gpu-backend",
        default="off",
        choices=["off", "auto", "paddle", "torch", "tensorrt", "onnxruntime_cuda_experimental"],
        help="Optional experimental GPU backend",
    )
    parser.add_argument("--gpu-device-id", type=int, default=0, help="GPU device id")
    parser.add_argument(
        "--gpu-fp16", action="store_true", help="Use FP16 for supported GPU engines"
    )
    parser.add_argument(
        "--no-gpu-fallback",
        action="store_true",
        help="Fail instead of falling back to CPU when the requested GPU backend is unavailable",
    )
    parser.add_argument("--threads", type=int, default=0, help="Number of threads (0=auto)")
    parser.add_argument(
        "--full-resolution",
        action="store_true",
        help="Use full resolution for detection (limit_type=min instead of max)",
    )

    args = parser.parse_args()

    use_openvino = not args.no_openvino
    threads = args.threads if args.threads > 0 else max(2, os.cpu_count() or 4)

    if args.persistent:
        run_persistent(args)
    else:
        if args.batch:
            results = run_ocr_batch(args.images, args.language, args.limit_side_len, use_openvino)
            print(json.dumps({"batch": True, "results": results}))
        elif args.images:
            result = run_ocr_full(
                image_path=args.images[0],
                **_engine_options_from_args(args, threads),
            )
            print(json.dumps(result))
        else:
            print(json.dumps({"success": False, "error": "No image paths provided"}))


if __name__ == "__main__":
    main()
