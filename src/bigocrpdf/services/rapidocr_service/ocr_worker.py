#!/usr/bin/env python3
"""
Standalone OCR worker script.

This script is called via subprocess to run OCR in an isolated environment,
avoiding GTK/GLib interference with ONNX Runtime.

Usage:
    python ocr_worker.py <image_path> [--language latin] [--limit_side_len 4000]
    python ocr_worker.py --batch <img1.png> <img2.png> ... [--language latin]

Output:
    JSON on stdout with OCR results
"""

import argparse
import json
import os
import sys

# Add parent paths for imports
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from bigocrpdf.utils.python_compat import (
    setup_python_compatibility,
)

# Setup Python version compatibility before any rapidocr imports
setup_python_compatibility()


def run_ocr_batch(
    image_paths: list, language: str, limit_side_len: int, use_openvino: bool = True
) -> list:
    """Run OCR on multiple images with shared RapidOCR instance."""
    import cv2

    # Import rapidocr with fallback
    try:
        from rapidocr import EngineType, LangRec, RapidOCR
    except ModuleNotFoundError:
        from bigocrpdf.utils.python_compat import get_module_from_attribute

        EngineType, LangRec, RapidOCR = get_module_from_attribute(
            "rapidocr", "EngineType", "LangRec", "RapidOCR"
        )

    # Map language string to enum
    lang_map = {
        "latin": LangRec.LATIN,
        "en": LangRec.EN,
        "ch": LangRec.CH,
        "chinese_cht": LangRec.CHINESE_CHT,
        "japan": LangRec.JAPAN,
        "korean": LangRec.KOREAN,
        "arabic": LangRec.ARABIC,
    }
    lang_rec = lang_map.get(language, LangRec.LATIN)

    # Create single RapidOCR instance for all images
    params = {
        "Det.limit_side_len": limit_side_len,
        "Global.text_score": 0.3,
        "Rec.lang_type": lang_rec,
    }
    if use_openvino:
        params["Det.engine_type"] = EngineType.OPENVINO
        params["Rec.engine_type"] = EngineType.OPENVINO

    rapid = RapidOCR(params=params)

    results = []
    for image_path in image_paths:
        try:
            img = cv2.imread(image_path)
            if img is None:
                results.append({"success": False, "error": f"Failed to load: {image_path}"})
                continue

            result = rapid(img)

            boxes = []
            txts = []
            scores = []

            if hasattr(result, "boxes") and result.boxes is not None:
                boxes = [
                    box.tolist() if hasattr(box, "tolist") else list(box) for box in result.boxes
                ]
                txts = list(result.txts) if result.txts else []
                scores = [float(s) for s in result.scores] if result.scores else []

            results.append(
                {
                    "success": True,
                    "boxes": boxes,
                    "txts": txts,
                    "scores": scores,
                    "count": len(txts),
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

        try:
            from rapidocr import EngineType, LangRec, OCRVersion, RapidOCR
        except ModuleNotFoundError:
            from bigocrpdf.utils.python_compat import get_module_from_attribute

            EngineType, LangRec, RapidOCR = get_module_from_attribute(
                "rapidocr", "EngineType", "LangRec", "RapidOCR"
            )
            OCRVersion = get_module_from_attribute("rapidocr", "OCRVersion")[0]

        # Map language string to enum
        lang_map = {
            "latin": LangRec.LATIN,
            "en": LangRec.EN,
            "ch": LangRec.CH,
            "chinese_cht": LangRec.CHINESE_CHT,
            "japan": LangRec.JAPAN,
            "korean": LangRec.KOREAN,
            "arabic": LangRec.ARABIC,
        }
        lang_rec = lang_map.get(language, LangRec.LATIN)

        img = cv2.imread(image_path)
        if img is None:
            return {"error": f"Could not load image: {image_path}"}

        engine_type = EngineType.OPENVINO if use_openvino else EngineType.ONNXRUNTIME

        params = {
            # Detection settings
            "Det.engine_type": engine_type,
            "Det.box_thresh": box_thresh,
            "Det.unclip_ratio": unclip_ratio,
            "Det.score_mode": score_mode,
            "Det.limit_side_len": limit_side_len,
            "Det.limit_type": "max",
            # Recognition settings
            "Rec.engine_type": engine_type,
            "Rec.lang_type": lang_rec,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.rec_batch_num": 8,
            # Global settings
            "Global.use_cls": False,
            "Global.text_score": text_score,
            "Global.max_side_len": limit_side_len,
        }

        # Engine-specific threading configuration
        if use_openvino:
            params.update(
                {
                    "Det.engine_cfg.inference_num_threads": threads,
                    "Rec.engine_cfg.inference_num_threads": threads,
                }
            )
        else:
            params.update(
                {
                    "Det.engine_cfg.intra_op_num_threads": threads,
                    "Det.engine_cfg.inter_op_num_threads": 2,
                    "Rec.engine_cfg.intra_op_num_threads": threads,
                    "Rec.engine_cfg.inter_op_num_threads": 2,
                }
            )

        # Add model paths if provided
        if rec_model_path:
            params["Rec.model_path"] = rec_model_path
        if rec_keys_path:
            params["Rec.rec_keys_path"] = rec_keys_path
        if det_model_path:
            params["Det.model_path"] = det_model_path
        if font_path and os.path.exists(font_path):
            params["Global.font_path"] = font_path

        ocr = RapidOCR(params=params)
        # Pass text_score and box_thresh both at init AND per-call to ensure
        # they are definitely applied (per-call overrides take precedence)
        result = ocr(img, use_cls=False, text_score=text_score, box_thresh=box_thresh)

        if result.boxes is None:
            return {"boxes": None}

        boxes = [b.tolist() if hasattr(b, "tolist") else list(b) for b in result.boxes]
        txts = list(result.txts) if result.txts else []
        scores = [float(s) for s in result.scores] if result.scores else []

        return {"boxes": boxes, "txts": txts, "scores": scores}

    except Exception as e:
        return {"error": str(e)}


def _create_ocr_engine(
    language: str = "latin",
    limit_side_len: int = 4000,
    use_openvino: bool = True,
    box_thresh: float = 0.5,
    unclip_ratio: float = 1.2,
    text_score: float = 0.3,
    score_mode: str = "slow",
    rec_model_path: str = "",
    rec_keys_path: str = "",
    det_model_path: str = "",
    font_path: str = "",
    threads: int = 4,
) -> object:
    """Create a RapidOCR engine with full parameters.

    This factory is shared by all OCR modes (single, batch, persistent)
    to ensure consistent parameter handling.
    """
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
    lang_rec = lang_map.get(language, LangRec.LATIN)

    engine_type = EngineType.OPENVINO if use_openvino else EngineType.ONNXRUNTIME

    params = {
        "Det.engine_type": engine_type,
        "Det.box_thresh": box_thresh,
        "Det.unclip_ratio": unclip_ratio,
        "Det.score_mode": score_mode,
        "Det.limit_side_len": limit_side_len,
        "Det.limit_type": "max",
        "Rec.engine_type": engine_type,
        "Rec.lang_type": lang_rec,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Rec.rec_batch_num": 8,
        "Global.use_cls": False,
        "Global.text_score": text_score,
        "Global.max_side_len": limit_side_len,
    }

    if use_openvino:
        params.update(
            {
                "Det.engine_cfg.inference_num_threads": threads,
                "Rec.engine_cfg.inference_num_threads": threads,
            }
        )
    else:
        params.update(
            {
                "Det.engine_cfg.intra_op_num_threads": threads,
                "Det.engine_cfg.inter_op_num_threads": 2,
                "Rec.engine_cfg.intra_op_num_threads": threads,
                "Rec.engine_cfg.inter_op_num_threads": 2,
            }
        )

    if rec_model_path:
        params["Rec.model_path"] = rec_model_path
    if rec_keys_path:
        params["Rec.rec_keys_path"] = rec_keys_path
    if det_model_path:
        params["Det.model_path"] = det_model_path
    if font_path and os.path.exists(font_path):
        params["Global.font_path"] = font_path

    return RapidOCR(params=params)


def _ocr_single_image(
    engine: object, image_path: str, text_score: float = 0.3, box_thresh: float = 0.5
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

        result = engine(img, use_cls=False, text_score=text_score, box_thresh=box_thresh)

        # Release image memory immediately
        del img

        if result.boxes is None:
            return {"boxes": None}

        boxes = [b.tolist() if hasattr(b, "tolist") else list(b) for b in result.boxes]
        txts = list(result.txts) if result.txts else []
        scores = [float(s) for s in result.scores] if result.scores else []

        return {"boxes": boxes, "txts": txts, "scores": scores}
    except Exception as e:
        return {"error": str(e)}


def run_persistent(args: argparse.Namespace) -> None:
    """Persistent OCR mode: reads image paths from stdin, writes results to stdout.

    The model is loaded ONCE at startup. Each line on stdin is an image path;
    the corresponding JSON result is written to stdout (one JSON per line).
    This eliminates model loading overhead for multi-page PDFs.

    Memory usage: ~400 MB (single model instance) vs ~2+ GB (subprocess per page).
    """
    import sys

    # Redirect any stray library output away from our JSON protocol
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    use_openvino = not args.no_openvino
    threads = args.threads if args.threads > 0 else max(2, os.cpu_count() or 4)

    try:
        engine = _create_ocr_engine(
            language=args.language,
            limit_side_len=args.limit_side_len,
            use_openvino=use_openvino,
            box_thresh=args.box_thresh,
            unclip_ratio=args.unclip_ratio,
            text_score=args.text_score,
            score_mode=args.score_mode,
            rec_model_path=args.rec_model_path,
            rec_keys_path=args.rec_keys_path,
            det_model_path=args.det_model_path,
            font_path=args.font_path,
            threads=threads,
        )
    except Exception as e:
        real_stdout.write(json.dumps({"fatal": str(e)}) + "\n")
        real_stdout.flush()
        return

    # Signal readiness
    real_stdout.write(json.dumps({"ready": True}) + "\n")
    real_stdout.flush()

    # Process images from stdin (one path per line)
    import gc

    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue

        result = _ocr_single_image(engine, path, args.text_score, args.box_thresh)
        real_stdout.write(json.dumps(result) + "\n")
        real_stdout.flush()

        # Prevent memory accumulation across pages
        gc.collect()
        # Force glibc to return freed pages to OS (Linux-specific)
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


def main():
    # Set reduced CPU priority so interactive applications stay responsive
    try:
        os.nice(10)
    except OSError:
        pass

    parser = argparse.ArgumentParser(description="Standalone OCR worker")
    parser.add_argument("images", nargs="*", help="Path to image file(s)")
    parser.add_argument("--batch", action="store_true", help="Batch mode (multiple images)")
    parser.add_argument("--persistent", action="store_true", help="Persistent mode (stdin/stdout)")
    parser.add_argument("--language", default="latin", help="Language code")
    parser.add_argument("--limit_side_len", type=int, default=4000, help="Max side length")
    parser.add_argument("--no-openvino", action="store_true", help="Disable OpenVINO")
    parser.add_argument("--box-thresh", type=float, default=0.5, help="Box threshold")
    parser.add_argument("--unclip-ratio", type=float, default=1.2, help="Unclip ratio")
    parser.add_argument("--text-score", type=float, default=0.3, help="Text score threshold")
    parser.add_argument("--score-mode", default="slow", help="Score mode (fast/slow)")
    parser.add_argument("--rec-model-path", default="", help="Recognition model path")
    parser.add_argument("--rec-keys-path", default="", help="Recognition keys path")
    parser.add_argument("--det-model-path", default="", help="Detection model path")
    parser.add_argument("--font-path", default="", help="Font path")
    parser.add_argument("--threads", type=int, default=0, help="Number of threads (0=auto)")

    args = parser.parse_args()

    use_openvino = not args.no_openvino
    threads = args.threads if args.threads > 0 else max(2, os.cpu_count() or 4)

    if args.persistent:
        run_persistent(args)
    elif args.batch:
        results = run_ocr_batch(args.images, args.language, args.limit_side_len, use_openvino)
        print(json.dumps({"batch": True, "results": results}))
    elif args.images:
        result = run_ocr_full(
            image_path=args.images[0],
            language=args.language,
            limit_side_len=args.limit_side_len,
            use_openvino=use_openvino,
            box_thresh=args.box_thresh,
            unclip_ratio=args.unclip_ratio,
            text_score=args.text_score,
            score_mode=args.score_mode,
            rec_model_path=args.rec_model_path,
            rec_keys_path=args.rec_keys_path,
            det_model_path=args.det_model_path,
            font_path=args.font_path,
            threads=threads,
        )
        print(json.dumps(result))
    else:
        print(json.dumps({"success": False, "error": "No image paths provided"}))


if __name__ == "__main__":
    main()
