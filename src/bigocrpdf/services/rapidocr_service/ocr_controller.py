"""One-shot OCR execution controller for ProfessionalPDFOCR."""

import json as _json
import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TypeGuard

import cv2
import numpy as np

from bigocrpdf.services.rapidocr_service.config import OCRConfig, OCRResult
from bigocrpdf.services.rapidocr_service.ocr_postprocess import (
    fix_vertical_overlaps,
    refine_ocr_results,
)
from bigocrpdf.services.rapidocr_service.ocr_worker_engine import build_ocr_worker_command
from bigocrpdf.services.rapidocr_service.pdf_assembly import convert_to_pdfa
from bigocrpdf.utils.logger import logger


def _save_ocr_temp_image(image: np.ndarray) -> str:
    fd, temp_img_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    if cv2.imwrite(temp_img_path, image):
        return temp_img_path
    Path(temp_img_path).unlink(missing_ok=True)
    raise OSError(f"OpenCV could not write OCR image: {temp_img_path}")


def _parse_ocr_worker_json(stdout: str, image_path: str, log_errors: bool = True) -> dict | None:
    try:
        return _json.loads(stdout.strip())
    except _json.JSONDecodeError as json_error:
        if log_errors:
            logger.error(f"OCR subprocess returned invalid JSON for {image_path}: {json_error}")
        return None


def _raw_ocr_result_has_boxes(raw_result: dict | None) -> TypeGuard[dict]:
    if not raw_result:
        return False
    if raw_result.get("error"):
        logger.error(f"OCR worker error: {raw_result['error']}")
        return False
    if not raw_result.get("boxes"):
        logger.debug("RapidOCR subprocess returned no results")
        return False
    return True


def _raw_ocr_results_to_boxes(raw_result: dict, padding: tuple) -> list[OCRResult]:
    results = []
    pad_top, pad_left = padding[0], padding[3]
    for index, raw_box in enumerate(raw_result["boxes"]):
        box = np.array(raw_box)
        if pad_top > 0 or pad_left > 0:
            box[:, 0] -= pad_left
            box[:, 1] -= pad_top
        results.append(
            OCRResult(
                raw_result["txts"][index],
                box.tolist(),
                raw_result["scores"][index],
            )
        )
    return results


class OCRController:
    """Run isolated OCR workers and PDF/A conversion."""

    def __init__(self, config: OCRConfig, openvino_checker: Callable[[], bool]) -> None:
        self._config = config
        self._openvino_checker = openvino_checker

    def fix_vertical_overlaps(self, results: list[OCRResult]) -> list[OCRResult]:
        """Fix vertical overlaps between text lines by adjusting their bounding boxes."""
        return fix_vertical_overlaps(results)

    def build_command(self, temp_img_path: str) -> list[str]:
        """Build the OCR subprocess command with all configuration parameters."""
        cpu_count = os.cpu_count() or 4
        return build_ocr_worker_command(
            self._config,
            image_path=temp_img_path,
            threads=max(2, cpu_count),
            openvino_available=(
                self._config.engine_type != "onnxruntime" and self._openvino_checker()
            ),
        )

    def run(self, image: np.ndarray, padding: tuple = (0, 0, 0, 0)) -> list[OCRResult]:
        """Run OCR on an image via subprocess for GTK isolation.

        NOTE: OCR is run in a subprocess to avoid GTK/ONNX Runtime conflicts.
        GTK's threading model interferes with ONNX Runtime causing detection to fail.
        """

        temp_img_path = ""
        try:
            temp_img_path = _save_ocr_temp_image(image)
            raw_result = self._run_ocr_worker_json(temp_img_path)
            if not _raw_ocr_result_has_boxes(raw_result):
                return []

            raw_result = refine_ocr_results(
                raw_result,
                temp_img_path,
                lambda crop_path: self._run_ocr_worker_json(crop_path, log_errors=False),
            )
            results = _raw_ocr_results_to_boxes(raw_result, padding)
            results = self._filter_low_confidence_results(results)
            return results

        except FileNotFoundError:
            logger.error(f"OCR failed: image file not found: {temp_img_path}")
            return []
        except PermissionError:
            logger.error(f"OCR failed: permission denied reading: {temp_img_path}")
            return []
        except Exception as e:
            logger.error(f"OCR failed for {temp_img_path}: {type(e).__name__}: {e}")
            return []
        finally:
            try:
                os.unlink(temp_img_path)
            except (OSError, UnboundLocalError):
                pass

    def _run_ocr_worker_json(self, image_path: str, *, log_errors: bool = True) -> dict | None:
        proc = subprocess.run(
            self.build_command(image_path),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            if log_errors:
                logger.error(f"OCR subprocess failed: {proc.stderr}")
            return None
        return _parse_ocr_worker_json(proc.stdout, image_path, log_errors=log_errors)

    def _filter_low_confidence_results(self, results: list[OCRResult]) -> list[OCRResult]:
        min_score = self._config.text_score_threshold
        before = len(results)
        results = [result for result in results if result.confidence >= min_score]
        filtered = before - len(results)
        if filtered > 0:
            logger.info(
                f"Filtered {filtered}/{before} low-confidence regions (threshold={min_score:.2f})"
            )
        return results

    def convert_to_pdfa(self, input_pdf: Path, output_pdf: Path) -> None:
        """Convert PDF to PDF/A-2b format using Ghostscript."""
        convert_to_pdfa(input_pdf, output_pdf, getattr(self._config, "page_layout", "default"))
