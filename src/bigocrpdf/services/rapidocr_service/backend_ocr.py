"""OCR Execution Mixin for ProfessionalPDFOCR."""

import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from bigocrpdf.services.rapidocr_service.config import OCRResult
from bigocrpdf.services.rapidocr_service.ocr_postprocess import (
    fix_vertical_overlaps,
    refine_ocr_results,
)
from bigocrpdf.services.rapidocr_service.pdf_assembly import convert_to_pdfa
from bigocrpdf.utils.logger import logger


class BackendOCRMixin:
    """Mixin providing OCR execution and searchable PDF creation methods."""

    def _fix_vertical_overlaps(self, results: list[OCRResult]) -> list[OCRResult]:
        """Fix vertical overlaps between text lines by adjusting their bounding boxes."""
        return fix_vertical_overlaps(results)

    def _build_ocr_command(self, temp_img_path: str) -> list[str]:
        """Build the OCR subprocess command with all configuration parameters."""
        worker_script = str(Path(__file__).parent / "ocr_worker.py")
        cpu_count = os.cpu_count() or 4
        optimal_threads = max(2, cpu_count)

        cmd = [
            "python3",
            worker_script,
            temp_img_path,
            "--language",
            self.config.language,
            "--limit_side_len",
            str(self.config.detection_limit_side_len),
            "--box-thresh",
            str(self.config.box_thresh),
            "--unclip-ratio",
            str(self.config.unclip_ratio),
            "--text-score",
            str(self.config.text_score_threshold),
            "--score-mode",
            self.config.score_mode,
            "--threads",
            str(optimal_threads),
        ]

        # Only disable OpenVINO in subprocess if main process also can't use it
        if not self._check_openvino_available():
            cmd.append("--no-openvino")

        for flag, getter in [
            ("--rec-model-path", self.config.get_rec_model_path),
            ("--rec-keys-path", self.config.get_rec_keys_path),
            ("--det-model-path", self.config.get_det_model_path),
            ("--font-path", self.config.get_font_path),
        ]:
            path = getter()
            if path:
                cmd.extend([flag, str(path)])

        return cmd

    def _run_ocr(self, image: np.ndarray, padding: tuple = (0, 0, 0, 0)) -> list[OCRResult]:
        """Run OCR on an image via subprocess for GTK isolation.

        NOTE: OCR is run in a subprocess to avoid GTK/ONNX Runtime conflicts.
        GTK's threading model interferes with ONNX Runtime causing detection to fail.
        """

        try:
            fd, temp_img_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            cv2.imwrite(temp_img_path, image)

            cmd = self._build_ocr_command(temp_img_path)
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if proc.returncode != 0:
                logger.error(f"OCR subprocess failed: {proc.stderr}")
                return []

            # Parse the raw JSON result for refinement before converting
            import json as _json

            try:
                raw_result = _json.loads(proc.stdout.strip())
            except _json.JSONDecodeError as jde:
                logger.error(f"OCR subprocess returned invalid JSON for {temp_img_path}: {jde}")
                return []

            if raw_result.get("error") or not raw_result.get("boxes"):
                if raw_result.get("error"):
                    logger.error(f"OCR worker error: {raw_result['error']}")
                else:
                    logger.debug("RapidOCR subprocess returned no results")
                return []

            # Refine low-confidence oversized detections before parsing
            def _subprocess_ocr(crop_path: str) -> dict | None:
                crop_cmd = self._build_ocr_command(crop_path)
                crop_proc = subprocess.run(crop_cmd, capture_output=True, text=True, timeout=120)
                if crop_proc.returncode != 0:
                    return None
                try:
                    return _json.loads(crop_proc.stdout.strip())
                except _json.JSONDecodeError:
                    return None

            raw_result = refine_ocr_results(raw_result, temp_img_path, _subprocess_ocr)

            # Convert refined raw result to OCRResult list
            results = []
            pad_top, pad_left = padding[0], padding[3]
            for i in range(len(raw_result["boxes"])):
                box = np.array(raw_result["boxes"][i])
                if pad_top > 0 or pad_left > 0:
                    box[:, 0] -= pad_left
                    box[:, 1] -= pad_top
                results.append(
                    OCRResult(
                        raw_result["txts"][i],
                        box.tolist(),
                        raw_result["scores"][i],
                    )
                )

            # Post-processing confidence filter — additional safety net
            # RapidOCR applies text_score internally, but we double-check here
            # to ensure low-confidence noise (e.g. signatures) is removed
            min_score = self.config.text_score_threshold
            before = len(results)
            results = [r for r in results if r.confidence >= min_score]
            filtered = before - len(results)
            if filtered > 0:
                logger.info(
                    f"Filtered {filtered}/{before} low-confidence regions "
                    f"(threshold={min_score:.2f})"
                )

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

    def _convert_to_pdfa(self, input_pdf: Path, output_pdf: Path) -> None:
        """Convert PDF to PDF/A-2b format using Ghostscript."""
        convert_to_pdfa(input_pdf, output_pdf)
