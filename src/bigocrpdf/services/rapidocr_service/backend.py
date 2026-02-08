#!/usr/bin/env python3
"""
Professional PDF OCR Application
High-quality searchable PDF generation using RapidOCR PP-OCRv5

Features:
- High-resolution image extraction (300 DPI)
- Adaptive image preprocessing
- Server-grade OCR models for maximum accuracy
- Precise invisible text layer positioning with rotation support
- PDF/A-2b compliance
- Progress tracking and comprehensive logging

Author: Professional OCR Suite
License: MIT
"""

import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Import rapidocr with fallback to other Python versions
# Import unified OCRConfig and data classes from the single source of truth
from bigocrpdf.services.rapidocr_service.config import (
    OCRBoxData,
    OCRConfig,
    OCRResult,
    ProcessingStats,
)

# Import extracted logic
from bigocrpdf.services.rapidocr_service.pdf_extractor import (
    PDFImageExtractor,
    TextLayerRenderer,
    has_native_text,
)

# Import ImagePreprocessor from dedicated module (single source of truth)
from bigocrpdf.services.rapidocr_service.preprocessor import ImagePreprocessor

# Import unified rotation module
from bigocrpdf.utils.python_compat import (
    get_module_from_attribute,
    setup_python_compatibility,
)

# Setup compatibility paths before importing rapidocr
setup_python_compatibility()

# Import rapidocr components with fallback
try:
    from rapidocr import EngineType, LangRec, OCRVersion, RapidOCR
except ModuleNotFoundError:
    # Use fallback import
    EngineType, LangRec, OCRVersion, RapidOCR = get_module_from_attribute(
        "rapidocr", "EngineType", "LangRec", "OCRVersion", "RapidOCR"
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# === Enum conversion helpers ===
# The canonical OCRConfig uses strings for language and engine_type.
# These helpers convert to rapidocr enums when calling the OCR engine.

_LANG_MAP: dict[str, Any] = {
    "latin": LangRec.LATIN,
    "en": LangRec.EN,
    "ch": LangRec.CH,
    "chinese_cht": LangRec.CHINESE_CHT,
    "japan": LangRec.JAPAN,
    "korean": LangRec.KOREAN,
    "arabic": LangRec.ARABIC,
    "cyrillic": LangRec.CYRILLIC,
    "devanagari": LangRec.DEVANAGARI,
    "el": LangRec.EL,
    "eslav": LangRec.ESLAV,
    "ta": LangRec.TA,
    "te": LangRec.TE,
    "th": LangRec.TH,
    "ka": LangRec.KA,
}

_ENGINE_MAP: dict[str, Any] = {
    "openvino": EngineType.OPENVINO,
    "onnx": EngineType.ONNXRUNTIME,
    "onnxruntime": EngineType.ONNXRUNTIME,
}


def _to_lang_enum(language: str) -> Any:
    """Convert language string to LangRec enum."""
    return _LANG_MAP.get(language, LangRec.LATIN)


def _to_engine_enum(engine_type: str) -> Any:
    """Convert engine type string to EngineType enum."""
    return _ENGINE_MAP.get(engine_type, EngineType.OPENVINO)


from bigocrpdf.services.rapidocr_service.backend_ocr import BackendOCRMixin
from bigocrpdf.services.rapidocr_service.backend_pipeline import BackendPipelineMixin
from bigocrpdf.services.rapidocr_service.backend_text_layer import BackendTextLayerMixin

# Re-export for backward compatibility (used by tests and external modules via patching)
from bigocrpdf.services.rapidocr_service.rotation import (  # noqa: F401
    apply_final_rotation_to_pdf,
    extract_page_rotations,
)


class ProfessionalPDFOCR(
    BackendPipelineMixin,
    BackendTextLayerMixin,
    BackendOCRMixin,
):
    """High-quality PDF OCR engine for professional document processing."""

    # Class-level cache for OpenVINO availability (check once per process)
    _openvino_available: bool | None = None

    def __init__(self, config: OCRConfig | None = None):
        self.config = config or OCRConfig()
        self.cancel_event = threading.Event()  # Cooperative cancellation
        self.preprocessor = ImagePreprocessor(self.config)
        self.extractor = PDFImageExtractor(self.config.dpi)
        self.renderer = TextLayerRenderer(self.config.dpi)
        self.engine = None  # Lazy-loaded: only when _run_ocr_direct() is called
        logger.info("Initializing RapidOCR engine...")

    def _format_ocr_text(self, ocr_results: list[OCRResult], page_width: float) -> str:
        """Format OCR results into readable text with line breaks and paragraphs."""
        if not ocr_results:
            return ""

        sorted_results = self.renderer._sort_for_reading_order(ocr_results, page_width)
        if not sorted_results:
            return ""

        formatted_text = ""
        prev_y = -1
        prev_bottom = -1

        for r in sorted_results:
            ys = [p[1] for p in r.box]
            curr_top = min(ys)
            curr_bottom = max(ys)
            curr_h = curr_bottom - curr_top
            center_y = (curr_top + curr_bottom) / 2

            if prev_y != -1:
                # Column break (moved UP significantly)
                if center_y < prev_y - (curr_h * 2):
                    formatted_text += "\n\n"
                # Paragraph break (vertical gap > 60% of line height)
                elif (curr_top - prev_bottom) > (curr_h * 0.6):
                    formatted_text += "\n\n"
                # Line break (moved DOWN significantly)
                elif center_y > prev_bottom:
                    formatted_text += "\n"
                elif center_y > prev_y + (curr_h * 0.5):
                    formatted_text += "\n"
                else:
                    formatted_text += " "

            formatted_text += r.text
            prev_y = center_y
            prev_bottom = curr_bottom

        return formatted_text

    def _collect_ocr_boxes(
        self,
        ocr_results: list[OCRResult],
        page_num: int,
        page_width: float,
        page_height: float,
    ) -> list[OCRBoxData]:
        """Collect structured OCR box data for high-fidelity export.

        Converts pixel coordinates to percentages (x, y, width) and
        calculates height in points for font size estimation.
        """
        boxes = []
        dpi = self.config.dpi or 300  # Default DPI for conversion

        for r in ocr_results:
            xs = [p[0] for p in r.box]
            ys = [p[1] for p in r.box]
            box_x = min(xs)
            box_y = min(ys)
            box_w = max(xs) - box_x
            box_h = max(ys) - box_y

            x_pct = (box_x / page_width) * 100 if page_width > 0 else 0
            y_pct = (box_y / page_height) * 100 if page_height > 0 else 0
            w_pct = (box_w / page_width) * 100 if page_width > 0 else 0

            # Convert height from pixels to points (1 inch = 72 points)
            # height_pts = box_h_pixels * (72 points/inch) / (dpi pixels/inch)
            height_pts = (box_h * 72) / dpi

            boxes.append(
                OCRBoxData(
                    text=r.text,
                    x=x_pct,
                    y=y_pct,
                    width=w_pct,
                    height=height_pts,
                    confidence=r.confidence,
                    page_num=page_num,
                )
            )
        return boxes

    @classmethod
    def _check_openvino_available(cls) -> bool:
        """Check if OpenVINO is available and compatible with current Python version.

        Result is cached to avoid import system corruption on repeated failed imports.
        """
        if cls._openvino_available is not None:
            return cls._openvino_available

        try:
            from openvino._pyopenvino import AxisSet  # noqa: F401

            cls._openvino_available = True
        except (ImportError, ModuleNotFoundError, KeyError):
            cls._openvino_available = False

        return cls._openvino_available

    def _init_engine(self):
        """Initialize RapidOCR engine with optimized configuration.

        Performance optimizations applied:
        - OpenVINO backend for faster CPU inference
        - Optimized threading configuration
        - PP-OCRv5 models for best accuracy
        - Classifier disabled (causes alignment issues)
        """
        logger.info("Initializing RapidOCR engine...")

        # Use all available cores for maximum throughput.
        # Worker processes run at lowest priority (nice 19) to avoid
        # impacting other applications.
        cpu_count = os.cpu_count() or 4
        optimal_threads = max(2, cpu_count)

        # Convert string config values to rapidocr enums
        effective_engine_type = _to_engine_enum(self.config.engine_type)

        # Pre-check: If OpenVINO requested but not available, switch to ONNX immediately
        # This avoids state corruption from failed OpenVINO initialization
        if effective_engine_type == EngineType.OPENVINO and not self._check_openvino_available():
            logger.warning(
                "OpenVINO not compatible with current Python version. Using ONNX Runtime directly."
            )
            effective_engine_type = EngineType.ONNXRUNTIME

        # Convert model paths to strings for rapidocr (returns Path | None)
        rec_model = self.config.get_rec_model_path()
        rec_keys = self.config.get_rec_keys_path()
        det_model = self.config.get_det_model_path()
        font_path = self.config.get_font_path()

        params = {
            # === Detection settings ===
            "Det.engine_type": effective_engine_type,
            "Det.box_thresh": self.config.box_thresh,
            "Det.unclip_ratio": self.config.unclip_ratio,
            "Det.score_mode": self.config.score_mode,  # 'slow' for better accuracy
            "Det.limit_side_len": self.config.detection_limit_side_len,
            "Det.limit_type": "max",  # Use max to ensure large images are processed
            # === Recognition settings ===
            "Rec.engine_type": effective_engine_type,
            "Rec.lang_type": _to_lang_enum(self.config.language),
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "Rec.model_path": str(rec_model) if rec_model else None,
            "Rec.rec_keys_path": str(rec_keys) if rec_keys else None,
            "Rec.rec_batch_num": 8,  # Batch more recognition requests for throughput
            # === Classifier settings (use same engine even if disabled) ===
            "Cls.engine_type": effective_engine_type,
            # === Global settings ===
            "Global.font_path": str(font_path) if font_path else None,
            "Global.use_cls": False,  # Disable classifier (causes alignment issues)
            "Global.text_score": self.config.text_score_threshold,  # Lower threshold catches more text
            "Global.max_side_len": self.config.detection_limit_side_len,
        }

        # Add detection model path if we have a custom one
        if det_model:
            params["Det.model_path"] = str(det_model)
            logger.info(f"Using detection model: {det_model}")

        # === Engine-specific optimizations ===
        if effective_engine_type == EngineType.OPENVINO:
            # OpenVINO threading optimization
            params.update(
                {
                    "Det.engine_cfg.inference_num_threads": optimal_threads,
                    "Rec.engine_cfg.inference_num_threads": optimal_threads,
                }
            )
            logger.info(f"OpenVINO: Using {optimal_threads} inference threads")
        else:
            # ONNXRuntime threading optimization
            params.update(
                {
                    "Det.engine_cfg.intra_op_num_threads": optimal_threads,
                    "Det.engine_cfg.inter_op_num_threads": 2,
                    "Rec.engine_cfg.intra_op_num_threads": optimal_threads,
                    "Rec.engine_cfg.inter_op_num_threads": 2,
                }
            )
            logger.info(f"ONNXRuntime: Using {optimal_threads} intra-op threads")

        logger.debug(f"Engine params: {params}")

        try:
            self.engine = RapidOCR(params=params)
            logger.info("RapidOCR engine initialized successfully")
        except Exception as e:
            # Check if this is an OpenVINO compatibility error
            error_msg = str(e)
            if effective_engine_type == EngineType.OPENVINO and (
                "openvino" in error_msg.lower()
                or "AxisSet" in error_msg
                or "_pyopenvino" in error_msg
            ):
                logger.warning(
                    f"OpenVINO initialization failed: {e}. Falling back to ONNX Runtime..."
                )
                # Retry with ONNX Runtime — update string config for consistency
                self.config.engine_type = "onnxruntime"
                params["Det.engine_type"] = EngineType.ONNXRUNTIME
                params["Rec.engine_type"] = EngineType.ONNXRUNTIME
                # Update threading config for ONNX
                params.pop("Det.engine_cfg.inference_num_threads", None)
                params.pop("Rec.engine_cfg.inference_num_threads", None)
                params.update(
                    {
                        "Det.engine_cfg.intra_op_num_threads": optimal_threads,
                        "Det.engine_cfg.inter_op_num_threads": 2,
                        "Rec.engine_cfg.intra_op_num_threads": optimal_threads,
                        "Rec.engine_cfg.inter_op_num_threads": 2,
                    }
                )
                try:
                    # Log params for debugging
                    logger.info(f"Fallback ONNX params: {params}")
                    self.engine = RapidOCR(params=params)
                    logger.info("RapidOCR engine initialized with ONNX Runtime (fallback)")
                except Exception as fallback_error:
                    fallback_msg = str(fallback_error).lower()
                    if "onnxruntime is not installed" in fallback_msg:
                        import sys

                        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
                        logger.error(
                            f"OCR dependencies not compatible with Python {py_version}. "
                            f"Install python-onnxruntime-cuda or python-onnxruntime-cpu: "
                            f"sudo pacman -S python-onnxruntime-cuda"
                        )
                        raise RuntimeError(
                            f"OCR engine requires ONNX Runtime for Python {py_version}.\n"
                            f"Install with: sudo pacman -S python-onnxruntime-cuda"
                        ) from fallback_error
                    logger.error(
                        f"Failed to initialize RapidOCR with ONNX fallback: {fallback_error}"
                    )
                    raise
            else:
                logger.error(f"Failed to initialize RapidOCR: {e}")
                raise

    def process(
        self,
        input_pdf: Path,
        output_pdf: Path,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> ProcessingStats:
        """Process PDF and create searchable version.

        This method detects the PDF type and uses the appropriate strategy:
        - Image-only PDFs: Extract images, OCR all, create searchable PDF
        - Mixed content PDFs: Preserve original structure, OCR only images in place

        Args:
            input_pdf: Path to input PDF file
            output_pdf: Path for output searchable PDF
            progress_callback: Optional callback(current, total, status_message)

        Returns:
            ProcessingStats with processing details
        """
        input_pdf = Path(input_pdf)
        output_pdf = Path(output_pdf)

        if not input_pdf.exists():
            raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

        logger.info(f"Processing: {input_pdf}")
        logger.info(f"Output: {output_pdf}")

        # Check if PDF has native text (mixed content)
        if (
            not self.config.force_full_ocr
            and not self.config.replace_existing_ocr
            and has_native_text(input_pdf)
        ):
            logger.info("Detected mixed content PDF (text + images). Using preservation mode.")
            return self._process_mixed_content_pdf(input_pdf, output_pdf, progress_callback)
        else:
            if self.config.force_full_ocr:
                logger.info("Force full OCR mode (editor-merged file). Using full OCR mode.")
            elif self.config.replace_existing_ocr and has_native_text(input_pdf):
                # Mixed-content PDF with replace_existing_ocr: need full-page rendering
                self.config.render_full_pages = True
                logger.info("Mixed-content PDF with replace OCR. Using full-page rendering mode.")
            else:
                logger.info("Detected image-only PDF. Using full OCR mode.")
            return self._process_image_only_pdf(input_pdf, output_pdf, progress_callback)
