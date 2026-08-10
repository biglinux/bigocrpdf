"""
RapidOCR Configuration and Data Types.

This module contains configuration dataclasses and result types
for the RapidOCR integration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from bigocrpdf.services.rapidocr_service.resource_paths import find_font_dir, find_model_dir

# Model and font locations come from resource_paths, resolved per instance so a
# relocatable build finds the copy it ships with instead of the host's.

# --- OCR Processing Defaults (single source of truth) ---
# Shared by OCRConfig and OcrSettings to avoid default drift.
DEFAULT_LANGUAGE = "latin"
DEFAULT_DPI = 300
DEFAULT_BOX_THRESH = 0.5
DEFAULT_UNCLIP_RATIO = 1.6
DEFAULT_DETECTION_LIMIT_SIDE_LEN = 4096
DEFAULT_SCORE_MODE = "slow"
DEFAULT_TEXT_SCORE_THRESHOLD = 0.3
DEFAULT_ENGINE_TYPE = "openvino"
DEFAULT_MODEL_TYPE = "small"
DEFAULT_REC_BATCH_NUM = 1
DEFAULT_USE_TEXTLINE_CLS = False
DEFAULT_TEXT_LAYER_RENDERER = "unicode"
DEFAULT_PDF_MODE = "auto"
DEFAULT_GPU_BACKEND = "off"
DEFAULT_GPU_DEVICE_ID = 0
DEFAULT_GPU_FP16 = True
DEFAULT_GPU_FALLBACK_TO_CPU = True
DEFAULT_FALLBACK_RENDER_DPI = 300
DEFAULT_RETRY_RENDER_DPI = 350
DEFAULT_MAX_RENDER_MEGAPIXELS = 45
DEFAULT_MAX_IMAGE_MEGAPIXELS = 128.0
DEFAULT_MAX_PDF_PAGES = 2000
# Geometric corrections (ON by default)
DEFAULT_ENABLE_PERSPECTIVE_CORRECTION = True
DEFAULT_ENABLE_DESKEW = True
DEFAULT_ENABLE_BASELINE_DEWARP = True
DEFAULT_ENABLE_ORIENTATION_DETECTION = True
# Color/enhancement stays off by default to preserve the source for OCR.
DEFAULT_ENABLE_PREPROCESSING = False
DEFAULT_ENABLE_AUTO_CONTRAST = False
DEFAULT_ENABLE_AUTO_BRIGHTNESS = False
DEFAULT_ENABLE_DENOISE = False
DEFAULT_ENABLE_BORDER_CLEAN = False
DEFAULT_ENABLE_SCANNER_EFFECT = True
DEFAULT_SCANNER_EFFECT_STRENGTH = 1.0
DEFAULT_ENABLE_VINTAGE_LOOK = False
DEFAULT_VINTAGE_BW = False
# Output
DEFAULT_CONVERT_TO_PDFA = True
DEFAULT_MAX_FILE_SIZE_MB = 0
# Viewer page layout written to the PDF catalog /PageLayout.
# One of: "default" (omit), "single", "continuous", "two_page".
DEFAULT_PAGE_LAYOUT = "default"
# Image export
DEFAULT_IMAGE_EXPORT_FORMAT = "original"
DEFAULT_IMAGE_EXPORT_QUALITY = 85
DEFAULT_AUTO_DETECT_QUALITY = True
# Bilevel compression (JBIG2/CCITT)
DEFAULT_ENABLE_BILEVEL_COMPRESSION = True
DEFAULT_FORCE_BILEVEL_COMPRESSION = False
# Detection resolution: False = capped at 2000px (faster), True = full resolution (more accurate)
DEFAULT_DETECTION_FULL_RESOLUTION = False
# Execution
DEFAULT_WORKERS = 0
DEFAULT_REPLACE_EXISTING_OCR = False
DEFAULT_ENHANCE_EMBEDDED_IMAGES = False


@dataclass
class OCRConfig:
    """Configuration for RapidOCR processing.

    Attributes:
        language: Language code for recognition (latin, ch, japan, korean, etc.)
        dpi: Resolution for image extraction
        box_thresh: Detection threshold for text boxes
        unclip_ratio: Box expansion ratio
        detection_limit_side_len: Max side length for detection
        engine_type: Inference engine (openvino only)
        model_base_path: Base path for model files
        font_base_path: Base path for font files
        enable_deskew: Correct skewed documents
        enable_orientation_detection: Detect and fix page rotation
        enable_preprocessing: Master switch for color enhancements
        enable_auto_contrast: CLAHE for low-contrast images
        enable_denoise: Apply denoising
        convert_to_pdfa: Convert output to PDF/A-2b
        workers: Number of parallel workers
    """

    # === Core Settings ===
    language: str = DEFAULT_LANGUAGE
    dpi: int = DEFAULT_DPI

    # === Detection Thresholds ===
    box_thresh: float = DEFAULT_BOX_THRESH
    unclip_ratio: float = DEFAULT_UNCLIP_RATIO
    detection_limit_side_len: int = DEFAULT_DETECTION_LIMIT_SIDE_LEN
    detection_full_resolution: bool = DEFAULT_DETECTION_FULL_RESOLUTION
    score_mode: str = DEFAULT_SCORE_MODE
    text_score_threshold: float = DEFAULT_TEXT_SCORE_THRESHOLD

    # === Model Settings ===
    engine_type: str = DEFAULT_ENGINE_TYPE
    model_type: str = DEFAULT_MODEL_TYPE
    rec_batch_num: int = DEFAULT_REC_BATCH_NUM
    use_textline_cls: bool = DEFAULT_USE_TEXTLINE_CLS

    # === Text Layer ===
    text_layer_renderer: str = DEFAULT_TEXT_LAYER_RENDERER
    pdf_mode: Literal["ocr", "geometric", "auto", "auto_verified"] = DEFAULT_PDF_MODE

    # === Optional GPU Inference ===
    gpu_backend: str = DEFAULT_GPU_BACKEND
    gpu_device_id: int = DEFAULT_GPU_DEVICE_ID
    gpu_fp16: bool = DEFAULT_GPU_FP16
    gpu_fallback_to_cpu: bool = DEFAULT_GPU_FALLBACK_TO_CPU

    # === Adaptive Rendering ===
    fallback_render_dpi: int = DEFAULT_FALLBACK_RENDER_DPI
    retry_render_dpi: int = DEFAULT_RETRY_RENDER_DPI
    max_render_megapixels: int = DEFAULT_MAX_RENDER_MEGAPIXELS
    max_image_megapixels: float = DEFAULT_MAX_IMAGE_MEGAPIXELS
    max_pdf_pages: int = DEFAULT_MAX_PDF_PAGES

    # === Paths (BigLinux standard) ===
    model_base_path: Path = field(default_factory=find_model_dir)
    font_base_path: Path = field(default_factory=find_font_dir)

    # === Preprocessing Options ===
    # Geometric corrections
    enable_perspective_correction: bool = DEFAULT_ENABLE_PERSPECTIVE_CORRECTION
    enable_deskew: bool = DEFAULT_ENABLE_DESKEW
    enable_baseline_dewarp: bool = DEFAULT_ENABLE_BASELINE_DEWARP
    enable_orientation_detection: bool = DEFAULT_ENABLE_ORIENTATION_DETECTION
    # Color/enhancement: off by default to preserve the source for OCR.
    enable_preprocessing: bool = DEFAULT_ENABLE_PREPROCESSING
    enable_auto_contrast: bool = DEFAULT_ENABLE_AUTO_CONTRAST
    enable_auto_brightness: bool = DEFAULT_ENABLE_AUTO_BRIGHTNESS
    enable_denoise: bool = DEFAULT_ENABLE_DENOISE
    enable_border_clean: bool = DEFAULT_ENABLE_BORDER_CLEAN
    enable_scanner_effect: bool = DEFAULT_ENABLE_SCANNER_EFFECT
    scanner_effect_strength: float = DEFAULT_SCANNER_EFFECT_STRENGTH
    enable_vintage_look: bool = DEFAULT_ENABLE_VINTAGE_LOOK
    vintage_bw: bool = DEFAULT_VINTAGE_BW

    # === Output Options ===
    convert_to_pdfa: bool = DEFAULT_CONVERT_TO_PDFA
    max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB
    page_layout: str = DEFAULT_PAGE_LAYOUT
    enable_bilevel_compression: bool = DEFAULT_ENABLE_BILEVEL_COMPRESSION
    force_bilevel_compression: bool = DEFAULT_FORCE_BILEVEL_COMPRESSION

    # === Image Export Options ===
    image_export_format: str = DEFAULT_IMAGE_EXPORT_FORMAT
    image_export_quality: int = DEFAULT_IMAGE_EXPORT_QUALITY
    auto_detect_quality: bool = DEFAULT_AUTO_DETECT_QUALITY

    # === Execution Options ===
    workers: int = DEFAULT_WORKERS
    page_range: tuple[int, int] | None = None
    page_modifications: list[dict] | None = None
    force_full_ocr: bool = False
    replace_existing_ocr: bool = DEFAULT_REPLACE_EXISTING_OCR
    enhance_embedded_images: bool = DEFAULT_ENHANCE_EMBEDDED_IMAGES

    def get_font_path(self) -> Path:
        """Get correct font path based on language."""
        font_map = {
            "latin": "latin.ttf",
            "english": "latin.ttf",
            "ch": "FZYTK.TTF",
            "chinese_cht": "FZYTK.TTF",
            "japan": "japan.ttf",
            "korean": "korean.ttf",
            "arabic": "arabic.ttf",
            "cyrillic": "latin.ttf",
            "devanagari": "devanagari.ttf",
            "greek": "latin.ttf",
            "tamil": "tamil.ttf",
            "telugu": "telugu.ttf",
            "thai": "thai.ttf",
        }
        font_name = font_map.get(self.language, "latin.ttf")
        font_path = self.font_base_path / font_name

        # Fallback to latin if specific font not found
        if not font_path.exists():
            font_path = self.font_base_path / "latin.ttf"

        return font_path

    def get_rec_model_path(self) -> Path | None:
        """Get the installed PP-OCRv6 unified recognition model."""
        path = self.model_base_path / f"PP-OCRv6_rec_{self.model_type}.onnx"
        return path if path.exists() else None

    def get_det_model_path(self) -> Path | None:
        """Get detection model path."""
        path = self.model_base_path / f"PP-OCRv6_det_{self.model_type}.onnx"
        return path if path.exists() else None

    def get_rec_keys_path(self) -> Path | None:
        """PP-OCRv6's unified model owns its recognition dictionary."""
        return None


@dataclass
class OCRResult:
    """Result from OCR processing of a single text region."""

    text: str
    box: list[list[float]] = field(default_factory=list)  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    confidence: float = 0.0


@dataclass
class OcrWord:
    """Positioned OCR word used by structured document exports."""

    text: str
    bbox: list[float] = field(default_factory=list)  # [x1, y1, x2, y2] in page pixels
    confidence: float = 0.0


@dataclass
class OcrLine:
    """Positioned OCR text line in reading order."""

    text: str
    bbox: list[float] = field(default_factory=list)  # [x1, y1, x2, y2] in page pixels
    words: list[OcrWord] = field(default_factory=list)
    reading_order: int = 0
    source: str = "ocr"


@dataclass
class OcrLayoutBlock:
    """Structured document block used by TXT, Markdown, and ODT exports."""

    kind: str
    text: str = ""
    rows: list[list[str]] = field(default_factory=list)
    raw_lines: list[str] = field(default_factory=list)
    indent_chars: int = 0
    y_top: float = 0.0
    reading_order: int = 0


@dataclass
class OCRBoxData:
    """Structured OCR box data for high-fidelity export."""

    text: str
    x: float  # X position as percentage of page width (0-100)
    y: float  # Y position as percentage of page height (0-100)
    width: float  # Width as percentage of page width
    height: float  # Height in points (estimated font size)
    confidence: float = 0.0
    page_num: int = 0
    is_bold: bool = False
    is_underlined: bool = False


@dataclass
class OcrPage:
    """Canonical OCR data for one processed page."""

    page_index: int
    width_px: int
    height_px: int
    dpi: int
    text_results: list[OCRResult] = field(default_factory=list)
    lines: list[OcrLine] = field(default_factory=list)
    layout_blocks: list[OcrLayoutBlock] = field(default_factory=list)
    native_text: str = ""
    text_layer_quality: str = "absent"
    retry_level: int = 0
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass
class OcrDocument:
    """Canonical OCR document data shared by exports and benchmarks."""

    pages: list[OcrPage] = field(default_factory=list)
    diagnostics: dict[str, object] = field(default_factory=dict)

    def append_page(self, page: OcrPage) -> None:
        self.pages.append(page)

    def full_text(self) -> str:
        parts = []
        for page in self.pages:
            if page.native_text:
                page_text = page.native_text
            elif page.lines:
                page_text = "\n".join(line.text for line in page.lines)
            else:
                page_text = "\n".join(result.text for result in page.text_results)
            if page_text.strip():
                parts.append(page_text.strip())
        return "\n\n".join(parts)


@dataclass
class ProcessingStats:
    """Statistics from OCR processing."""

    pages_total: int = 0
    pages_processed: int = 0
    total_text_regions: int = 0
    average_confidence: float = 0.0
    processing_time_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    # Additional fields used by processor
    total_words: int = 0
    total_chars: int = 0
    # Fields from backend processing
    full_text: str = ""
    ocr_boxes: list[OCRBoxData] = field(default_factory=list)
    ocr_document: OcrDocument = field(default_factory=OcrDocument)
    error: str | None = None
    # Split output files (when max_file_size_mb is set and exceeded)
    split_output_files: list[str] = field(default_factory=list)
