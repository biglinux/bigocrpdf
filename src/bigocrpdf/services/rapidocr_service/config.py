"""
RapidOCR Configuration and Data Types.

This module contains configuration dataclasses and result types
for the RapidOCR integration.
"""

from dataclasses import dataclass, field
from pathlib import Path

# Constants for model paths (BigLinux standard)
DEFAULT_MODEL_PATH = Path("/usr/share/rapidocr/models")
DEFAULT_FONT_PATH = Path("/usr/share/rapidocr/fonts")

# --- OCR Processing Defaults (single source of truth) ---
# Used by OCRConfig, PreprocessingConfig and OcrSettings to avoid drift.
DEFAULT_LANGUAGE = "latin"
DEFAULT_DPI = 300
DEFAULT_BOX_THRESH = 0.5
DEFAULT_UNCLIP_RATIO = 1.2
DEFAULT_DETECTION_LIMIT_SIDE_LEN = 4096
DEFAULT_SCORE_MODE = "slow"
DEFAULT_TEXT_SCORE_THRESHOLD = 0.3
DEFAULT_ENGINE_TYPE = "openvino"
# Geometric corrections (ON by default)
DEFAULT_ENABLE_PERSPECTIVE_CORRECTION = True
DEFAULT_ENABLE_DESKEW = True
DEFAULT_ENABLE_BASELINE_DEWARP = True
DEFAULT_ENABLE_ORIENTATION_DETECTION = True
# Color/enhancement (OFF by default – PP-OCRv5 works best without)
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
DEFAULT_CONVERT_TO_PDFA = False
DEFAULT_MAX_FILE_SIZE_MB = 0
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
        use_server_models: Use high-quality server models if available
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
    use_server_models: bool = False
    engine_type: str = DEFAULT_ENGINE_TYPE

    # === Paths (BigLinux standard) ===
    model_base_path: Path = field(default_factory=lambda: DEFAULT_MODEL_PATH)
    font_base_path: Path = field(default_factory=lambda: DEFAULT_FONT_PATH)

    # === Preprocessing Options ===
    # Geometric corrections
    enable_perspective_correction: bool = DEFAULT_ENABLE_PERSPECTIVE_CORRECTION
    enable_deskew: bool = DEFAULT_ENABLE_DESKEW
    enable_baseline_dewarp: bool = DEFAULT_ENABLE_BASELINE_DEWARP
    enable_orientation_detection: bool = DEFAULT_ENABLE_ORIENTATION_DETECTION
    # Color/Enhancement: OFF by default (PP-OCRv5 works best without)
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
        """Get recognition model path based on language."""
        # Try server model first if requested
        if self.use_server_models:
            server_path = self.model_base_path / f"{self.language}_PP-OCRv5_rec_server_infer.onnx"
            if server_path.exists():
                return server_path

        # Mobile model
        mobile_path = self.model_base_path / f"{self.language}_PP-OCRv5_rec_infer.onnx"
        if mobile_path.exists():
            return mobile_path

        # Fallback patterns
        patterns = [
            f"{self.language}_PP-OCRv5_rec_mobile_infer.onnx",
            f"{self.language}_ppocr_mobile_v2.0_rec_infer.onnx",
        ]

        for pattern in patterns:
            path = self.model_base_path / pattern
            if path.exists():
                return path

        return None

    def get_det_model_path(self) -> Path | None:
        """Get detection model path."""
        # Try server model first if requested
        if self.use_server_models:
            server_path = self.model_base_path / "ch_PP-OCRv5_server_det.onnx"
            if server_path.exists():
                return server_path

        # Mobile model
        mobile_path = self.model_base_path / "ch_PP-OCRv5_mobile_det.onnx"
        if mobile_path.exists():
            return mobile_path

        return None

    def get_rec_keys_path(self) -> Path | None:
        """Get recognition dictionary path based on language."""
        dict_path = self.model_base_path / f"ppocrv5_{self.language}_dict.txt"
        if dict_path.exists():
            return dict_path

        # Fallback patterns
        patterns = [
            f"{self.language}_dict.txt",
            f"ppocr_keys_{self.language}.txt",
        ]
        for pattern in patterns:
            path = self.model_base_path / pattern
            if path.exists():
                return path

        return None


@dataclass
class OCRResult:
    """Result from OCR processing of a single text region."""

    text: str
    box: list[list[float]] = field(default_factory=list)  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    confidence: float = 0.0


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
    total_time: float = 0.0  # Alias for processing_time_seconds
    # Fields from backend processing
    full_text: str = ""
    ocr_boxes: list[OCRBoxData] = field(default_factory=list)
    error: str | None = None
    # Split output files (when max_file_size_mb is set and exceeded)
    split_output_files: list[str] = field(default_factory=list)
