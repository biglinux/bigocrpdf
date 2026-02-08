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
    language: str = "latin"
    dpi: int = 300

    # === Detection Thresholds ===
    box_thresh: float = 0.5
    unclip_ratio: float = 1.2
    detection_limit_side_len: int = 4096
    score_mode: str = "slow"
    text_score_threshold: float = 0.3  # Lower threshold to catch more text (reference: 0.3)

    # === Model Settings ===
    use_server_models: bool = False
    engine_type: str = "openvino"  # OpenVINO only (no ONNX fallback)

    # === Paths (BigLinux standard) ===
    model_base_path: Path = field(default_factory=lambda: DEFAULT_MODEL_PATH)
    font_base_path: Path = field(default_factory=lambda: DEFAULT_FONT_PATH)

    # === Preprocessing Options ===
    # Auto-detect: when ON, corrections only apply if detection indicates need
    # When OFF, all enabled corrections always apply
    enable_auto_detect: bool = True
    # Geometric corrections
    enable_perspective_correction: bool = False  # Perspective correction for photographed documents
    enable_deskew: bool = True  # Correct skewed documents
    enable_orientation_detection: bool = True  # Detect and fix page rotation
    # Color/Enhancement: OFF by default (PP-OCRv5 works best without)
    enable_preprocessing: bool = False  # Master switch for color enhancements
    enable_auto_contrast: bool = False  # CLAHE (only if preprocessing enabled)
    enable_auto_brightness: bool = False  # Brightness adjustment
    enable_denoise: bool = False
    enable_border_clean: bool = False
    enable_scanner_effect: bool = False
    scanner_effect_strength: float = 1.0
    enable_vintage_look: bool = False
    vintage_bw: bool = False

    # === Output Options ===
    convert_to_pdfa: bool = False
    max_file_size_mb: int = 0  # 0 = no limit; split output into parts if exceeded

    # === Image Export Options ===
    image_export_format: str = "original"  # original, jpeg, png, jp2
    image_export_quality: int = 85  # Image quality (1-100) for lossy formats
    auto_detect_quality: bool = True  # Auto-detect original image quality and format

    # === Execution Options ===
    workers: int = 0  # 0 = Auto (all CPU cores, low priority), 1 = Single process
    page_range: tuple[int, int] | None = None  # (start, end) 1-indexed, None for all pages
    page_modifications: list[dict] | None = None  # List of page states from editor
    force_full_ocr: bool = False  # Force image-only OCR mode (skip mixed content detection)
    replace_existing_ocr: bool = False  # Replace existing OCR text in PDFs that already have it
    render_full_pages: bool = (
        False  # Use pdftoppm for full-page rendering (mixed-content + replace OCR)
    )

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
