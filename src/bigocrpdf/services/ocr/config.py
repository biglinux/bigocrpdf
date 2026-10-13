"""OCRConfig, its defaults, and the OCR result types shared by the pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Shared by OCRConfig and OcrSettings to avoid default drift.
DEFAULT_DPI = 300
DEFAULT_BOX_THRESH = 0.5
# Each step of box expansion adds margin above and below the text, and the
# PP-OCRv6 recognizer reads long lines over textured paper as blanks once the
# crop carries too much of it.
DEFAULT_UNCLIP_RATIO = 1.5
DEFAULT_DETECTION_LIMIT_SIDE_LEN = 4096
DEFAULT_TEXT_SCORE_THRESHOLD = 0.3
DEFAULT_PDF_MODE = "auto"
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
# Detection resolution. False lets the detector take the page at most 2000
# pixels long, which downscales a 300 dpi page; True sends it as rendered. Off
# by default: it about doubles each worker's peak memory for a small accuracy
# gain, and memory already caps the worker count. `--full-resolution` and the
# Full Resolution Detection switch turn it on.
DEFAULT_DETECTION_FULL_RESOLUTION = False
# Execution
DEFAULT_WORKERS = 0
DEFAULT_REPLACE_EXISTING_OCR = False
DEFAULT_ENHANCE_EMBEDDED_IMAGES = False


def _installed_font(variable: str, relative: str) -> Path:
    """``variable`` names the file for prefixes such as Nix; an AppImage uses
    its own copy, never the host's."""
    if override := os.environ.get(variable):
        return Path(override)
    return Path(os.environ.get("APPDIR") or "/") / relative


def text_layer_font() -> Path:
    """Noto Sans (Arch: noto-fonts), which covers the Latin, Greek and Cyrillic
    text the recogniser returns."""
    return _installed_font("BIGOCRPDF_TEXT_FONT", "usr/share/fonts/noto/NotoSans-Regular.ttf")


def cjk_text_layer_font() -> Path:
    """WenQuanYi Micro Hei (Arch: wqy-microhei), for lines with Chinese or
    Japanese characters Noto Sans lacks. Optional: without it those characters
    stay out of the text layer."""
    return _installed_font(
        "BIGOCRPDF_CJK_FONT", "usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc"
    )


@dataclass
class OCRConfig:
    """Configuration of one OCR job.

    Attributes:
        dpi: Resolution for image extraction
        box_thresh: Detection threshold for text boxes
        unclip_ratio: Box expansion ratio
        detection_limit_side_len: Longest side the page is resized to before
            detection and crop extraction. The detector itself sees at most
            2000 pixels unless ``detection_full_resolution`` is set
        font_path: TrueType font the text layer embeds; ``None`` writes the
            built-in Helvetica, which only encodes Latin-1
        cjk_font_path: Font for lines ``font_path`` cannot encode, used when
            the file exists
        enable_deskew: Correct skewed documents
        enable_orientation_detection: Detect and fix page rotation
        enable_preprocessing: Master switch for color enhancements
        enable_auto_contrast: CLAHE for low-contrast images
        enable_denoise: Apply denoising
        convert_to_pdfa: Convert output to PDF/A-2b
        workers: Number of parallel workers
    """

    dpi: int = DEFAULT_DPI

    box_thresh: float = DEFAULT_BOX_THRESH
    unclip_ratio: float = DEFAULT_UNCLIP_RATIO
    detection_limit_side_len: int = DEFAULT_DETECTION_LIMIT_SIDE_LEN
    detection_full_resolution: bool = DEFAULT_DETECTION_FULL_RESOLUTION
    text_score_threshold: float = DEFAULT_TEXT_SCORE_THRESHOLD

    pdf_mode: Literal["ocr", "geometric", "auto", "auto_verified"] = DEFAULT_PDF_MODE

    fallback_render_dpi: int = DEFAULT_FALLBACK_RENDER_DPI
    retry_render_dpi: int = DEFAULT_RETRY_RENDER_DPI
    max_render_megapixels: int = DEFAULT_MAX_RENDER_MEGAPIXELS
    max_image_megapixels: float = DEFAULT_MAX_IMAGE_MEGAPIXELS
    max_pdf_pages: int = DEFAULT_MAX_PDF_PAGES

    font_path: Path | None = field(default_factory=text_layer_font)
    cjk_font_path: Path = field(default_factory=cjk_text_layer_font)

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

    convert_to_pdfa: bool = DEFAULT_CONVERT_TO_PDFA
    max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB
    page_layout: str = DEFAULT_PAGE_LAYOUT
    enable_bilevel_compression: bool = DEFAULT_ENABLE_BILEVEL_COMPRESSION
    force_bilevel_compression: bool = DEFAULT_FORCE_BILEVEL_COMPRESSION

    image_export_format: str = DEFAULT_IMAGE_EXPORT_FORMAT
    image_export_quality: int = DEFAULT_IMAGE_EXPORT_QUALITY
    auto_detect_quality: bool = DEFAULT_AUTO_DETECT_QUALITY

    workers: int = DEFAULT_WORKERS
    page_range: tuple[int, int] | None = None
    page_modifications: list[dict] | None = None
    force_full_ocr: bool = False
    replace_existing_ocr: bool = DEFAULT_REPLACE_EXISTING_OCR
    enhance_embedded_images: bool = DEFAULT_ENHANCE_EMBEDDED_IMAGES


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
    # Pages left without text because OCR failed on them, with the cause.
    failed_pages: dict[int, str] = field(default_factory=dict)
    full_text: str = ""
    ocr_boxes: list[OCRBoxData] = field(default_factory=list)
    ocr_document: OcrDocument = field(default_factory=OcrDocument)
    error: str | None = None
    # Split output files (when max_file_size_mb is set and exceeded)
    split_output_files: list[str] = field(default_factory=list)
