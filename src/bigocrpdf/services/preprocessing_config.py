"""OCR engine and image preprocessing configuration."""

from __future__ import annotations

from bigocrpdf.services.rapidocr_service.config import (
    DEFAULT_BOX_THRESH,
    DEFAULT_DPI,
    DEFAULT_ENABLE_AUTO_BRIGHTNESS,
    DEFAULT_ENABLE_AUTO_CONTRAST,
    DEFAULT_ENABLE_BASELINE_DEWARP,
    DEFAULT_ENABLE_BORDER_CLEAN,
    DEFAULT_ENABLE_DENOISE,
    DEFAULT_ENABLE_DESKEW,
    DEFAULT_ENABLE_ORIENTATION_DETECTION,
    DEFAULT_ENABLE_PERSPECTIVE_CORRECTION,
    DEFAULT_ENABLE_PREPROCESSING,
    DEFAULT_ENABLE_SCANNER_EFFECT,
    DEFAULT_ENABLE_VINTAGE_LOOK,
    DEFAULT_LANGUAGE,
    DEFAULT_SCANNER_EFFECT_STRENGTH,
    DEFAULT_TEXT_SCORE_THRESHOLD,
    DEFAULT_UNCLIP_RATIO,
    DEFAULT_VINTAGE_BW,
    DEFAULT_WORKERS,
)


class PreprocessingConfig:
    """OCR engine and image preprocessing configuration."""

    def __init__(self) -> None:
        self.dpi: int = DEFAULT_DPI
        self.ocr_language: str = DEFAULT_LANGUAGE
        # Master switch (color/enhancement features)
        self.enable_preprocessing: bool = DEFAULT_ENABLE_PREPROCESSING
        # Geometric corrections
        self.enable_deskew: bool = DEFAULT_ENABLE_DESKEW
        self.enable_baseline_dewarp: bool = DEFAULT_ENABLE_BASELINE_DEWARP
        self.enable_perspective_correction: bool = DEFAULT_ENABLE_PERSPECTIVE_CORRECTION
        self.enable_orientation_detection: bool = DEFAULT_ENABLE_ORIENTATION_DETECTION
        # Enhancement (only if enable_preprocessing=True)
        self.enable_auto_contrast: bool = DEFAULT_ENABLE_AUTO_CONTRAST
        self.enable_auto_brightness: bool = DEFAULT_ENABLE_AUTO_BRIGHTNESS
        self.enable_denoise: bool = DEFAULT_ENABLE_DENOISE
        self.enable_scanner_effect: bool = DEFAULT_ENABLE_SCANNER_EFFECT
        self.scanner_effect_strength: float = DEFAULT_SCANNER_EFFECT_STRENGTH
        self.enable_border_clean: bool = DEFAULT_ENABLE_BORDER_CLEAN
        self.enable_vintage_look: bool = DEFAULT_ENABLE_VINTAGE_LOOK
        self.vintage_bw: bool = DEFAULT_VINTAGE_BW
        # OCR thresholds
        self.text_score_threshold: float = DEFAULT_TEXT_SCORE_THRESHOLD
        self.box_thresh: float = DEFAULT_BOX_THRESH
        self.unclip_ratio: float = DEFAULT_UNCLIP_RATIO
        self.ocr_profile: str = "balanced"
        # Behavior
        self.replace_existing_ocr: bool = False
        self.enhance_embedded_images: bool = False
        self.parallel_workers: int = DEFAULT_WORKERS
