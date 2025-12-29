"""
BigOcrPdf - OCR Options Builder Module

This module provides a builder pattern for constructing OCRmyPDF options.
"""

import os
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from services.settings import OcrSettings

from utils.logger import logger

# OCR compression format mapping
OCR_COMPRESSION_FORMATS = {
    "normal": "jpeg",
    "economic": "jpeg",
    "economicplus": "jpeg",
}

# OCR optimization level mapping
OCR_OPTIMIZATION_LEVELS = {
    "normal": 0,
    "economic": 1,
    "economicplus": 3,
}


class OcrOptionsBuilder:
    """Builder class for constructing OCRmyPDF options.
    
    This class provides a fluent interface for building OCR options
    dictionaries used by ocrmypdf.
    
    Usage:
        builder = OcrOptionsBuilder(settings)
        options = (builder
            .with_language()
            .with_quality()
            .with_alignment()
            .with_sidecar(output_file)
            .build())
    """

    def __init__(self, settings: "OcrSettings"):
        """Initialize the builder with settings.
        
        Args:
            settings: OcrSettings instance containing user preferences
        """
        self.settings = settings
        self._options: Dict[str, Any] = {
            "force_ocr": True,
            "progress_bar": False,
        }
        self._temp_dir: Optional[str] = None

    def with_language(self) -> "OcrOptionsBuilder":
        """Add language option from settings.
        
        Returns:
            Self for method chaining
        """
        if hasattr(self.settings, 'lang') and self.settings.lang:
            self._options["language"] = self.settings.lang
        return self

    def with_quality(self) -> "OcrOptionsBuilder":
        """Add quality-related options from settings.
        
        Returns:
            Self for method chaining
        """
        quality = getattr(self.settings, 'quality', 'normal')
        
        if quality in OCR_COMPRESSION_FORMATS:
            self._options["pdfa_image_compression"] = OCR_COMPRESSION_FORMATS[quality]
            self._options["optimize"] = OCR_OPTIMIZATION_LEVELS[quality]
            
            # Extra optimization for economicplus
            if quality == "economicplus":
                self._options["oversample"] = 300
        
        return self

    def with_alignment(self) -> "OcrOptionsBuilder":
        """Add alignment/rotation options from settings.
        
        Returns:
            Self for method chaining
        """
        align = getattr(self.settings, 'align', '')
        
        if align == "align":
            self._options["deskew"] = True
        elif align == "rotate":
            self._options["rotate_pages"] = True
        elif align == "alignrotate":
            self._options["deskew"] = True
            self._options["rotate_pages"] = True
        
        return self

    def with_sidecar(self, output_file: str) -> "OcrOptionsBuilder":
        """Add sidecar (text extraction) options.
        
        Args:
            output_file: Path to the output PDF file
            
        Returns:
            Self for method chaining
        """
        # Create temp directory for temporary sidecar
        self._temp_dir = os.path.join(os.path.dirname(output_file), ".temp")
        os.makedirs(self._temp_dir, exist_ok=True)
        
        # Default temp sidecar location
        base_name = os.path.splitext(os.path.basename(output_file))[0]
        temp_sidecar = os.path.join(self._temp_dir, f"temp_{base_name}.txt")
        self._options["sidecar"] = temp_sidecar
        
        return self

    def with_text_extraction(self, output_file: str) -> "OcrOptionsBuilder":
        """Add text extraction options if enabled in settings.
        
        Args:
            output_file: Path to the output PDF file
            
        Returns:
            Self for method chaining
        """
        if not getattr(self.settings, 'save_txt', False):
            return self
        
        # Determine sidecar file location
        use_separate_folder = (
            getattr(self.settings, 'separate_txt_folder', False) and
            getattr(self.settings, 'txt_folder', None)
        )
        
        if use_separate_folder:
            txt_folder = self.settings.txt_folder
            os.makedirs(txt_folder, exist_ok=True)
            txt_filename = os.path.splitext(os.path.basename(output_file))[0] + ".txt"
            sidecar_file = os.path.join(txt_folder, txt_filename)
        else:
            sidecar_file = os.path.splitext(output_file)[0] + ".txt"
        
        self._options["sidecar"] = sidecar_file
        
        return self

    def with_custom_option(self, key: str, value: Any) -> "OcrOptionsBuilder":
        """Add a custom option.
        
        Args:
            key: Option key
            value: Option value
            
        Returns:
            Self for method chaining
        """
        self._options[key] = value
        return self

    def without_option(self, key: str) -> "OcrOptionsBuilder":
        """Remove an option if it exists.
        
        Args:
            key: Option key to remove
            
        Returns:
            Self for method chaining
        """
        self._options.pop(key, None)
        return self

    def reset(self) -> "OcrOptionsBuilder":
        """Reset options to defaults.
        
        Returns:
            Self for method chaining
        """
        self._options = {
            "force_ocr": True,
            "progress_bar": False,
        }
        self._temp_dir = None
        return self

    def build(self) -> Dict[str, Any]:
        """Build and return the options dictionary.
        
        Returns:
            Dictionary of OCR options
        """
        return self._options.copy()

    def build_for_file(self, output_file: str) -> Dict[str, Any]:
        """Build complete options for processing a specific file.
        
        This is a convenience method that applies all standard options.
        
        Args:
            output_file: Path to the output PDF file
            
        Returns:
            Dictionary of OCR options
        """
        return (self
            .with_language()
            .with_quality()
            .with_alignment()
            .with_sidecar(output_file)
            .with_text_extraction(output_file)
            .build())

    def get_temp_dir(self) -> Optional[str]:
        """Get the temporary directory path if one was created.
        
        Returns:
            Path to temp directory or None
        """
        return self._temp_dir

    def __str__(self) -> str:
        """String representation of current options."""
        return f"OcrOptionsBuilder({self._options})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"OcrOptionsBuilder(settings={type(self.settings).__name__}, options={self._options})"
