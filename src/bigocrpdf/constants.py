"""
BigOcrPdf - Numeric Constants

Simple numeric constants with ZERO internal imports to avoid circular dependencies.
For application-level constants (strings, paths), use config.py.
"""

from typing import Final

# ============================================================================
# Size Constants
# ============================================================================

BYTES_PER_MB: Final[int] = 1024 * 1024

# ============================================================================
# PDF Font Sizing
# ============================================================================

FONT_SIZE_SCALE_FACTOR: Final[float] = 0.85
MIN_FONT_SIZE: Final[float] = 4.0
MAX_FONT_SIZE: Final[float] = 72.0

# ============================================================================
# Image Dimension & Size Thresholds
# ============================================================================

MIN_IMAGE_DIMENSION_PX: Final[int] = 64
MIN_IMAGE_FILE_SIZE_BYTES: Final[int] = 2048

# ============================================================================
# Resource Management (Pipeline)
# ============================================================================

RESOURCE_TIER_CONSTRAINED_GB: Final[float] = 2.0
RESOURCE_TIER_MODERATE_GB: Final[float] = 6.0
OCR_SUBPROCESS_OVERHEAD_MB: Final[int] = 400
BASE_PROCESS_OVERHEAD_MB: Final[int] = 150
PER_WORKER_COST_MB: Final[int] = 200

# ============================================================================
# Processing Defaults
# ============================================================================

DEFAULT_DPI: Final[int] = 300
DEFAULT_PDF_RESOLUTION: Final[int] = 150
DEFAULT_JPEG_QUALITY: Final[int] = 85

# ============================================================================
# Text Layer Rendering Thresholds
# ============================================================================

MIN_TEXT_BOX_WIDTH_PX: Final[int] = 50
MIN_TEXT_BOX_HEIGHT_PX: Final[int] = 30
MIN_IMAGE_BOX_SIZE_PX: Final[int] = 50

# ============================================================================
# Subprocess Timeouts (seconds)
# ============================================================================

JBIG2_ENCODER_TIMEOUT_SECS: Final[int] = 30
