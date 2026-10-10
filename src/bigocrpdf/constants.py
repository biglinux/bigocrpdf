"""Numeric constants. This module imports nothing from the package, so any
module can use it without an import cycle; strings and paths live in config.py.
"""

from typing import Final

BYTES_PER_MB: Final[int] = 1024 * 1024


FONT_SIZE_SCALE_FACTOR: Final[float] = 0.85
MIN_FONT_SIZE: Final[float] = 4.0
MAX_FONT_SIZE: Final[float] = 72.0


MIN_IMAGE_DIMENSION_PX: Final[int] = 64
MIN_IMAGE_FILE_SIZE_BYTES: Final[int] = 2048


RESOURCE_TIER_CONSTRAINED_GB: Final[float] = 2.0
RESOURCE_TIER_MODERATE_GB: Final[float] = 6.0
OCR_SUBPROCESS_OVERHEAD_MB: Final[int] = 400
BASE_PROCESS_OVERHEAD_MB: Final[int] = 150
PER_WORKER_COST_MB: Final[int] = 200


DEFAULT_DPI: Final[int] = 300
DEFAULT_PDF_RESOLUTION: Final[int] = 150
DEFAULT_JPEG_QUALITY: Final[int] = 85


MIN_TEXT_BOX_WIDTH_PX: Final[int] = 50
MIN_TEXT_BOX_HEIGHT_PX: Final[int] = 30
MIN_IMAGE_BOX_SIZE_PX: Final[int] = 50

# Subprocess timeouts, in seconds.

JBIG2_ENCODER_TIMEOUT_SECS: Final[int] = 30
PDF_TOOL_TIMEOUT_SECS: Final[int] = 120
