"""Numeric constants. This module imports nothing from the package, so any
module can use it without an import cycle; strings and paths live in config.py.
"""

from typing import Final

FONT_SIZE_SCALE_FACTOR: Final[float] = 0.85
MIN_FONT_SIZE: Final[float] = 4.0
MAX_FONT_SIZE: Final[float] = 72.0


MIN_IMAGE_DIMENSION_PX: Final[int] = 64


RESOURCE_TIER_CONSTRAINED_GB: Final[float] = 2.0
RESOURCE_TIER_MODERATE_GB: Final[float] = 6.0
OCR_SUBPROCESS_OVERHEAD_MB: Final[int] = 400
BASE_PROCESS_OVERHEAD_MB: Final[int] = 150
PER_WORKER_COST_MB: Final[int] = 900
# One OCR subprocess reads preprocessed pages one at a time, so beyond this
# many workers the extra pages only wait in memory.
MAX_AUTO_PREPROCESS_WORKERS: Final[int] = 2


DEFAULT_DPI: Final[int] = 300


MIN_TEXT_BOX_WIDTH_PX: Final[int] = 50
MIN_TEXT_BOX_HEIGHT_PX: Final[int] = 30
MIN_IMAGE_BOX_SIZE_PX: Final[int] = 50

# Subprocess timeouts, in seconds.

JBIG2_ENCODER_TIMEOUT_SECS: Final[int] = 30
PDF_TOOL_TIMEOUT_SECS: Final[int] = 120
