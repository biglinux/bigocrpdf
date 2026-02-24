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
