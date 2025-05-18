"""
BigOcrPdf - Utils Package

This package contains utility functions and helpers used throughout the application.
"""

# Primeiro inicializar logger (pois i18n pode precisar dele)
from .logger import logger

# Depois inicializar i18n
from .i18n import _, setup_i18n

# Depois outros módulos
from .timer import safe_remove_source

# Exportar tudo
__all__ = ['logger', '_', 'setup_i18n', 'safe_remove_source']