"""
Minimal python-bidi shim using the system fribidi C library via ctypes.

Provides bidi.algorithm.get_display() for RapidOCR Arabic text
reordering without requiring the python-bidi pip package.
"""
