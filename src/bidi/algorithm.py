"""
BiDi algorithm implementation using the system fribidi C library.

Provides get_display() compatible with python-bidi's API,
backed by the native fribidi library (libfribidi.so).
"""

from __future__ import annotations

import ctypes
import ctypes.util

# Locate and load the fribidi C library
_fribidi_lib_name = ctypes.util.find_library("fribidi")
if not _fribidi_lib_name:
    raise ImportError("fribidi C library not found. Install it with: sudo pacman -S fribidi")

_fribidi = ctypes.CDLL(_fribidi_lib_name)

# fribidi constants
FRIBIDI_PAR_ON = 0  # Auto-detect paragraph direction
FRIBIDI_FLAGS_DEFAULT = 0x001F  # Default reordering flags

# fribidi function signatures
_fribidi.fribidi_log2vis.argtypes = [
    ctypes.POINTER(ctypes.c_uint32),  # input string (FriBidiChar*)
    ctypes.c_int,  # length
    ctypes.POINTER(ctypes.c_int),  # paragraph base direction
    ctypes.POINTER(ctypes.c_uint32),  # output visual string
    ctypes.POINTER(ctypes.c_int),  # position L-to-V map (can be NULL)
    ctypes.POINTER(ctypes.c_int),  # position V-to-L map (can be NULL)
    ctypes.POINTER(ctypes.c_uint8),  # embedding levels (can be NULL)
]
_fribidi.fribidi_log2vis.restype = ctypes.c_int


def get_display(text: str | bytes, base_dir: str = "L") -> str | bytes:
    """Apply the Unicode BiDi algorithm for visual display ordering.

    This is a drop-in replacement for bidi.algorithm.get_display()
    from the python-bidi package, using the native fribidi C library.

    Args:
        text: Input text (logical order). Can be str or bytes (UTF-8).
        base_dir: Base paragraph direction ('L', 'R', or 'N' for auto).

    Returns:
        Text reordered for visual display, same type as input.
    """
    is_bytes = isinstance(text, bytes)
    if is_bytes:
        text_str = text.decode("utf-8", errors="replace")
    else:
        text_str = text

    if not text_str:
        return text

    length = len(text_str)

    # Convert Python string to array of Unicode codepoints (uint32)
    input_arr = (ctypes.c_uint32 * length)(*[ord(c) for c in text_str])
    output_arr = (ctypes.c_uint32 * length)()

    # Map base direction
    dir_map = {"L": 272, "R": 273, "N": 0}  # FRIBIDI_PAR_LTR/RTL/ON
    pbase_dir = ctypes.c_int(dir_map.get(base_dir, 0))

    # Run fribidi log2vis
    success = _fribidi.fribidi_log2vis(
        input_arr,
        length,
        ctypes.byref(pbase_dir),
        output_arr,
        None,  # position L-to-V map
        None,  # position V-to-L map
        None,  # embedding levels
    )

    if not success:
        return text  # fallback to original on failure

    # Convert back to Python string
    result = "".join(chr(output_arr[i]) for i in range(length))

    if is_bytes:
        return result.encode("utf-8")
    return result
