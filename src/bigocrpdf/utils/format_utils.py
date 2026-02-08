"""
BigOcrPdf - Format Utilities Module

This module provides shared utility functions for formatting values.
Centralizes formatting logic to avoid code duplication.
"""


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    if size_bytes < 0:
        return "0 B"

    if size_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]

    size = float(size_bytes)
    unit_index = 0

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    # Format with appropriate precision
    if unit_index == 0:  # Bytes - no decimals
        return f"{int(size)} {units[unit_index]}"
    elif size >= 100:  # Large values - no decimals
        return f"{int(size)} {units[unit_index]}"
    elif size >= 10:  # Medium values - 1 decimal
        return f"{size:.1f} {units[unit_index]}"
    else:  # Small values - 2 decimals
        return f"{size:.2f} {units[unit_index]}"


def format_elapsed_time(seconds: int) -> str:
    """Format elapsed time in human-readable format.

    Args:
        seconds: Elapsed time in seconds

    Returns:
        Formatted time string (e.g., "2m 30s" or "45s")
    """
    if seconds < 0:
        seconds = 0

    if seconds < 60:
        return f"{seconds}s"

    minutes = seconds // 60
    remaining_seconds = seconds % 60

    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"

    hours = minutes // 60
    remaining_minutes = minutes % 60

    return f"{hours}h {remaining_minutes}m {remaining_seconds}s"
