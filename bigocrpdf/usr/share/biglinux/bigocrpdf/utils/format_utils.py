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


def format_time_mmss(seconds: int) -> str:
    """Format time as MM:SS.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "02:30")
    """
    if seconds < 0:
        seconds = 0
    
    minutes = int(seconds / 60)
    remaining_seconds = int(seconds % 60)
    
    return f"{minutes:02d}:{remaining_seconds:02d}"


def format_progress_percentage(fraction: float) -> str:
    """Format a fraction as a percentage string.
    
    Args:
        fraction: Value between 0.0 and 1.0
        
    Returns:
        Formatted percentage string (e.g., "75%")
    """
    percentage = max(0, min(100, int(fraction * 100)))
    return f"{percentage}%"


def format_count_with_label(count: int, singular: str, plural: str = None) -> str:
    """Format a count with appropriate singular/plural label.
    
    Args:
        count: The count value
        singular: Label for count of 1
        plural: Label for count != 1 (if None, adds 's' to singular)
        
    Returns:
        Formatted string (e.g., "1 file" or "3 files")
    """
    if plural is None:
        plural = singular + "s"
    
    label = singular if count == 1 else plural
    return f"{count} {label}"


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate a string to maximum length with suffix.
    
    Args:
        text: String to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add when truncated (default: "...")
        
    Returns:
        Truncated string if needed, original string otherwise
    """
    if not text or len(text) <= max_length:
        return text
    
    if max_length <= len(suffix):
        return suffix[:max_length]
    
    return text[:max_length - len(suffix)] + suffix


def format_file_list(file_paths: list, max_items: int = 3) -> str:
    """Format a list of file paths for display.
    
    Args:
        file_paths: List of file paths
        max_items: Maximum items to show before truncating
        
    Returns:
        Formatted string showing files
    """
    import os
    
    if not file_paths:
        return "No files"
    
    count = len(file_paths)
    
    if count <= max_items:
        names = [os.path.basename(p) for p in file_paths]
        return ", ".join(names)
    
    # Show first max_items and count remaining
    names = [os.path.basename(p) for p in file_paths[:max_items]]
    remaining = count - max_items
    
    return f"{', '.join(names)} and {remaining} more"
