"""Elapsed-time formatting for the progress page."""


def format_elapsed_time(seconds: int) -> str:
    """Format elapsed time in human-readable format.

    Args:
        seconds: Elapsed time in seconds

    Returns:
        Formatted time string (e.g., "2m 30s" or "45s")
    """
    if isinstance(seconds, bool) or seconds < 0:
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
