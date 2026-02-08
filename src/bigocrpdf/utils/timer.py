"""
BigOcrPdf - Timer Utilities

This module provides utility functions for handling GLib timers safely.
Centralizes timer management to prevent memory leaks and crashes.
"""

from gi import require_version

require_version("Gtk", "4.0")
from gi.repository import GLib


def safe_remove_source(source_id: int | None) -> bool:
    """Safely remove a GLib source timer without generating warnings.

    Args:
        source_id: The GLib source ID to remove

    Returns:
        True if the source was removed, False otherwise
    """
    if source_id is None or source_id <= 0:
        return False

    try:
        # Check if it's a valid source ID
        context = GLib.MainContext.default()
        if context.find_source_by_id(source_id):
            return GLib.source_remove(source_id)
        return False
    except Exception:
        # Catch any exceptions to prevent app crashes
        return False


class TimerManager:
    """Centralized manager for GLib timers.

    This class tracks all timers created by the application and ensures
    they are properly cleaned up when no longer needed.
    """

    def __init__(self) -> None:
        """Initialize the timer manager."""
        self._timers: dict[str, int] = {}

    def remove_timer(self, name: str) -> bool:
        """Remove a specific timer by name.

        Args:
            name: The timer name to remove

        Returns:
            True if timer was removed, False otherwise
        """
        timer_id = self._timers.pop(name, None)

        if timer_id is not None:
            return safe_remove_source(timer_id)

        return False

    def remove_all(self) -> int:
        """Remove all tracked timers.

        Returns:
            Number of timers removed
        """
        count = 0

        for name in tuple(self._timers.keys()):
            if self.remove_timer(name):
                count += 1

        self._timers.clear()
        return count

    def __del__(self) -> None:
        """Cleanup on destruction."""
        self.remove_all()
