"""
BigOcrPdf - Timer Utilities

This module provides utility functions for handling GLib timers safely.
Centralizes timer management to prevent memory leaks and crashes.
"""

from collections.abc import Callable

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

    def add_timeout(
        self,
        name: str,
        interval_ms: int,
        callback: Callable,
        *args,
        replace: bool = True,
    ) -> int | None:
        """Add a timeout timer.

        Args:
            name: Unique identifier for this timer
            interval_ms: Interval in milliseconds
            callback: Function to call when timer fires
            *args: Additional arguments for the callback
            replace: If True, remove existing timer with same name

        Returns:
            The timer source ID, or None if failed
        """
        if replace and name in self._timers:
            self.remove_timer(name)

        try:
            if args:
                timer_id = GLib.timeout_add(interval_ms, callback, *args)
            else:
                timer_id = GLib.timeout_add(interval_ms, callback)

            self._timers[name] = timer_id
            return timer_id

        except Exception:
            return None

    def add_idle(self, name: str, callback: Callable, *args, replace: bool = True) -> int | None:
        """Add an idle callback.

        Args:
            name: Unique identifier for this timer
            callback: Function to call when idle
            *args: Additional arguments for the callback
            replace: If True, remove existing timer with same name

        Returns:
            The timer source ID, or None if failed
        """
        if replace and name in self._timers:
            self.remove_timer(name)

        try:
            if args:
                timer_id = GLib.idle_add(callback, *args)
            else:
                timer_id = GLib.idle_add(callback)

            self._timers[name] = timer_id
            return timer_id

        except Exception:
            return None

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

    def has_timer(self, name: str) -> bool:
        """Check if a timer exists.

        Args:
            name: The timer name to check

        Returns:
            True if timer exists, False otherwise
        """
        return name in self._timers

    def get_timer_id(self, name: str) -> int | None:
        """Get the source ID for a timer.

        Args:
            name: The timer name

        Returns:
            The source ID, or None if timer doesn't exist
        """
        return self._timers.get(name)

    def get_timer_count(self) -> int:
        """Get the number of active timers.

        Returns:
            Number of tracked timers
        """
        return len(self._timers)

    def __del__(self) -> None:
        """Cleanup on destruction."""
        self.remove_all()
