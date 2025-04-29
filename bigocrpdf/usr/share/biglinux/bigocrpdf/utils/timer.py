"""
BigOcrPdf - Timer Utilities

This module provides utility functions for handling GLib timers.
"""

from gi import require_version

require_version("Gtk", "4.0")
from gi.repository import GLib


def safe_remove_source(source_id):
    """Safely remove a GLib source timer without generating warnings

    Args:
        source_id: The GLib source ID to remove

    Returns:
        bool: True if the source was removed, False otherwise
    """
    if source_id and source_id > 0:
        try:
            # Only remove if it's a valid source ID
            if GLib.MainContext.default().find_source_by_id(source_id):
                return GLib.source_remove(source_id)
            return False
        except Exception:
            # Catch any exceptions to prevent app crashes
            return False
    return False
