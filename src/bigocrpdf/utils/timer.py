"""Safe removal of GLib sources owned by application components."""

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
        source = GLib.MainContext.default().find_source_by_id(source_id)
    except (OverflowError, TypeError, ValueError):
        return False

    if source is None:
        return False

    source.destroy()
    return True
