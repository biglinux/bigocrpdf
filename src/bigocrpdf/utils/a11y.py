"""Accessibility helpers for GTK4 widgets.

Provides a single ``set_a11y_label`` function that sets the accessible label
on any Gtk.Widget, keeping the verbose ``update_property`` boilerplate in
one place.
"""

from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402


def set_a11y_label(widget: Gtk.Widget, label: str) -> None:
    """Set the accessible label that screen readers (e.g. Orca) announce.

    Args:
        widget: Any GTK4 widget.
        label: Human-readable text for assistive technology.
    """
    widget.update_property([Gtk.AccessibleProperty.LABEL], [label])
