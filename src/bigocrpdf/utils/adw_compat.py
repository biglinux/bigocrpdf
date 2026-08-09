"""Widgets that need a fallback on older libadwaita.

The application targets current libadwaita but must also run on the stack it is
bundled against in AppImage builds, which is whatever the build container ships
-- Ubuntu 24.04 carries libadwaita 1.5, so anything newer is simply absent from
the introspection data.

Availability is probed with ``hasattr`` rather than
``Adw.get_minor_version()``: a symbol can be missing because the library is old
*or* because its typelib was bundled incompletely, and ``hasattr`` catches both.
Probes run once at import, so the cost is paid a single time.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gtk

# Adw.Spinner landed in libadwaita 1.6, Adw.WrapBox in 1.7, and the
# Adw.ShortcutsDialog family in 1.8 (GNOME 49).
HAS_ADW_SPINNER = hasattr(Adw, "Spinner")
HAS_ADW_WRAP_BOX = hasattr(Adw, "WrapBox")
HAS_ADW_SHORTCUTS_DIALOG = (
    hasattr(Adw, "ShortcutsDialog")
    and hasattr(Adw, "ShortcutsSection")
    and hasattr(Adw, "ShortcutsItem")
)

# Adw.ViewStack itself is old, but it only learned to animate page changes in
# libadwaita 1.7, so the methods have to be probed rather than the class.
HAS_VIEW_STACK_TRANSITIONS = hasattr(Adw.ViewStack, "set_enable_transitions")


def enable_view_stack_transitions(stack: Adw.ViewStack, duration_ms: int) -> bool:
    """Animate page changes on ``stack`` where libadwaita supports it.

    Purely decorative: on older stacks pages simply switch instantly, which is
    how Adw.ViewStack behaved before the feature existed.  Returns whether the
    animation was enabled.
    """
    if not HAS_VIEW_STACK_TRANSITIONS:
        return False

    stack.set_enable_transitions(True)
    stack.set_transition_duration(duration_ms)
    return True


def create_spinner(size: int = 40) -> Gtk.Widget:
    """A spinning progress indicator, sized to a square of ``size`` pixels."""
    if HAS_ADW_SPINNER:
        return Adw.Spinner(
            width_request=size,
            height_request=size,
            halign=Gtk.Align.CENTER,
        )

    # Gtk.Spinner has to be started explicitly; Adw.Spinner animates on its own.
    spinner = Gtk.Spinner(
        width_request=size,
        height_request=size,
        halign=Gtk.Align.CENTER,
    )
    spinner.start()
    return spinner


def create_wrap_box(
    *,
    child_spacing: int = 0,
    line_spacing: int = 0,
    **kwargs,
) -> Gtk.Widget:
    """A horizontal container whose children flow onto a new line when needed.

    The fallback is a ``Gtk.FlowBox`` in non-selectable, non-homogeneous mode,
    which wraps the same way.  Both accept children through ``append()``.
    """
    if HAS_ADW_WRAP_BOX:
        return Adw.WrapBox(
            orientation=Gtk.Orientation.HORIZONTAL,
            child_spacing=child_spacing,
            line_spacing=line_spacing,
            **kwargs,
        )

    flow_box = Gtk.FlowBox(
        orientation=Gtk.Orientation.HORIZONTAL,
        column_spacing=child_spacing,
        row_spacing=line_spacing,
        selection_mode=Gtk.SelectionMode.NONE,
        homogeneous=False,
        max_children_per_line=30,
        **kwargs,
    )
    # Gtk.FlowBox only exposes insert(); give callers the WrapBox spelling so
    # the call sites stay identical on both paths.
    flow_box.append = lambda child: flow_box.insert(child, -1)  # type: ignore[method-assign]
    return flow_box


def build_shortcuts_dialog(groups) -> Adw.Dialog:
    """Build the keyboard shortcuts dialog from ``(title, ((label, accel), ...))``.

    On libadwaita 1.8+ this is the native ``Adw.ShortcutsDialog``.  Older stacks
    get an ``Adw.PreferencesDialog`` whose rows carry a ``Gtk.ShortcutLabel``,
    which renders accelerators the same way and is not deprecated.
    """
    if HAS_ADW_SHORTCUTS_DIALOG:
        dialog = Adw.ShortcutsDialog()
        for title, shortcuts in groups:
            section = Adw.ShortcutsSection(title=title)
            for shortcut_title, accelerator in shortcuts:
                section.add(Adw.ShortcutsItem.new(shortcut_title, accelerator))
            dialog.add(section)
        return dialog

    dialog = Adw.PreferencesDialog()
    dialog.set_title(_shortcuts_dialog_title())
    page = Adw.PreferencesPage()
    for title, shortcuts in groups:
        group = Adw.PreferencesGroup(title=title)
        for shortcut_title, accelerator in shortcuts:
            row = Adw.ActionRow(title=shortcut_title)
            row.add_suffix(Gtk.ShortcutLabel(accelerator=accelerator, valign=Gtk.Align.CENTER))
            group.add(row)
        page.add(group)
    dialog.add(page)
    return dialog


def _shortcuts_dialog_title() -> str:
    from bigocrpdf.utils.i18n import _

    return _("Keyboard Shortcuts")
