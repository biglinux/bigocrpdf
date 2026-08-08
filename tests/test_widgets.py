"""Shared UI widget regressions."""

from unittest.mock import MagicMock, patch

from bigocrpdf.ui import widgets


def test_svg_illustration_is_accessibility_presentation() -> None:
    image = MagicMock()

    with (
        patch.object(widgets.os.path, "exists", return_value=False),
        patch.object(widgets.Gtk, "Image", return_value=image),
    ):
        result = widgets.load_svg_picture("missing.svg")

    assert result is image
    image.set_accessible_role.assert_called_once_with(widgets.Gtk.AccessibleRole.PRESENTATION)


def test_css_is_registered_only_once() -> None:
    provider = MagicMock()
    display = MagicMock()

    with (
        patch.object(widgets.os.path, "exists", return_value=True),
        patch.object(widgets.Gtk, "CssProvider", return_value=provider),
        patch.object(widgets.Gdk.Display, "get_default", return_value=display),
        patch.object(widgets.Gtk.StyleContext, "add_provider_for_display") as add_provider,
    ):
        widgets._css_provider = None
        assert widgets.load_css() is True
        assert widgets.load_css() is True

    add_provider.assert_called_once_with(
        display,
        provider,
        widgets.Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
    )
    widgets._css_provider = None
