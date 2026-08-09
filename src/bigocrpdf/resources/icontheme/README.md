# Bundled icon theme

Every symbolic icon the interface references is shipped here as a private icon
theme named `bigocrpdf`, registered at startup by
`src/bigocrpdf/utils/icons.py`.

## Why

Relying on the host icon theme makes the interface look different — and often
broken — from one system to the next. On a machine using `bigicons-papient`,
four of the names the application asks for (`clock-symbolic`,
`document-multiple-symbolic`, `format-text-uppercase-symbolic`,
`window-new-symbolic`) already resolved to `image-missing`. Inside an AppImage,
where the host theme is whatever the user happens to have installed, the problem
is worse.

Registering a theme rather than loading each SVG by path keeps every existing
`icon-name` call site working, including widgets that accept only a name:
`Adw.ButtonContent`, `Adw.StatusPage`, `Gio.MenuItem` and `Gio.Notification`.

`index.theme` declares `Inherits=Adwaita,hicolor`, so any name that is not
bundled still resolves through the host.

## Adding an icon

Drop the SVG in `bigocrpdf/scalable/actions/`, named exactly as the string
passed to `set_icon_name()`. `tests/test_icons.py` fails if any name used in the
source tree has no bundled file.

## Sources and licensing

35 icons come from the **Adwaita** icon theme (GNOME) — CC0-1.0 / CC-BY-SA-3.0.

5 icons come from the **Breeze** icon theme (KDE) — LGPL-3.0-or-later:
`clock-symbolic`, `document-multiple-symbolic`, `emblem-ok-symbolic`,
`emblem-synchronizing-symbolic`, `format-text-uppercase-symbolic`.

Both are compatible with this project's GPL-3.0-or-later license.
