# Bundled icon theme

Every symbolic icon the interface references is shipped here as a `hicolor`
theme, added to the icon search path at startup by
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

## How

Every icon theme's lookup chain ends at `hicolor`, so these act as the last
resort: a name the user's theme provides comes from their theme, and a name it
lacks comes from here. Measured on a host running `bigicons-papient`, the four
names above resolve to nothing without this directory and resolve with it,
while `folder-open-symbolic` and `document-send-symbolic` keep coming from the
host theme.

The bundle used to be a private theme selected through `gtk-icon-theme-name`.
That did make the interface identical on every system, but it did so by
discarding the user's choice: the chain became `bigocrpdf → Adwaita → hicolor`
and a Papirus or Breeze user never saw one of their own icons.

## Adding an icon

Drop the SVG in `hicolor/scalable/actions/`, named exactly as the string
passed to `set_icon_name()`. `tests/test_icons.py` fails if any name used in the
source tree has no bundled file.

## Sources and licensing

35 icons come from the **Adwaita** icon theme (GNOME) — CC0-1.0 / CC-BY-SA-3.0.

5 icons come from the **Breeze** icon theme (KDE) — LGPL-3.0-or-later:
`clock-symbolic`, `document-multiple-symbolic`, `emblem-ok-symbolic`,
`emblem-synchronizing-symbolic`, `format-text-uppercase-symbolic`.

Both are compatible with this project's GPL-3.0-or-later license.
