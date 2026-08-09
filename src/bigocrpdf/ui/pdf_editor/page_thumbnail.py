"""
BigOcrPdf - PDF Page Thumbnail Widget

A GTK4 widget displaying a PDF page thumbnail with controls for
selection, OCR checkbox, rotation indicator, and deletion overlay.
"""

from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
gi.require_version("Gdk", "4.0")
from gi.repository import Gdk, GdkPixbuf, Gio, GLib, GObject, Gtk

from bigocrpdf.ui.pdf_editor.page_model import PageState
from bigocrpdf.ui.pdf_editor.page_thumbnail_export import PageThumbnailExporter
from bigocrpdf.ui.pdf_editor.thumbnail_renderer import ThumbnailRequest, get_thumbnail_renderer
from bigocrpdf.utils.i18n import _


class PageThumbnail(Gtk.Box):
    """Widget representing a single PDF page thumbnail.

    Displays:
    - Thumbnail image of the page
    - Page number label
    - OCR checkbox
    - Rotate and delete buttons
    - Rotation indicator (when rotated)
    - Deleted overlay (when marked for deletion)
    - Selection border (when selected)

    Signals:
        ocr-toggled: Emitted when OCR checkbox is toggled
        thumbnail-clicked: Emitted when the thumbnail is clicked
        rotate-left-clicked: Emitted when rotate left button is clicked
        rotate-right-clicked: Emitted when rotate right button is clicked
    """

    __gsignals__ = {
        "ocr-toggled": (GObject.SignalFlags.RUN_FIRST, None, (bool,)),
        "thumbnail-clicked": (GObject.SignalFlags.RUN_FIRST, None, ()),
        "rotate-left-clicked": (GObject.SignalFlags.RUN_FIRST, None, ()),
        "rotate-right-clicked": (GObject.SignalFlags.RUN_FIRST, None, ()),
        "flip-horizontal-clicked": (GObject.SignalFlags.RUN_FIRST, None, ()),
        "flip-vertical-clicked": (GObject.SignalFlags.RUN_FIRST, None, ()),
    }

    def __init__(
        self,
        page_state: PageState,
        pdf_path: str,
        size: int = 150,
    ) -> None:
        """Initialize the page thumbnail widget.

        Args:
            page_state: The page state data
            pdf_path: Path to the PDF file
            size: Thumbnail width in pixels
        """
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=4)

        self._page_state = page_state
        self._pdf_path = pdf_path
        self._size = size
        self._selected = False
        self._thumbnail_loaded = False
        self._thumbnail_loading = False
        self._thumbnail_generation = 0
        self._thumbnail_retry_count = 0
        self._thumbnail_request: ThumbnailRequest | None = None
        self._current_rotation: int | None = None
        self.on_before_mutate: Callable[[], None] | None = None
        self._grid_handler_ids: list[int] = []
        self._exporter = PageThumbnailExporter(self)

        # Calculate height for A4 aspect ratio
        self._height = int(size * 1.414)

        self.add_css_class("page-thumbnail")
        self.set_size_request(size + 16, self._height + 50)

        # Accessible name for screen readers
        self.update_property(
            [Gtk.AccessibleProperty.LABEL],
            [_("Page {}").format(page_state.page_number)],
        )

        # Set Grab cursor
        self.set_cursor(Gdk.Cursor.new_from_name("grab", None))

        self._setup_ui()
        self._update_appearance()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        # Main container with overlay for delete indicator
        self._overlay = Gtk.Overlay()
        self._overlay.set_halign(Gtk.Align.CENTER)

        # Frame for border/selection highlight
        self._frame = Gtk.Frame()
        self._frame.add_css_class("thumbnail-frame")
        self._frame.set_halign(Gtk.Align.CENTER)

        # Image container
        self._image_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self._image_box.set_size_request(self._size, self._height)
        self._image_box.add_css_class("thumbnail-image-box")

        # Placeholder/Loading spinner
        self._spinner = Gtk.Spinner()
        self._spinner.set_size_request(32, 32)
        self._spinner.set_halign(Gtk.Align.CENTER)
        self._spinner.set_valign(Gtk.Align.CENTER)
        self._spinner.set_visible(False)

        # Image widget
        self._image = Gtk.Picture()
        self._image.set_size_request(self._size, self._height)
        self._image.set_content_fit(Gtk.ContentFit.CONTAIN)
        self._image.set_visible(False)

        self._image_box.append(self._spinner)
        self._image_box.append(self._image)
        self._frame.set_child(self._image_box)

        # Rotation indicator overlay
        self._rotation_badge = Gtk.Label()
        self._rotation_badge.add_css_class("rotation-badge")
        self._rotation_badge.set_halign(Gtk.Align.END)
        self._rotation_badge.set_valign(Gtk.Align.START)
        self._rotation_badge.set_margin_top(4)
        self._rotation_badge.set_margin_end(4)
        self._rotation_badge.set_visible(False)

        # Excluded overlay
        self._deleted_overlay = Gtk.Box()
        self._deleted_overlay.add_css_class("excluded-overlay")
        self._deleted_overlay.set_halign(Gtk.Align.FILL)
        self._deleted_overlay.set_valign(Gtk.Align.FILL)
        self._deleted_overlay.set_visible(False)

        excluded_label = Gtk.Label(label=_("Excluded"))
        excluded_label.add_css_class("excluded-label")
        excluded_label.set_halign(Gtk.Align.CENTER)
        excluded_label.set_valign(Gtk.Align.CENTER)
        self._deleted_overlay.append(excluded_label)

        self._overlay.set_child(self._frame)
        self._overlay.add_overlay(self._rotation_badge)
        self._overlay.add_overlay(self._deleted_overlay)

        self.append(self._overlay)

        # Bottom info bar: checkbox + page number + rotate buttons
        info_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        info_box.set_halign(Gtk.Align.CENTER)
        info_box.set_margin_top(2)

        # OCR/Include checkbox
        self._ocr_check = Gtk.CheckButton()
        # Active means NOT deleted (Included)
        self._ocr_check.set_active(not self._page_state.deleted)
        self._ocr_check.set_tooltip_text(_("Include this page in the final document"))
        self._ocr_check.update_property(
            [Gtk.AccessibleProperty.LABEL],
            [_("Include page {} in the final document").format(self._page_state.page_number)],
        )
        self._ocr_check.connect("toggled", self._on_ocr_toggled)
        info_box.append(self._ocr_check)

        # Page number label
        self._page_label = Gtk.Label()
        self._page_label.add_css_class("page-number-label")
        self._update_page_label()
        info_box.append(self._page_label)

        # Rotate left button
        self._rotate_left_btn = Gtk.Button()
        self._rotate_left_btn.set_icon_name("object-rotate-left-symbolic")
        self._rotate_left_btn.add_css_class("flat")
        self._rotate_left_btn.add_css_class("circular")
        self._rotate_left_btn.set_tooltip_text(_("Rotate this page to the left"))
        self._rotate_left_btn.update_property(
            [Gtk.AccessibleProperty.LABEL],
            [_("Rotate page {} to the left").format(self._page_state.page_number)],
        )
        self._rotate_left_btn.connect("clicked", self._on_rotate_left_clicked)
        info_box.append(self._rotate_left_btn)

        # Rotate right button
        self._rotate_right_btn = Gtk.Button()
        self._rotate_right_btn.set_icon_name("object-rotate-right-symbolic")
        self._rotate_right_btn.add_css_class("flat")
        self._rotate_right_btn.add_css_class("circular")
        self._rotate_right_btn.set_tooltip_text(_("Rotate this page to the right"))
        self._rotate_right_btn.update_property(
            [Gtk.AccessibleProperty.LABEL],
            [_("Rotate page {} to the right").format(self._page_state.page_number)],
        )
        self._rotate_right_btn.connect("clicked", self._on_rotate_right_clicked)
        info_box.append(self._rotate_right_btn)

        # Flip (Mirror) button using a MenuButton
        self._flip_btn = Gtk.MenuButton()
        self._flip_btn.set_icon_name("object-flip-horizontal-symbolic")
        self._flip_btn.add_css_class("flat")
        self._flip_btn.add_css_class("circular")

        # Actions for flip menu inside the thumbnail (using simple callbacks since it's inside the widget)
        flip_menu = Gio.Menu()
        flip_menu.append(_("Horizontal Flip"), "thumbnail.flip-h")
        flip_menu.append(_("Vertical Flip"), "thumbnail.flip-v")
        self._flip_btn.set_menu_model(flip_menu)

        # Create action group for the menu button
        action_group = Gio.SimpleActionGroup()

        # Horizontal action
        flip_h_action = Gio.SimpleAction.new("flip-h", None)
        flip_h_action.connect("activate", self._on_flip_horizontal_clicked)
        action_group.add_action(flip_h_action)

        # Vertical action
        flip_v_action = Gio.SimpleAction.new("flip-v", None)
        flip_v_action.connect("activate", self._on_flip_vertical_clicked)
        action_group.add_action(flip_v_action)

        self.insert_action_group("thumbnail", action_group)

        self._flip_btn.set_tooltip_text(_("Mirror this page (Horizontal/Vertical)"))
        self._flip_btn.update_property(
            [Gtk.AccessibleProperty.LABEL],
            [_("Mirror page {}").format(self._page_state.page_number)],
        )
        info_box.append(self._flip_btn)

        self.append(info_box)

        # Click gesture for selection (released so drags can claim the sequence first)
        click_gesture = Gtk.GestureClick()
        click_gesture.connect("released", self._on_clicked)
        self.add_controller(click_gesture)

        # Right-click gesture for context menu
        right_click = Gtk.GestureClick()
        right_click.set_button(3)  # secondary button
        right_click.connect("released", self._on_right_clicked)
        self.add_controller(right_click)

        # Keyboard context menu (Shift+F10 / Menu key)
        key_ctrl = Gtk.EventControllerKey()
        key_ctrl.connect("key-pressed", self._on_key_for_context_menu)
        self.add_controller(key_ctrl)

        # Save page actions
        save_img_action = Gio.SimpleAction.new("save-image", None)
        save_img_action.connect("activate", lambda *_: self._save_page_as_image())
        action_group.add_action(save_img_action)

        save_pdf_action = Gio.SimpleAction.new("save-pdf", None)
        save_pdf_action.connect("activate", lambda *_: self._save_page_as_pdf())
        action_group.add_action(save_pdf_action)

        # Context menu model
        ctx_menu = Gio.Menu()
        ctx_menu.append(_("Save page as image"), "thumbnail.save-image")
        ctx_menu.append(_("Save page as PDF"), "thumbnail.save-pdf")
        self._ctx_popover = Gtk.PopoverMenu.new_from_model(ctx_menu)
        self._ctx_popover.set_parent(self._overlay)
        self._ctx_popover.set_has_arrow(True)

        # Drag source for reordering
        drag_source = Gtk.DragSource()
        drag_source.set_actions(Gdk.DragAction.MOVE)
        drag_source.connect("prepare", self._on_drag_prepare)
        drag_source.connect("drag-begin", self._on_drag_begin)
        self.add_controller(drag_source)

    def _on_drag_prepare(
        self, source: Gtk.DragSource, x: float, y: float
    ) -> Gdk.ContentProvider | None:
        """Prepare drag data.

        Args:
            source: The drag source
            x: X coordinate
            y: Y coordinate

        Returns:
            ContentProvider with the page position
        """
        # Store position as string value
        value = GObject.Value(GObject.TYPE_INT, self._page_state.position)
        return Gdk.ContentProvider.new_for_value(value)

    def _on_drag_begin(self, source: Gtk.DragSource, _drag: Gdk.Drag) -> None:
        """Handle drag start.

        Args:
            source: The drag source
            _drag: The drag object
        """
        # Create drag icon from the thumbnail
        if self._page_state.thumbnail_pixbuf:
            texture = Gdk.Texture.new_for_pixbuf(self._page_state.thumbnail_pixbuf)
            source.set_icon(texture, self._size // 2, self._height // 2)

    def _on_clicked(self, gesture: Gtk.GestureClick, _n_press: int, x: float, y: float) -> None:
        """Handle click on the thumbnail.

        Args:
            gesture: The gesture controller
            n_press: Number of presses
            x: X coordinate
            y: Y coordinate
        """
        self.emit("thumbnail-clicked")

    def _on_key_for_context_menu(
        self,
        _controller: Gtk.EventControllerKey,
        keyval: int,
        _keycode: int,
        state: Gdk.ModifierType,
    ) -> bool:
        """Show context menu via Shift+F10 or Menu key."""
        if keyval == Gdk.KEY_Menu or (
            keyval == Gdk.KEY_F10 and state & Gdk.ModifierType.SHIFT_MASK
        ):
            rect = Gdk.Rectangle()
            rect.x = self.get_width() // 2
            rect.y = self.get_height() // 2
            rect.width = 1
            rect.height = 1
            self._ctx_popover.set_pointing_to(rect)
            self._ctx_popover.popup()
            return True
        return False

    def _on_right_clicked(
        self, gesture: Gtk.GestureClick, _n_press: int, x: float, y: float
    ) -> None:
        """Show context menu on right-click."""
        rect = Gdk.Rectangle()
        rect.x = int(x)
        rect.y = int(y)
        rect.width = 1
        rect.height = 1
        self._ctx_popover.set_pointing_to(rect)
        self._ctx_popover.popup()

    def _on_ocr_toggled(self, check: Gtk.CheckButton) -> None:
        """Handle Include checkbox toggle.

        Args:
            check: The checkbox widget
        """
        active = check.get_active()
        if self._page_state.deleted == (not active) and self._page_state.included_for_ocr == active:
            return
        if self.on_before_mutate:
            self.on_before_mutate()
        # Active = Included = Not Deleted
        self._page_state.deleted = not active
        # Also sync OCR state (if kept, default to OCR enabled)
        self._page_state.included_for_ocr = active

        self.emit("ocr-toggled", active)
        self._update_appearance()

    def _on_rotate_left_clicked(self, button: Gtk.Button) -> None:
        """Handle rotate left button click."""
        self.emit("rotate-left-clicked")

    def _on_rotate_right_clicked(self, button: Gtk.Button) -> None:
        """Handle rotate right button click."""
        self.emit("rotate-right-clicked")

    def _on_flip_horizontal_clicked(self, _action, _param) -> None:
        """Handle flip horizontal click."""
        self.emit("flip-horizontal-clicked")

    def _on_flip_vertical_clicked(self, _action, _param) -> None:
        """Handle flip vertical click."""
        self.emit("flip-vertical-clicked")

    def _update_page_label(self) -> None:
        """Update the page number label text and accessible description."""
        page_num = self._page_state.page_number
        self._page_label.set_text(str(page_num))
        # Build accessible label with state for screen readers
        parts = [_("Page {n}").format(n=page_num)]
        if self._page_state.deleted:
            parts.append(_("Excluded"))
        if self._page_state.rotation:
            parts.append(_("Rotated {deg}°").format(deg=self._page_state.rotation))
        if self._page_state.flip_horizontal:
            parts.append(_("Horizontal Flip"))
        if self._page_state.flip_vertical:
            parts.append(_("Vertical Flip"))
        self.update_property(
            [Gtk.AccessibleProperty.LABEL],
            [" — ".join(parts)],
        )

    def _update_appearance(self) -> None:
        """Update widget appearance based on current state."""
        # Update rotation & flip badge
        badge_text = []
        if self._page_state.rotation != 0:
            badge_text.append(f"↻{self._page_state.rotation}°")
        if self._page_state.flip_horizontal:
            badge_text.append("↔")
        if self._page_state.flip_vertical:
            badge_text.append("↕")

        if badge_text:
            self._rotation_badge.set_text(" ".join(badge_text))
            self._rotation_badge.set_visible(True)
        else:
            self._rotation_badge.set_visible(False)

        # Update checkbox (checked if NOT deleted)
        self._ocr_check.set_active(not self._page_state.deleted)

        # Update selection border
        if self._selected:
            self._frame.add_css_class("selected")
        else:
            self._frame.remove_css_class("selected")

        # Dim excluded pages
        if self._page_state.deleted:
            self.add_css_class("excluded")
            self._deleted_overlay.set_visible(True)
        else:
            self.remove_css_class("excluded")
            self._deleted_overlay.set_visible(False)

    def load_thumbnail(self) -> None:
        """Load the thumbnail image asynchronously."""
        if self._thumbnail_loaded or self._thumbnail_loading:
            return

        self._thumbnail_loading = True
        self._thumbnail_generation += 1
        generation = self._thumbnail_generation
        requested_rotation = self._page_state.rotation
        self._spinner.set_visible(True)
        self._spinner.start()
        renderer = get_thumbnail_renderer()
        self._thumbnail_request = renderer.render_page_thumbnail_async(
            self._pdf_path,
            self._page_state.page_number - 1,  # Convert to 0-indexed
            lambda pixbuf: self._on_thumbnail_loaded(
                pixbuf,
                generation=generation,
                requested_rotation=requested_rotation,
            ),
            self._size,
            requested_rotation,
        )

    def _on_thumbnail_loaded(
        self,
        pixbuf: GdkPixbuf.Pixbuf | None,
        *,
        generation: int,
        requested_rotation: int | None = None,
    ) -> None:
        """Handle thumbnail rendering completion.

        Args:
            pixbuf: The rendered thumbnail pixbuf
            generation: Widget request generation that owns this callback
            requested_rotation: Rotation applied by the renderer
        """
        if generation != self._thumbnail_generation:
            return
        self._thumbnail_request = None
        self._thumbnail_loading = False
        self._spinner.stop()
        self._spinner.set_visible(False)

        if pixbuf is None:
            self._thumbnail_loaded = False
            self._current_rotation = None
            self._page_state.thumbnail_pixbuf = None
            self._image.set_paintable(None)
            self._image.set_visible(False)
            if getattr(self, "_thumbnail_retry_count", 0) < 2:
                self._thumbnail_retry_count = getattr(self, "_thumbnail_retry_count", 0) + 1
                GLib.timeout_add(50, self._retry_invalidated_thumbnail, generation)
            return

        self._thumbnail_retry_count = 0
        displayed_pixbuf = pixbuf
        if self._page_state.flip_horizontal:
            displayed_pixbuf = displayed_pixbuf.flip(True) or displayed_pixbuf
        if self._page_state.flip_vertical:
            displayed_pixbuf = displayed_pixbuf.flip(False) or displayed_pixbuf
        texture = Gdk.Texture.new_for_pixbuf(displayed_pixbuf)
        self._image.set_paintable(texture)
        self._page_state.thumbnail_pixbuf = displayed_pixbuf
        self._thumbnail_loaded = True
        self._current_rotation = requested_rotation
        self._image.set_visible(True)

        if self._current_rotation != self._page_state.rotation:
            self.reload_thumbnail()

    def _retry_invalidated_thumbnail(self, generation: int) -> bool:
        """Retry a render invalidated by an atomic source-file replacement."""
        if (
            generation == self._thumbnail_generation
            and not self._thumbnail_loaded
            and not self._thumbnail_loading
        ):
            self.load_thumbnail()
        return False

    def reload_thumbnail(self) -> None:
        """Force reload of the thumbnail."""
        self._discard_thumbnail(show_spinner=True)

        renderer = get_thumbnail_renderer()
        renderer.clear_page_cache(
            self._pdf_path,
            self._page_state.page_number - 1,
        )

        self.load_thumbnail()

    def unload_thumbnail(self) -> None:
        """Release off-screen thumbnail pixels and invalidate any widget callback."""
        self._discard_thumbnail(show_spinner=False)

    def _discard_thumbnail(self, *, show_spinner: bool) -> None:
        self._thumbnail_generation += 1
        if self._thumbnail_request is not None:
            self._thumbnail_request.cancel()
            self._thumbnail_request = None
        self._thumbnail_loading = False
        self._thumbnail_loaded = False
        self._thumbnail_retry_count = 0
        self._current_rotation = None
        self._page_state.thumbnail_pixbuf = None
        self._image.set_paintable(None)
        self._image.set_visible(False)
        self._spinner.stop()
        self._spinner.set_visible(show_spinner)
        if show_spinner:
            self._spinner.start()

    def rotate_thumbnail_in_place(self, degrees: int) -> None:
        """Rotate the existing thumbnail image in memory.

        Args:
            degrees: Rotation angle (90, 180, 270, or -90)
        """
        if self._page_state.thumbnail_pixbuf is None:
            self.reload_thumbnail()
            return

        try:
            pixbuf = self._page_state.thumbnail_pixbuf

            angle = degrees % 360
            if angle == 90:
                new_pixbuf = pixbuf.rotate_simple(GdkPixbuf.PixbufRotation.CLOCKWISE)
            elif angle == 180:
                new_pixbuf = pixbuf.rotate_simple(GdkPixbuf.PixbufRotation.UPSIDEDOWN)
            elif angle == 270:
                new_pixbuf = pixbuf.rotate_simple(GdkPixbuf.PixbufRotation.COUNTERCLOCKWISE)
            else:
                return

            if new_pixbuf is None:
                self.reload_thumbnail()
                return
            texture = Gdk.Texture.new_for_pixbuf(new_pixbuf)
            self._image.set_paintable(texture)
            self._page_state.thumbnail_pixbuf = new_pixbuf
            self._current_rotation = self._page_state.rotation

        except Exception:
            self.reload_thumbnail()

    def flip_thumbnail_in_place(self, horizontal: bool = True) -> None:
        """Flip the existing thumbnail image in memory.

        Args:
            horizontal: True for horizontal flip, False for vertical flip
        """
        if self._page_state.thumbnail_pixbuf is None:
            self.reload_thumbnail()
            return

        try:
            pixbuf = self._page_state.thumbnail_pixbuf
            new_pixbuf = pixbuf.flip(horizontal)

            if new_pixbuf is None:
                self.reload_thumbnail()
                return
            texture = Gdk.Texture.new_for_pixbuf(new_pixbuf)
            self._image.set_paintable(texture)
            self._page_state.thumbnail_pixbuf = new_pixbuf
            self._current_rotation = self._page_state.rotation

        except Exception:
            self.reload_thumbnail()

    @property
    def page_state(self) -> PageState:
        return self._page_state

    @property
    def selected(self) -> bool:
        return self._selected

    @selected.setter
    def selected(self, value: bool) -> None:
        if self._selected != value:
            self._selected = value
            self._update_appearance()

    def set_size(self, size: int) -> None:
        if size == self._size:
            return
        self._size = size
        self._height = int(size * 1.414)
        self._image_box.set_size_request(size, self._height)
        self._image.set_size_request(size, self._height)
        self.set_size_request(size + 16, self._height + 50)
        self.reload_thumbnail()

    def resize_without_reload(self, size: int) -> None:
        """Resize the thumbnail widget without triggering a reload.

        Used during batch operations like zoom changes where the
        cache is cleared once externally.
        """
        self._discard_thumbnail(show_spinner=False)
        self._size = size
        self._height = int(size * 1.414)
        self._image_box.set_size_request(size, self._height)
        self._image.set_size_request(size, self._height)
        self.set_size_request(size + 16, self._height + 50)
        # The caller schedules a bounded reload after layout settles.

    def update_from_state(self) -> None:
        self._update_appearance()
        self._update_page_label()

        if self._thumbnail_loaded and self._page_state.rotation != self._current_rotation:
            self.reload_thumbnail()

    @property
    def pdf_path(self) -> str:
        return self._pdf_path

    def _save_page_as_image(self) -> None:
        self._exporter._save_page_as_image()

    def _save_page_as_pdf(self) -> None:
        self._exporter._save_page_as_pdf()
