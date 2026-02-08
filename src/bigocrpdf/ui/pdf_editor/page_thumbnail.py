"""
BigOcrPdf - PDF Page Thumbnail Widget

A GTK4 widget displaying a PDF page thumbnail with controls for
selection, OCR checkbox, rotation indicator, and deletion overlay.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gdk, GdkPixbuf, GObject, Gtk

from bigocrpdf.ui.pdf_editor.page_model import PageState
from bigocrpdf.ui.pdf_editor.thumbnail_renderer import get_thumbnail_renderer
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.tooltip_helper import get_tooltip_helper


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
        delete-clicked: Emitted when delete button is clicked
    """

    __gsignals__ = {
        "ocr-toggled": (GObject.SignalFlags.RUN_FIRST, None, (bool,)),
        "thumbnail-clicked": (GObject.SignalFlags.RUN_FIRST, None, ()),
        "rotate-left-clicked": (GObject.SignalFlags.RUN_FIRST, None, ()),
        "rotate-right-clicked": (GObject.SignalFlags.RUN_FIRST, None, ()),
        "delete-clicked": (GObject.SignalFlags.RUN_FIRST, None, ()),
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
        self._current_rotation = None

        # Calculate height for A4 aspect ratio
        self._height = int(size * 1.414)

        self.add_css_class("page-thumbnail")
        self.set_size_request(size + 16, self._height + 50)

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
        self._spinner.start()

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

        # Bottom info bar with controls
        info_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        info_box.set_halign(Gtk.Align.CENTER)
        info_box.set_margin_top(4)

        # OCR/Include checkbox
        self._ocr_check = Gtk.CheckButton()
        # Active means NOT deleted (Included)
        self._ocr_check.set_active(not self._page_state.deleted)
        get_tooltip_helper().add_tooltip(
            self._ocr_check, _("Include this page in the final document")
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
        get_tooltip_helper().add_tooltip(self._rotate_left_btn, _("Rotate this page to the left"))
        self._rotate_left_btn.connect("clicked", self._on_rotate_left_clicked)
        info_box.append(self._rotate_left_btn)

        # Rotate right button
        self._rotate_right_btn = Gtk.Button()
        self._rotate_right_btn.set_icon_name("object-rotate-right-symbolic")
        self._rotate_right_btn.add_css_class("flat")
        self._rotate_right_btn.add_css_class("circular")
        get_tooltip_helper().add_tooltip(self._rotate_right_btn, _("Rotate this page to the right"))
        self._rotate_right_btn.connect("clicked", self._on_rotate_right_clicked)
        info_box.append(self._rotate_right_btn)

        self.append(info_box)

        # Click gesture for selection
        click_gesture = Gtk.GestureClick()
        click_gesture.connect("pressed", self._on_clicked)
        self.add_controller(click_gesture)

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

    def _on_ocr_toggled(self, check: Gtk.CheckButton) -> None:
        """Handle Include checkbox toggle.

        Args:
            check: The checkbox widget
        """
        active = check.get_active()
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

    def _update_page_label(self) -> None:
        """Update the page number label text."""
        self._page_label.set_text(str(self._page_state.page_number))

    def _update_appearance(self) -> None:
        """Update widget appearance based on current state."""
        # Update rotation badge
        if self._page_state.rotation != 0:
            self._rotation_badge.set_text(f"↻{self._page_state.rotation}°")
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
        if self._thumbnail_loaded:
            return

        renderer = get_thumbnail_renderer()
        renderer.render_page_thumbnail_async(
            self._pdf_path,
            self._page_state.page_number - 1,  # Convert to 0-indexed
            self._on_thumbnail_loaded,
            self._size,
            self._page_state.rotation,
        )

    def _on_thumbnail_loaded(self, pixbuf: GdkPixbuf.Pixbuf | None) -> None:
        """Handle thumbnail rendering completion.

        Args:
            pixbuf: The rendered thumbnail pixbuf
        """
        self._thumbnail_loaded = True
        self._current_rotation = self._page_state.rotation
        self._spinner.stop()
        self._spinner.set_visible(False)

        if pixbuf is not None:
            texture = Gdk.Texture.new_for_pixbuf(pixbuf)
            self._image.set_paintable(texture)
            self._page_state.thumbnail_pixbuf = pixbuf
        else:
            self._image.set_resource(
                "/org/gtk/libgtk/icons/32x32/status/image-missing-symbolic.png"
            )

        self._image.set_visible(True)

        if self._current_rotation != self._page_state.rotation:
            self.reload_thumbnail()

    def reload_thumbnail(self) -> None:
        """Force reload of the thumbnail."""
        self._thumbnail_loaded = False
        self._spinner.set_visible(True)
        self._spinner.start()
        self._image.set_visible(False)

        renderer = get_thumbnail_renderer()
        renderer.clear_document_cache(self._pdf_path)

        self.load_thumbnail()

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

            texture = Gdk.Texture.new_for_pixbuf(new_pixbuf)
            self._image.set_paintable(texture)
            self._page_state.thumbnail_pixbuf = new_pixbuf
            self._current_rotation = degrees

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
        self._size = size
        self._height = int(size * 1.414)
        self._image_box.set_size_request(size, self._height)
        self._image.set_size_request(size, self._height)
        self.set_size_request(size + 16, self._height + 50)
        # Mark for reload but don't start it — caller handles batch reload
        self._thumbnail_loaded = False

    def update_from_state(self) -> None:
        self._update_appearance()
        self._update_page_label()

        if self._thumbnail_loaded and self._page_state.rotation != self._current_rotation:
            self.reload_thumbnail()
