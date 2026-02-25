"""
BigOcrPdf - PDF Page Grid Widget

A FlowBox-based widget for displaying and managing PDF page thumbnails
with multi-select support, zoom control, and enhanced drag-and-drop.
"""

from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gdk, GLib, GObject, Gtk

from bigocrpdf.ui.pdf_editor.page_model import PageState, PDFDocument
from bigocrpdf.ui.pdf_editor.page_thumbnail import PageThumbnail
from bigocrpdf.ui.pdf_editor.thumbnail_renderer import get_thumbnail_renderer
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class PageGrid(Gtk.ScrolledWindow):
    """Grid display of PDF page thumbnails.

    Features:
    - FlowBox layout that wraps based on window width
    - Multi-select with Ctrl+Click and Shift+Click and Rubberband
    - Zoom control (50% - 250%)
    - Lazy loading of visible thumbnails
    - Keyboard navigation
    - Enhanced Drag-and-Drop with visual feedback
    """

    __gsignals__ = {
        "selection-changed": (GObject.SignalFlags.RUN_FIRST, None, ()),
        "page-ocr-toggled": (GObject.SignalFlags.RUN_FIRST, None, (int, bool)),
    }

    DEFAULT_THUMBNAIL_SIZE = 150  # Base size at 100% zoom

    def __init__(self, document: PDFDocument | None = None) -> None:
        """Initialize the page grid.

        Args:
            document: Optional PDFDocument to display
        """
        super().__init__()

        self._document = document
        self._thumbnails: list[PageThumbnail] = []
        self._selected_indices: set[int] = set()
        self._last_selected_index: int | None = None
        self._zoom_level = 100  # Percentage
        self._thumbnail_size = self.DEFAULT_THUMBNAIL_SIZE
        self.on_before_mutate: Callable[[], None] | None = None

        # Drag and Drop state
        self._drop_target_index: int | None = None
        self._drop_indicator_rect: tuple[float, float, float, float] | None = None
        self._auto_scroll_source_id: int | None = None
        self._scroll_direction: int = 0  # -1=up, 0=none, 1=down

        # Rubberband state
        self._rubberband_active: bool = False

        # Track initial selection state at start of rubberband
        self._rubberband_initial_selection: set[int] = set()

        self.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.set_vexpand(True)
        self.set_hexpand(True)

        self._setup_ui()
        self._setup_interactions()
        self._setup_keyboard_shortcuts()

        if document:
            self.load_document(document)

    def _setup_ui(self) -> None:
        """Set up the grid UI."""
        # Main overlay for rubberband and drop indicator
        self._overlay = Gtk.Overlay()
        self.set_child(self._overlay)

        # Main container inside overlay
        self._main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self._main_box.set_margin_top(8)
        self._main_box.set_margin_bottom(8)
        self._main_box.set_margin_start(8)
        self._main_box.set_margin_end(8)
        self._overlay.set_child(self._main_box)

        # FlowBox for page thumbnails
        self._flowbox = Gtk.FlowBox()
        self._flowbox.set_selection_mode(Gtk.SelectionMode.NONE)  # Manual selection
        self._flowbox.set_homogeneous(False)

        self._flowbox.set_row_spacing(8)
        self._flowbox.set_column_spacing(8)
        self._flowbox.set_min_children_per_line(1)
        self._flowbox.set_max_children_per_line(30)
        self._flowbox.set_valign(Gtk.Align.START)
        self._flowbox.set_halign(Gtk.Align.FILL)
        self._flowbox.add_css_class("page-flowbox")
        self._flowbox.set_accessible_role(Gtk.AccessibleRole.LIST)
        self._flowbox.update_property(
            [Gtk.AccessibleProperty.LABEL],
            [
                _(
                    "Page thumbnails. Ctrl+A select all, Ctrl+Up/Down reorder, "
                    "Delete toggle page, Ctrl+L/R rotate, Ctrl+Z undo"
                )
            ],
        )

        self._main_box.append(self._flowbox)

        # Connect to scroll position changes for lazy loading
        vadj = self.get_vadjustment()
        vadj.connect("value-changed", self._on_scroll_changed)
        vadj.connect("notify::page-size", self._on_scroll_changed)

        # Drop Indicator Overlay (a simple DrawingArea that renders a line)
        self._drop_indicator = Gtk.DrawingArea()
        self._drop_indicator.set_visible(False)
        self._drop_indicator.set_can_target(False)
        self._drop_indicator.set_draw_func(self._draw_drop_indicator)
        self._overlay.add_overlay(self._drop_indicator)

        # Selection rectangle overlay
        self._selection_area = Gtk.DrawingArea()
        self._selection_area.set_visible(False)
        self._selection_area.set_can_target(False)
        self._selection_area.set_draw_func(self._draw_selection_rect)
        self._overlay.add_overlay(self._selection_area)

    def _setup_interactions(self) -> None:
        """Set up drag-drop and selection interactions."""
        # 1. Drop Target for page reordering (on FlowBox)
        drop_target = Gtk.DropTarget.new(GObject.TYPE_INT, Gdk.DragAction.MOVE)
        drop_target.connect("drop", self._on_page_drop)
        drop_target.connect("enter", self._on_drag_enter)
        drop_target.connect("motion", self._on_drag_motion)
        drop_target.connect("leave", self._on_drag_leave)
        self._flowbox.add_controller(drop_target)

        # 2. Rubberband Selection (on Overlay, BUBBLE phase for lower priority)
        self._rubberband_gesture = Gtk.GestureDrag()
        self._rubberband_gesture.set_button(1)
        self._rubberband_gesture.set_propagation_phase(Gtk.PropagationPhase.BUBBLE)
        self._rubberband_gesture.connect("drag-begin", self._on_rubberband_begin)
        self._rubberband_gesture.connect("drag-update", self._on_rubberband_update)
        self._rubberband_gesture.connect("drag-end", self._on_rubberband_end)
        self._overlay.add_controller(self._rubberband_gesture)

        self._rubberband_start: tuple[float, float] | None = None
        self._rubberband_current: tuple[float, float] | None = None

    def _setup_keyboard_shortcuts(self) -> None:
        """Set up keyboard event handling."""
        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(key_controller)

    def _on_key_pressed(
        self,
        controller: Gtk.EventControllerKey,
        keyval: int,
        _keycode: int,
        state: Gdk.ModifierType,
    ) -> bool:
        """Handle keyboard events."""
        ctrl = state & Gdk.ModifierType.CONTROL_MASK

        if keyval == Gdk.KEY_a and ctrl:
            self.select_all()
            return True
        elif keyval == Gdk.KEY_d and ctrl:
            self.deselect_all()
            return True
        elif keyval == Gdk.KEY_space:
            self.toggle_ocr_for_selected()
            return True
        elif keyval in (Gdk.KEY_Left, Gdk.KEY_Right, Gdk.KEY_Up, Gdk.KEY_Down):
            self._handle_arrow_navigation(keyval, state)
            return True
        elif keyval == Gdk.KEY_Home:
            if self._thumbnails:
                self._select_single(0)
            return True
        elif keyval == Gdk.KEY_End:
            if self._thumbnails:
                self._select_single(len(self._thumbnails) - 1)
            return True

        return False

    def _handle_arrow_navigation(self, keyval: int, state: Gdk.ModifierType) -> None:
        if not self._thumbnails:
            return

        shift = state & Gdk.ModifierType.SHIFT_MASK
        current = self._last_selected_index or 0
        columns = self._get_visible_columns()
        new_index = current

        if keyval == Gdk.KEY_Left:
            new_index = max(0, current - 1)
        elif keyval == Gdk.KEY_Right:
            new_index = min(len(self._thumbnails) - 1, current + 1)
        elif keyval == Gdk.KEY_Up:
            new_index = max(0, current - columns)
        elif keyval == Gdk.KEY_Down:
            new_index = min(len(self._thumbnails) - 1, current + columns)

        if shift:
            self._extend_selection_to(new_index)
        else:
            self._select_single(new_index)
        self._scroll_to_index(new_index)

    def _get_visible_columns(self) -> int:
        width = self.get_allocated_width()
        item_width = self._thumbnail_size + 28
        return max(1, width // item_width)

    def _scroll_to_index(self, index: int) -> None:
        if not 0 <= index < len(self._thumbnails):
            return

        child = self._flowbox.get_child_at_index(index)
        if child:
            allocation = child.get_allocation()
            vadj = self.get_vadjustment()
            visible_top = vadj.get_value()
            visible_bottom = visible_top + vadj.get_page_size()

            if allocation.y < visible_top:
                vadj.set_value(allocation.y)
            elif allocation.y + allocation.height > visible_bottom:
                vadj.set_value(allocation.y + allocation.height - vadj.get_page_size())

    def load_document(self, document: PDFDocument) -> None:
        self._document = document
        # Show ALL pages (including excluded) so they remain visible
        all_pages = sorted(document.pages, key=lambda p: p.position)

        # Preserve selection state across refresh
        old_selected = set(self._selected_indices)

        self._thumbnails.clear()
        self._remove_all_children()

        for page in all_pages:
            thumbnail = self._create_thumbnail(page)
            self._flowbox.append(thumbnail)
            self._thumbnails.append(thumbnail)

        # Restore selection
        self._selected_indices.clear()
        for idx in old_selected:
            if 0 <= idx < len(self._thumbnails):
                self._selected_indices.add(idx)
                self._thumbnails[idx].selected = True

        # Only load visible thumbnails (lazy loading)
        GLib.timeout_add(50, self._load_visible_thumbnails)
        self.emit("selection-changed")

    def reorder_in_place(self) -> None:
        """Re-sort existing thumbnails to match their page_state.position
        without destroying/recreating widgets, preserving the scroll position.
        """
        if not self._document:
            return

        # Build new order from current page positions
        indexed = [(t.page_state.position, i, t) for i, t in enumerate(self._thumbnails)]
        indexed.sort(key=lambda x: x[0])
        new_order = [t for _, _, t in indexed]

        # Detach all thumbnails from their FlowBoxChild wrappers,
        # then remove the empty FlowBoxChild from the FlowBox.
        # This is necessary because FlowBox.append() creates a new
        # FlowBoxChild wrapper and requires the widget to have no parent.
        for thumb in self._thumbnails:
            fb_child = thumb.get_parent()  # GtkFlowBoxChild
            if fb_child:
                fb_child.set_child(None)  # unparent thumbnail
                self._flowbox.remove(fb_child)  # remove empty wrapper

        # Re-append in new order (creates fresh FlowBoxChild wrappers)
        for thumb in new_order:
            self._flowbox.append(thumb)

        # Rebuild internal list and remap selection indices
        old_to_new = {}
        for new_idx, (_key, old_idx, _thumb) in enumerate(indexed):
            old_to_new[old_idx] = new_idx

        new_selected = set()
        for old_idx in self._selected_indices:
            if old_idx in old_to_new:
                new_selected.add(old_to_new[old_idx])
        self._selected_indices = new_selected

        self._thumbnails = new_order
        if self._last_selected_index is not None and self._last_selected_index in old_to_new:
            self._last_selected_index = old_to_new[self._last_selected_index]

        self.emit("selection-changed")

    def move_page_in_grid(self, source_pos: int, final_target: int) -> None:
        """Move a single page from source to target in the FlowBox.

        Only one widget is removed/re-inserted so the FlowBox never
        becomes empty and the scroll position is preserved.

        Args:
            source_pos: Current index of the page to move.
            final_target: Target index (after removal adjustment).
        """
        if source_pos < 0 or source_pos >= len(self._thumbnails):
            return

        thumb = self._thumbnails[source_pos]

        # Remove the single widget from FlowBox
        fb_child = thumb.get_parent()
        if fb_child:
            fb_child.set_child(None)
            self._flowbox.remove(fb_child)

        # Insert at the target position
        self._flowbox.insert(thumb, final_target)

        # Update internal list
        self._thumbnails.pop(source_pos)
        self._thumbnails.insert(final_target, thumb)

        # Remap selection indices
        new_selected: set[int] = set()
        for idx in self._selected_indices:
            if idx == source_pos:
                new_selected.add(final_target)
            elif source_pos < final_target:
                # Moved forward: items in (source, target] shift back by 1
                if source_pos < idx <= final_target:
                    new_selected.add(idx - 1)
                else:
                    new_selected.add(idx)
            else:
                # Moved backward: items in [target, source) shift forward by 1
                if final_target <= idx < source_pos:
                    new_selected.add(idx + 1)
                else:
                    new_selected.add(idx)
        self._selected_indices = new_selected

        if self._last_selected_index == source_pos:
            self._last_selected_index = final_target

        self.emit("selection-changed")

    def swap_pages_in_grid(self, swaps: list[tuple[int, int]]) -> None:
        """Swap thumbnail content between FlowBoxChild wrappers.

        No widgets are removed from or added to the FlowBox, so the
        scroll position is completely unaffected.

        Args:
            swaps: List of (index_a, index_b) pairs to swap.
        """
        n = len(self._thumbnails)
        for idx_a, idx_b in swaps:
            if not (0 <= idx_a < n and 0 <= idx_b < n):
                continue

            fb_a = self._flowbox.get_child_at_index(idx_a)
            fb_b = self._flowbox.get_child_at_index(idx_b)
            if not fb_a or not fb_b:
                continue

            thumb_a = self._thumbnails[idx_a]
            thumb_b = self._thumbnails[idx_b]

            # Detach both thumbnails from their wrappers
            fb_a.set_child(None)
            fb_b.set_child(None)

            # Cross-assign
            fb_a.set_child(thumb_b)
            fb_b.set_child(thumb_a)

            # Update internal list
            self._thumbnails[idx_a] = thumb_b
            self._thumbnails[idx_b] = thumb_a

        self.emit("selection-changed")

    def move_pages_in_grid(self, source_indices: list[int], insert_at: int) -> None:
        """Move multiple pages from source positions to a target position.

        Removes source thumbnails from FlowBox and re-inserts them at the
        target position. The FlowBox is never fully emptied (assuming there
        are non-selected pages), so the scroll position is preserved.

        Args:
            source_indices: Sorted list of current indices of pages to move.
            insert_at: Target index (after removal adjustment) for the first page.
        """
        source_set = set(source_indices)
        moving = [self._thumbnails[i] for i in source_indices]

        # Freeze rendering during batch operations
        self._flowbox.set_visible(False)

        # Remove from FlowBox (reverse order to preserve indices)
        for i in sorted(source_set, reverse=True):
            thumb = self._thumbnails[i]
            fb_child = thumb.get_parent()
            if fb_child:
                fb_child.set_child(None)
                self._flowbox.remove(fb_child)

        # Remove from internal list (reverse order)
        for i in sorted(source_set, reverse=True):
            self._thumbnails.pop(i)

        # Re-insert at target position
        for offset, thumb in enumerate(moving):
            pos = insert_at + offset
            self._flowbox.insert(thumb, pos)
            self._thumbnails.insert(pos, thumb)

        # Unfreeze rendering
        self._flowbox.set_visible(True)

        # Update selection to new positions
        count = len(moving)
        self._selected_indices = set(range(insert_at, insert_at + count))
        self._last_selected_index = insert_at

        self.emit("selection-changed")

    def _remove_all_children(self):
        for thumb in self._thumbnails:
            for hid in getattr(thumb, "_grid_handler_ids", []):
                thumb.disconnect(hid)
        child = self._flowbox.get_first_child()
        while child is not None:
            next_child = child.get_next_sibling()
            self._flowbox.remove(child)
            child = next_child

    def _create_thumbnail(self, page_state: PageState) -> PageThumbnail:
        thumbnail = PageThumbnail(
            page_state=page_state,
            pdf_path=page_state.source_file or (self._document.path if self._document else ""),
            size=self._thumbnail_size,
        )
        thumbnail.on_before_mutate = self.on_before_mutate
        handler_ids = [
            thumbnail.connect("thumbnail-clicked", self._on_thumbnail_clicked),
            thumbnail.connect("ocr-toggled", self._on_thumbnail_ocr_toggled, page_state),
            thumbnail.connect("rotate-left-clicked", self._on_thumbnail_rotate_left, page_state),
            thumbnail.connect("rotate-right-clicked", self._on_thumbnail_rotate_right, page_state),
            thumbnail.connect(
                "flip-horizontal-clicked", self._on_thumbnail_flip_horizontal, page_state
            ),
            thumbnail.connect(
                "flip-vertical-clicked", self._on_thumbnail_flip_vertical, page_state
            ),
        ]
        thumbnail._grid_handler_ids = handler_ids
        return thumbnail

    def _on_thumbnail_clicked(self, thumbnail: PageThumbnail) -> None:
        # Ignore clicks if rubberband is active
        if self._rubberband_active:
            return

        index = self._thumbnails.index(thumbnail) if thumbnail in self._thumbnails else -1
        if index == -1:
            return

        display = Gdk.Display.get_default()
        seat = display.get_default_seat() if display else None
        keyboard = seat.get_keyboard() if seat else None

        ctrl_pressed = False
        shift_pressed = False
        if keyboard:
            modifier_state = keyboard.get_modifier_state()
            ctrl_pressed = bool(modifier_state & Gdk.ModifierType.CONTROL_MASK)
            shift_pressed = bool(modifier_state & Gdk.ModifierType.SHIFT_MASK)

        if ctrl_pressed:
            self._toggle_selection(index)
        elif shift_pressed and self._last_selected_index is not None:
            self._select_range(self._last_selected_index, index)
        else:
            self._select_single(index)

    def _on_thumbnail_ocr_toggled(
        self, thumbnail: PageThumbnail, active: bool, page_state: PageState
    ) -> None:
        if self._document:
            self._document.mark_modified()
        self.emit("page-ocr-toggled", page_state.page_number, active)
        self.emit("selection-changed")

    def _on_thumbnail_rotate_left(self, thumbnail: PageThumbnail, page_state: PageState) -> None:
        if self.on_before_mutate:
            self.on_before_mutate()
        page_state.rotate(-90)
        thumbnail.rotate_thumbnail_in_place(270)
        thumbnail._update_appearance()
        if self._document:
            self._document.mark_modified()

    def _on_thumbnail_rotate_right(self, thumbnail: PageThumbnail, page_state: PageState) -> None:
        if self.on_before_mutate:
            self.on_before_mutate()
        page_state.rotate(90)
        thumbnail.rotate_thumbnail_in_place(90)
        thumbnail._update_appearance()
        if self._document:
            self._document.mark_modified()

    def _on_thumbnail_flip_horizontal(
        self, thumbnail: PageThumbnail, page_state: PageState
    ) -> None:
        if self.on_before_mutate:
            self.on_before_mutate()
        page_state.toggle_flip_horizontal()
        thumbnail.flip_thumbnail_in_place(horizontal=True)
        thumbnail._update_appearance()
        if self._document:
            self._document.mark_modified()

    def _on_thumbnail_flip_vertical(self, thumbnail: PageThumbnail, page_state: PageState) -> None:
        if self.on_before_mutate:
            self.on_before_mutate()
        page_state.toggle_flip_vertical()
        thumbnail.flip_thumbnail_in_place(horizontal=False)
        thumbnail._update_appearance()
        if self._document:
            self._document.mark_modified()

    # --- Selection Logic ---

    def _select_single(self, index: int) -> None:
        for i in self._selected_indices:
            if 0 <= i < len(self._thumbnails):
                self._thumbnails[i].selected = False
        self._selected_indices.clear()
        self._selected_indices.add(index)
        self._last_selected_index = index
        if 0 <= index < len(self._thumbnails):
            self._thumbnails[index].selected = True
        self.emit("selection-changed")

    def _toggle_selection(self, index: int) -> None:
        if index in self._selected_indices:
            self._selected_indices.discard(index)
            if 0 <= index < len(self._thumbnails):
                self._thumbnails[index].selected = False
        else:
            self._selected_indices.add(index)
            if 0 <= index < len(self._thumbnails):
                self._thumbnails[index].selected = True
        self._last_selected_index = index
        self.emit("selection-changed")

    def _select_range(self, start: int, end: int) -> None:
        for i in self._selected_indices:
            if 0 <= i < len(self._thumbnails):
                self._thumbnails[i].selected = False
        self._selected_indices.clear()
        min_idx = min(start, end)
        max_idx = max(start, end)
        for i in range(min_idx, max_idx + 1):
            self._selected_indices.add(i)
            if 0 <= i < len(self._thumbnails):
                self._thumbnails[i].selected = True
        self._last_selected_index = end
        self.emit("selection-changed")

    def _extend_selection_to(self, index: int) -> None:
        if self._last_selected_index is None:
            self._select_single(index)
            return
        self._select_range(self._last_selected_index, index)

    def select_all(self) -> None:
        self._selected_indices.clear()
        for i, thumbnail in enumerate(self._thumbnails):
            self._selected_indices.add(i)
            thumbnail.selected = True
        if self._thumbnails:
            self._last_selected_index = len(self._thumbnails) - 1
        self.emit("selection-changed")

    def deselect_all(self) -> None:
        for thumbnail in self._thumbnails:
            thumbnail.selected = False
        self._selected_indices.clear()
        self._last_selected_index = None
        self.emit("selection-changed")

    def get_selected_indices(self) -> list[int]:
        return sorted(self._selected_indices)

    def get_ocr_count(self) -> int:
        return sum(1 for t in self._thumbnails if not t.page_state.deleted)

    def get_total_pages(self) -> int:
        return len(self._thumbnails)

    def select_all_for_ocr(self) -> None:
        self.set_ocr_for_all(True)

    def deselect_all_for_ocr(self) -> None:
        self.set_ocr_for_all(False)

    def set_ocr_for_all(self, included: bool) -> None:
        if self.on_before_mutate:
            self.on_before_mutate()
        for thumbnail in self._thumbnails:
            thumbnail.page_state.deleted = not included
            thumbnail.page_state.included_for_ocr = included
            thumbnail.update_from_state()
        if self._document:
            self._document.mark_modified()
        self.emit("selection-changed")

    # --- Zoom ---

    def set_zoom_level(self, level: int) -> None:
        level = max(50, min(400, level))
        if level == self._zoom_level:
            return
        self._zoom_level = level
        self._thumbnail_size = int(self.DEFAULT_THUMBNAIL_SIZE * level / 100)

        # Clear cache once for all affected documents
        renderer = get_thumbnail_renderer()
        seen_paths: set[str] = set()
        for thumbnail in self._thumbnails:
            path = thumbnail._pdf_path
            if path not in seen_paths:
                renderer.clear_document_cache(path)
                seen_paths.add(path)

        # Resize all thumbnails without individual cache clears
        for thumbnail in self._thumbnails:
            thumbnail.resize_without_reload(self._thumbnail_size)

        # Schedule batch reload after layout settles
        GLib.timeout_add(50, self._load_visible_thumbnails)
        logger.info(f"Zoom level set to {level}%")

    # --- Lazy Load ---

    def _on_scroll_changed(self, *_args) -> None:
        self._load_visible_thumbnails()

    def _load_visible_thumbnails(self) -> bool:
        vadj = self.get_vadjustment()
        visible_top = vadj.get_value()
        visible_bottom = visible_top + vadj.get_page_size()
        margin = vadj.get_page_size() * 0.5
        load_top = visible_top - margin
        load_bottom = visible_bottom + margin
        for i, thumbnail in enumerate(self._thumbnails):
            child = self._flowbox.get_child_at_index(i)
            if child:
                allocation = child.get_allocation()
                if allocation.height == 0 or (
                    allocation.y + allocation.height >= load_top and allocation.y <= load_bottom
                ):
                    thumbnail.load_thumbnail()
        return False

    def refresh(self):
        if self._document:
            self.load_document(self._document)

    def toggle_ocr_for_selected(self) -> None:
        """Toggle OCR inclusion for all selected pages."""
        if self.on_before_mutate:
            self.on_before_mutate()
        for idx in self._selected_indices:
            if 0 <= idx < len(self._thumbnails):
                thumb = self._thumbnails[idx]
                page = thumb.page_state
                page.deleted = not page.deleted
                page.included_for_ocr = not page.deleted
                thumb.update_from_state()
        if self._document:
            self._document.mark_modified()
        self.emit("selection-changed")

    # --- Drag and Drop Logic (Overlay-based) ---

    def _on_drag_enter(self, _target: Gtk.DropTarget, x: float, y: float) -> Gdk.DragAction:
        self._drop_target_index = None
        self._drop_indicator.set_visible(True)
        return Gdk.DragAction.MOVE

    def _on_drag_leave(self, _target: Gtk.DropTarget) -> None:
        self._drop_indicator.set_visible(False)
        self._drop_target_index = None
        self._drop_indicator_rect = None
        self._stop_auto_scroll()

    def _on_drag_motion(self, _target: Gtk.DropTarget, x: float, y: float) -> Gdk.DragAction:
        # Calculate drop index and visual position
        drop_index, indicator_rect = self._get_drop_target_info(x, y)

        self._drop_target_index = drop_index
        self._drop_indicator_rect = indicator_rect
        self._drop_indicator.queue_draw()

        # Update auto-scroll based on VISIBLE position
        # DropTarget coordinates are absolute (scrolled content).
        # We need relative-to-viewport to detect edges.
        vadj = self.get_vadjustment()
        visible_y = y - vadj.get_value()
        self._update_auto_scroll(visible_y)

        return Gdk.DragAction.MOVE

    def _draw_drop_indicator(self, area, cr, width, height) -> None:
        """Draw the drop position indicator line."""
        if not self._drop_indicator_rect:
            return

        x, y, w, h = self._drop_indicator_rect

        # Draw a vertical bar
        cr.set_source_rgba(0.2, 0.6, 1.0, 0.9)
        cr.rectangle(x, y, w, h)
        cr.fill()

    def _find_child_at_flowbox_coords(self, x: float, y: float) -> tuple[int, Gtk.Widget | None]:
        """Find child at FlowBox coordinates by iterating allocations.

        Returns (index, child) or (-1, None) if not found.
        """
        for i in range(len(self._thumbnails)):
            child = self._flowbox.get_child_at_index(i)
            if child:
                alloc = child.get_allocation()
                if alloc.x <= x < alloc.x + alloc.width and alloc.y <= y < alloc.y + alloc.height:
                    return i, child
        return -1, None

    def _get_drop_target_info(
        self, x: float, y: float
    ) -> tuple[int, tuple[float, float, float, float] | None]:
        """Calculate drop target index and indicator rectangle.

        Args:
            x, y: Coordinates from DropTarget (already FlowBox-relative and absolute in content space)

        Returns:
            Tuple of (target_index, indicator_rect_in_overlay_coords)
            All overlay coordinates are absolute content coordinates.
        """
        if not self._thumbnails:
            return 0, None

        margin = 12

        # Find child at position using allocation iteration
        idx, child = self._find_child_at_flowbox_coords(x, y)

        if child and idx >= 0:
            alloc = child.get_allocation()

            # Use absolute allocation Y. DO NOT subtract scroll_y for drawing in Overlay
            # (which is also absolute content space).
            absolute_y = alloc.y

            # Decide left or right half
            center_x = alloc.x + alloc.width / 2

            if x < center_x:
                # Drop before this child
                indicator_x = margin + alloc.x - 6
                indicator_y = margin + absolute_y
                return idx, (indicator_x, indicator_y, 4, alloc.height)
            else:
                # Drop after this child
                indicator_x = margin + alloc.x + alloc.width + 2
                indicator_y = margin + absolute_y
                return idx + 1, (indicator_x, indicator_y, 4, alloc.height)

        # If no child hit, find closest by checking row positions
        # Default: append at end
        if self._thumbnails:
            last_child = self._flowbox.get_child_at_index(len(self._thumbnails) - 1)
            if last_child:
                alloc = last_child.get_allocation()
                absolute_y = alloc.y
                indicator_x = margin + alloc.x + alloc.width + 2
                indicator_y = margin + absolute_y
                return len(self._thumbnails), (indicator_x, indicator_y, 4, alloc.height)

        return len(self._thumbnails), None

    def _update_auto_scroll(self, visible_y: float) -> None:
        """Check edge proximity and manage scroll timer.

        Args:
            visible_y: Y position relative to visible viewport (0 = top of visible area)
        """
        height = self.get_allocated_height()
        margin = 60

        if visible_y < margin:
            self._scroll_direction = -1  # Scroll up
            if self._auto_scroll_source_id is None:
                self._auto_scroll_source_id = GLib.timeout_add(40, self._on_auto_scroll_tick)
        elif visible_y > height - margin:
            self._scroll_direction = 1  # Scroll down
            if self._auto_scroll_source_id is None:
                self._auto_scroll_source_id = GLib.timeout_add(40, self._on_auto_scroll_tick)
        else:
            self._stop_auto_scroll()

    def _stop_auto_scroll(self) -> None:
        """Stop the auto-scroll timer."""
        if self._auto_scroll_source_id is not None:
            GLib.source_remove(self._auto_scroll_source_id)
            self._auto_scroll_source_id = None
        self._scroll_direction = 0

    def _on_auto_scroll_tick(self) -> bool:
        """Timer callback for continuous scrolling."""
        if self._scroll_direction == 0:
            return False  # Stop timer

        vadj = self.get_vadjustment()
        current = vadj.get_value()
        step = 20 * self._scroll_direction
        new_value = current + step

        # Clamp
        min_val = vadj.get_lower()
        max_val = vadj.get_upper() - vadj.get_page_size()
        new_value = max(min_val, min(new_value, max_val))

        if new_value != current:
            vadj.set_value(new_value)
        return True  # Keep timer running

    def _on_page_drop(self, _target: Gtk.DropTarget, value: int, x: float, y: float) -> bool:
        self._stop_auto_scroll()
        self._drop_indicator.set_visible(False)

        target_pos = self._drop_target_index
        self._drop_target_index = None
        self._drop_indicator_rect = None

        if not self._document or target_pos is None:
            return False

        source_pos = value

        # Same position = no-op
        if source_pos == target_pos or source_pos == target_pos - 1:
            return False

        if self.on_before_mutate:
            self.on_before_mutate()

        try:
            pages = self._document.pages

            # Multi-page move: dragged page is part of a multi-selection
            if source_pos in self._selected_indices and len(self._selected_indices) > 1:
                selected = sorted(self._selected_indices)
                moving_pages = [pages[i] for i in selected]

                # Remove selected pages (reverse order to preserve indices)
                for i in reversed(selected):
                    pages.pop(i)

                # Adjust target for removed pages that were before the target
                adjusted_target = target_pos - sum(1 for i in selected if i < target_pos)
                adjusted_target = max(0, min(adjusted_target, len(pages)))

                # Insert all pages at target
                for offset, page in enumerate(moving_pages):
                    pages.insert(adjusted_target + offset, page)

                # Re-index
                for i, p in enumerate(pages):
                    p.position = i

                self._document.mark_modified()
                self.move_pages_in_grid(selected, adjusted_target)
                logger.info(f"Moved {len(selected)} pages to position {adjusted_target}")
                return True

            # Single page move
            if 0 <= source_pos < len(pages):
                page = pages.pop(source_pos)

                # Adjust target because we removed source
                final_target = target_pos
                if source_pos < target_pos:
                    final_target -= 1

                final_target = max(0, min(final_target, len(pages)))
                pages.insert(final_target, page)

                # Re-index
                for i, p in enumerate(pages):
                    p.position = i

                self._document.mark_modified()
                self.move_page_in_grid(source_pos, final_target)
                logger.info(f"Page reordered from {source_pos} to {final_target}")
                return True

        except Exception as e:
            logger.error(f"Error reordering pages: {e}")

        return False

    # --- Rubberband Selection ---

    def _on_rubberband_begin(self, gesture: Gtk.GestureDrag, x: float, y: float) -> None:
        """Begin rubberband selection, but only if not over a thumbnail."""

        # x, y are relative to Overlay (Absolute content coordinates because Overlay is in Viewport)
        # Convert to FlowBox coordinates for child detection
        # FlowBox has margin 12 inside MainBox inside Overlay
        flowbox_x = x - 12
        flowbox_y = y - 12

        # Check if we hit a thumbnail
        idx, child = self._find_child_at_flowbox_coords(flowbox_x, flowbox_y)
        if child:
            # Clicked on a thumbnail - don't start rubberband
            gesture.set_state(Gtk.EventSequenceState.DENIED)
            self._rubberband_start = None
            self._rubberband_active = False
            return

        # Clicked on empty space - start rubberband selection
        self._rubberband_active = True
        self._rubberband_start = (x, y)
        self._rubberband_current = (x, y)

        # Capture initial selection state for toggle logic
        self._rubberband_initial_selection = set(self._selected_indices)

        self._selection_area.set_visible(True)
        self._selection_area.queue_draw()
        gesture.set_state(Gtk.EventSequenceState.CLAIMED)

    def _on_rubberband_update(
        self, gesture: Gtk.GestureDrag, offset_x: float, offset_y: float
    ) -> None:
        if not self._rubberband_start or not self._rubberband_active:
            return

        start_x, start_y = self._rubberband_start
        self._rubberband_current = (start_x + offset_x, start_y + offset_y)
        self._selection_area.queue_draw()

        # Update selection VISUALLY while dragging
        # We restore to initial then apply toggle based on rect
        end_x = start_x + offset_x
        end_y = start_y + offset_y
        self._preview_selection_in_rect(start_x, start_y, end_x, end_y)

        # Auto-scroll during rubberband
        # Need visible Y for auto-scroll check
        vadj = self.get_vadjustment()
        scroll_y = vadj.get_value()
        curr_y = start_y + offset_y
        # Convert absolute content Y to visible Y
        visible_y = curr_y - scroll_y
        self._update_auto_scroll(visible_y)

    def _on_rubberband_end(
        self, gesture: Gtk.GestureDrag, offset_x: float, offset_y: float
    ) -> None:
        self._stop_auto_scroll()
        if not self._rubberband_start or not self._rubberband_active:
            self._cleanup_rubberband()
            return

        start_x, start_y = self._rubberband_start
        end_x = start_x + offset_x
        end_y = start_y + offset_y

        # Finalize selection
        self._preview_selection_in_rect(start_x, start_y, end_x, end_y)

        self._cleanup_rubberband()

    def _cleanup_rubberband(self) -> None:
        """Clean up rubberband state."""
        self._selection_area.set_visible(False)
        self._rubberband_start = None
        self._rubberband_current = None
        self._rubberband_active = False

    def _draw_selection_rect(self, area, cr, width, height) -> None:
        if (
            not self._rubberband_start
            or not self._rubberband_current
            or not self._rubberband_active
        ):
            return

        x1, y1 = self._rubberband_start
        x2, y2 = self._rubberband_current

        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)

        cr.set_source_rgba(0.2, 0.4, 0.8, 0.3)
        cr.rectangle(x, y, w, h)
        cr.fill_preserve()
        cr.set_source_rgba(0.2, 0.4, 0.8, 0.8)
        cr.set_line_width(1)
        cr.stroke()

    def _preview_selection_in_rect(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Update selection based on rect (Toggle logic)."""
        rect_x = min(x1, x2)
        rect_y = min(y1, y2)
        rect_w = abs(x2 - x1)
        rect_h = abs(y2 - y1)

        # Convert overlay rect to FlowBox content coords (Absolute to Absolute)
        content_x = rect_x - 12
        content_y = rect_y - 12

        # Start with initial state
        new_selection = set(self._rubberband_initial_selection)

        # Find pages that intersect the selection rect
        for i, _thumbnail in enumerate(self._thumbnails):
            child = self._flowbox.get_child_at_index(i)
            if not child:
                continue

            alloc = child.get_allocation()

            # Check intersection
            if (
                alloc.x < content_x + rect_w
                and alloc.x + alloc.width > content_x
                and alloc.y < content_y + rect_h
                and alloc.y + alloc.height > content_y
            ):
                # Toggle: if was in initial, remove it. If not, add it.
                if i in self._rubberband_initial_selection:
                    if i in new_selection:
                        new_selection.remove(i)
                else:
                    new_selection.add(i)

        if new_selection != self._selected_indices:
            self._selected_indices = new_selection
            for i, thumb in enumerate(self._thumbnails):
                thumb.selected = i in self._selected_indices
            self.emit("selection-changed")
