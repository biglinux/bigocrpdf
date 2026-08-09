"""Main-window UI ownership and layout construction."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from typing import TYPE_CHECKING

from gi.repository import Adw, Gio, Gtk

from bigocrpdf.config import APP_ICON_NAME
from bigocrpdf.ui.conclusion_page import ConclusionPageManager
from bigocrpdf.ui.dialogs_manager import DialogsManager
from bigocrpdf.ui.header_bar import HeaderBar
from bigocrpdf.ui.settings_page import SettingsPageManager
from bigocrpdf.ui.terminal_page import TerminalPageManager
from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.adw_compat import enable_view_stack_transitions
from bigocrpdf.utils.i18n import _, ngettext
from bigocrpdf.utils.logger import logger

if TYPE_CHECKING:
    from bigocrpdf.window import BigOcrPdfWindow

# ── UI Layout Constants ────────────────────────────────────────────
_MIN_SIDEBAR_WIDTH = 300  # pixels
_MAX_SIDEBAR_WIDTH = 420  # pixels
_SIDEBAR_WIDTH_FRACTION = 0.35
_VIEWSTACK_TRANSITION_MS = 250
_RESPONSIVE_BREAKPOINT = 900  # collapse before both panes exceed the available width
_MIN_CONTENT_HEIGHT = 450  # pixels
_CONCLUSION_CLAMP_MAX = 800  # pixels
_CONCLUSION_CLAMP_TIGHT = 600  # pixels
_APP_ICON_SIZE = 20  # pixels


class BigOcrPdfUI:
    """Own the main-window widgets and page managers."""

    def __init__(self, window: "BigOcrPdfWindow") -> None:
        self.window = window
        self.settings_page_manager = SettingsPageManager(window)
        self.terminal_page_manager = TerminalPageManager(window)
        self.conclusion_page_manager = ConclusionPageManager(window)

    def setup(self) -> None:
        """Set up the main user interface with responsive split view layout."""
        self.dialogs_manager = DialogsManager(
            self.window,
            self.window.settings,
            self.show_toast,
        )
        self.toast_overlay = Adw.ToastOverlay()

        # Create responsive split view layout (sidebar + content)
        self.split_view = Adw.OverlaySplitView()
        self.split_view.set_sidebar_position(Gtk.PackType.START)
        self.split_view.set_min_sidebar_width(_MIN_SIDEBAR_WIDTH)
        self.split_view.set_max_sidebar_width(_MAX_SIDEBAR_WIDTH)
        self.split_view.set_sidebar_width_fraction(_SIDEBAR_WIDTH_FRACTION)
        self.split_view.set_enable_hide_gesture(True)
        self.split_view.set_enable_show_gesture(True)

        # Create left sidebar pane
        self._create_left_sidebar()

        # Create right content pane with header bar
        self._create_right_content_area()

        # Create master ViewStack for main view and other pages
        self.main_stack = Adw.ViewStack()
        enable_view_stack_transitions(self.main_stack, _VIEWSTACK_TRANSITION_MS)

        # Add split_view as primary view
        self.main_stack.add_titled(self.split_view, "main_view", _("Main"))

        # Set up window-level actions for keyboard shortcuts
        self.window.actions.setup_window_actions()

        # Set up global drag and drop
        self.window.actions.setup_global_drag_drop()

        # Set up responsive breakpoint — collapse sidebar when window is narrow
        breakpoint = Adw.Breakpoint.new(
            Adw.BreakpointCondition.parse(f"max-width: {_RESPONSIVE_BREAKPOINT}sp")
        )
        breakpoint.add_setter(self.split_view, "collapsed", True)
        self.window.add_breakpoint(breakpoint)

        self.main_stack.connect("notify::visible-child", self._on_main_stack_changed)

        self.toast_overlay.set_child(self.main_stack)
        self.window.set_content(self.toast_overlay)

        # Set up pages (must be after UI structure is created)
        self._setup_pages()

        # Initialize queue controls from restored state.
        self.custom_header_bar.update_queue_size(len(self.window.settings.selected_files))

        # Bind sidebar toggle to split view for responsive collapse
        self.custom_header_bar.bind_split_view(self.split_view)

    def show_toast(self, message: str, timeout: int = 3) -> None:
        """Show a notification over the main window content."""
        toast = Adw.Toast.new(message)
        toast.set_use_markup(False)
        toast.set_timeout(timeout)
        self.toast_overlay.add_toast(toast)

    def update_file_info(self) -> None:
        """Publish the current queue count and refresh its visible page."""
        file_count = len(self.window.settings.selected_files)
        self.window.announce_status(
            ngettext(
                "{count} file in queue",
                "{count} files in queue",
                file_count,
            ).format(count=file_count)
        )
        self.custom_header_bar.update_queue_size(file_count)

        if self.stack.get_visible_child_name() == "settings":
            self.settings_page_manager.refresh_queue_status()
            logger.info("Queue status refreshed with %s files", file_count)

    def _create_left_sidebar(self) -> None:
        """Create the left sidebar with header bar (video-converter style)."""
        # Create ToolbarView for left pane with sidebar styling
        left_toolbar_view = Adw.ToolbarView()
        left_toolbar_view.add_css_class("sidebar")

        # Detect window button layout
        window_buttons_left = self.window.actions.window_buttons_on_left()

        # Create HeaderBar for the sidebar
        left_header = Adw.HeaderBar()
        left_header.add_css_class("sidebar")
        left_header.set_show_title(True)
        # Keep title buttons only in the pane selected by the system layout.
        if not window_buttons_left:
            left_header.set_decoration_layout("")

        # Create title with app icon
        if not window_buttons_left:
            center_box = Gtk.CenterBox()
            center_box.set_hexpand(True)
            app_icon = Gtk.Image.new_from_icon_name(APP_ICON_NAME)
            app_icon.set_pixel_size(_APP_ICON_SIZE)
            app_icon.set_halign(Gtk.Align.START)
            app_icon.set_valign(Gtk.Align.CENTER)
            app_icon.set_hexpand(False)
            app_icon.set_accessible_role(Gtk.AccessibleRole.PRESENTATION)
            center_box.set_start_widget(app_icon)
            title_label = Gtk.Label(label="Big OCR PDF")
            title_label.set_halign(Gtk.Align.CENTER)
            title_label.set_valign(Gtk.Align.CENTER)
            title_label.set_hexpand(True)
            center_box.set_center_widget(title_label)
            left_header.set_title_widget(center_box)
        else:
            title_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            title_label = Gtk.Label(label="Big OCR PDF")
            title_box.append(title_label)
            expander = Gtk.Box()
            expander.set_hexpand(True)
            title_box.append(expander)
            left_header.set_title_widget(title_box)

        left_toolbar_view.add_top_bar(left_header)

        self.split_view.set_sidebar(left_toolbar_view)
        self.left_toolbar_view = left_toolbar_view

    def _create_right_content_area(self) -> None:
        """Create the right content area with header bar."""
        # Create ToolbarView for right pane
        right_toolbar_view = Adw.ToolbarView()

        # Detect window button layout
        window_buttons_left = self.window.actions.window_buttons_on_left()

        # Create custom header bar (Add Files, Start OCR, etc)
        self.custom_header_bar = HeaderBar(self.window, window_buttons_left)
        right_toolbar_view.add_top_bar(self.custom_header_bar)

        # Set up help action
        help_action = Gio.SimpleAction.new("help", None)
        help_action.connect("activate", lambda *_: self.window.actions.welcome.show())
        self.window.add_action(help_action)

        # Set up reset settings action
        reset_action = Gio.SimpleAction.new("reset_settings", None)
        reset_action.connect("activate", lambda *_: self.window.actions.reset.confirm())
        self.window.add_action(reset_action)

        # Create stack for right content (file queue only for now)
        self.stack = Adw.ViewStack()
        self.stack.set_vexpand(True)
        enable_view_stack_transitions(self.stack, _VIEWSTACK_TRANSITION_MS)

        # Content scroll for the file queue
        content_scroll = Gtk.ScrolledWindow()
        content_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        content_scroll.set_propagate_natural_height(True)
        content_scroll.set_min_content_height(_MIN_CONTENT_HEIGHT)
        content_scroll.set_vexpand(True)
        content_scroll.set_child(self.stack)

        right_toolbar_view.set_content(content_scroll)

        self.split_view.set_content(right_toolbar_view)

    def _setup_pages(self) -> None:
        """Set up the application pages."""
        # File queue goes into the right stack
        settings_page = self.create_settings_page()
        self.stack.add_named(settings_page, "settings")

        # Terminal page goes into main_stack (full-width) - with Cancel button
        terminal_toolbar = self._create_full_width_page_with_header(is_terminal=True)
        terminal_content = self.terminal_page_manager.create_terminal_page()
        terminal_toolbar.set_content(terminal_content)
        self.main_stack.add_titled(terminal_toolbar, "terminal", _("Processing"))

        # Conclusion page goes into main_stack (full-width) - with Back button
        conclusion_toolbar = self._create_full_width_page_with_header()
        conclusion_content = self.conclusion_page_manager.create_conclusion_page()
        conclusion_clamp = Adw.Clamp()
        conclusion_clamp.set_maximum_size(_CONCLUSION_CLAMP_MAX)
        conclusion_clamp.set_tightening_threshold(_CONCLUSION_CLAMP_TIGHT)
        conclusion_clamp.set_child(conclusion_content)
        conclusion_toolbar.set_content(conclusion_clamp)
        self.main_stack.add_titled(conclusion_toolbar, "conclusion", _("Results"))

        # Populate sidebar settings
        self._populate_sidebar_settings()

    def _create_full_width_page_with_header(self, *, is_terminal: bool = False) -> Adw.ToolbarView:
        """Create a full-width page with its own header bar.

        Args:
            is_terminal: If True, no navigation button is added (Cancel is in the content area)

        Returns:
            ToolbarView with header
        """
        toolbar_view = Adw.ToolbarView()

        # Create a simpler header bar for non-settings pages
        header = Adw.HeaderBar()

        if not is_terminal:
            # Back button for conclusion page
            back_button = Gtk.Button()
            back_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            back_icon = Gtk.Image.new_from_icon_name("go-previous-symbolic")
            back_icon.set_accessible_role(Gtk.AccessibleRole.PRESENTATION)
            back_label = Gtk.Label(label=_("Back"))
            back_box.append(back_icon)
            back_box.append(back_label)
            back_button.set_child(back_box)
            back_button.add_css_class("suggested-action")
            set_a11y_label(back_button, _("Back"))
            back_button.connect("clicked", lambda _: self.window._return_to_main_view())
            header.pack_start(back_button)

        # Title
        header.set_title_widget(Gtk.Label(label="Big OCR PDF"))

        # Menu button
        menu_button = Gtk.MenuButton()
        menu_button.set_icon_name("open-menu-symbolic")
        menu_button.set_tooltip_text(_("Menu"))
        set_a11y_label(menu_button, _("Menu"))
        menu = Gio.Menu.new()
        menu.append(_("Reset Settings"), "win.reset_settings")
        menu.append(_("Help"), "win.help")
        menu.append(_("About"), "app.about")
        menu.append(_("Quit"), "app.quit")
        menu_button.set_menu_model(menu)
        header.pack_end(menu_button)

        toolbar_view.add_top_bar(header)

        return toolbar_view

    def _populate_sidebar_settings(self) -> None:
        """Populate the sidebar with OCR settings content."""
        settings_content = self.settings_page_manager.create_sidebar_content()
        self.left_toolbar_view.set_content(settings_content)

    def create_settings_page(self) -> Gtk.Widget:
        """Create the queue page."""
        return self.settings_page_manager.create_settings_page()

    def _on_main_stack_changed(self, stack: Adw.ViewStack, _param) -> None:
        """Handle main stack page changes."""
        current_page = stack.get_visible_child_name()
        self.window.actions.sync_for_page(current_page)
        logger.debug(f"Main stack changed to: {current_page}")

    def cleanup(self) -> None:
        """Clean up resources owned by page managers."""
        try:
            self.settings_page_manager.cleanup()
        except Exception as error:
            logger.error("Error cleaning settings page: %s", error)

        try:
            self.terminal_page_manager.cleanup()
        except Exception as error:
            logger.error("Error cleaning terminal page: %s", error)

        try:
            self.conclusion_page_manager.reset_page()
        except Exception as error:
            logger.error("Error cleaning conclusion page: %s", error)
