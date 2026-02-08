"""Window UI Setup Mixin."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, Gtk

from bigocrpdf.config import APP_ICON_NAME
from bigocrpdf.ui.header_bar import HeaderBar
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class WindowUISetupMixin:
    """Mixin providing UI setup and layout creation for the main window."""

    def setup_ui(self) -> None:
        """Set up the main user interface with video-converter style layout."""
        # Setup CSS for sidebar styling
        self._setup_sidebar_css()

        self.toast_overlay = Adw.ToastOverlay()

        # Create main paned layout (like video-converter)
        self.main_paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        self.main_paned.set_vexpand(True)

        # Create left sidebar pane
        self._create_left_sidebar()

        # Create right content pane with header bar
        self._create_right_content_area()

        # Set initial paned position
        self.main_paned.set_position(380)

        # Create master ViewStack for main view and other pages
        self.main_stack = Adw.ViewStack()

        # Add main_paned as primary view
        self.main_stack.add_titled(self.main_paned, "main_view", _("Main"))

        # Create terminal and conclusion pages
        self._create_non_settings_pages()

        # Set up window-level actions for keyboard shortcuts
        self._setup_window_actions()

        # Set up global drag and drop
        self._setup_global_drag_drop()

        self.main_stack.connect("notify::visible-child", self._on_main_stack_changed)

        self.toast_overlay.set_child(self.main_stack)
        self.set_content(self.toast_overlay)

        # Set up pages (must be after UI structure is created)
        self.setup_pages()

        # Initialize header bar view for settings page
        self.custom_header_bar.set_view("queue")

    def _setup_sidebar_css(self) -> None:
        """Setup CSS for sidebar styling to match video-converter."""
        css_provider = Gtk.CssProvider()
        css_provider.load_from_string(
            """
        .sidebar {
            background-color: @sidebar_bg_color;
        }
        """
        )
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

    def _create_left_sidebar(self) -> None:
        """Create the left sidebar with header bar (video-converter style)."""
        # Create ToolbarView for left pane with sidebar styling
        left_toolbar_view = Adw.ToolbarView()
        left_toolbar_view.add_css_class("sidebar")

        # Detect window button layout
        window_buttons_left = self._window_buttons_on_left()

        # Create HeaderBar for the sidebar
        left_header = Adw.HeaderBar()
        left_header.add_css_class("sidebar")
        left_header.set_show_title(True)
        # Configure decoration layout
        left_header.set_decoration_layout(
            "close,maximize,minimize:menu" if window_buttons_left else ""
        )

        # Create title with app icon
        if not window_buttons_left:
            center_box = Gtk.CenterBox()
            center_box.set_hexpand(True)
            app_icon = Gtk.Image.new_from_icon_name(APP_ICON_NAME)
            app_icon.set_pixel_size(20)
            app_icon.set_halign(Gtk.Align.START)
            app_icon.set_valign(Gtk.Align.CENTER)
            app_icon.set_hexpand(False)
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

        # Create sidebar content (OCR settings)
        sidebar_content = self._create_sidebar_settings()
        left_toolbar_view.set_content(sidebar_content)

        self.main_paned.set_start_child(left_toolbar_view)
        self.left_toolbar_view = left_toolbar_view  # Store reference for later settings

    def _create_sidebar_settings(self) -> Gtk.Widget:
        """Create a placeholder for sidebar settings.

        The actual content is set by _populate_sidebar_settings().

        Returns:
            Widget placeholder for OCR settings
        """
        # Just a placeholder box - the real content is set later
        self.sidebar_content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.sidebar_content_box.set_vexpand(True)
        return self.sidebar_content_box

    def _create_right_content_area(self) -> None:
        """Create the right content area with header bar."""
        # Create ToolbarView for right pane
        right_toolbar_view = Adw.ToolbarView()

        # Detect window button layout
        window_buttons_left = self._window_buttons_on_left()

        # Create custom header bar (Add Files, Start OCR, etc)
        self.custom_header_bar = HeaderBar(self, window_buttons_left)
        right_toolbar_view.add_top_bar(self.custom_header_bar)

        # Set up help action
        help_action = Gio.SimpleAction.new("help", None)
        help_action.connect("activate", lambda *_: self.show_welcome_dialog())
        self.add_action(help_action)

        # Set up reset settings action
        reset_action = Gio.SimpleAction.new("reset_settings", None)
        reset_action.connect("activate", lambda *_: self._confirm_reset_settings())
        self.add_action(reset_action)

        # Create stack for right content (file queue only for now)
        self.stack = Adw.ViewStack()
        self.stack.set_vexpand(True)
        self.stack.set_transition_duration(300)

        # Content scroll for the file queue
        content_scroll = Gtk.ScrolledWindow()
        content_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        content_scroll.set_propagate_natural_height(True)
        content_scroll.set_min_content_height(450)
        content_scroll.set_vexpand(True)
        content_scroll.set_child(self.stack)

        right_toolbar_view.set_content(content_scroll)

        self.main_paned.set_end_child(right_toolbar_view)

    def _create_non_settings_pages(self) -> None:
        """Create terminal and conclusion pages as separate full-width views."""

    def setup_pages(self) -> None:
        """Set up the application pages."""
        # File queue goes into the right stack
        settings_page = self.ui.create_settings_page()
        self.stack.add_named(settings_page, "settings")

        # Terminal page goes into main_stack (full-width) - with Cancel button
        terminal_toolbar = self._create_full_width_page_with_header("terminal", is_terminal=True)
        terminal_content = self.ui.create_terminal_page()
        terminal_toolbar.set_content(terminal_content)
        self.main_stack.add_titled(terminal_toolbar, "terminal", _("Processing"))

        # Conclusion page goes into main_stack (full-width) - with Back button
        conclusion_toolbar = self._create_full_width_page_with_header("complete")
        conclusion_content = self.ui.create_conclusion_page()
        conclusion_toolbar.set_content(conclusion_content)
        self.main_stack.add_titled(conclusion_toolbar, "conclusion", _("Results"))

        # Populate sidebar settings
        self._populate_sidebar_settings()

    def _create_full_width_page_with_header(
        self, _view_mode: str, is_terminal: bool = False
    ) -> Adw.ToolbarView:
        """Create a full-width page with its own header bar.

        Args:
            view_mode: The view mode for the header bar
            is_terminal: If True, no navigation button is added (Cancel is in the content area)

        Returns:
            ToolbarView with header
        """
        toolbar_view = Adw.ToolbarView()

        # Create a simpler header bar for non-settings pages
        header = Adw.HeaderBar()
        header.set_decoration_layout("menu:minimize,maximize,close")

        if not is_terminal:
            # Back button for conclusion page
            back_button = Gtk.Button()
            back_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            back_icon = Gtk.Image.new_from_icon_name("go-previous-symbolic")
            back_label = Gtk.Label(label=_("Back"))
            back_box.append(back_icon)
            back_box.append(back_label)
            back_button.set_child(back_box)
            back_button.add_css_class("suggested-action")
            back_button.connect("clicked", lambda _: self._return_to_main_view())
            header.pack_start(back_button)

        # Title
        header.set_title_widget(Gtk.Label(label="Big OCR PDF"))

        # Menu button
        menu_button = Gtk.MenuButton()
        menu_button.set_icon_name("open-menu-symbolic")
        menu = Gio.Menu.new()
        menu.append(_("Reset Settings"), "win.reset_settings")
        menu.append(_("Help"), "win.help")
        menu.append(_("About"), "app.about")
        menu.append(_("Quit"), "app.quit")
        menu_button.set_menu_model(menu)
        header.pack_end(menu_button)

        toolbar_view.add_top_bar(header)

        return toolbar_view

    def _return_to_main_view(self) -> None:
        """Return to the main view (settings page) and clear processed files."""
        # Reset conclusion page labels to avoid stale data on next run
        if hasattr(self, "ui") and hasattr(self.ui, "conclusion_page_manager"):
            self.ui.conclusion_page_manager.reset_page()

        # Clear processing results via shared method
        self.settings.reset_processing_state()

        # Clear the file queue (files have been processed)
        self.clear_file_queue()

        # Navigate back to main view
        self.main_stack.set_visible_child_name("main_view")
        self.custom_header_bar.set_view("queue")

    def _populate_sidebar_settings(self) -> None:
        """Populate the sidebar with OCR settings content."""
        if hasattr(self.ui, "settings_page_manager") and self.ui.settings_page_manager:
            settings_content = self.ui.settings_page_manager.create_sidebar_content()
            # Set content directly on the toolbar view
            self.left_toolbar_view.set_content(settings_content)

    def _on_main_stack_changed(self, stack: Adw.ViewStack, _param) -> None:
        """Handle main stack page changes."""
        current_page = stack.get_visible_child_name()
        logger.debug(f"Main stack changed to: {current_page}")
