"""
BigOcrPdf - Base View Classes

Abstract base classes for UI views to ensure consistency and proper
separation of concerns across the application.

Based on BigLinux ARCHITECTURAL RULES:
- UI (`ui/`): Rendering ONLY. No system calls.
- Service (`services/`): OS interactions ONLY. Injected into UI.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gtk

if TYPE_CHECKING:
    from window import BigOcrPdfWindow

from bigocrpdf.utils.logger import logger


class BaseView(ABC):
    """Abstract base class for all views/pages.

    This class provides common functionality for views including:
    - Reference to the main window
    - Lazy widget creation
    - Logging utilities
    - Lifecycle methods (refresh, cleanup)

    Subclasses must implement:
    - create_view(): Create and return the view widget
    - refresh(): Refresh the view's contents
    - cleanup(): Clean up resources when view is destroyed

    Usage:
        class SettingsView(BaseView):
            def create_view(self) -> Gtk.Widget:
                box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
                # ... build UI
                return box

            def refresh(self) -> None:
                # Update UI with current data
                pass

            def cleanup(self) -> None:
                # Clean up resources
                pass
    """

    def __init__(self, window: "BigOcrPdfWindow"):
        """Initialize the view.

        Args:
            window: Reference to the main application window
        """
        self.window = window
        self._widget: Gtk.Widget | None = None

    @abstractmethod
    def create_view(self) -> Gtk.Widget:
        """Create and return the view widget.

        This method is called lazily when the widget property is first accessed.
        Subclasses should build their entire UI hierarchy here.

        Returns:
            The main widget for this view
        """
        pass

    @abstractmethod
    def refresh(self) -> None:
        """Refresh the view's contents.

        Called when the view's data needs to be updated, such as when
        returning to this view from another page or when data changes.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources when view is destroyed.

        Called during application shutdown or when the view is no longer needed.
        Should clean up:
        - Timers and timeouts
        - Signal handlers
        - External resources
        """
        pass

    @property
    def widget(self) -> Gtk.Widget:
        """Get or create the view widget.

        Implements lazy initialization - the widget is only created
        when first accessed.

        Returns:
            The view's main widget
        """
        if self._widget is None:
            self._widget = self.create_view()
            self.log_info("View widget created")
        return self._widget

    def log_info(self, message: str) -> None:
        """Log an info message with view context.

        Args:
            message: Message to log
        """
        logger.info(f"[{self.__class__.__name__}] {message}")

    def log_error(self, message: str) -> None:
        """Log an error message with view context.

        Args:
            message: Message to log
        """
        logger.error(f"[{self.__class__.__name__}] {message}")

    def log_debug(self, message: str) -> None:
        """Log a debug message with view context.

        Args:
            message: Message to log
        """
        logger.debug(f"[{self.__class__.__name__}] {message}")

    def show_toast(self, message: str, timeout: int = 3) -> None:
        """Show a toast notification via the window.

        Args:
            message: Toast message
            timeout: Display duration in seconds
        """
        if hasattr(self.window, "show_toast"):
            self.window.show_toast(message, timeout)


class BasePageView(BaseView):
    """Base class for stack page views with navigation support.

    Extends BaseView with page-specific functionality:
    - Page name and title properties
    - Navigation lifecycle hooks (on_enter, on_leave)

    Subclasses must implement:
    - page_name: Stack page identifier
    - page_title: Human-readable page title
    - create_view(), refresh(), cleanup() from BaseView

    Usage:
        class TerminalPageView(BasePageView):
            @property
            def page_name(self) -> str:
                return "terminal"

            @property
            def page_title(self) -> str:
                return _("Processing")

            def on_enter(self) -> None:
                super().on_enter()
                self.start_progress_monitor()

            def on_leave(self) -> None:
                super().on_leave()
                self.stop_progress_monitor()
    """

    @property
    @abstractmethod
    def page_name(self) -> str:
        """Return the page name for the stack.

        This is the identifier used with stack.add_named() and
        stack.set_visible_child_name().

        Returns:
            Page identifier string
        """
        pass

    @property
    @abstractmethod
    def page_title(self) -> str:
        """Return the page title for display.

        This is the human-readable title shown in step indicators
        or headers.

        Returns:
            Localized page title
        """
        pass

    def on_enter(self) -> None:
        """Called when navigating to this page.

        Override this method to perform actions when the page becomes visible,
        such as starting timers, refreshing data, or updating the header bar.

        Always call super().on_enter() when overriding.
        """
        self.log_info(f"Entering page: {self.page_name}")

    def on_leave(self) -> None:
        """Called when navigating away from this page.

        Override this method to perform cleanup when the page becomes hidden,
        such as stopping timers or saving temporary state.

        Always call super().on_leave() when overriding.
        """
        self.log_info(f"Leaving page: {self.page_name}")


class BaseDialog(ABC):
    """Base class for dialog windows.

    Provides common dialog functionality:
    - Transient window setup
    - Standard dialog styling
    - Response handling

    Usage:
        class OptionsDialog(BaseDialog):
            def __init__(self, parent: Gtk.Window):
                super().__init__(parent, _("Options"), 500, 400)

            def create_content(self) -> Gtk.Widget:
                # Build dialog content
                return content_box

            def on_response(self, response_id: str) -> None:
                if response_id == "apply":
                    self.apply_changes()
    """

    def __init__(
        self,
        parent: Gtk.Window,
        title: str,
        default_width: int = 400,
        default_height: int = 300,
        modal: bool = True,
        resizable: bool = True,
    ):
        """Initialize the dialog.

        Args:
            parent: Parent window
            title: Dialog title
            default_width: Default width
            default_height: Default height
            modal: Whether dialog is modal
            resizable: Whether dialog can be resized
        """
        self.parent = parent
        self.title = title
        self.default_width = default_width
        self.default_height = default_height
        self.modal = modal
        self.resizable = resizable
        self._dialog: Adw.Window | None = None

    @abstractmethod
    def create_content(self) -> Gtk.Widget:
        """Create the dialog content.

        Returns:
            Widget containing dialog content
        """
        pass

    def on_response(self, response_id: str) -> None:  # noqa: B027
        """Handle dialog response.

        This is an optional hook method that subclasses can override to handle
        specific responses like "apply", "cancel", etc.
        Default implementation does nothing - this is intentional.

        Args:
            response_id: The response identifier
        """
        # Default: no-op, subclasses override as needed

    def present(self) -> None:
        """Show the dialog."""
        if self._dialog is None:
            self._dialog = self._create_dialog()
        self._dialog.present()

    def close(self) -> None:
        """Close the dialog."""
        if self._dialog:
            self._dialog.close()

    def _create_dialog(self) -> Adw.Window:
        """Create the dialog window.

        Returns:
            Configured Adw.Window
        """
        dialog = Adw.Window()
        dialog.set_default_size(self.default_width, self.default_height)
        dialog.set_modal(self.modal)
        dialog.set_transient_for(self.parent)
        dialog.set_resizable(self.resizable)
        dialog.set_title(self.title)

        # Create content
        content = self.create_content()
        dialog.set_content(content)

        return dialog

    def log_info(self, message: str) -> None:
        """Log an info message with dialog context."""
        logger.info(f"[{self.__class__.__name__}] {message}")

    def log_error(self, message: str) -> None:
        """Log an error message with dialog context."""
        logger.error(f"[{self.__class__.__name__}] {message}")
