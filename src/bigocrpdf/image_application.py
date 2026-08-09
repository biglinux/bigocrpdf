"""
BigOcrPdf - Image OCR Application Module

Standalone GTK application for Image OCR, separate from the PDF application.
Uses a different application_id for proper Wayland taskbar grouping.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk

from bigocrpdf import OcrDependencyState
from bigocrpdf.config import (
    APP_DEVELOPERS,
    APP_ISSUES,
    APP_VERSION,
    APP_WEBSITE,
    IMAGE_APP_ICON_NAME,
    IMAGE_APP_ID,
    SHORTCUTS,
)
from bigocrpdf.ui.image_ocr_window import ImageOcrWindow
from bigocrpdf.ui.widgets import load_css
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.icons import setup_icons
from bigocrpdf.utils.logger import logger


class ImageOcrApp(Adw.Application):
    """Standalone application for Image OCR.

    Uses a separate application_id (br.com.biglinux.bigocrimage) so that
    it appears as a separate application in the Wayland taskbar.
    """

    def __init__(self, *, ocr_dependency: OcrDependencyState) -> None:
        """Initialize the image OCR application."""
        super().__init__(
            application_id=IMAGE_APP_ID,
            flags=Gio.ApplicationFlags.HANDLES_OPEN,
        )
        self.ocr_dependency = ocr_dependency

        # Add version command line option
        self.add_main_option(
            "version",
            ord("v"),
            GLib.OptionFlags.NONE,
            GLib.OptionArg.NONE,
            _("Print version information and exit"),
            None,
        )

        self.connect("activate", self.on_activate)
        self.connect("open", self.on_open)
        self.connect("handle-local-options", self.on_handle_local_options)
        self.connect("shutdown", self._on_shutdown)

        # Set up application icon
        self._setup_actions()

    def do_startup(self) -> None:
        """Register the bundled icon theme before any widget is created."""
        Adw.Application.do_startup(self)
        setup_icons()

    def _setup_actions(self) -> None:
        """Set up application actions."""
        # About action
        about_action = Gio.SimpleAction.new("about", None)
        about_action.connect("activate", self.on_about_action)
        self.add_action(about_action)

        # Quit action
        quit_action = Gio.SimpleAction.new("quit", None)
        quit_action.connect("activate", self._on_quit_action)
        self.add_action(quit_action)

        # Set keyboard shortcuts
        self.set_accels_for_action("app.quit", [SHORTCUTS.get("quit", "<Control>q")])
        self.set_accels_for_action("app.about", [SHORTCUTS.get("about", "F1")])
        self.set_accels_for_action(
            "win.paste-clipboard",
            [SHORTCUTS.get("paste-clipboard", "<Control>v")],
        )
        self.set_accels_for_action(
            "win.cancel-processing",
            [SHORTCUTS.get("cancel-processing", "Escape")],
        )

    def _prepare_image_windows_for_shutdown(self) -> None:
        """Cancel window-owned workers before application teardown."""
        for window in self.get_windows():
            if isinstance(window, ImageOcrWindow):
                window.prepare_close()

    def _on_quit_action(self, *_args: object) -> None:
        """Prepare image windows before the explicit quit action."""
        self._prepare_image_windows_for_shutdown()
        self.quit()

    def _on_shutdown(self, *_args: object) -> None:
        """Defensively prepare windows for non-action shutdown paths."""
        self._prepare_image_windows_for_shutdown()

    def on_handle_local_options(self, app: Adw.Application, options: GLib.VariantDict) -> int:
        """Handle command line options."""
        if options.contains("version"):
            print(f"Big Image OCR {APP_VERSION}")
            return 0
        return -1

    def on_activate(self, app: Adw.Application) -> None:
        """Callback for application activation."""
        load_css()
        win = self.get_active_window()
        if not isinstance(win, ImageOcrWindow):
            win = ImageOcrWindow(app, ocr_dependency=self.ocr_dependency)
            logger.info("Started Image OCR application")
        win.present()
        if not self.ocr_dependency.is_available:
            GLib.idle_add(win.show_ocr_unavailable_dialog)

    def on_open(
        self,
        app: Adw.Application,
        files: list[Gio.File],
        n_files: int,
        _hint: str,
    ) -> None:
        """Handle opening files."""
        logger.debug(f"on_open called with {n_files} files")
        self.on_activate(app)
        win = self.get_active_window()

        if isinstance(win, ImageOcrWindow) and files:
            file_path = files[0].get_path()
            logger.debug(f"on_open: file_path={file_path}, uri={files[0].get_uri()}")
            if file_path:
                win.open_image(file_path)
                logger.debug(f"Opened image: {file_path}")
            else:
                logger.warning(f"Could not open image URI: {files[0].get_uri()}")
        else:
            logger.warning(f"on_open: win={win}, files={files}")

    def on_about_action(self, _action: Gio.SimpleAction, _param: None) -> None:
        """Show the About dialog."""
        about = Adw.AboutDialog.new()
        about.set_application_name("Big Image OCR")
        about.set_application_icon(IMAGE_APP_ICON_NAME)
        about.set_version(APP_VERSION)
        about.set_developer_name(_("BigLinux Team"))
        about.set_comments(_("Extract text from images using OCR"))
        about.set_website(APP_WEBSITE)
        about.set_issue_url(APP_ISSUES)
        about.set_developers(APP_DEVELOPERS)
        about.set_license_type(Gtk.License.GPL_3_0)
        about.present(self.get_active_window())
