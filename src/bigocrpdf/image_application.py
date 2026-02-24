"""
BigOcrPdf - Image OCR Application Module

Standalone GTK application for Image OCR, separate from the PDF application.
Uses a different application_id for proper Wayland taskbar grouping.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk

from bigocrpdf.ui.image_ocr_window import ImageOcrWindow
from bigocrpdf.ui.widgets import load_css
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger

# Constants for the standalone image app (avoid circular import from config)
IMAGE_APP_ID = "br.com.biglinux.bigocrimage"
IMAGE_APP_ICON = "bigocrimage"
IMAGE_APP_VERSION = "3.0.0"


class ImageOcrApp(Adw.Application):
    """Standalone application for Image OCR.

    Uses a separate application_id (br.com.biglinux.bigocrimage) so that
    it appears as a separate application in the Wayland taskbar.
    """

    def __init__(self) -> None:
        """Initialize the image OCR application."""
        super().__init__(
            application_id=IMAGE_APP_ID,
            flags=Gio.ApplicationFlags.HANDLES_OPEN,
        )

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

        # Set up application icon
        self._setup_actions()

    def _setup_actions(self) -> None:
        """Set up application actions."""
        # About action
        about_action = Gio.SimpleAction.new("about", None)
        about_action.connect("activate", self.on_about_action)
        self.add_action(about_action)

        # Quit action
        quit_action = Gio.SimpleAction.new("quit", None)
        quit_action.connect("activate", lambda *_: self.quit())
        self.add_action(quit_action)

        # Paste from clipboard action
        paste_action = Gio.SimpleAction.new("paste-clipboard", None)
        paste_action.connect("activate", self._on_paste_action)
        self.add_action(paste_action)

        # Set keyboard shortcuts
        self.set_accels_for_action("app.quit", ["<Control>q"])
        self.set_accels_for_action("app.about", ["F1"])
        self.set_accels_for_action("app.paste-clipboard", ["<Control>v"])

    def _on_paste_action(self, _action: Gio.SimpleAction, _param: None) -> None:
        """Handle paste from clipboard action."""
        win = self.get_active_window()
        if win and hasattr(win, "paste_from_clipboard"):
            win.paste_from_clipboard()

    def on_handle_local_options(self, app: Adw.Application, options: GLib.VariantDict) -> int:
        """Handle command line options."""
        if options.contains("version"):
            print(f"Big Image OCR {IMAGE_APP_VERSION}")
            return 0
        return -1

    def on_activate(self, app: Adw.Application) -> None:
        """Callback for application activation."""
        try:
            load_css()
            win = self.get_active_window()
            if not win:
                win = ImageOcrWindow(app)
                logger.info("Started Image OCR application")
            win.present()
        except Exception as e:
            logger.error(f"Error activating Image OCR: {e}")

    def on_open(
        self,
        app: Adw.Application,
        files: list[Gio.File],
        n_files: int,
        _hint: str,
    ) -> None:
        """Handle opening files."""
        logger.info(f"on_open called with {n_files} files")
        self.on_activate(app)
        win = self.get_active_window()

        if win and files:
            file_path = files[0].get_path()
            logger.info(f"on_open: file_path={file_path}, uri={files[0].get_uri()}")
            if file_path and hasattr(win, "open_image"):
                win.open_image(file_path)
                logger.info(f"Opened image: {file_path}")
            else:
                logger.warning(f"Could not open: path={file_path}, has_open_image={hasattr(win, 'open_image')}")
        else:
            logger.warning(f"on_open: win={win}, files={files}")

    def on_about_action(self, _action: Gio.SimpleAction, _param: None) -> None:
        """Show the About dialog."""
        about = Adw.AboutDialog.new()
        about.set_application_name("Big Image OCR")
        about.set_application_icon(IMAGE_APP_ICON)
        about.set_version(IMAGE_APP_VERSION)
        about.set_comments(_("Extract text from images using OCR"))
        about.set_website("https://www.biglinux.com.br")
        about.set_developers(["BigLinux https://github.com/biglinux/bigocrpdf"])
        about.set_license_type(Gtk.License.GPL_3_0)
        about.present(self.get_active_window())
