"""
BigOcrPdf - Application Module

This module contains the main application class for the BigOcrPdf application.
"""

from typing import Any

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk

from bigocrpdf.config import (
    APP_DEVELOPERS,
    APP_ICON_NAME,
    APP_ID,
    APP_ISSUES,
    APP_NAME,
    APP_VERSION,
    APP_WEBSITE,
    SHORTCUTS,
    get_app_description,
    init_config,
)
from bigocrpdf.ui.image_ocr_window import ImageOcrWindow
from bigocrpdf.ui.widgets import load_css
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger
from bigocrpdf.window import BigOcrPdfWindow


class BigOcrPdfApp(Adw.Application):
    """Application class for BigOcrPdf."""

    def __init__(self, edit_mode: bool = False) -> None:
        """Initialize the application."""
        # Editor mode uses a separate application-id so it runs as an
        # independent process from the OCR window (different D-Bus name).
        app_id = f"{APP_ID}.editor" if edit_mode else APP_ID
        super().__init__(application_id=app_id, flags=Gio.ApplicationFlags.HANDLES_OPEN)

        self._edit_mode = edit_mode

        # Add command line handling
        self.add_main_option(
            "version",
            ord("v"),
            GLib.OptionFlags.NONE,
            GLib.OptionArg.NONE,
            _("Print version information and exit"),
            None,
        )
        self.add_main_option(
            "edit",
            ord("e"),
            GLib.OptionFlags.NONE,
            GLib.OptionArg.NONE,
            _("Open files directly in the PDF editor"),
            None,
        )

        # Setup signals
        self.connect("activate", self.on_activate)
        self.connect("open", self.on_open)
        self.connect("handle-local-options", self.on_handle_local_options)

        # Set up application actions
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

        # Image OCR action
        image_ocr_action = Gio.SimpleAction.new("image-ocr", None)
        image_ocr_action.connect("activate", self.on_image_ocr_action)
        self.add_action(image_ocr_action)

        # Keyboard Shortcuts dialog action
        shortcuts_action = Gio.SimpleAction.new("shortcuts", None)
        shortcuts_action.connect("activate", self._on_shortcuts_action)
        self.add_action(shortcuts_action)

        # Set up keyboard shortcuts
        self._setup_keyboard_shortcuts()

    def _setup_keyboard_shortcuts(self) -> None:
        """Set up application-level keyboard shortcuts."""
        try:
            # Application-level shortcuts
            self.set_accels_for_action("app.quit", [SHORTCUTS.get("quit", "<Control>q")])
            self.set_accels_for_action("app.about", [SHORTCUTS.get("about", "F1")])
            self.set_accels_for_action("app.shortcuts", ["<Control>question"])

            # Window-level shortcuts (win. prefix)
            self.set_accels_for_action("win.add-files", [SHORTCUTS.get("add-files", "<Control>o")])
            self.set_accels_for_action(
                "win.start-processing", [SHORTCUTS.get("start-processing", "<Control>Return")]
            )
            self.set_accels_for_action(
                "win.cancel-processing", [SHORTCUTS.get("cancel-processing", "Escape")]
            )
            self.set_accels_for_action(
                "win.remove-all-files", [SHORTCUTS.get("remove-all-files", "<Control>r")]
            )
            self.set_accels_for_action(
                "win.paste-clipboard", [SHORTCUTS.get("paste-clipboard", "<Control>v")]
            )

            logger.info("Keyboard shortcuts configured successfully")
        except Exception as e:
            logger.error(f"Failed to setup keyboard shortcuts: {e}")

    def on_handle_local_options(self, app: Adw.Application, options: GLib.VariantDict) -> int:
        """Handle command line options.

        Args:
            app: The application
            options: Command line options

        Returns:
            Integer value indicating if processing should continue
        """
        if options.contains("version"):
            print(f"{APP_NAME} {APP_VERSION}")
            return 0  # Exit successfully

        if options.contains("edit"):
            self._edit_mode = True

        return -1  # Continue processing

    def on_activate(self, app: Adw.Application) -> None:
        """Callback for application activation.

        Args:
            app: The application instance
        """
        try:
            # Ensure configuration directory exists
            init_config()

            # Load custom CSS
            load_css()

            # --edit mode without files: open standalone editor
            if self._edit_mode:
                from bigocrpdf.ui.pdf_editor.editor_window import PDFEditorWindow

                win = PDFEditorWindow(
                    application=app,
                    standalone=True,
                )
                win.present()
                logger.info("Opened standalone PDF editor (no file)")
                return

            # Check if we already have a window open
            win = self.get_active_window()
            if not win:
                # Create the main window
                win = BigOcrPdfWindow(app)

            # Show the window
            win.present()

            # Check if we should show the welcome dialog
            if hasattr(win, "should_show_welcome_dialog") and win.should_show_welcome_dialog():
                # Use a small delay to ensure the window is fully drawn
                GLib.timeout_add(300, lambda: win.show_welcome_dialog())

            # Check for resumable session (after welcome dialog)
            if hasattr(win, "check_resumable_session"):
                GLib.timeout_add(500, lambda: win.check_resumable_session())

            logger.info(_("Application started successfully"))

        except Exception as e:
            logger.error(f"{_('Error activating application')}: {e}")
            error_dialog = Gtk.AlertDialog()
            error_dialog.set_message(_("Error starting application"))
            error_dialog.set_detail(str(e))
            error_dialog.show()

    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

    @staticmethod
    def _categorize_files(files) -> tuple[list[str], list[str]]:
        """Split GFile objects into (pdf_paths, image_paths)."""
        import os

        pdf_paths: list[str] = []
        image_paths: list[str] = []
        for gfile in files:
            path = gfile.get_path()
            if path:
                ext = os.path.splitext(path)[1].lower()
                if ext in BigOcrPdfApp._IMAGE_EXTENSIONS:
                    image_paths.append(path)
                else:
                    pdf_paths.append(path)
        return pdf_paths, image_paths

    def _open_edit_mode(self, app, pdf_paths, image_paths):
        """Handle --edit mode file opening."""
        if pdf_paths:
            from bigocrpdf.ui.pdf_editor.editor_window import PDFEditorWindow

            for pdf_path in pdf_paths:
                win = PDFEditorWindow(application=app, pdf_path=pdf_path, standalone=True)
                win.present()
                logger.info(f"Opened PDF editor for: {pdf_path}")

            if image_paths:
                win = self.get_active_window()
                if win and hasattr(win, "_add_files_to_document"):
                    GLib.timeout_add(200, lambda: win._add_files_to_document(image_paths) or False)
            return

        if image_paths:
            self._open_images_in_editor(app, image_paths)

    def _open_normal_mode(self, app, pdf_paths, image_paths):
        """Handle normal mode file opening."""
        if image_paths and not pdf_paths:
            win = ImageOcrWindow(app, image_path=image_paths[0])
            win.present()
            logger.info(f"Opened image OCR window with: {image_paths[0]}")
            return

        win = self.get_active_window()
        if not win or isinstance(win, ImageOcrWindow):
            win = BigOcrPdfWindow(app)
        win.present()

        if pdf_paths:

            def add_files_when_ready():
                try:
                    if hasattr(win, "settings"):
                        added = win.settings.add_files(pdf_paths)
                        if added > 0:
                            logger.info(f"Added {added} file(s) from command line")
                            if hasattr(win, "update_file_info"):
                                win.update_file_info()
                except Exception as e:
                    logger.error(f"Error adding files: {e}")
                return False

            GLib.timeout_add(100, add_files_when_ready)

    def on_open(self, app: Adw.Application, files: list, n_files: int, _hint: str) -> None:
        """Callback for opening files from command line or file manager."""
        try:
            load_css()
            pdf_paths, image_paths = self._categorize_files(files)

            if self._edit_mode:
                self._open_edit_mode(app, pdf_paths, image_paths)
            else:
                self._open_normal_mode(app, pdf_paths, image_paths)

            logger.info(_("Opened {0} file(s)").format(n_files))

        except Exception as e:
            logger.error(f"{_('Error opening files')}: {e}")

    def _open_images_in_editor(self, app: Adw.Application, image_paths: list[str]) -> None:
        """Open images in the PDF editor to create a new PDF.

        Converts images to a temporary PDF and opens the editor.

        Args:
            app: The application instance
            image_paths: List of image file paths
        """
        import os
        import tempfile

        try:
            from bigocrpdf.ui.pdf_editor.thumbnail_renderer import get_thumbnail_renderer

            get_thumbnail_renderer()  # Initialize singleton

            # Create a temporary PDF from the first image to bootstrap the editor
            first_path = image_paths[0]
            fd, tmp_pdf = tempfile.mkstemp(suffix=".pdf", prefix="bigocr_images_")
            os.close(fd)

            # Use pikepdf + Pillow to create a minimal PDF from the first image
            from PIL import Image as PILImage

            img = PILImage.open(first_path)
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            img.save(tmp_pdf, "PDF", resolution=150)
            img.close()

            from bigocrpdf.ui.pdf_editor.editor_window import PDFEditorWindow

            win = PDFEditorWindow(
                application=app,
                pdf_path=tmp_pdf,
                standalone=True,
            )
            win.present()

            # If there are additional images, add them after the window is ready
            if len(image_paths) > 1:
                remaining = image_paths[1:]

                def add_remaining():
                    win._add_files_to_document(remaining)
                    return False

                GLib.timeout_add(500, add_remaining)

            logger.info(f"Opened PDF editor with {len(image_paths)} image(s)")

        except Exception as e:
            logger.error(f"Failed to open images in editor: {e}")

    def _standalone_editor_save(self, doc, original_path: str) -> None:
        """Save callback when editor is used in standalone mode.

        Args:
            doc: The PDFDocument with changes
            original_path: Original file path
        """
        import os
        import shutil
        import tempfile

        from bigocrpdf.ui.pdf_editor.page_operations import apply_changes_to_pdf

        fd, tmp = tempfile.mkstemp(suffix=".pdf", prefix="bigocr_edit_")
        os.close(fd)

        try:
            if apply_changes_to_pdf(doc, tmp):
                shutil.move(tmp, original_path)
                logger.info("Saved edited PDF: %s", original_path)
            else:
                logger.error("Failed to save PDF in standalone editor mode")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def _on_shortcuts_action(self, _action: Gio.SimpleAction, _param: Any) -> None:
        """Show the keyboard shortcuts dialog."""
        win = self.get_active_window()
        shortcuts_win = self._build_shortcuts_window()
        shortcuts_win.set_transient_for(win)
        shortcuts_win.present()

    def _build_shortcuts_window(self) -> Gtk.ShortcutsWindow:
        """Build a Gtk.ShortcutsWindow with categorized shortcut descriptions."""
        window = Gtk.ShortcutsWindow()

        # Main section
        section = Gtk.ShortcutsSection(section_name="main", title=_("Shortcuts"))
        section.set_visible(True)

        # File group
        file_group = Gtk.ShortcutsGroup(title=_("File"))
        file_group.set_visible(True)
        for accel, title in [
            (SHORTCUTS.get("add-files", "<Control>o"), _("Add files")),
            (SHORTCUTS.get("paste-clipboard", "<Control>v"), _("Paste from clipboard")),
            (SHORTCUTS.get("quit", "<Control>q"), _("Quit")),
        ]:
            sc = Gtk.ShortcutsShortcut(
                shortcut_type=Gtk.ShortcutType.ACCELERATOR,
                accelerator=accel,
                title=title,
            )
            sc.set_visible(True)
            file_group.append(sc)
        section.append(file_group)

        # Processing group
        proc_group = Gtk.ShortcutsGroup(title=_("Processing"))
        proc_group.set_visible(True)
        for accel, title in [
            (SHORTCUTS.get("start-processing", "<Control>Return"), _("Start OCR")),
            (SHORTCUTS.get("cancel-processing", "Escape"), _("Cancel processing")),
            (SHORTCUTS.get("remove-all-files", "<Control>r"), _("Clear file queue")),
        ]:
            sc = Gtk.ShortcutsShortcut(
                shortcut_type=Gtk.ShortcutType.ACCELERATOR,
                accelerator=accel,
                title=title,
            )
            sc.set_visible(True)
            proc_group.append(sc)
        section.append(proc_group)

        # General group
        gen_group = Gtk.ShortcutsGroup(title=_("General"))
        gen_group.set_visible(True)
        for accel, title in [
            ("<Control>question", _("Keyboard shortcuts")),
            (SHORTCUTS.get("about", "F1"), _("About")),
        ]:
            sc = Gtk.ShortcutsShortcut(
                shortcut_type=Gtk.ShortcutType.ACCELERATOR,
                accelerator=accel,
                title=title,
            )
            sc.set_visible(True)
            gen_group.append(sc)
        section.append(gen_group)

        window.set_child(section)
        return window

    def on_about_action(self, _action: Gio.SimpleAction, _param: Any) -> None:
        """Show about dialog.

        Args:
            _action: The action that triggered this callback (unused)
            _param: Action parameters (unused)
        """
        # Get active window as the parent
        win = self.get_active_window()

        # Create an about dialog following GNOME guidelines
        about = Adw.AboutDialog()
        about.set_application_name(APP_NAME)
        about.set_version(APP_VERSION)
        about.set_developer_name(_("BigLinux Team"))
        about.set_license_type(Gtk.License.GPL_3_0)
        about.set_comments(get_app_description())
        about.set_website(APP_WEBSITE)
        about.set_issue_url(APP_ISSUES)

        # Legal information
        about.add_legal_section(
            _("Interface"),
            None,
            Gtk.License.GPL_3_0,
            None,
        )
        about.add_legal_section(
            _("Third-party Components"),
            _(
                "The OCR engine and other libraries used by this application "
                "are independent projects, each distributed under its own license."
            ),
            Gtk.License.CUSTOM,
            None,
        )

        # Use app icon for the about dialog
        about.set_application_icon(APP_ICON_NAME)

        # Add credits
        about.add_credit_section(_("Developers"), APP_DEVELOPERS)

        # Acknowledge base projects
        about.add_credit_section(
            _("Powered by"),
            [
                "RapidOCR https://github.com/RapidAI/RapidOCR",
                "PaddleOCR (PP-OCRv5) https://github.com/PaddlePaddle/PaddleOCR",
                "OpenCV https://opencv.org",
                "OpenVINO https://github.com/openvinotoolkit/openvino",
                "pikepdf https://github.com/pikepdf/pikepdf",
                "Pillow https://python-pillow.org",
            ],
        )

        # Show the about dialog
        about.present(win)

    def on_image_ocr_action(self, _action: Gio.SimpleAction, _param: Any) -> None:
        """Open the independent Image OCR window.

        Args:
            _action: The action that triggered this callback
            _param: Action parameters
        """
        win = ImageOcrWindow(self)
        win.present()
