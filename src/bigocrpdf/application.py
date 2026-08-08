"""
BigOcrPdf - Application Module

allow-noisy-log: --version prints user-facing CLI output.

This module contains the main application class for the BigOcrPdf application.
"""

from typing import Any

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk

from bigocrpdf import OcrDependencyState
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
from bigocrpdf.ui.pdf_editor.thumbnail_renderer import shutdown_thumbnail_renderer
from bigocrpdf.ui.widgets import load_css
from bigocrpdf.utils.adw_compat import build_shortcuts_dialog
from bigocrpdf.utils.i18n import _, ngettext
from bigocrpdf.utils.icons import setup_icons
from bigocrpdf.utils.logger import logger
from bigocrpdf.window import BigOcrPdfWindow


class BigOcrPdfApp(Adw.Application):
    """Application class for BigOcrPdf."""

    def __init__(self, *, ocr_dependency: OcrDependencyState) -> None:
        """Initialize the application."""
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.HANDLES_OPEN)

        self.ocr_dependency = ocr_dependency
        self._edit_mode = False

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
        self.connect("shutdown", self._on_shutdown)

        # Set up application actions
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

    def _prepare_windows_for_shutdown(self) -> None:
        """Release resources owned by every application window."""
        from bigocrpdf.ui.pdf_editor.editor_window import PDFEditorWindow

        for window in self.get_windows():
            if isinstance(window, (BigOcrPdfWindow, ImageOcrWindow)):
                window.prepare_close()
            elif isinstance(window, PDFEditorWindow):
                window._prepare_close()

    def _on_quit_action(self, *_args: object) -> None:
        """Release window resources before explicit application quit."""
        self._prepare_windows_for_shutdown()
        self.quit()

    def _on_shutdown(self, *_args: object) -> None:
        """Release window resources for every shutdown path."""
        self._prepare_windows_for_shutdown()
        shutdown_thumbnail_renderer(wait=True)

    def _setup_keyboard_shortcuts(self) -> None:
        """Set up application-level keyboard shortcuts."""
        self.set_accels_for_action("app.quit", [SHORTCUTS.get("quit", "<Control>q")])
        self.set_accels_for_action("app.about", [SHORTCUTS.get("about", "F1")])
        self.set_accels_for_action("app.shortcuts", ["<Control>question"])
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
            if not isinstance(win, BigOcrPdfWindow):
                # Create the main window
                win = BigOcrPdfWindow(app, ocr_dependency=self.ocr_dependency)

            # Show the window
            win.present()

            if self.ocr_dependency.is_available:
                # Check if we should show the welcome dialog
                if win.actions.welcome.should_show():
                    # Use a small delay to ensure the window is fully drawn
                    GLib.timeout_add(300, lambda: win.actions.welcome.show())

                # Check for resumable session (after welcome dialog)
                GLib.timeout_add(500, lambda: win.actions.sessions.check())
            else:
                GLib.idle_add(win.show_ocr_unavailable_dialog)

            logger.info(_("Application started successfully"))

        except Exception as e:
            logger.error(f"{_('Error activating application')}: {e}")
            error_dialog = Gtk.AlertDialog()
            error_dialog.set_message(_("Error starting application"))
            error_dialog.set_detail(str(e))
            error_dialog.show()

    _IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".avif"}

    @staticmethod
    def _categorize_files(files) -> tuple[list[str], list[str], list[str]]:
        """Split GFile objects into supported PDFs, images, and rejected paths."""
        import os

        pdf_paths: list[str] = []
        image_paths: list[str] = []
        unsupported_paths: list[str] = []
        for gfile in files:
            path = gfile.get_path()
            if path:
                ext = os.path.splitext(path)[1].lower()
                if ext in BigOcrPdfApp._IMAGE_EXTENSIONS:
                    image_paths.append(path)
                elif ext == ".pdf":
                    pdf_paths.append(path)
                else:
                    unsupported_paths.append(path)
        return pdf_paths, image_paths, unsupported_paths

    def _open_edit_mode(self, app, pdf_paths, image_paths):
        """Handle --edit mode file opening."""
        if len(pdf_paths) > 1 and not image_paths:
            self._show_multi_pdf_open_dialog(app, pdf_paths)
            return

        if pdf_paths:
            self._open_pdf_files_individually(app, pdf_paths, image_paths)
            return

        if image_paths:
            self._open_images_in_editor(app, image_paths)

    def _show_multi_pdf_open_dialog(self, app: Adw.Application, pdf_paths: list[str]) -> None:
        """Ask how multiple selected PDFs should be opened in edit mode."""
        from bigocrpdf.ui.pdf_editor.open_options_dialog import MultiPdfOpenDialog

        dialog = MultiPdfOpenDialog(
            application=app,
            file_paths=pdf_paths,
            on_open_individual=lambda paths: self._open_pdf_files_individually(app, paths, []),
            on_open_combined=lambda paths: self._open_pdf_files_combined(app, paths),
        )
        dialog.present()
        logger.info(f"Prompting for multi-PDF edit mode with {len(pdf_paths)} files")

    def _open_pdf_files_individually(
        self, app: Adw.Application, pdf_paths: list[str], image_paths: list[str]
    ) -> None:
        """Open each PDF in its own standalone editor window."""
        from bigocrpdf.ui.pdf_editor.editor_window import PDFEditorWindow

        for pdf_path in pdf_paths:
            win = PDFEditorWindow(application=app, pdf_path=pdf_path, standalone=True)
            win.present()
            logger.info(f"Opened PDF editor for: {pdf_path}")

        if image_paths:
            win = self.get_active_window()
            if isinstance(win, PDFEditorWindow):

                def add_images_to_editor() -> bool:
                    win._add_files_to_document(image_paths)
                    return False

                GLib.timeout_add(200, add_images_to_editor)

    def _open_pdf_files_combined(self, app: Adw.Application, pdf_paths: list[str]) -> None:
        """Open several PDFs in one editor window, preserving the given file order."""
        if not pdf_paths:
            return

        from bigocrpdf.ui.pdf_editor.editor_window import PDFEditorWindow

        first_pdf = pdf_paths[0]
        remaining_pdfs = pdf_paths[1:]
        win = PDFEditorWindow(application=app, pdf_path=first_pdf, standalone=True)
        win.present()

        if remaining_pdfs:

            def add_remaining_files():
                try:
                    added_pages = win._add_files_to_document(remaining_pdfs)
                    if added_pages == 0:
                        logger.error("No pages from the additional PDFs could be imported")
                        return False
                    win.set_title(_("PDF Editor - Combined PDF"))
                    win._filename_label.set_text(
                        ngettext(
                            "Combined PDF ({count} file)",
                            "Combined PDF ({count} files)",
                            len(pdf_paths),
                        ).format(count=len(pdf_paths))
                    )
                    logger.info(f"Opened combined PDF editor with {len(pdf_paths)} files")
                except Exception as e:
                    logger.error(f"Failed to combine PDFs in editor: {e}")
                return False

            GLib.timeout_add(200, add_remaining_files)

    def _open_normal_mode(self, app, pdf_paths, image_paths):
        """Handle normal mode file opening."""
        if image_paths and not pdf_paths:
            win = ImageOcrWindow(
                app,
                image_path=image_paths[0],
                ocr_dependency=self.ocr_dependency,
            )
            win.present()
            if not self.ocr_dependency.is_available:
                GLib.idle_add(win.show_ocr_unavailable_dialog)
            logger.info(f"Opened image OCR window with: {image_paths[0]}")
            return

        win = self.get_active_window()
        if not isinstance(win, BigOcrPdfWindow):
            win = BigOcrPdfWindow(app, ocr_dependency=self.ocr_dependency)
        win.present()
        if not self.ocr_dependency.is_available:
            GLib.idle_add(win.show_ocr_unavailable_dialog)

        input_paths = [*pdf_paths, *image_paths]
        if input_paths:

            def add_files_when_ready():
                try:
                    added = win.settings.add_files(input_paths)
                    if added > 0:
                        logger.info(
                            ngettext(
                                "Added {count} file from command line",
                                "Added {count} files from command line",
                                added,
                            ).format(count=added)
                        )
                        win.ui.update_file_info()
                except Exception as e:
                    logger.error(f"Error adding files: {e}")
                return False

            GLib.timeout_add(100, add_files_when_ready)

    def on_open(self, app: Adw.Application, files: list, n_files: int, _hint: str) -> None:
        """Callback for opening files from command line or file manager."""
        try:
            load_css()
            pdf_paths, image_paths, unsupported_paths = self._categorize_files(files)

            if not pdf_paths and not image_paths:
                self.on_activate(app)

            if self._edit_mode:
                self._open_edit_mode(app, pdf_paths, image_paths)
            else:
                self._open_normal_mode(app, pdf_paths, image_paths)

            opened_count = len(pdf_paths) + len(image_paths)
            if opened_count:
                logger.info(
                    ngettext(
                        "Opened {count} file",
                        "Opened {count} files",
                        opened_count,
                    ).format(count=opened_count)
                )
            if unsupported_paths:
                GLib.idle_add(self._show_unsupported_files_dialog, unsupported_paths)

        except Exception as e:
            logger.error(f"{_('Error opening files')}: {e}")

    def _show_unsupported_files_dialog(self, paths: list[str]) -> bool:
        """Explain which command-line or file-manager inputs were rejected."""
        parent = self.get_active_window()
        if parent is None:
            logger.warning(_("Unsupported file type: {0}").format(", ".join(paths)))
            return False

        dialog = Adw.AlertDialog(
            heading=_("Unsupported file format"),
            body=_("Unsupported file type: {0}").format("\n".join(paths)),
        )
        dialog.add_response("close", _("Close"))
        dialog.set_default_response("close")
        dialog.set_close_response("close")
        dialog.present(parent)
        return False

    def _open_images_in_editor(self, app: Adw.Application, image_paths: list[str]) -> None:
        """Open images in the PDF editor to create a new PDF.

        Converts images to a temporary PDF and opens the editor.

        Args:
            app: The application instance
            image_paths: List of image file paths
        """
        import os

        from bigocrpdf.utils.temp_manager import mkstemp, remove_file

        tmp_pdf: str | None = None
        transferred = False
        try:
            from bigocrpdf.ui.pdf_editor.thumbnail_renderer import get_thumbnail_renderer

            get_thumbnail_renderer()

            # Create a temporary PDF from the first image to bootstrap the editor
            first_path = image_paths[0]
            fd, tmp_pdf = mkstemp(suffix=".pdf", prefix="bigocr_images_")
            os.close(fd)

            # Use pikepdf + Pillow to create a minimal PDF from the first image
            from PIL import Image as PILImage

            with PILImage.open(first_path) as source_image:
                img = source_image
                if source_image.mode in ("RGBA", "LA", "P"):
                    img = source_image.convert("RGB")
                try:
                    # Embed at native pixel resolution (1 px = 1 pt, i.e. 72 DPI) so the
                    # bootstrap cover page matches the pages appended later via
                    # _add_image_page. A non-native resolution (e.g. 150) here, or an
                    # image carrying its own DPI metadata, would make this first page a
                    # different physical size than the rest, which mobile viewers render
                    # at a different scale.
                    img.save(tmp_pdf, "PDF", resolution=72.0)
                finally:
                    if img is not source_image:
                        img.close()

            from bigocrpdf.ui.pdf_editor.editor_window import PDFEditorWindow

            win = PDFEditorWindow(
                application=app,
                pdf_path=tmp_pdf,
                standalone=True,
            )
            win.present()
            transferred = True

            # If there are additional images, add them after the window is ready
            if len(image_paths) > 1:
                remaining = image_paths[1:]

                def add_remaining():
                    added_pages = win._add_files_to_document(remaining)
                    logger.info(f"Added {added_pages} additional image page(s) to the editor")
                    return False

                GLib.timeout_add(500, add_remaining)

            logger.info("Opened PDF editor with the first image")

        except Exception as e:
            if tmp_pdf is not None and not transferred:
                remove_file(tmp_pdf)
            logger.error(f"Failed to open images in editor: {e}")

    def _on_shortcuts_action(self, _action: Gio.SimpleAction, _param: Any) -> None:
        """Show the keyboard shortcuts dialog."""
        self._build_shortcuts_dialog().present(self.get_active_window())

    def _build_shortcuts_dialog(self) -> Adw.Dialog:
        """Build the keyboard shortcuts dialog."""
        groups = (
            (
                _("File"),
                (
                    (_("Add files"), SHORTCUTS.get("add-files", "<Control>o")),
                    (_("Paste from clipboard"), SHORTCUTS.get("paste-clipboard", "<Control>v")),
                    (_("Quit"), SHORTCUTS.get("quit", "<Control>q")),
                ),
            ),
            (
                _("Processing"),
                (
                    (_("Start OCR"), SHORTCUTS.get("start-processing", "<Control>Return")),
                    (_("Cancel processing"), SHORTCUTS.get("cancel-processing", "Escape")),
                    (_("Clear file queue"), SHORTCUTS.get("remove-all-files", "<Control>r")),
                ),
            ),
            (
                _("General"),
                (
                    (_("Keyboard shortcuts"), "<Control>question"),
                    (_("About"), SHORTCUTS.get("about", "F1")),
                ),
            ),
        )

        return build_shortcuts_dialog(groups)

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
                "PaddleOCR (PP-OCRv6) https://github.com/PaddlePaddle/PaddleOCR",
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
        win = ImageOcrWindow(self, ocr_dependency=self.ocr_dependency)
        win.present()
        if not self.ocr_dependency.is_available:
            GLib.idle_add(win.show_ocr_unavailable_dialog)
