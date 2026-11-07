"""GUI entry points, with the GTK, libadwaita and OCR checks they run first."""

import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from bigocrpdf.utils.i18n import _, setup_i18n

if TYPE_CHECKING:
    from gi.repository import Gio

__version__ = "4.0.0"

# The real floor, not the version we develop against.  Widgets introduced after
# this point are used through bigocrpdf.utils.adw_compat, which falls back when
# they are absent: an AppImage runs on whatever its build container shipped,
# and Ubuntu 24.04 carries GTK 4.14 with libadwaita 1.5.
_MIN_GTK_VERSION = (4, 14)
_MIN_ADW_VERSION = (1, 5)


def run_quitting_on_termination(app: "Gio.Application") -> int:
    """Run ``app``, quitting it through its quit action on SIGTERM and SIGHUP
    as closing its windows does, which cancels a running OCR job.

    Once the main loop has returned nothing would handle a second signal while
    the interpreter waits for a job's thread to finish its cleanup, so either
    signal then removes the temporary files this process tracks, which exit
    handlers would have removed, and ends it with its default action.
    """
    import signal

    import gi
    from gi.repository import GLib

    try:
        gi.require_version("GLibUnix", "2.0")
        from gi.repository import GLibUnix

        signal_add = GLibUnix.signal_add
    except (ImportError, ValueError):  # GLib older than 2.80
        signal_add = GLib.unix_signal_add

    def quit_app(*_args: object) -> bool:
        # A remote instance, which forwards its command line to the running
        # one, has no quit action of its own: it only ends itself.
        action = app.lookup_action("quit")
        if action is None:
            app.quit()
        else:
            action.activate(None)
        return GLib.SOURCE_CONTINUE

    def end_now(signum: int, _frame: object) -> None:
        from bigocrpdf.utils.temp_manager import cleanup_all

        cleanup_all()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    terminations = (signal.SIGTERM, signal.SIGHUP)
    for signum in terminations:
        signal_add(GLib.PRIORITY_DEFAULT, signum, quit_app)
    try:
        return app.run(sys.argv)
    finally:
        for signum in terminations:
            signal.signal(signum, end_now)


@dataclass(frozen=True)
class OcrDependencyState:
    """Resolved startup availability of the OCR engine."""

    is_available: bool
    error: str = ""


def set_process_environment() -> None:
    """Settings this process and its children must have before they import
    anything that reads them.

    Multiprocessing children put their cwd first on sys.path, and file
    managers start the app in the opened file's directory, so child
    interpreters do not import from it. OpenCV reads its decoders' pixel limit
    when it is imported: it gets the one Pillow's checks use.
    """
    from bigocrpdf.services.ocr.config import DEFAULT_MAX_IMAGE_MEGAPIXELS

    os.environ["PYTHONSAFEPATH"] = "1"
    os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(int(DEFAULT_MAX_IMAGE_MEGAPIXELS * 1_000_000))


def _check_ocr_dependencies() -> OcrDependencyState:
    """Whether the OCR engine is installed, and what to tell the user when it is not."""
    from bigocrpdf.services.ocr import ppocr_native
    from bigocrpdf.utils.logger import logger

    if ppocr_native.find_worker() is not None:
        return OcrDependencyState(is_available=True)
    error = _(
        "The OCR engine, {worker}, is not installed, or its model files are missing from {path}."
    ).format(worker=ppocr_native.WORKER, path=ppocr_native.plans_dir())
    logger.error(f"OCR Dependency Error: {error}")
    print(f"\n*** OCR Dependency Error ***\n{error}\n", file=sys.stderr)
    return OcrDependencyState(is_available=False, error=error)


def _check_gtk_dependencies() -> bool:
    """Check if supported GTK dependencies are available.

    Returns:
        True if dependencies are met, False otherwise
    """
    try:
        import gi

        gi.require_version("Gtk", "4.0")
        gi.require_version("Adw", "1")

        from gi.repository import Adw, Gtk

        gtk_version = (
            Gtk.get_major_version(),
            Gtk.get_minor_version(),
            Gtk.get_micro_version(),
        )
        adw_version = (
            Adw.get_major_version(),
            Adw.get_minor_version(),
            Adw.get_micro_version(),
        )

        errors = []
        if gtk_version[:2] < _MIN_GTK_VERSION:
            errors.append(
                f"GTK {_MIN_GTK_VERSION[0]}.{_MIN_GTK_VERSION[1]} or newer is required "
                f"(found {'.'.join(map(str, gtk_version))})"
            )
        if adw_version[:2] < _MIN_ADW_VERSION:
            errors.append(
                f"libadwaita {_MIN_ADW_VERSION[0]}.{_MIN_ADW_VERSION[1]} or newer is required "
                f"(found {'.'.join(map(str, adw_version))})"
            )

        if errors:
            print("Error: Unsupported graphical runtime:", file=sys.stderr)
            for error in errors:
                print(f"  {error}", file=sys.stderr)
            return False

        return True
    except (ImportError, ValueError) as e:
        # We can't use translations yet as dependencies are missing
        print(f"Error: Missing dependencies: {e}", file=sys.stderr)
        print(
            f"Please install GTK {_MIN_GTK_VERSION[0]}.{_MIN_GTK_VERSION[1]} or newer "
            f"and libadwaita {_MIN_ADW_VERSION[0]}.{_MIN_ADW_VERSION[1]} or newer",
            file=sys.stderr,
        )
        return False


def main() -> int:
    """Main entry point for the application.

    Returns:
        The application exit code.
    """
    set_process_environment()
    # Configure locale before building translated UI.
    setup_i18n()

    from bigocrpdf.config import (
        CONFIG_DIR,
        SELECTED_FILE_PATH,
        setup_environment,
    )
    from bigocrpdf.utils.logger import logger, setup_logger

    # Parse arguments before configuring the requested log level.
    args = setup_environment()
    setup_logger()

    # Check for image mode flag
    if getattr(args, "image_mode", False):
        if "--image-mode" in sys.argv:
            sys.argv.remove("--image-mode")
        return main_image()

    # Check GTK first as we need it for UI
    if not _check_gtk_dependencies():
        return 1

    from bigocrpdf.application import BigOcrPdfApp

    # The GUI starts without the OCR engine and shows why it is missing.
    ocr_dependency = _check_ocr_dependencies()

    os.makedirs(CONFIG_DIR, exist_ok=True)

    # Always start with a clean file queue
    try:
        if os.path.exists(SELECTED_FILE_PATH):
            os.remove(SELECTED_FILE_PATH)
            logger.info("Cleared file queue at startup")
    except Exception as e:
        logger.error(f"Error clearing file queue: {e}")

    # Run application
    try:
        app = BigOcrPdfApp(ocr_dependency=ocr_dependency)

        if hasattr(args, "files") and args.files:
            logger.debug(f"Files provided in arguments: {args.files}")

        # Run the application
        return run_quitting_on_termination(app)
    except Exception as e:
        logger.error(f"Critical error starting application: {e}")
        return 1


def main_image() -> int:
    """Entry point for the Image OCR application (standalone).

    This uses a separate application_id for proper Wayland taskbar grouping.

    Returns:
        The application exit code.
    """
    set_process_environment()
    setup_i18n()

    from bigocrpdf.utils.logger import logger, setup_logger

    setup_logger()
    if not _check_gtk_dependencies():
        return 1

    from bigocrpdf.image_application import ImageOcrApp

    ocr_dependency = _check_ocr_dependencies()

    try:
        app = ImageOcrApp(ocr_dependency=ocr_dependency)
        return run_quitting_on_termination(app)
    except Exception as e:
        logger.error(f"Critical error starting Image OCR: {e}")
        return 1


__all__ = [
    "OcrDependencyState",
    "main",
    "main_image",
    "__version__",
    "setup_i18n",
]
