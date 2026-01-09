"""
BigOcrPdf - Screen Capture OCR Service

This module provides screen capture and OCR functionality using external tools
(spectacle, gnome-screenshot, flameshot) and Tesseract OCR.
"""

import os
import subprocess
import tempfile
import threading
from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class ScreenCaptureService:
    """Service to capture screen regions and extract text using OCR."""

    def __init__(self, parent_window: Gtk.Window | None = None) -> None:
        """Initialize the screen capture service.

        Args:
            parent_window: Optional parent window for dialogs
        """
        self._parent_window = parent_window
        self._pending_callback: Callable[[str | None, str | None], None] | None = None
        self._pending_processing_callback: Callable[[], None] | None = None

    def set_parent_window(self, window: Gtk.Window) -> None:
        """Set the parent window for dialogs.

        Args:
            window: The parent window
        """
        self._parent_window = window

    def process_image_file(
        self,
        image_path: str,
        callback: Callable[[str | None, str | None], None],
        on_processing: Callable[[], None] | None = None,
        lang: str = "eng",
        psm: int = 3,
        oem: int = 3,
    ) -> None:
        """Process an existing image file and extract text.

        Args:
            image_path: Path to the image file
            callback: Callback function to receive the result
            on_processing: Optional callback invoked when processing starts
            lang: Language code for OCR (default: "eng")
            psm: Page segmentation mode (default: 3 - fully automatic)
            oem: OCR engine mode (default: 3 - default based on available)
        """
        self._pending_callback = callback
        self._pending_processing_callback = on_processing

        thread = threading.Thread(target=self._run_image_process, args=(image_path, lang, psm, oem))
        thread.daemon = True
        thread.start()

    def _run_image_process(
        self, image_path: str, lang: str = "eng", psm: int = 3, oem: int = 3
    ) -> None:
        """Execute the image processing in a thread."""
        self._invoke_processing_callback()
        text = self.extract_text_from_image(image_path, lang, psm, oem)
        self._invoke_callback(text, None)

    def capture_screen_region(
        self,
        callback: Callable[[str | None, str | None], None],
        on_processing: Callable[[], None] | None = None,
        lang: str = "eng",
        psm: int = 3,
        oem: int = 3,
    ) -> None:
        """Capture a region of the screen and extract text from it.

        Args:
            callback: Callback function to receive the result (text, error)
            on_processing: Optional callback invoked when processing starts
            lang: Language code for OCR (default: "eng")
            psm: Page segmentation mode (default: 3 - fully automatic)
            oem: OCR engine mode (default: 3 - default based on available)
        """
        self._pending_callback = callback
        self._pending_processing_callback = on_processing

        # Run capture in a separate thread to avoid freezing the UI
        thread = threading.Thread(target=self._run_capture_thread, args=(lang, psm, oem))
        thread.daemon = True
        thread.start()

    def _run_capture_thread(self, lang: str, psm: int = 3, oem: int = 3) -> None:
        """Execute the capture and OCR process in a thread."""
        try:
            # Generate a temporary file path
            fd, temp_path = tempfile.mkstemp(suffix=".png", prefix="bigocrpdf_capture_")
            os.close(fd)

            # Capture screen
            tool_executed = self._capture_with_cli_tools(temp_path)

            if not tool_executed:
                self._cleanup_temp_file(temp_path)
                self._invoke_callback(
                    None,
                    _(
                        "No screenshot tool available. Please install spectacle, gnome-screenshot, or flameshot."
                    ),
                )
                return

            # Check if file has content (i.e., screenshot was actually taken, not just cancelled)
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                # Invoke processing callback (in main thread)
                self._invoke_processing_callback()

                # Extract text
                text = self.extract_text_from_image(temp_path, lang, psm, oem)
                self._cleanup_temp_file(temp_path)
                self._invoke_callback(text, None)
            else:
                # Likely cancelled by user
                self._cleanup_temp_file(temp_path)
                self._invoke_callback(None, _("Screenshot was cancelled"))

        except Exception as e:
            logger.error(f"Screenshot capture error: {e}")
            self._invoke_callback(None, _(f"An unexpected error occurred during capture: {e}"))

    def _capture_with_cli_tools(self, temp_path: str) -> bool:
        """Capture screen using CLI tools (spectacle, gnome-screenshot, flameshot).

        Args:
            temp_path: The path where the screenshot should be saved.

        Returns:
            True if a tool was found and executed (screenshot taken or cancelled by user),
            False if no tool was found.
        """
        try:
            # Prioritize tools based on likely environment
            desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()

            # Default order
            commands = []

            if "kde" in desktop:
                # On KDE, spectacle is the native tool
                commands.append(["spectacle", "-r", "-b", "-n", "-o", temp_path])
                commands.append(["flameshot", "gui", "--raw"])
                commands.append(["gnome-screenshot", "-a", "-f", temp_path])
            else:
                commands.append(["gnome-screenshot", "-a", "-f", temp_path])
                commands.append(["flameshot", "gui", "--raw"])
                commands.append(["spectacle", "-r", "-b", "-n", "-o", temp_path])

            tool_found_and_executed = False
            for cmd in commands:
                try:
                    if cmd[0] == "flameshot":
                        # Flameshot outputs to stdout
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            timeout=60,
                        )
                        if result.returncode == 0 and result.stdout:
                            with open(temp_path, "wb") as f:
                                f.write(result.stdout)
                            tool_found_and_executed = True
                            break
                        elif result.returncode != 0:
                            # Flameshot cancelled or failed
                            logger.debug(
                                f"Flameshot exited with code {result.returncode}: {result.stderr.decode().strip()}"
                            )
                            tool_found_and_executed = True  # Tool was executed, even if cancelled
                            break
                    else:
                        tool_name = cmd[0]
                        # Check availability first
                        if subprocess.call(["which", tool_name], stdout=subprocess.DEVNULL) != 0:
                            continue

                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            timeout=60,
                        )
                        if result.returncode == 0:
                            tool_found_and_executed = True
                            break
                        elif result.returncode != 0:
                            # Tool failed or cancelled
                            logger.debug(
                                f"{tool_name} exited with code {result.returncode}: {result.stderr.decode().strip()}"
                            )
                            tool_found_and_executed = True  # Tool was executed, even if cancelled
                            break

                except FileNotFoundError:
                    continue
                except subprocess.TimeoutExpired:
                    logger.warning(f"Screenshot tool {cmd[0]} timed out.")
                    continue
                except Exception as e:
                    logger.warning(f"Error running screenshot tool {cmd[0]}: {e}")
                    continue

            return tool_found_and_executed

        except Exception as e:
            logger.error(f"Screenshot capture error: {e}")
            return False

    def extract_text_from_image(
        self, image_path: str, lang: str = "eng", psm: int = 3, oem: int = 3
    ) -> str | None:
        """Extract text from an image using Tesseract OCR.

        Args:
            image_path: Path to the image file
            lang: Language to use for OCR (default: "eng")
            psm: Page segmentation mode (default: 3 - fully automatic)
            oem: OCR engine mode (default: 3 - default based on available)

        Returns:
            Extracted text or None on error
        """
        try:
            # Check if tesseract is installed
            if not self.check_tesseract():
                logger.error("Tesseract not found or not in PATH.")
                self._invoke_callback(None, _("Tesseract OCR engine not found. Please install it."))
                return None

            # Direct tesseract execution with psm and oem
            args = [
                "tesseract",
                image_path,
                "stdout",
                "-l",
                lang,
                "--psm",
                str(psm),
                "--oem",
                str(oem),
            ]

            logger.info(f"Executing OCR: {' '.join(args)}")
            result = subprocess.run(args, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                text = result.stdout.strip()
                if text:
                    return text
                return _("No text found in the selected region")
            else:
                logger.error(
                    f"Tesseract direct execution failed with code {result.returncode}: {result.stderr.strip()}"
                )
                # Try to return something if stdout has content even with error
                if result.stdout:
                    return result.stdout.strip()

                # Check for common errors
                err_msg = result.stderr.strip()
                if "eng" in err_msg and "not found" in err_msg:
                    self._invoke_callback(
                        None,
                        _(
                            f"Language data for '{lang}' not found. Please install tesseract-data-{lang}."
                        ),
                    )
                else:
                    self._invoke_callback(None, _(f"OCR failed: {err_msg}"))
                return None

        except FileNotFoundError:
            logger.error("Tesseract not installed")
            return None
        except subprocess.TimeoutExpired:
            logger.error("Tesseract OCR timed out")
            return None
        except Exception as e:
            logger.error(f"OCR processing error: {e}")
            return None

        return None

    def _cleanup_temp_file(self, path: str) -> None:
        """Clean up a temporary file.

        Args:
            path: Path to the file to delete
        """
        try:
            if path and os.path.exists(path):
                os.unlink(path)
        except Exception as e:
            logger.debug(f"Could not remove temp file {path}: {e}")

    def _invoke_callback(self, text: str | None, error: str | None) -> None:
        """Invoke the pending callback with the result.

        Args:
            text: Extracted text or None
            error: Error message or None
        """
        if self._pending_callback:
            # Use idle_add to ensure we're in the main thread
            GLib.idle_add(self._pending_callback, text, error)

            # We don't clear callback immediately if we want to support multiple calls?
            # But here it's one-shot.
            # However, since we are in a thread, clearing it here (which runs in thread)
            # vs idle_add (runs in main) is tricky.
            # Wrapper function for idle_add:
            def callback_wrapper():
                if self._pending_callback:
                    self._pending_callback(text, error)
                    self._pending_callback = None
                return False

            GLib.idle_add(callback_wrapper)

    def _invoke_processing_callback(self) -> None:
        """Invoke the processing callback."""
        if self._pending_processing_callback:

            def callback_wrapper():
                if self._pending_processing_callback:
                    self._pending_processing_callback()
                    # Keep it? usually one shot.
                    self._pending_processing_callback = None
                return False

            GLib.idle_add(callback_wrapper)

    @staticmethod
    def is_available() -> bool:
        """Check if screen capture is available.

        Returns:
            True if at least one screenshot tool is available
        """
        # Check for screenshot tools
        tools = ["gnome-screenshot", "spectacle", "flameshot"]
        for tool in tools:
            try:
                if subprocess.call(["which", tool], stdout=subprocess.DEVNULL) == 0:
                    return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue

        return False

    @staticmethod
    def check_tesseract() -> bool:
        """Check if Tesseract OCR is installed.

        Returns:
            True if Tesseract is available
        """
        try:
            result = subprocess.run(
                ["tesseract", "--version"],
                capture_output=True,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError):
            return False
