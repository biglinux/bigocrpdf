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
    ) -> None:
        """Process an existing image file and extract text.

        Args:
            image_path: Path to the image file
            callback: Callback function to receive the result
            on_processing: Optional callback invoked when processing starts
        """
        self._pending_callback = callback
        self._pending_processing_callback = on_processing

        thread = threading.Thread(target=self._run_image_process, args=(image_path,))
        thread.daemon = True
        thread.start()

    def _run_image_process(self, image_path: str) -> None:
        """Execute the image processing in a thread."""
        self._invoke_processing_callback()
        text = self.extract_text_from_image(image_path)
        self._invoke_callback(text, None)

    def capture_screen_region(
        self,
        callback: Callable[[str | None, str | None], None],
        on_processing: Callable[[], None] | None = None,
    ) -> None:
        """Capture a rectangular region of the screen and extract text.

        The callback receives two arguments:
        - extracted_text: The extracted text, or None on error
        - error_message: Error message if failed, or None on success

        Args:
            callback: Callback function to receive the result
            on_processing: Optional callback invoked when screenshot is taken and OCR starts
        """
        self._pending_callback = callback
        self._pending_processing_callback = on_processing

        # Run capture in a separate thread to avoid freezing UI during screenshot/OCR
        thread = threading.Thread(target=self._run_capture_flow)
        thread.daemon = True
        thread.start()

    def _run_capture_flow(self) -> None:
        """Execute the capture flow in a thread."""
        if not self._capture_with_cli_tools():
            self._invoke_callback(
                None,
                _(
                    "No screenshot tool available. Please install spectacle, gnome-screenshot, or flameshot."
                ),
            )

    def _capture_with_cli_tools(self) -> bool:
        """Capture screen using CLI tools (spectacle, gnome-screenshot, flameshot).

        Returns:
            True if a tool was found and executed (screenshot taken or cancelled by user),
            False if no tool was found.
        """
        try:
            # Create a temporary file for the screenshot
            fd, temp_path = tempfile.mkstemp(suffix=".png", prefix="bigocrpdf_capture_")
            os.close(fd)

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

            tool_found = False
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
                            tool_found = True
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
                        if (
                            result.returncode == 0
                            and os.path.exists(temp_path)
                            and os.path.getsize(temp_path) > 0
                        ):
                            tool_found = True
                            break
                        elif result.returncode != 0:
                            # Tool failed or cancelled
                            continue

                except FileNotFoundError:
                    continue
                except subprocess.TimeoutExpired:
                    continue

            if not tool_found:
                self._cleanup_temp_file(temp_path)
                return False

            # Check if file has content
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                # Invoke processing callback (in main thread)
                self._invoke_processing_callback()

                # Extract text
                text = self.extract_text_from_image(temp_path)
                self._cleanup_temp_file(temp_path)
                self._invoke_callback(text, None)
                return True
            else:
                # Likely cancelled
                self._cleanup_temp_file(temp_path)
                self._invoke_callback(None, _("Screenshot was cancelled"))
                return True

        except Exception as e:
            logger.error(f"Screenshot capture error: {e}")
            return False

    def extract_text_from_image(self, image_path: str) -> str | None:
        """Extract text from an image using Tesseract OCR.

        Args:
            image_path: Path to the image file

        Returns:
            Extracted text or None on error
        """
        try:
            # First try using tesseract directly (faster, no Python dependencies)
            result = subprocess.run(
                ["tesseract", image_path, "stdout", "-l", "eng+por+spa"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                text = result.stdout.strip()
                if text:
                    return text
                return _("No text found in the selected region")

            # If tesseract fails, log the error
            logger.warning(f"Tesseract failed: {result.stderr}")

        except FileNotFoundError:
            logger.error("Tesseract not installed")
            return None
        except subprocess.TimeoutExpired:
            logger.error("Tesseract OCR timed out")
            return None
        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
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
