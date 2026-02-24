"""
BigOcrPdf - Screen Capture OCR Service

This module provides screen capture and OCR functionality using external tools
(spectacle, gnome-screenshot, flameshot) and RapidOCR PP-OCRv5 for text extraction.
"""

import json
import os
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

import cv2
import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from bigocrpdf.services.rapidocr_service.config import OCRConfig, OCRResult
from bigocrpdf.services.rapidocr_service.preprocessor import ImagePreprocessor
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class ScreenCaptureService:
    """Service to capture screen regions and extract text using RapidOCR."""

    def __init__(self, parent_window: Gtk.Window | None = None) -> None:
        """Initialize the screen capture service.

        Args:
            parent_window: Optional parent window for dialogs
        """
        self._parent_window = parent_window
        self._pending_callback: Callable[[str | None, str | None], None] | None = None
        self._pending_processing_callback: Callable[[], None] | None = None

    def process_image_file(
        self,
        image_path: str,
        callback: Callable[[str | None, str | None], None],
        on_processing: Callable[[], None] | None = None,
        language: str = "latin",
    ) -> None:
        """Process an existing image file and extract text using RapidOCR.

        Args:
            image_path: Path to the image file
            callback: Callback function to receive the result (text, error)
            on_processing: Optional callback invoked when processing starts
            language: Language/script code for OCR (default: "latin")
        """
        self._pending_callback = callback
        self._pending_processing_callback = on_processing

        thread = threading.Thread(
            target=self._run_image_process,
            args=(image_path, language),
        )
        thread.daemon = True
        thread.start()

    def _run_image_process(self, image_path: str, language: str = "latin") -> None:
        """Execute the image processing in a thread."""
        self._invoke_processing_callback()
        text = self.extract_text_from_image(image_path, language)
        self._invoke_callback(text, None)

    def capture_screen_region(
        self,
        callback: Callable[[str | None, str | None], None],
        on_processing: Callable[[], None] | None = None,
        language: str = "latin",
    ) -> None:
        """Capture a region of the screen and extract text from it.

        Args:
            callback: Callback function to receive the result (text, error)
            on_processing: Optional callback invoked when processing starts
            language: Language/script code for OCR (default: "latin")
        """
        self._pending_callback = callback
        self._pending_processing_callback = on_processing

        thread = threading.Thread(
            target=self._run_capture_thread,
            args=(language,),
        )
        thread.daemon = True
        thread.start()

    def _run_capture_thread(self, language: str) -> None:
        """Execute the capture and OCR process in a thread."""
        temp_path = None
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
                        "No screenshot tool available. Please install spectacle, "
                        "gnome-screenshot, or flameshot."
                    ),
                )
                return

            # Check if file has content (screenshot was taken, not cancelled)
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                self._invoke_processing_callback()

                text = self.extract_text_from_image(temp_path, language)
                self._cleanup_temp_file(temp_path)
                self._invoke_callback(text, None)
            else:
                self._cleanup_temp_file(temp_path)
                self._invoke_callback(None, _("Screenshot was cancelled"))

        except Exception as e:
            if temp_path:
                self._cleanup_temp_file(temp_path)
            logger.error(f"Screenshot capture error: {e}")
            self._invoke_callback(
                None, _("An unexpected error occurred during capture: {0}").format(e)
            )

    # ── Screenshot Capture ──────────────────────────────────────────────

    def _capture_with_cli_tools(self, temp_path: str) -> bool:
        """Capture screen using CLI tools (spectacle, gnome-screenshot, flameshot).

        Args:
            temp_path: The path where the screenshot should be saved.

        Returns:
            True if a tool was found and executed, False if no tool was found.
        """
        try:
            commands = self._get_screenshot_commands(temp_path)
            return self._try_screenshot_tools(commands, temp_path)
        except Exception as e:
            logger.error(f"Screenshot capture error: {e}")
            return False

    def _get_screenshot_commands(self, temp_path: str) -> list[list[str]]:
        """Get ordered list of screenshot commands based on desktop environment.

        Args:
            temp_path: The path where the screenshot should be saved.

        Returns:
            List of command arrays to try in order.
        """
        desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()

        if "kde" in desktop:
            return [
                ["spectacle", "-r", "-b", "-n", "-o", temp_path],
                ["flameshot", "gui", "--raw"],
                ["gnome-screenshot", "-a", "-f", temp_path],
            ]
        return [
            ["gnome-screenshot", "-a", "-f", temp_path],
            ["flameshot", "gui", "--raw"],
            ["spectacle", "-r", "-b", "-n", "-o", temp_path],
        ]

    def _try_screenshot_tools(self, commands: list[list[str]], temp_path: str) -> bool:
        """Try each screenshot tool until one succeeds or is executed.

        Args:
            commands: List of command arrays to try.
            temp_path: The path where the screenshot should be saved.

        Returns:
            True if a tool was executed, False if no tool was found.
        """
        for cmd in commands:
            result = self._try_single_tool(cmd, temp_path)
            if result is not None:
                return result
        return False

    def _try_single_tool(self, cmd: list[str], temp_path: str) -> bool | None:
        """Try to execute a single screenshot tool.

        Args:
            cmd: Command array to execute.
            temp_path: The path where the screenshot should be saved.

        Returns:
            True if tool executed successfully or was cancelled,
            None if tool should be skipped (not available or error).
        """
        try:
            if cmd[0] == "flameshot":
                return self._run_flameshot(cmd, temp_path)
            return self._run_standard_tool(cmd)
        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired:
            logger.warning(f"Screenshot tool {cmd[0]} timed out.")
            return None
        except Exception as e:
            logger.warning(f"Error running screenshot tool {cmd[0]}: {e}")
            return None

    def _run_flameshot(self, cmd: list[str], temp_path: str) -> bool:
        """Run flameshot and save output to file.

        Args:
            cmd: Flameshot command array.
            temp_path: The path where the screenshot should be saved.

        Returns:
            True (flameshot was executed, regardless of success/cancel).
        """
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode == 0 and result.stdout:
            with open(temp_path, "wb") as f:
                f.write(result.stdout)
        else:
            logger.debug(
                f"Flameshot exited with code {result.returncode}: {result.stderr.decode().strip()}"
            )
        return True

    def _run_standard_tool(self, cmd: list[str]) -> bool | None:
        """Run a standard screenshot tool (spectacle, gnome-screenshot).

        Args:
            cmd: Command array to execute.

        Returns:
            True if tool was executed, None if tool not available.
        """
        tool_name = cmd[0]
        if subprocess.call(["which", tool_name], stdout=subprocess.DEVNULL) != 0:
            return None

        result = subprocess.run(cmd, capture_output=True, timeout=60)
        if result.returncode != 0:
            logger.debug(
                f"{tool_name} exited with code {result.returncode}: "
                f"{result.stderr.decode().strip()}"
            )
        return True

    # ── RapidOCR Image Processing ───────────────────────────────────────

    def extract_text_from_image(
        self,
        image_path: str,
        language: str = "latin",
    ) -> str | None:
        """Extract text from an image using RapidOCR PP-OCRv5.

        Applies geometric corrections (deskew, orientation detection, border trimming)
        before running OCR for optimal accuracy with photographed documents.

        Args:
            image_path: Path to the image file
            language: Language/script code for OCR (default: "latin")

        Returns:
            Extracted text or None on error
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                self._invoke_callback(None, _("Could not load image file."))
                return None

            # Create OCR config with appropriate defaults
            config = OCRConfig(language=language)

            # Apply geometric corrections (deskew, orientation, border trim)
            # Color enhancements stay OFF (PP-OCRv5 works best without)
            preprocessor = ImagePreprocessor(config)
            img = preprocessor.process(img)

            # Write preprocessed image to temp file for OCR worker subprocess
            fd, temp_img_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            cv2.imwrite(temp_img_path, img)

            # Get image dimensions for text formatting
            h, w = img.shape[:2]

            try:
                # Run OCR via subprocess (avoids GTK/ONNX Runtime conflicts)
                cmd = self._build_ocr_command(temp_img_path, config)
                logger.info(f"Running image OCR: language={language}")

                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

                if proc.returncode != 0:
                    logger.error(f"OCR subprocess failed: {proc.stderr}")
                    self._invoke_callback(None, _("OCR processing failed."))
                    return None

                # Parse OCR results
                results = self._parse_ocr_results(proc.stdout)
                if not results:
                    return _("No text found in the image")

                # Format text with reading order and paragraph detection
                text = self._format_text(results, w)
                return text if text.strip() else _("No text found in the image")

            finally:
                try:
                    os.unlink(temp_img_path)
                except Exception:
                    pass

        except FileNotFoundError:
            logger.error("RapidOCR worker not found")
            self._invoke_callback(
                None, _("OCR engine not available. Please check your installation.")
            )
            return None
        except subprocess.TimeoutExpired:
            logger.error("OCR processing timed out")
            self._invoke_callback(None, _("OCR processing timed out."))
            return None
        except Exception as e:
            logger.error(f"OCR processing error: {e}")
            return None

    def _build_ocr_command(self, image_path: str, config: OCRConfig) -> list[str]:
        """Build the OCR subprocess command.

        Args:
            image_path: Path to the image file to process
            config: OCR configuration

        Returns:
            Command list for subprocess.run()
        """
        worker_script = str(Path(__file__).parent / "rapidocr_service" / "ocr_worker.py")
        cpu_count = os.cpu_count() or 4
        optimal_threads = max(2, cpu_count)

        cmd = [
            "python3",
            worker_script,
            image_path,
            "--language",
            config.language,
            "--limit_side_len",
            str(config.detection_limit_side_len),
            "--box-thresh",
            str(config.box_thresh),
            "--unclip-ratio",
            str(config.unclip_ratio),
            "--text-score",
            str(config.text_score_threshold),
            "--score-mode",
            config.score_mode,
            "--threads",
            str(optimal_threads),
        ]

        # Add model paths if available
        for flag, getter in [
            ("--rec-model-path", config.get_rec_model_path),
            ("--rec-keys-path", config.get_rec_keys_path),
            ("--det-model-path", config.get_det_model_path),
            ("--font-path", config.get_font_path),
        ]:
            path = getter()
            if path:
                cmd.extend([flag, str(path)])

        return cmd

    @staticmethod
    def _parse_ocr_results(stdout: str) -> list[OCRResult]:
        """Parse JSON output from OCR subprocess into OCRResult list.

        Args:
            stdout: Raw stdout from OCR worker

        Returns:
            List of OCRResult objects
        """
        try:
            raw = json.loads(stdout.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OCR result: {e}")
            return []

        if raw.get("error"):
            logger.error(f"OCR worker error: {raw['error']}")
            return []

        if not raw or not raw.get("boxes"):
            logger.debug("RapidOCR returned no results")
            return []

        boxes = raw["boxes"]
        txts = raw["txts"]
        scores = raw["scores"]
        logger.info(f"RapidOCR found {len(boxes)} text regions")

        results = []
        for i in range(len(boxes)):
            results.append(
                OCRResult(
                    text=txts[i],
                    box=boxes[i],
                    confidence=scores[i],
                )
            )
        return results

    @staticmethod
    def _format_text(results: list[OCRResult], page_width: float) -> str:
        """Format OCR results into readable text with line breaks and paragraphs.

        Sorts text boxes by reading order (top-to-bottom, left-to-right)
        and inserts appropriate line/paragraph breaks based on vertical spacing.

        Args:
            results: List of OCR results with text and bounding boxes
            page_width: Image width in pixels (used for column detection)

        Returns:
            Formatted text string
        """
        if not results:
            return ""

        # Sort by reading order: top-to-bottom, left-to-right
        def sort_key(r: OCRResult) -> tuple[float, float]:
            ys = [p[1] for p in r.box]
            xs = [p[0] for p in r.box]
            return (min(ys), min(xs))

        sorted_results = sorted(results, key=sort_key)

        text = ""
        prev_y = -1.0
        prev_bottom = -1.0

        for r in sorted_results:
            ys = [p[1] for p in r.box]
            curr_top = min(ys)
            curr_bottom = max(ys)
            curr_h = curr_bottom - curr_top
            center_y = (curr_top + curr_bottom) / 2

            if prev_y != -1:
                # Column break (moved UP significantly)
                if center_y < prev_y - (curr_h * 2):
                    text += "\n\n"
                # Paragraph break (vertical gap > 60% of line height)
                elif (curr_top - prev_bottom) > (curr_h * 0.6):
                    text += "\n\n"
                # Line break (moved DOWN past previous bottom)
                elif center_y > prev_bottom:
                    text += "\n"
                elif center_y > prev_y + (curr_h * 0.5):
                    text += "\n"
                else:
                    text += " "

            text += r.text
            prev_y = center_y
            prev_bottom = curr_bottom

        return text

    # ── Callback Helpers ────────────────────────────────────────────────

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
                    self._pending_processing_callback = None
                return False

            GLib.idle_add(callback_wrapper)
