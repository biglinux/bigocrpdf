"""
BigOcrPdf - OCR API Service

This module provides direct integration with the OCRmyPDF Python API.
"""

import os
import logging
import threading
import multiprocessing
import signal
import subprocess
import time
from typing import Dict, Callable, List, Optional, Any

import ocrmypdf

from utils.logger import logger
from utils.i18n import _

# Constants for process status
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# Constants for progress tracking
PROGRESS_ALMOST_DONE = 0.95
PROGRESS_COMPLETE = 1.0

# Constants for monitor thread
MONITOR_SLEEP_INTERVAL = 0.5


def configure_logging() -> None:
    """Configure logging to use our application logger"""
    # Set up OCRmyPDF logging to use our logger
    ocrmypdf_logger = logging.getLogger("ocrmypdf")
    ocrmypdf_logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    for handler in ocrmypdf_logger.handlers:
        ocrmypdf_logger.removeHandler(handler)

    # Add our handler
    for handler in logger.handlers:
        ocrmypdf_logger.addHandler(handler)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text directly from a PDF file using pdftotext

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text or empty string if extraction failed
    """
    try:
        result = subprocess.run(
            ["pdftotext", pdf_path, "-"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout and len(result.stdout.strip()) > 0:
            logger.info(
                _("Found existing text in PDF, length: {0}").format(len(result.stdout))
            )
            return result.stdout
    except Exception as e:
        logger.warning(_("Failed to extract existing text from PDF: {0}").format(e))
    return ""


def read_text_from_sidecar(sidecar_path: str) -> Optional[str]:
    """Read text from a sidecar file

    Args:
        sidecar_path: Path to the sidecar file

    Returns:
        Text content or None if file couldn't be read
    """
    if not sidecar_path or not os.path.exists(sidecar_path):
        return None

    try:
        with open(sidecar_path, "r", encoding="utf-8") as f:
            text = f.read()
        logger.info(
            _("Read {0} characters of extracted text from sidecar file").format(
                len(text)
            )
        )
        return text
    except Exception as e:
        logger.error(_("Error reading sidecar file: {0}").format(e))
        return None


def setup_sigbus_handler() -> None:
    """Set up SIGBUS handler as recommended in OCRmyPDF docs"""
    if hasattr(signal, "SIGBUS"):

        def handle_sigbus(signum, frame):
            raise RuntimeError(_("SIGBUS received - memory-mapped file access error"))

        signal.signal(signal.SIGBUS, handle_sigbus)


class SimpleProgressIndicator:
    """Simple, reliable progress indicator for OCRmyPDF"""

    def __init__(self, progress_file: str):
        self.progress_file = progress_file
        self.start_time = time.time()
        self.is_running = True
        self.last_update_time = time.time()

    def start_progress_updates(self):
        """Start background thread to update progress smoothly"""

        def update_loop():
            # Initial write to indicate process has started
            try:
                with open(self.progress_file, "w") as f:
                    f.write("0.0")
                logger.debug(f"Initial progress file created: {self.progress_file}")
            except Exception as e:
                logger.error(f"Failed to create initial progress file: {e}")
                return

            while self.is_running:
                try:
                    elapsed = time.time() - self.start_time
                    # More conservative progress that reaches 80% after 120 seconds
                    # This gives OCRmyPDF more time to actually start working
                    progress = min(0.80, elapsed / 120.0)

                    with open(self.progress_file, "w") as f:
                        f.write(f"{progress:.3f}")

                    self.last_update_time = time.time()
                    time.sleep(3)  # Update every 3 seconds for better reliability

                except Exception as e:
                    logger.debug(f"Progress update error: {e}")
                    time.sleep(1)  # Wait before retrying

        self.progress_thread = threading.Thread(target=update_loop, daemon=True)
        self.progress_thread.start()
        logger.debug("Progress indicator thread started")

    def stop_progress_updates(self):
        """Stop background progress updates"""
        self.is_running = False
        if hasattr(self, "progress_thread"):
            self.progress_thread.join(timeout=2)
        logger.debug("Progress indicator thread stopped")

    def set_complete(self):
        """Mark as 100% complete"""
        try:
            with open(self.progress_file, "w") as f:
                f.write("1.0")
            logger.debug("Progress marked as complete")
        except Exception as e:
            logger.debug(f"Failed to mark progress as complete: {e}")

    def is_healthy(self) -> bool:
        """Check if the progress indicator is working properly"""
        try:
            # Check if progress file exists and was updated recently
            if not os.path.exists(self.progress_file):
                return False

            # Check if file was updated in the last 10 seconds
            file_mtime = os.path.getmtime(self.progress_file)
            if time.time() - file_mtime > 10:
                return False

            return True
        except Exception:
            return False


class OcrProcess:
    """Manages an OCR process for a single file"""

    def __init__(
        self,
        input_file: str,
        output_file: str,
        options: Dict[str, Any],
        page_count: int = 1,
    ):
        """Initialize the OCR process

        Args:
            input_file: Path to the input PDF file
            output_file: Path to save the output PDF file
            options: Dictionary of options to pass to OCRmyPDF
            page_count: Number of pages in the input file
        """
        self.input_file = input_file
        self.output_file = output_file
        self.options = options
        self.process = None
        self.status = STATUS_PENDING
        self.progress = 0.0
        self.start_time = None
        self.end_time = None
        self.error = None
        self.page_count = page_count if page_count > 0 else 1  # Avoid division by zero
        self.extracted_text = ""  # Will hold extracted text from the OCR process

    def cleanup_previous_process(self) -> None:
        """Clean up any previous process and reset state"""
        # Terminate any existing process
        if self.process and self.process.is_alive():
            logger.info("Terminating previous OCR process...")
            self.process.terminate()
            try:
                self.process.join(timeout=5)  # Wait up to 5 seconds
            except Exception:
                pass
            if self.process.is_alive():
                logger.warning("Force killing previous OCR process...")
                self.process.kill()

        # Clean up progress file from previous run
        if hasattr(self, "progress_file") and self.progress_file:
            try:
                if os.path.exists(self.progress_file):
                    os.unlink(self.progress_file)
            except Exception as e:
                logger.debug(f"Failed to clean up progress file: {e}")

        # Reset state
        self.process = None
        self.status = STATUS_PENDING
        self.progress = 0.0
        self.start_time = None
        self.end_time = None
        self.error = None
        self.extracted_text = ""

        logger.debug("Previous process cleanup completed")

    def start(self) -> None:
        """Start the OCR process in a separate process with robust validation"""
        # First, clean up any previous process
        self.cleanup_previous_process()

        self.start_time = time.time()
        self.status = STATUS_RUNNING

        # Create temporary file for progress tracking
        import tempfile

        max_retries = 3
        for retry_attempt in range(max_retries):
            try:
                # Create new progress file for each attempt
                self.progress_file = tempfile.mktemp(
                    prefix="ocr_progress_", suffix=".txt"
                )

                # Verify input file before proceeding
                if not os.path.exists(self.input_file):
                    raise FileNotFoundError(f"Input file not found: {self.input_file}")
                if not os.access(self.input_file, os.R_OK):
                    raise PermissionError(f"Cannot read input file: {self.input_file}")

                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

                # Verify output directory is writable
                output_dir = os.path.dirname(self.output_file)
                if not os.access(output_dir, os.W_OK):
                    raise PermissionError(
                        f"Cannot write to output directory: {output_dir}"
                    )

                # Log retry attempt
                if retry_attempt > 0:
                    logger.info(
                        f"Retry attempt {retry_attempt + 1} for OCR process: {os.path.basename(self.input_file)}"
                    )
                else:
                    logger.info(
                        f"Starting OCR process for {os.path.basename(self.input_file)}"
                    )

                # Start the process
                self.process = multiprocessing.Process(
                    target=self._run_ocr,
                    args=(
                        self.input_file,
                        self.output_file,
                        self.options,
                        self.progress_file,
                    ),
                )
                self.process.start()

                # Enhanced process startup validation with more robust checks
                if not self._validate_process_startup():
                    if retry_attempt < max_retries - 1:
                        logger.warning(
                            f"Process startup validation failed, retrying... (attempt {retry_attempt + 1}/{max_retries})"
                        )
                        self._cleanup_failed_attempt()
                        time.sleep(1)  # Wait before retry
                        continue
                    else:
                        raise RuntimeError(
                            "Process startup validation failed after all retries"
                        )

                # If we get here, the process started successfully
                logger.info(
                    _("Successfully started OCR process for {0} (PID: {1})").format(
                        os.path.basename(self.input_file), self.process.pid
                    )
                )
                return

            except Exception as e:
                if retry_attempt < max_retries - 1:
                    logger.warning(
                        f"Failed to start OCR process (attempt {retry_attempt + 1}/{max_retries}): {e}"
                    )
                    self._cleanup_failed_attempt()
                    time.sleep(1)  # Wait before retry
                    continue
                else:
                    # Final attempt failed
                    self.status = STATUS_FAILED
                    self.error = f"Failed to start OCR process after {max_retries} attempts: {str(e)}"
                    logger.error(
                        f"Error starting OCR process for {os.path.basename(self.input_file)}: {self.error}"
                    )

                    # Write error to progress file if possible
                    if hasattr(self, "progress_file"):
                        try:
                            with open(self.progress_file, "w") as f:
                                f.write("ERROR")
                        except Exception as pf_error:
                            logger.debug(
                                f"Could not write to progress file: {pf_error}"
                            )

                    raise

    def _validate_process_startup(self) -> bool:
        """Validate that the OCR process has started successfully

        Returns:
            True if process is running properly, False otherwise
        """
        try:
            startup_timeout = 3.0  # Increased timeout for better reliability
            check_interval = 0.1  # Check every 100ms
            checks = int(startup_timeout / check_interval)

            progress_file_found = False
            process_working = False

            for attempt in range(checks):
                time.sleep(check_interval)

                # Check if process is still alive
                if not self.process.is_alive():
                    if self.process.exitcode is not None:
                        logger.error(
                            f"OCR process exited with code {self.process.exitcode}"
                        )
                        return False
                    else:
                        logger.warning(
                            "OCR process appears to have died without exit code"
                        )
                        return False

                # Check for progress file creation (indicates process is working)
                if not progress_file_found and os.path.exists(self.progress_file):
                    progress_file_found = True
                    logger.debug(
                        f"Progress file created for {os.path.basename(self.input_file)}"
                    )

                # After 1 second, verify the process is actually doing work
                if attempt >= 10:  # After 1 second
                    if progress_file_found:
                        # Read progress file to ensure it contains valid data
                        try:
                            with open(self.progress_file, "r") as f:
                                content = f.read().strip()
                                if content and content != "ERROR":
                                    try:
                                        progress_val = float(content)
                                        if 0 <= progress_val <= 1.0:
                                            process_working = True
                                            logger.debug(
                                                f"Process working correctly, progress: {progress_val}"
                                            )
                                    except ValueError:
                                        logger.debug(
                                            f"Invalid progress value: {content}"
                                        )
                        except Exception as e:
                            logger.debug(f"Could not read progress file: {e}")

                # If we have confirmed the process is working, we can break early
                if process_working:
                    break

            # Final validation
            if not self.process.is_alive():
                logger.error("Process died during startup validation")
                return False

            # Additional check: ensure process has been alive for at least 0.5 seconds
            time.sleep(0.5)
            if not self.process.is_alive():
                logger.error("Process died shortly after apparent successful startup")
                return False

            # Check if we detected the process is working
            if process_working:
                logger.info(
                    f"Process startup validated successfully for {os.path.basename(self.input_file)}"
                )
                return True
            elif progress_file_found:
                logger.info(
                    f"Process started with progress file created for {os.path.basename(self.input_file)}"
                )
                return True
            else:
                logger.warning(
                    f"Process appears alive but no progress file found for {os.path.basename(self.input_file)}"
                )
                # Still consider it successful if process is alive - progress file might come later
                return True

        except Exception as e:
            logger.error(f"Error during process startup validation: {e}")
            return False

    def _cleanup_failed_attempt(self) -> None:
        """Clean up after a failed process start attempt"""
        try:
            # Terminate the process if it exists
            if self.process and self.process.is_alive():
                logger.debug("Terminating failed process attempt")
                self.process.terminate()
                try:
                    self.process.join(timeout=2)
                except Exception:
                    pass
                if self.process.is_alive():
                    self.process.kill()

            # Clean up progress file
            if (
                hasattr(self, "progress_file")
                and self.progress_file
                and os.path.exists(self.progress_file)
            ):
                try:
                    os.unlink(self.progress_file)
                except Exception as e:
                    logger.debug(f"Failed to clean up progress file: {e}")

            # Reset process reference
            self.process = None

        except Exception as e:
            logger.debug(f"Error during failed attempt cleanup: {e}")

    def _run_ocr(
        self,
        input_file: str,
        output_file: str,
        options: Dict[str, Any],
        progress_file: str,
    ) -> None:
        """Run OCRmyPDF in a separate process

        Args:
            input_file: Path to the input PDF file
            output_file: Path to save the output PDF file
            options: Dictionary of options to pass to OCRmyPDF
            progress_file: Path to file for progress communication
        """
        progress_indicator = None
        try:
            # Log that the OCR process has actually started
            logger.info(
                f"OCR subprocess started for {os.path.basename(input_file)} (PID: {os.getpid()})"
            )

            # Configure logging
            configure_logging()

            # Set up SIGBUS handler
            setup_sigbus_handler()

            # Get the sidecar path from options
            sidecar_path = options.get("sidecar", None)

            # Try to extract text from input PDF first
            extracted_text = extract_text_from_pdf(input_file)
            logger.debug(f"Extracted {len(extracted_text)} characters from input PDF")

            # Make a copy of options to modify safely
            ocr_options = options.copy()

            # Set additional options to improve text extraction and reliability
            ocr_options["force_ocr"] = True
            ocr_options["progress_bar"] = False  # We handle progress ourselves

            # Create and start progress indicator
            progress_indicator = SimpleProgressIndicator(progress_file)
            progress_indicator.start_progress_updates()

            # Verify input file exists and is readable
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file not found: {input_file}")
            if not os.access(input_file, os.R_OK):
                raise PermissionError(f"Cannot read input file: {input_file}")

            # Check if output directory exists and is writable
            output_dir = os.path.dirname(output_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            if not os.access(output_dir, os.W_OK):
                raise PermissionError(f"Cannot write to output directory: {output_dir}")

            # Log process health before starting OCR
            logger.info(
                f"Process health check - PID: {os.getpid()}, Progress indicator healthy: {progress_indicator.is_healthy()}"
            )

            logger.info(
                f"Calling ocrmypdf.ocr with options: {list(ocr_options.keys())}"
            )

            # Run OCRmyPDF - this is the critical call
            ocrmypdf.ocr(input_file, output_file, **ocr_options)

            logger.info(
                f"OCRmyPDF completed successfully for {os.path.basename(input_file)}"
            )

            # Mark as complete
            if progress_indicator:
                progress_indicator.set_complete()

            # Read the text from the sidecar file if it exists
            if sidecar_path and os.path.exists(sidecar_path):
                sidecar_text = read_text_from_sidecar(sidecar_path)
                if sidecar_text:
                    logger.info(
                        f"Read {len(sidecar_text)} characters from sidecar file"
                    )
                    # Note: In multiprocessing context, we can't set self.extracted_text
                    # The text will be read again in the main process if needed

                # Handle temporary sidecar files
                self._cleanup_temp_sidecar(sidecar_path)
            else:
                self._log_sidecar_issues(sidecar_path)

            logger.info(f"Completed OCR processing for {os.path.basename(input_file)}")

        except Exception as e:
            logger.error(
                f"OCR processing failed for {os.path.basename(input_file)}: {str(e)}"
            )
            # Write error to progress file so main process knows about the failure
            try:
                with open(progress_file, "w") as f:
                    f.write("ERROR")
            except Exception as pf_error:
                logger.error(f"Could not write error to progress file: {pf_error}")
            raise

        finally:
            # Always stop progress updates
            if progress_indicator:
                try:
                    progress_indicator.stop_progress_updates()
                except Exception as e:
                    logger.debug(f"Error stopping progress indicator: {e}")

            # Clean up progress file
            try:
                if os.path.exists(progress_file):
                    os.unlink(progress_file)
            except Exception as e:
                logger.debug(f"Error cleaning up progress file: {e}")

    def _cleanup_temp_sidecar(self, sidecar_path: str) -> None:
        """Clean up temporary sidecar files

        Args:
            sidecar_path: Path to the sidecar file
        """
        if ".temp" in sidecar_path:
            try:
                os.remove(sidecar_path)
                logger.info(
                    _("Deleted temporary sidecar file: {0}").format(sidecar_path)
                )
            except Exception as e:
                logger.warning(
                    _("Failed to delete temporary sidecar file: {0}").format(e)
                )

    def _log_sidecar_issues(self, sidecar_path: str) -> None:
        """Log issues with sidecar files

        Args:
            sidecar_path: Path to the sidecar file
        """
        if sidecar_path:
            logger.warning(_("No sidecar file found at {0}").format(sidecar_path))
        else:
            logger.warning(_("No sidecar file was specified in options"))

    def check_status(self) -> str:
        """Check the status of the OCR process

        Returns:
            The current status of the process
        """
        if self.status == STATUS_RUNNING and self.process:
            if not self.process.is_alive():
                if self.process.exitcode == 0:
                    self.status = STATUS_COMPLETED
                    self.progress = PROGRESS_COMPLETE
                    self.end_time = time.time()
                else:
                    self.status = STATUS_FAILED
                    self.error = _("Process exited with code {0}").format(
                        self.process.exitcode
                    )

                # Clean up progress file when process completes
                if hasattr(self, "progress_file"):
                    try:
                        os.unlink(self.progress_file)
                    except:
                        pass

        return self.status

    def get_progress(self) -> float:
        """Get the progress of the OCR process

        Returns:
            Progress as a float between 0 and 1
        """
        # Check status first to update progress if needed
        status = self.check_status()

        # If the process is complete, return 1.0
        if status == STATUS_COMPLETED:
            return PROGRESS_COMPLETE

        # For running processes, try to read progress from file
        if status == STATUS_RUNNING and hasattr(self, "progress_file"):
            try:
                if os.path.exists(self.progress_file):
                    with open(self.progress_file, "r") as f:
                        content = f.read().strip()
                        if content:
                            # Check for error condition
                            if content == "ERROR":
                                self.status = STATUS_FAILED
                                self.error = _("OCR process encountered an error")
                                logger.error(
                                    f"Error detected in progress file for {os.path.basename(self.input_file)}"
                                )
                                return self.progress

                            # Try to parse as float
                            try:
                                new_progress = float(content)
                                if 0 <= new_progress <= 1.0:
                                    self.progress = new_progress
                            except ValueError:
                                logger.debug(
                                    f"Invalid progress value in file: {content}"
                                )
            except IOError as e:
                # If we can't read progress, keep the last known value
                logger.debug(f"Failed to read progress file: {e}")

        return self.progress

    def terminate(self) -> None:
        """Terminate the OCR process"""
        if self.process and self.process.is_alive():
            logger.info(
                _("Terminating OCR process for {0}").format(
                    os.path.basename(self.input_file)
                )
            )
            self.process.terminate()
            try:
                self.process.join(
                    timeout=5
                )  # Wait up to 5 seconds for graceful shutdown
            except Exception:
                pass

            # Force kill if still alive
            if self.process.is_alive():
                logger.warning("Force killing OCR process...")
                self.process.kill()

            self.status = STATUS_FAILED
            self.error = _("Process was terminated")
            self.end_time = time.time()

        # Clean up progress file
        if hasattr(self, "progress_file") and self.progress_file:
            try:
                if os.path.exists(self.progress_file):
                    os.unlink(self.progress_file)
            except Exception as e:
                logger.debug(f"Failed to clean up progress file: {e}")

        logger.debug("OCR process termination completed")


class OcrQueue:
    """Manages a queue of OCR processes"""

    def __init__(self, max_concurrent: int = 1):
        """Initialize the OCR queue

        Args:
            max_concurrent: Maximum number of concurrent OCR processes
        """
        self.max_concurrent = max_concurrent
        self.queue: List[OcrProcess] = []
        self.running: List[OcrProcess] = []
        self.completed: List[OcrProcess] = []
        self.failed: List[OcrProcess] = []
        self.lock = threading.Lock()
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_monitor = False
        self.on_file_complete: Optional[Callable] = None
        self.on_all_complete: Optional[Callable] = None
        self._total_files = 0
        self._processed_files = 0
        self.total_pages = 0

    def add_file(
        self,
        input_file: str,
        output_file: str,
        options: Dict[str, Any],
        page_count: int,
    ) -> None:
        """Add a file to the OCR queue

        Args:
            input_file: Path to the input PDF file
            output_file: Path to save the output PDF file
            options: Dictionary of options to pass to OCRmyPDF
            page_count: Number of pages in the input file
        """
        with self.lock:
            process = OcrProcess(input_file, output_file, options, page_count)
            self.queue.append(process)
            self._total_files += 1
            self.total_pages += page_count
            logger.info(
                _("Added {0} to OCR queue ({1} pages)").format(
                    os.path.basename(input_file), page_count
                )
            )

    def start(self) -> None:
        """Start processing the OCR queue"""
        # Clean up any previous state before starting
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.info("Cleaning up previous OCR queue state...")
            self.stop_monitor = True
            try:
                self.monitor_thread.join(timeout=2)
            except Exception:
                pass

        with self.lock:
            self.stop_monitor = False

            # Validate that we have files to process
            if not self.queue:
                logger.warning("No files in OCR queue to process")
                return

            # Log initial queue state
            logger.info(f"Starting OCR queue with {len(self.queue)} files")

        # Start the monitor thread
        if not self.monitor_thread or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(
                target=self._monitor_queue, daemon=True
            )
            self.monitor_thread.start()
            logger.info(_("Started OCR queue processing"))

            # Verify the monitor thread actually started
            time.sleep(0.1)
            if not self.monitor_thread.is_alive():
                logger.error("Failed to start OCR queue monitor thread")
                raise RuntimeError("Failed to start OCR queue monitor thread")
        else:
            logger.warning("OCR queue monitor thread already running")

    def cleanup_all_processes(self) -> None:
        """Clean up all processes and reset queue state"""
        with self.lock:
            logger.info("Cleaning up all OCR processes...")

            # Terminate all running processes
            for process in self.running:
                process.terminate()

            # Clear all lists
            for process in self.queue + self.running + self.completed + self.failed:
                if hasattr(process, "cleanup_previous_process"):
                    try:
                        process.cleanup_previous_process()
                    except Exception as e:
                        logger.debug(f"Error during process cleanup: {e}")

            # Reset all state
            self.queue.clear()
            self.running.clear()
            self.completed.clear()
            self.failed.clear()
            self._total_files = 0
            self._processed_files = 0
            self.total_pages = 0
            self.stop_monitor = True

            # Stop monitor thread if it exists
            if self.monitor_thread and self.monitor_thread.is_alive():
                try:
                    self.monitor_thread.join(timeout=2)
                except Exception:
                    pass

            self.monitor_thread = None
            logger.info("OCR process cleanup completed")

    def stop(self) -> None:
        """Stop processing the OCR queue"""
        logger.info(_("Stopping OCR queue processing"))
        self.cleanup_all_processes()

    def _monitor_queue(self) -> None:
        """Monitor the OCR queue and start new processes as needed"""
        last_activity_time = time.time()
        stall_warning_logged = False

        try:
            logger.info("OCR queue monitor thread started")

            while True:
                current_time = time.time()

                # Check for stop signal at the beginning of the loop
                with self.lock:
                    if self.stop_monitor:
                        logger.info("OCR queue monitor received stop signal")
                        break

                completed_now = []
                is_all_done = False
                had_activity = False

                with self.lock:
                    # Check status of running processes
                    still_running = []
                    for process in self.running:
                        status = process.check_status()
                        if status == STATUS_COMPLETED:
                            completed_now.append(process)
                            self.completed.append(process)
                            self._processed_files += 1
                            had_activity = True
                            logger.info(
                                f"Process completed: {os.path.basename(process.input_file)}"
                            )
                        elif status == STATUS_FAILED:
                            # Also treat failed as "completed" for queue purposes
                            completed_now.append(process)
                            self.failed.append(process)
                            self._processed_files += 1
                            had_activity = True
                            logger.error(
                                f"Process failed: {os.path.basename(process.input_file)} - {process.error}"
                            )
                        else:
                            # Process is still running - perform health checks
                            is_healthy = self._check_process_health(process)
                            if not is_healthy:
                                logger.warning(
                                    f"Process health check failed for {os.path.basename(process.input_file)}"
                                )
                                # Don't immediately fail - give it more time

                            # Check for stalled processes (running too long without progress)
                            if hasattr(process, "start_time") and process.start_time:
                                runtime = current_time - process.start_time
                                if runtime > 300:  # 5 minutes timeout
                                    logger.warning(
                                        f"Process appears stalled after {runtime:.1f} seconds: {os.path.basename(process.input_file)}"
                                    )
                                    if (
                                        runtime > 900
                                    ):  # 15 minutes - force failure (increased from 10)
                                        logger.error(
                                            f"Terminating stalled process: {os.path.basename(process.input_file)}"
                                        )
                                        process.terminate()
                                        process.status = STATUS_FAILED
                                        process.error = (
                                            "Process terminated due to timeout"
                                        )
                                        completed_now.append(process)
                                        self.failed.append(process)
                                        self._processed_files += 1
                                        had_activity = True
                                        continue
                            still_running.append(process)

                    # Update running processes list
                    self.running = still_running

                    # Start new processes if there's room
                    started_new = self._start_new_processes()
                    if started_new > 0:
                        had_activity = True
                        logger.info(f"Started {started_new} new OCR processes")

                    # Check if all processes are complete
                    if not self.running and not self.queue:
                        is_all_done = True
                        logger.info("All OCR processes completed")

                # Update activity tracking
                if had_activity:
                    last_activity_time = current_time
                    stall_warning_logged = False

                # Check for stalled queue (no activity for too long)
                time_since_activity = current_time - last_activity_time
                if time_since_activity > 60 and not stall_warning_logged:  # 1 minute
                    with self.lock:
                        if self.running or self.queue:
                            logger.warning(
                                f"OCR queue appears stalled - no activity for {time_since_activity:.1f} seconds"
                            )
                            logger.info(
                                f"Queue state: {len(self.queue)} queued, {len(self.running)} running, {len(self.completed)} completed, {len(self.failed)} failed"
                            )
                            stall_warning_logged = True

                            # If completely stalled for too long, abort
                            if time_since_activity > 300:  # 5 minutes
                                logger.error(
                                    "OCR queue completely stalled - aborting remaining processes"
                                )
                                # Mark all remaining processes as failed
                                for process in self.queue + self.running:
                                    process.status = STATUS_FAILED
                                    process.error = "Process aborted due to queue stall"
                                    self.failed.append(process)
                                    self._processed_files += 1
                                self.queue.clear()
                                self.running.clear()
                                is_all_done = True

                # Execute callbacks outside the lock to prevent deadlocks
                for process in completed_now:
                    try:
                        if self.on_file_complete:
                            extracted_text = self._get_extracted_text(
                                process, process.input_file, process.output_file
                            )
                            self.on_file_complete(
                                process.input_file, process.output_file, extracted_text
                            )
                    except Exception as e:
                        logger.error(f"Error in file completion callback: {e}")

                if is_all_done:
                    try:
                        if self.on_all_complete:
                            self.on_all_complete()
                    except Exception as e:
                        logger.error(f"Error in completion callback: {e}")
                    break  # Exit the monitor thread loop

                # Sleep before checking again
                time.sleep(MONITOR_SLEEP_INTERVAL)

        except Exception as e:
            logger.error(f"OCR queue monitor thread crashed: {e}")
            # Try to notify about the failure
            try:
                if self.on_all_complete:
                    self.on_all_complete()
            except:
                pass
        finally:
            logger.info("OCR queue monitor thread ended")

    def _start_new_processes(self) -> int:
        """Start new processes if there's room in the queue

        Returns:
            Number of processes successfully started
        """
        started_count = 0

        while len(self.running) < self.max_concurrent and self.queue:
            process = self.queue.pop(0)

            # Log process start attempt
            logger.info(
                f"Attempting to start OCR process for {os.path.basename(process.input_file)}"
            )

            try:
                process.start()
                self.running.append(process)
                started_count += 1
                logger.info(
                    f"Successfully started OCR process for {os.path.basename(process.input_file)} (PID: {process.process.pid if process.process else 'unknown'})"
                )

                # Additional validation: verify the process is actually working after a short delay
                def validate_process_later():
                    """Validate the process is still working after a short delay"""
                    time.sleep(2)  # Wait 2 seconds
                    try:
                        if process.process and not process.process.is_alive():
                            logger.error(
                                f"Process for {os.path.basename(process.input_file)} died shortly after startup"
                            )
                            # Move from running to failed if it's still in running
                            with self.lock:
                                if process in self.running:
                                    self.running.remove(process)
                                    process.status = STATUS_FAILED
                                    process.error = "Process died shortly after startup"
                                    self.failed.append(process)
                                    self._processed_files += 1
                    except Exception as e:
                        logger.debug(f"Error during delayed process validation: {e}")

                # Start validation in background thread
                validation_thread = threading.Thread(
                    target=validate_process_later, daemon=True
                )
                validation_thread.start()

            except Exception as e:
                logger.error(
                    f"Failed to start OCR process for {os.path.basename(process.input_file)}: {e}"
                )
                # Mark the process as failed
                process.status = STATUS_FAILED
                process.error = f"Failed to start: {str(e)}"
                self.failed.append(process)
                self._processed_files += 1

                # If we have more files in queue, try to continue with the next one
                if self.queue:
                    logger.info(
                        f"Continuing with next file after startup failure. {len(self.queue)} files remaining in queue."
                    )

        return started_count

    def _get_extracted_text(
        self, completed_process: Optional[OcrProcess], input_file: str, output_file: str
    ) -> str:
        """Get extracted text from a completed process

        Args:
            completed_process: The completed OCR process
            input_file: Input file path
            output_file: Output file path

        Returns:
            Extracted text from the PDF
        """
        # First try to get text from the completed process
        if completed_process:
            extracted_text = getattr(completed_process, "extracted_text", "")
            if extracted_text:
                logger.info(
                    _("Found extracted_text in process: {0} characters").format(
                        len(extracted_text)
                    )
                )
                return extracted_text

        # Try to get text from sidecar file
        extracted_text = self._try_get_text_from_sidecar(output_file)
        if extracted_text:
            return extracted_text

        # Try to get text from temporary file
        extracted_text = self._try_get_text_from_temp_file(output_file)
        if extracted_text:
            return extracted_text

        # Try direct extraction from PDF as last resort
        extracted_text = extract_text_from_pdf(output_file)
        if extracted_text:
            return extracted_text

        # Provide a placeholder if all methods failed
        return _("Text extraction completed.")

    def _try_get_text_from_sidecar(self, output_file: str) -> Optional[str]:
        """Try to get text from a sidecar file

        Args:
            output_file: Output PDF file path

        Returns:
            Extracted text or None if not found
        """
        sidecar_file = os.path.splitext(output_file)[0] + ".txt"
        if os.path.exists(sidecar_file):
            try:
                with open(sidecar_file, "r", encoding="utf-8") as f:
                    extracted_text = f.read()
                logger.info(
                    _("Read {0} characters from sidecar file before callback").format(
                        len(extracted_text)
                    )
                )
                return extracted_text
            except Exception as e:
                logger.error(
                    _("Error reading sidecar file before callback: {0}").format(e)
                )
        return None

    def _try_get_text_from_temp_file(self, output_file: str) -> Optional[str]:
        """Try to get text from a temporary file

        Args:
            output_file: Output PDF file path

        Returns:
            Extracted text or None if not found
        """
        temp_dir = os.path.join(os.path.dirname(output_file), ".temp")
        if os.path.exists(temp_dir):
            temp_file = os.path.join(
                temp_dir,
                f"temp_{os.path.basename(os.path.splitext(output_file)[0])}.txt",
            )
            if os.path.exists(temp_file):
                try:
                    with open(temp_file, "r", encoding="utf-8") as f:
                        extracted_text = f.read()
                    logger.info(
                        _(
                            "Read {0} characters from temporary file before callback"
                        ).format(len(extracted_text))
                    )
                    return extracted_text
                except Exception as e:
                    logger.error(
                        _("Error reading temporary file before callback: {0}").format(e)
                    )
        return None

    def get_progress(self) -> float:
        """Get the overall progress of the OCR queue based on pages processed

        Returns:
            Progress as a float between 0 and 1
        """
        with self.lock:
            if self.total_pages == 0:
                return 0.0

            # Calculate pages from completed processes
            completed_pages = sum(p.page_count for p in self.completed)

            # Calculate estimated pages from running processes
            running_pages = 0.0
            for process in self.running:
                # get_progress is file-level (0-1), multiply by its pages
                running_pages += process.get_progress() * process.page_count

            total_processed_pages = completed_pages + running_pages

            # Calculate overall progress
            return min(1.0, total_processed_pages / self.total_pages)

    def get_processed_page_count(self) -> int:
        """Get the number of processed pages (completed + running estimated)

        Returns:
            Number of processed pages as an integer
        """
        with self.lock:
            if self.total_pages == 0:
                return 0

            completed_pages = sum(p.page_count for p in self.completed)
            running_pages = sum(p.get_progress() * p.page_count for p in self.running)
            return int(completed_pages + running_pages)

    def get_total_page_count(self) -> int:
        """Get the total number of pages in the queue.

        Returns:
            Total number of pages.
        """
        with self.lock:
            return self.total_pages

    def get_processed_count(self) -> int:
        """Get the number of processed files

        Returns:
            Number of processed files
        """
        with self.lock:
            return self._processed_files

    def get_total_count(self) -> int:
        """Get the total number of files in the queue

        Returns:
            Total number of files
        """
        with self.lock:
            return self._total_files

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register a callback function for queue events

        Args:
            event: Event name ('file_complete' or 'all_complete')
            callback: Callback function
        """
        if event == "file_complete":
            self.on_file_complete = callback
        elif event == "all_complete":
            self.on_all_complete = callback

    def has_failed_processes(self) -> bool:
        """Check if any processes have failed

        Returns:
            True if any process has failed, False otherwise
        """
        with self.lock:
            # Check failed list
            if self.failed:
                return True

            # Check running processes for any that might have failed
            for process in self.running:
                if process.check_status() == STATUS_FAILED:
                    return True

            return False

    def get_failed_process_errors(self) -> List[str]:
        """Get error messages from failed processes

        Returns:
            List of error messages from failed processes
        """
        errors = []
        with self.lock:
            for process in self.failed:
                if process.error:
                    errors.append(
                        f"{os.path.basename(process.input_file)}: {process.error}"
                    )
                else:
                    errors.append(
                        f"{os.path.basename(process.input_file)}: Unknown error"
                    )

            # Also check running processes that might have failed
            for process in self.running:
                if process.check_status() == STATUS_FAILED and process.error:
                    errors.append(
                        f"{os.path.basename(process.input_file)}: {process.error}"
                    )

        return errors

    def _check_process_health(self, process: "OcrProcess") -> bool:
        """Check if a process is healthy and working properly

        Args:
            process: The OCR process to check

        Returns:
            True if process appears healthy, False otherwise
        """
        try:
            # Check if process is alive
            if not process.process or not process.process.is_alive():
                logger.warning(
                    f"Process not alive for {os.path.basename(process.input_file)}"
                )
                return False

            # Check if progress file exists and is being updated
            if hasattr(process, "progress_file") and process.progress_file:
                if not os.path.exists(process.progress_file):
                    # Process has been running but no progress file - might be starting up
                    runtime = (
                        time.time() - process.start_time if process.start_time else 0
                    )
                    if runtime > 30:  # After 30 seconds, we should have a progress file
                        logger.warning(
                            f"No progress file after {runtime:.1f}s for {os.path.basename(process.input_file)}"
                        )
                        return False
                else:
                    # Check if progress file is recent
                    try:
                        file_mtime = os.path.getmtime(process.progress_file)
                        if (
                            time.time() - file_mtime > 30
                        ):  # File not updated in 30 seconds
                            logger.warning(
                                f"Progress file stale for {os.path.basename(process.input_file)}"
                            )
                            return False
                    except Exception as e:
                        logger.debug(f"Error checking progress file time: {e}")

            return True

        except Exception as e:
            logger.debug(f"Error during process health check: {e}")
            return False
