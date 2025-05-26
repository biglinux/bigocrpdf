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
from typing import Dict, Callable, List, Optional, Any, Tuple

import ocrmypdf

from utils.logger import logger
from utils.i18n import _

# Constants for process status
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# Constants for progress estimation
INITIAL_STAGE_THRESHOLD = 5  # seconds
MAIN_STAGE_THRESHOLD = 15    # seconds
FINAL_STAGE_THRESHOLD = 30   # seconds
INITIAL_STAGE_MAX = 0.25
MAIN_STAGE_MAX = 0.5
FINAL_STAGE_MAX = 0.2
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
        logger.warning(
            _("Failed to extract existing text from PDF: {0}").format(e)
        )
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
            _("Read {0} characters of extracted text from sidecar file").format(len(text))
        )
        return text
    except Exception as e:
        logger.error(_("Error reading sidecar file: {0}").format(e))
        return None


def setup_sigbus_handler() -> None:
    """Set up SIGBUS handler as recommended in OCRmyPDF docs"""
    if hasattr(signal, "SIGBUS"):
        def handle_sigbus(signum, frame):
            raise RuntimeError(
                _("SIGBUS received - memory-mapped file access error")
            )
        signal.signal(signal.SIGBUS, handle_sigbus)


class OcrProcess:
    """Manages an OCR process for a single file"""

    def __init__(self, input_file: str, output_file: str, options: Dict[str, Any]):
        """Initialize the OCR process

        Args:
            input_file: Path to the input PDF file
            output_file: Path to save the output PDF file
            options: Dictionary of options to pass to OCRmyPDF
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
        self.extracted_text = ""  # Will hold extracted text from the OCR process

    def start(self) -> None:
        """Start the OCR process in a separate process"""
        self.start_time = time.time()
        self.status = STATUS_RUNNING

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        # Start the process
        self.process = multiprocessing.Process(
            target=self._run_ocr, 
            args=(self.input_file, self.output_file, self.options)
        )
        self.process.start()

        logger.info(_("Started OCR process for {0}").format(os.path.basename(self.input_file)))

    def _run_ocr(self, input_file: str, output_file: str, options: Dict[str, Any]) -> None:
        """Run OCRmyPDF in a separate process

        Args:
            input_file: Path to the input PDF file
            output_file: Path to save the output PDF file
            options: Dictionary of options to pass to OCRmyPDF
        """
        try:
            # Configure logging
            configure_logging()

            # Set up SIGBUS handler
            setup_sigbus_handler()

            # Get the sidecar path from options
            sidecar_path = options.get("sidecar", None)
            
            # Initialize extracted text
            self.extracted_text = ""
            
            # Try to extract text from input PDF first
            self.extracted_text = extract_text_from_pdf(input_file)

            # Make a copy of options to modify safely
            ocr_options = options.copy()

            # Set additional options to improve text extraction
            ocr_options["force_ocr"] = True
            ocr_options["optimize"] = 0

            # Run OCRmyPDF with optimized options
            ocrmypdf.ocr(input_file, output_file, **ocr_options)

            # Read the text from the sidecar file if it exists
            if sidecar_path and os.path.exists(sidecar_path):
                sidecar_text = read_text_from_sidecar(sidecar_path)
                if sidecar_text:
                    self.extracted_text = sidecar_text

                # Handle temporary sidecar files
                self._cleanup_temp_sidecar(sidecar_path)
            else:
                self._log_sidecar_issues(sidecar_path)

            logger.info(_("Completed OCR processing for {0}").format(os.path.basename(input_file)))
        except Exception as e:
            logger.error(
                _("OCR processing failed for {0}: {1}").format(os.path.basename(input_file), str(e))
            )
            raise

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
                    self.error = _("Process exited with code {0}").format(self.process.exitcode)

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

        # If the process is still running, estimate progress based on elapsed time
        if status == STATUS_RUNNING and self.start_time:
            return self._estimate_progress_by_time()

        return self.progress

    def _estimate_progress_by_time(self) -> float:
        """Estimate progress based on elapsed time
        
        Returns:
            Estimated progress as a float between 0 and 1
        """
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time < INITIAL_STAGE_THRESHOLD:
            # Initial processing stage
            self.progress = min(INITIAL_STAGE_MAX, elapsed_time / 20)
        elif elapsed_time < MAIN_STAGE_THRESHOLD:
            # Main OCR processing stage
            self.progress = INITIAL_STAGE_MAX + min(
                MAIN_STAGE_MAX, (elapsed_time - MAIN_STAGE_THRESHOLD) / 20
            )
        elif elapsed_time < FINAL_STAGE_THRESHOLD:
            # Final processing stage
            self.progress = (INITIAL_STAGE_MAX + MAIN_STAGE_MAX) + min(
                FINAL_STAGE_MAX, (elapsed_time - MAIN_STAGE_THRESHOLD) / 30
            )
        else:
            # Continue progressing slowly instead of stopping at 95%
            extra_time = elapsed_time - FINAL_STAGE_THRESHOLD
            additional_progress = min(0.04, extra_time / 600 * 0.04)  # Very slow progress over 10 minutes max
            self.progress = PROGRESS_ALMOST_DONE + additional_progress
            
        return min(0.99, self.progress)  # Cap at 99%, never show 100% until really done

    def terminate(self) -> None:
        """Terminate the OCR process"""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.status = STATUS_FAILED
            self.error = _("Process was terminated")
            logger.info(
                _("OCR process for {0} was terminated").format(os.path.basename(self.input_file))
            )


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

    def add_file(self, input_file: str, output_file: str, options: Dict[str, Any]) -> None:
        """Add a file to the OCR queue

        Args:
            input_file: Path to the input PDF file
            output_file: Path to save the output PDF file
            options: Dictionary of options to pass to OCRmyPDF
        """
        with self.lock:
            process = OcrProcess(input_file, output_file, options)
            self.queue.append(process)
            self._total_files += 1
            logger.info(_("Added {0} to OCR queue").format(os.path.basename(input_file)))

    def start(self) -> None:
        """Start processing the OCR queue"""
        with self.lock:
            self.stop_monitor = False

        if not self.monitor_thread or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(target=self._monitor_queue)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info(_("Started OCR queue processing"))

    def stop(self) -> None:
        """Stop processing the OCR queue"""
        with self.lock:
            self.stop_monitor = True

            # Terminate all running processes more aggressively
            for process in self.running:
                try:
                    process.terminate()
                    # Wait a bit, then force kill if still alive
                    if hasattr(process, 'process') and process.process:
                        if process.process.is_alive():
                            process.process.join(timeout=2)
                            if process.process.is_alive():
                                process.process.kill()
                except Exception as e:
                    logger.warning(f"Error terminating OCR process: {e}")

            # Clear all lists
            self.running.clear()
            self.queue.clear()
            self.completed.clear()
            self.failed.clear()

            # Reset counters
            self._total_files = 0
            self._processed_files = 0

            logger.info(_("Stopped OCR queue processing with aggressive cleanup"))

            # Force cleanup of monitor thread
            if self.monitor_thread and self.monitor_thread.is_alive():
                try:
                    self.monitor_thread.join(timeout=1)
                except Exception:
                    pass

    def _monitor_queue(self) -> None:
        """Monitor the OCR queue and start new processes as needed"""
        while True:
            file_completed = False
            all_completed = False
            callback_input_file = None
            callback_output_file = None
            completed_process = None

            with self.lock:
                if self.stop_monitor:
                    break

                # Check status of running processes
                still_running = []
                for process in self.running:
                    status = process.check_status()
                    if status == STATUS_COMPLETED:
                        self.completed.append(process)
                        self._processed_files += 1
                        # Mark for callback outside the lock
                        file_completed = True
                        callback_input_file = process.input_file
                        callback_output_file = process.output_file
                        completed_process = process
                    elif status == STATUS_FAILED:
                        self.failed.append(process)
                        self._processed_files += 1
                    else:
                        still_running.append(process)

                # Update running processes
                self.running = still_running

                # Start new processes if there's room
                self._start_new_processes()

                # Check if all processes are complete
                if not self.running and not self.queue:
                    all_completed = True
                    if not self.completed and not self.failed:
                        # Nothing was processed
                        break

            # Execute callbacks outside the lock to prevent deadlocks
            if (
                file_completed
                and self.on_file_complete
                and callback_input_file
                and callback_output_file
            ):
                extracted_text = self._get_extracted_text(
                    completed_process, callback_input_file, callback_output_file
                )
                self.on_file_complete(
                    callback_input_file, callback_output_file, extracted_text
                )

            if all_completed and self.on_all_complete:
                self.on_all_complete()
                break

            # Sleep before checking again
            time.sleep(MONITOR_SLEEP_INTERVAL)

    def _start_new_processes(self) -> None:
        """Start new processes if there's room in the queue"""
        while len(self.running) < self.max_concurrent and self.queue:
            process = self.queue.pop(0)
            process.start()
            self.running.append(process)

    def _get_extracted_text(
        self, 
        completed_process: Optional[OcrProcess], 
        input_file: str, 
        output_file: str
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
                    _("Found extracted_text in process: {0} characters").format(len(extracted_text))
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
                logger.error(_("Error reading sidecar file before callback: {0}").format(e))
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
                        _("Read {0} characters from temporary file before callback").format(
                            len(extracted_text)
                        )
                    )
                    return extracted_text
                except Exception as e:
                    logger.error(_("Error reading temporary file before callback: {0}").format(e))
        return None

    def get_progress(self) -> float:
        """Get the overall progress of the OCR queue

        Returns:
            Progress as a float between 0 and 1
        """
        with self.lock:
            if self._total_files == 0:
                return 0.0

            # Check if everything is processed - return 1.0 for 100% complete
            if self._processed_files >= self._total_files:
                return PROGRESS_COMPLETE

            # Check if we have completed files - let's make sure the UI shows completion
            if len(self.completed) > 0 and len(self.completed) == self._total_files:
                return PROGRESS_COMPLETE

            # For running processes, check their individual progress
            total_progress = self._processed_files
            for process in self.running:
                # Get progress from each running process
                total_progress += process.get_progress()

            # Calculate overall progress
            return min(0.99, total_progress / self._total_files)

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

    def count_remaining_files(self) -> int:
        """Get the number of files remaining in the queue

        Returns:
            Number of files still in the queue
        """
        with self.lock:
            queue_count = len(self.queue)
            running_count = len(self.running)
            return queue_count + running_count

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