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
from typing import Dict, Callable

import ocrmypdf

from ..utils.logger import logger


def configure_logging():
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


class OcrProcess:
    """Manages an OCR process for a single file"""

    def __init__(self, input_file: str, output_file: str, options: Dict):
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
        self.status = "pending"  # pending, running, completed, failed
        self.progress = 0.0
        self.start_time = None
        self.end_time = None
        self.error = None
        self.extracted_text = ""  # Will hold extracted text from the OCR process

    def start(self) -> None:
        """Start the OCR process in a separate process"""
        self.start_time = time.time()
        self.status = "running"

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        # Start the process
        self.process = multiprocessing.Process(
            target=self._run_ocr, args=(self.input_file, self.output_file, self.options)
        )
        self.process.start()

        logger.info(f"Started OCR process for {os.path.basename(self.input_file)}")

    def _run_ocr(self, input_file: str, output_file: str, options: Dict) -> None:
        """Run OCRmyPDF in a separate process

        Args:
            input_file: Path to the input PDF file
            output_file: Path to save the output PDF file
            options: Dictionary of options to pass to OCRmyPDF
        """
        try:
            # Configure logging
            configure_logging()

            # Add SIGBUS handler as recommended in OCRmyPDF docs
            if hasattr(signal, "SIGBUS"):

                def handle_sigbus(signum, frame):
                    raise RuntimeError(
                        "SIGBUS received - memory-mapped file access error"
                    )

                signal.signal(signal.SIGBUS, handle_sigbus)

            # Get the sidecar path from options before running OCR
            sidecar_path = options.get(
                "sidecar", None
            )  # Initialize extracted text with a default value (to ensure we always have something)
            self.extracted_text = ""

            # Try to extract text directly from the input PDF first (may contain text already)
            try:
                result = subprocess.run(
                    ["pdftotext", input_file, "-"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.stdout and len(result.stdout.strip()) > 0:
                    logger.info(
                        f"Found existing text in PDF, length: {len(result.stdout)}"
                    )
                    self.extracted_text = result.stdout
            except Exception as e:
                logger.warning(
                    f"Failed to extract existing text from PDF: {e}"
                )  # Run OCR with the provided options
            # Make a copy of options to modify safely
            ocr_options = options.copy()

            # Set additional options to improve text extraction
            # Using force_ocr=True to ensure OCR is always performed
            # Use optimize=0 to prioritize text quality over file size
            ocr_options["force_ocr"] = True
            ocr_options["optimize"] = 0

            # Run OCRmyPDF with optimized options
            ocrmypdf.ocr(input_file, output_file, **ocr_options)

            # Read the text from the sidecar file if it exists (created by --sidecar option)
            if sidecar_path and os.path.exists(sidecar_path):
                try:
                    with open(sidecar_path, "r", encoding="utf-8") as f:
                        self.extracted_text = f.read()
                    logger.info(
                        f"Read {len(self.extracted_text)} characters of extracted text from sidecar file"
                    )

                    # If this is a temporary sidecar file (contains .temp in path), delete it
                    if ".temp" in sidecar_path:
                        try:
                            os.remove(sidecar_path)
                            logger.info(
                                f"Deleted temporary sidecar file: {sidecar_path}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to delete temporary sidecar file: {e}"
                            )
                except Exception as e:
                    logger.error(f"Error reading sidecar file: {e}")
                    # Keep the default text we set earlier instead of wiping it out
            else:
                if sidecar_path:
                    logger.warning(f"No sidecar file found at {sidecar_path}")
                else:
                    logger.warning("No sidecar file was specified in options")
                # Keep the default text we set earlier

            logger.info(f"Completed OCR processing for {os.path.basename(input_file)}")
        except Exception as e:
            logger.error(
                f"OCR processing failed for {os.path.basename(input_file)}: {str(e)}"
            )
            raise

    def check_status(self) -> str:
        """Check the status of the OCR process

        Returns:
            The current status of the process
        """
        if self.status == "running" and self.process:
            if not self.process.is_alive():
                if self.process.exitcode == 0:
                    self.status = "completed"
                    self.progress = 1.0
                    self.end_time = time.time()
                else:
                    self.status = "failed"
                    self.error = f"Process exited with code {self.process.exitcode}"

        return self.status

    def get_progress(self) -> float:
        """Get the progress of the OCR process

        Returns:
            Progress as a float between 0 and 1
        """
        # Check status first to update progress if needed
        status = self.check_status()

        # If the process is complete, return 1.0
        if status == "completed":
            return 1.0

        # If the process is still running, estimate progress based on elapsed time
        # OCR typically takes 10-30 seconds per page, let's assume 20 seconds on average
        if status == "running" and self.start_time:
            elapsed_time = time.time() - self.start_time
            if elapsed_time < 5:
                # Initial processing stage
                self.progress = min(0.25, elapsed_time / 20)
            elif elapsed_time < 15:
                # Main OCR processing stage
                self.progress = 0.25 + min(0.5, (elapsed_time - 5) / 20)
            elif elapsed_time < 30:
                # Final processing stage
                self.progress = 0.75 + min(0.2, (elapsed_time - 15) / 30)
            else:
                # Almost done
                self.progress = 0.95

        return self.progress

    def terminate(self) -> None:
        """Terminate the OCR process"""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.status = "failed"
            self.error = "Process was terminated"
            logger.info(
                f"OCR process for {os.path.basename(self.input_file)} was terminated"
            )


class OcrQueue:
    """Manages a queue of OCR processes"""

    def __init__(self, max_concurrent: int = 1):
        """Initialize the OCR queue

        Args:
            max_concurrent: Maximum number of concurrent OCR processes
        """
        self.max_concurrent = max_concurrent
        self.queue = []
        self.running = []
        self.completed = []
        self.failed = []
        self.lock = threading.Lock()
        self.monitor_thread = None
        self.stop_monitor = False
        self.on_file_complete = None
        self.on_all_complete = None
        self._total_files = 0
        self._processed_files = 0

    def add_file(self, input_file: str, output_file: str, options: Dict) -> None:
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
            logger.info(f"Added {os.path.basename(input_file)} to OCR queue")

    def start(self) -> None:
        """Start processing the OCR queue"""
        with self.lock:
            self.stop_monitor = False

        if not self.monitor_thread or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(target=self._monitor_queue)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info("Started OCR queue processing")

    def stop(self) -> None:
        """Stop processing the OCR queue"""
        with self.lock:
            self.stop_monitor = True

            # Terminate all running processes
            for process in self.running:
                process.terminate()

            logger.info("Stopped OCR queue processing")

    def _monitor_queue(self) -> None:
        """Monitor the OCR queue and start new processes as needed"""
        while True:
            file_completed = False
            all_completed = False
            callback_input_file = None
            callback_output_file = None

            with self.lock:
                if self.stop_monitor:
                    break

                # Check status of running processes
                still_running = []
                for process in self.running:
                    status = process.check_status()
                    if status == "completed":
                        self.completed.append(process)
                        self._processed_files += 1
                        # Mark for callback outside the lock
                        file_completed = True
                        callback_input_file = process.input_file
                        callback_output_file = process.output_file
                    elif status == "failed":
                        self.failed.append(process)
                        self._processed_files += 1
                    else:
                        still_running.append(process)

                # Update running processes
                self.running = still_running

                # Start new processes if there's room
                while len(self.running) < self.max_concurrent and self.queue:
                    process = self.queue.pop(0)
                    process.start()
                    self.running.append(process)

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
                # Find the completed process to get its extracted text
                extracted_text = ""
                for process in self.completed:
                    if (
                        process.input_file == callback_input_file
                        and process.output_file == callback_output_file
                    ):
                        extracted_text = getattr(process, "extracted_text", "")
                        logger.info(
                            f"Found extracted_text in process: {len(extracted_text)} characters"
                        )
                        break

                logger.info(
                    f"Passing extracted_text to callback: {len(extracted_text)} characters"
                )

                # Pass the extracted text along with the file paths
                # Make sure we definitely have the extracted text
                if not extracted_text and os.path.exists(
                    os.path.splitext(callback_output_file)[0] + ".txt"
                ):
                    try:
                        with open(
                            os.path.splitext(callback_output_file)[0] + ".txt",
                            "r",
                            encoding="utf-8",
                        ) as f:
                            extracted_text = f.read()
                        logger.info(
                            f"Read {len(extracted_text)} characters from sidecar file before callback"
                        )
                    except Exception as e:
                        logger.error(f"Error reading sidecar file before callback: {e}")

                # Check for temporary files if no text was found
                if not extracted_text:
                    temp_dir = os.path.join(
                        os.path.dirname(callback_output_file), ".temp"
                    )
                    if os.path.exists(temp_dir):
                        temp_file = os.path.join(
                            temp_dir,
                            f"temp_{os.path.basename(os.path.splitext(callback_output_file)[0])}.txt",
                        )
                        if os.path.exists(temp_file):
                            try:
                                with open(temp_file, "r", encoding="utf-8") as f:
                                    extracted_text = f.read()
                                logger.info(
                                    f"Read {len(extracted_text)} characters from temporary file before callback"
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error reading temporary file before callback: {e}"
                                )

                # Use direct extraction from PDF as last resort
                if not extracted_text:
                    try:
                        result = subprocess.run(
                            ["pdftotext", callback_output_file, "-"],
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        if result.stdout and len(result.stdout.strip()) > 0:
                            extracted_text = result.stdout
                            logger.info(
                                f"Extracted {len(extracted_text)} characters directly from PDF as fallback"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract text directly from PDF as fallback: {e}"
                        )

                # Provide a placeholder if all methods failed
                if not extracted_text:
                    extracted_text = "Text extraction completed."

                self.on_file_complete(
                    callback_input_file, callback_output_file, extracted_text
                )

            if all_completed:
                if self.on_all_complete:
                    self.on_all_complete()
                break

            # Sleep before checking again
            time.sleep(0.5)

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
                return 1.0

            # Check if we have completed files - let's make sure the UI shows completion
            if len(self.completed) > 0 and len(self.completed) == self._total_files:
                return 1.0

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
