"""
BigOcrPdf - OCR Service

This module orchestrates the RapidOCR processing queue, providing
real-time progress information for both PDF and image inputs.
"""

import logging
import multiprocessing
import os
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from services.rapidocr_service import RapidOCREngine
from utils.i18n import _
from utils.logger import logger

# Constants for process status
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# Constants for progress calculation
PROGRESS_COMPLETE = 1.0
PROGRESS_UPDATE_INTERVAL = 1.0  # Update every second for smooth progress
MONITOR_SLEEP_INTERVAL = 1.0


def configure_logging() -> None:
    """Configure logging to use our application logger"""
    for name in ("rapidocr", "rapidocr_onnxruntime"):
        ocr_logger = logging.getLogger(name)
        ocr_logger.setLevel(logging.INFO)
        ocr_logger.handlers = []
        for handler in logger.handlers:
            ocr_logger.addHandler(handler)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text directly from a PDF file using pdftotext"""
    try:
        result = subprocess.run(
            ["pdftotext", pdf_path, "-"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout and len(result.stdout.strip()) > 0:
            logger.info(
                f"Found existing text in PDF, length: {len(result.stdout)}"
            )
            return result.stdout
    except Exception as e:
        logger.warning(
            f"Failed to extract existing text from PDF: {e}"
        )
    return ""


def read_text_from_sidecar(sidecar_path: str) -> Optional[str]:
    """Read text from a sidecar file"""
    if not sidecar_path or not os.path.exists(sidecar_path):
        return None
    
    try:
        with open(sidecar_path, "r", encoding="utf-8") as f:
            text = f.read()
        logger.info(
            f"Read {len(text)} characters of extracted text from sidecar file"
        )
        return text
    except Exception as e:
        logger.error(f"Error reading sidecar file: {e}")
        return None


class OcrProcess:
    """Manages an OCR process for a single file with real-time progress tracking"""

    def __init__(self, input_file: str, output_file: str, options: Dict[str, Any]):
        """Initialize the OCR process"""
        self.input_file = input_file
        self.output_file = output_file
        self.options = options
        self.input_type = (options.get("input_type") or "pdf").lower()
        self.process = None
        self.status = STATUS_PENDING
        self.start_time = None
        self.end_time = None
        self.error = None
        self.extracted_text = ""
        
        # File metadata for tracking
        self.file_size = os.path.getsize(input_file) if os.path.exists(input_file) else 0
        self.total_pages = max(1, self._get_page_count())
        
        # Real-time progress tracking
        self.current_pages_processed = 0
        self.progress_lock = threading.Lock()
        self._shared_processed = multiprocessing.Value("i", 0)
        
    def _get_page_count(self) -> int:
        """Get the total number of pages for this input"""
        if self.input_type == "image":
            return 1

        try:
            import fitz  # type: ignore
            with fitz.open(self.input_file) as doc:
                return max(1, doc.page_count)
        except Exception as exc:
            logger.debug("Failed to determine page count with PyMuPDF: %s", exc)

        return max(1, self.file_size // 50000)

    def start(self) -> None:
        """Start the OCR process in a separate process"""
        self.start_time = time.time()
        self.status = STATUS_RUNNING

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_file) or ".", exist_ok=True)

        # Start the process with stdout capture
        self.process = multiprocessing.Process(
            target=self._run_ocr_with_progress, 
            args=(self.input_file, self.output_file, self.options)
        )
        self.process.start()

        logger.info(f"Started OCR process for {os.path.basename(self.input_file)}")

    def _run_ocr_with_progress(self, input_file: str, output_file: str, options: Dict[str, Any]) -> None:
        """Execute RapidOCR for the provided input"""
        try:
            configure_logging()
            sidecar_path = options.get("sidecar")
            expected_pages = max(1, int(options.get("expected_pages", self.total_pages)))
            zoom = float(options.get("render_zoom", 2.0) or 2.0)

            engine = RapidOCREngine(zoom=zoom)

            def progress_callback(processed: int, total: int) -> None:
                total = total or expected_pages
                normalized = min(processed, total)
                with self._shared_processed.get_lock():
                    self._shared_processed.value = normalized
                with self.progress_lock:
                    self.current_pages_processed = normalized

            text_result = engine.process_document(
                input_file=input_file,
                output_file=output_file,
                input_mode=self.input_type,
                sidecar=sidecar_path,
                progress_cb=progress_callback if self.input_type != "image" else None,
            )

            # Ensure progress reaches completion
            with self._shared_processed.get_lock():
                self._shared_processed.value = self.total_pages
            with self.progress_lock:
                self.current_pages_processed = self.total_pages

            if sidecar_path and os.path.exists(sidecar_path):
                sidecar_text = read_text_from_sidecar(sidecar_path)
                if sidecar_text:
                    self.extracted_text = sidecar_text

            if not self.extracted_text:
                self.extracted_text = text_result or extract_text_from_pdf(output_file)

            logger.info(
                _("RapidOCR processing completed for {0}").format(
                    os.path.basename(input_file)
                )
            )
        except Exception as exc:
            logger.error(
                _("RapidOCR processing failed for {0}: {1}").format(
                    os.path.basename(input_file), exc
                )
            )
            self.error = str(exc)
            raise

    def get_pages_processed(self) -> int:
        """Get the number of pages processed so far"""
        shared_value = 0
        with self._shared_processed.get_lock():
            shared_value = self._shared_processed.value

        with self.progress_lock:
            local_value = self.current_pages_processed

        return max(shared_value, local_value)

    def check_status(self) -> str:
        """Check the status of the OCR process"""
        if self.status == STATUS_RUNNING and self.process:
            if not self.process.is_alive():
                if self.process.exitcode == 0:
                    self.status = STATUS_COMPLETED
                    # Mark all pages as completed
                    with self.progress_lock:
                        self.current_pages_processed = self.total_pages
                    with self._shared_processed.get_lock():
                        self._shared_processed.value = self.total_pages
                    self.end_time = time.time()
                else:
                    self.status = STATUS_FAILED
                    self.error = f"Process exited with code {self.process.exitcode}"

        return self.status

    def get_pages_processed(self) -> int:
        """Get the number of pages processed so far"""
        shared_value = 0
        with self._shared_processed.get_lock():
            shared_value = self._shared_processed.value

        with self.progress_lock:
            local_value = self.current_pages_processed

        return max(shared_value, local_value)

    def get_total_pages(self) -> int:
        """Get total pages in this file"""
        return self.total_pages

    def terminate(self) -> None:
        """Terminate the OCR process"""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.status = STATUS_FAILED
            self.error = "Process was terminated"
            logger.info(
                f"OCR process for {os.path.basename(self.input_file)} was terminated"
            )


class OcrQueue:
    """Manages a queue of OCR processes with incremental page-based progress"""

    def __init__(self, max_concurrent: int = 1):
        """Initialize the OCR queue"""
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
        
        # Page-based progress tracking for incremental updates
        self._total_files = 0
        self._completed_files = 0
        self._total_pages = 0
        self._completed_pages = 0  # Only from fully completed files
        
        self._start_time = None

    def add_file(self, input_file: str, output_file: str, options: Dict[str, Any]) -> None:
        """Add a file to the OCR queue"""
        with self.lock:
            process = OcrProcess(input_file, output_file, options)
            self.queue.append(process)
            self._total_files += 1

            # Get page count
            page_count = self._get_page_count(input_file, options.get("input_type"))
            process.total_pages = page_count
            self._total_pages += page_count

            if "expected_pages" not in options:
                options["expected_pages"] = page_count
            
            logger.info(f"Added {os.path.basename(input_file)} to OCR queue (pages: {page_count})")

    def _get_page_count(self, file_path: str, input_type: Optional[str]) -> int:
        """Get page count for PDFs or treat images as single-page documents"""
        normalized = (input_type or "pdf").lower()
        if normalized == "image":
            return 1

        try:
            import fitz  # type: ignore
            with fitz.open(file_path) as doc:
                return max(1, doc.page_count)
        except Exception as exc:
            logger.debug("Failed to count pages with PyMuPDF: %s", exc)

        return 1

    def start(self) -> None:
        """Start processing the OCR queue"""
        with self.lock:
            self.stop_monitor = False
            self._start_time = time.time()

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
                try:
                    process.terminate()
                    if hasattr(process, 'process') and process.process:
                        if process.process.is_alive():
                            process.process.join(timeout=2)
                            if process.process.is_alive():
                                process.process.kill()
                except Exception as e:
                    logger.warning(f"Error terminating OCR process: {e}")

            # Clear all lists and reset counters
            self.running.clear()
            self.queue.clear()
            self.completed.clear()
            self.failed.clear()
            self._total_files = 0
            self._completed_files = 0
            self._total_pages = 0
            self._completed_pages = 0

            logger.info("Stopped OCR queue processing with aggressive cleanup")

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
                        self._completed_files += 1
                        self._completed_pages += process.total_pages
                        # Mark for callback outside the lock
                        file_completed = True
                        callback_input_file = process.input_file
                        callback_output_file = process.output_file
                        completed_process = process
                    elif status == STATUS_FAILED:
                        self.failed.append(process)
                        self._completed_files += 1
                        self._completed_pages += process.total_pages
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
                        break

            # Execute callbacks outside the lock
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
        """Get extracted text from a completed process"""
        # Try to get text from the completed process
        if completed_process:
            extracted_text = getattr(completed_process, "extracted_text", "")
            if extracted_text:
                logger.info(
                    f"Found extracted_text in process: {len(extracted_text)} characters"
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
        
        return _("No text could be extracted from this file.")

    def _try_get_text_from_sidecar(self, output_file: str) -> Optional[str]:
        """Try to get text from a sidecar file"""
        sidecar_file = os.path.splitext(output_file)[0] + ".txt"
        if os.path.exists(sidecar_file):
            try:
                with open(sidecar_file, "r", encoding="utf-8") as f:
                    extracted_text = f.read()
                logger.info(
                    f"Read {len(extracted_text)} characters from sidecar file"
                )
                return extracted_text
            except Exception as e:
                logger.error(f"Error reading sidecar file: {e}")
        return None

    def _try_get_text_from_temp_file(self, output_file: str) -> Optional[str]:
        """Try to get text from a temporary file"""
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
                        f"Read {len(extracted_text)} characters from temporary file"
                    )
                    return extracted_text
                except Exception as e:
                    logger.error(f"Error reading temporary file: {e}")
        return None

    def get_progress(self) -> float:
        """Get incremental progress based on time and processing status"""
        with self.lock:
            if self._total_pages == 0:
                return 0.0

            # Calculate total pages processed from completed files
            total_pages_processed = self._completed_pages
            
            # Add estimated progress from currently running processes
            for process in self.running:
                if process.start_time:
                    # Calculate time-based progress for running processes
                    elapsed = time.time() - process.start_time
                    # Estimate 4 seconds per page on average
                    estimated_pages = min(process.total_pages, elapsed / 4.0)
                    total_pages_processed += estimated_pages
            
            # Calculate smooth progress from 0% to 100%
            progress = total_pages_processed / self._total_pages
            
            # Ensure we show 100% when everything is truly done
            if (self._completed_files >= self._total_files and 
                not self.running and not self.queue):
                return PROGRESS_COMPLETE
            
            # Clamp between 0 and 1, but allow for smooth incremental updates
            return max(0.0, min(0.99, progress))  # Cap at 99% until truly complete

    def get_processed_count(self) -> int:
        """Get the number of processed files"""
        with self.lock:
            return self._completed_files

    def get_total_count(self) -> int:
        """Get the total number of files in the queue"""
        with self.lock:
            return self._total_files

    def count_remaining_files(self) -> int:
        """Get the number of files remaining in the queue"""
        with self.lock:
            return len(self.queue) + len(self.running)

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register a callback function for queue events"""
        if event == "file_complete":
            self.on_file_complete = callback
        elif event == "all_complete":
            self.on_all_complete = callback
            
    def get_total_pages(self) -> int:
        """Get the total number of pages across all files"""
        with self.lock:
            return self._total_pages

    def get_processed_pages(self) -> int:
        """Get the total number of pages processed (including currently processing)"""
        with self.lock:
            total_pages_processed = self._completed_pages
            
            # Add pages from currently running processes
            for process in self.running:
                total_pages_processed += process.get_pages_processed()
            
            return total_pages_processed

    def get_current_file_info(self) -> Dict[str, Any]:
        """Get information about the currently processing file"""
        with self.lock:
            if self.running:
                # Return info for the most active process
                for process in self.running:
                    pages_processed = process.get_pages_processed()
                    if pages_processed > 0:  # This process is actively working
                        return {
                            "filename": os.path.basename(process.input_file),
                            "current_page": pages_processed,
                            "total_pages": process.get_total_pages(),
                            "file_number": self._completed_files + 1,
                            "total_files": self._total_files
                        }
                
                # Fallback to first process if none show progress yet
                if self.running:
                    current_process = self.running[0]
                    return {
                        "filename": os.path.basename(current_process.input_file),
                        "current_page": 0,
                        "total_pages": current_process.get_total_pages(),
                        "file_number": self._completed_files + 1,
                        "total_files": self._total_files
                    }
            return {}

    def get_processing_speed(self) -> float:
        """Get the current processing speed in pages per minute"""
        with self.lock:
            if not self._start_time:
                return 0.0
            
            elapsed_time = time.time() - self._start_time
            total_pages_processed = self.get_processed_pages()
            
            if elapsed_time > 0 and total_pages_processed > 0:
                return (total_pages_processed * 60) / elapsed_time  # pages per minute
            
            return 0.0

    def get_estimated_time_remaining(self) -> int:
        """Get estimated time remaining in seconds"""
        with self.lock:
            speed = self.get_processing_speed()
            if speed <= 0:
                return 0
            
            total_pages_processed = self.get_processed_pages()
            remaining_pages = self._total_pages - total_pages_processed
            
            if remaining_pages <= 0:
                return 0
            
            # Convert from pages per minute to seconds
            return int((remaining_pages / speed) * 60)
