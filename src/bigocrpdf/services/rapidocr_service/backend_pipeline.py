"""PDF Processing Pipeline Mixin for ProfessionalPDFOCR."""

import gc
import json
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import pikepdf

from bigocrpdf.services.rapidocr_service.config import OCRBoxData, ProcessingStats
from bigocrpdf.services.rapidocr_service.pdf_assembly import (
    smart_merge_pdfs,
)
from bigocrpdf.services.rapidocr_service.pdf_extractor import (
    extract_image_positions,
    load_image_with_exif_rotation,
)
from bigocrpdf.services.rapidocr_service.rotation import (
    PageRotation,
    apply_final_rotation_to_pdf,
    extract_page_rotations,
)
from bigocrpdf.services.rapidocr_service.rotation import (
    apply_editor_modifications as apply_editor_mods_to_rotations,
)
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger


class BackendPipelineMixin:
    """Mixin providing PDF processing pipeline methods."""

    def _build_ocr_subprocess_cmd(self) -> list[str]:
        """Build command-line args for persistent OCR subprocess.

        Returns:
            Command list ready for subprocess.Popen.
        """
        import multiprocessing

        worker_script = str(Path(__file__).parent / "ocr_worker.py")
        cpu_count = multiprocessing.cpu_count()

        cmd = [
            "python3",
            worker_script,
            "--persistent",
            "--language",
            self.config.language,
            "--limit_side_len",
            str(self.config.detection_limit_side_len),
            "--box-thresh",
            str(self.config.box_thresh),
            "--unclip-ratio",
            str(self.config.unclip_ratio),
            "--text-score",
            str(self.config.text_score_threshold),
            "--score-mode",
            self.config.score_mode,
            "--threads",
            str(max(2, cpu_count // 2)),
        ]

        try:
            if not self._check_openvino_available():
                cmd.append("--no-openvino")
        except Exception:
            cmd.append("--no-openvino")

        for flag, getter in [
            ("--rec-model-path", self.config.get_rec_model_path),
            ("--rec-keys-path", self.config.get_rec_keys_path),
            ("--det-model-path", self.config.get_det_model_path),
            ("--font-path", self.config.get_font_path),
        ]:
            path = getter()
            if path:
                cmd.extend([flag, str(path)])

        return cmd

    def _start_ocr_subprocess(self) -> subprocess.Popen:
        """Start a persistent OCR subprocess.

        The subprocess loads the RapidOCR model once and waits for image
        paths on stdin. This uses ~400 MB total (one model instance)
        vs ~2+ GB with subprocess-per-page approach.

        Returns:
            Popen instance with stdin/stdout pipes.
        """
        cmd = self._build_ocr_subprocess_cmd()
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Wait for "ready" signal (model loaded)
        ready_line = proc.stdout.readline()
        if not ready_line:
            stderr_out = proc.stderr.read()
            raise RuntimeError(f"OCR subprocess failed to start: {stderr_out[:500]}")

        try:
            ready = json.loads(ready_line.strip())
            if ready.get("fatal"):
                raise RuntimeError(f"OCR subprocess fatal error: {ready['fatal']}")
        except json.JSONDecodeError:
            logger.warning(f"Unexpected OCR subprocess output: {ready_line.strip()}")

        logger.info("Persistent OCR subprocess started (model loaded)")
        return proc

    def _ocr_image_via_subprocess(self, proc: subprocess.Popen, image_path: str) -> dict | None:
        """Send an image to the persistent OCR subprocess and get results.

        Args:
            proc: Running OCR subprocess with stdin/stdout pipes
            image_path: Path to preprocessed image file

        Returns:
            Dict with boxes/txts/scores, or None on failure.
        """
        try:
            proc.stdin.write(image_path + "\n")
            proc.stdin.flush()

            result_line = proc.stdout.readline()
            if not result_line:
                logger.error("OCR subprocess closed unexpectedly")
                return None

            result = json.loads(result_line.strip())
            if result.get("error"):
                logger.error(f"OCR error for {image_path}: {result['error']}")
                return None

            return result

        except (BrokenPipeError, OSError) as e:
            logger.error(f"OCR subprocess pipe error: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"OCR subprocess invalid JSON: {e}")
            return None

    def _stop_ocr_subprocess(self, proc: subprocess.Popen) -> None:
        """Gracefully stop the persistent OCR subprocess."""
        try:
            proc.stdin.close()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
            proc.wait(timeout=5)

    def _prepare_pages_for_processing(
        self,
        extracted_images: list[Path | None],
        page_rotations: list,
    ) -> tuple[list[Path | None], list[dict]]:
        """Filter extracted images by deletion flags and build rotation dicts.

        Returns:
            Tuple of (valid_images, valid_rotation_dicts)
        """
        start_page = 1
        if self.config.page_range:
            start_page = self.config.page_range[0]

        valid_images = []
        valid_rotations = []

        for i, img_path in enumerate(extracted_images):
            page_num = start_page + i
            original_idx = page_num - 1

            if 0 <= original_idx < len(page_rotations):
                rot = page_rotations[original_idx]
            else:
                rot = PageRotation(page_number=page_num)
                logger.warning(f"Page {page_num}: No rotation info found")

            if rot.deleted:
                logger.info(f"Skipping deleted page {page_num} (adding placeholder)")
                valid_images.append(None)
            elif img_path is None:
                logger.warning(f"No image found for page {page_num}, using placeholder")
                valid_images.append(None)
            else:
                valid_images.append(img_path)

            valid_rotations.append(
                {
                    "rotation": rot.original_pdf_rotation,
                    "mediabox": rot.mediabox,
                    "page_rotation": rot,
                }
            )

        return valid_images, valid_rotations

    def _process_image_only_pdf(
        self,
        input_pdf: Path,
        output_pdf: Path,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> ProcessingStats:
        """Process a PDF where pages are just images (scanned).

        Architecture (memory-optimized):
        - ThreadPoolExecutor for image preprocessing (no fork = no memory duplication)
        - Single persistent OCR subprocess (one model instance = ~400 MB)
        - Chunked extraction to minimize disk usage

        Total memory: ~600-700 MB vs ~4-10 GB with the old fork+subprocess approach.
        """
        CHUNK_SIZE = 5

        logger.info(f"Processing image-only PDF: {input_pdf}")
        stats = ProcessingStats()
        start_time = time.time()

        output_dir = output_pdf.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Images go to /tmp (RAM on Linux) for faster I/O
        images_temp_dir = tempfile.TemporaryDirectory(prefix="rapidocr_imgs_")
        images_dir = Path(images_temp_dir.name) / "chunk_imgs"
        images_dir.mkdir()

        # Temporary PDFs stay in the output directory (same filesystem = fast rename)
        # Use descriptive names so the user knows what they are
        merged_pdf = output_dir / f".{output_pdf.stem}_processing.pdf"
        text_layer_pdf = output_dir / f".{output_pdf.stem}_textlayer.pdf"

        try:
            if progress_callback:
                progress_callback(0, 100, _("Analyzing PDF..."))

            # 1. Metadata pass (fast, no images extracted)
            page_rotations = extract_page_rotations(input_pdf)
            if self.config.page_modifications:
                page_rotations = apply_editor_mods_to_rotations(
                    page_rotations, self.config.page_modifications
                )

            total_pages = len(page_rotations)
            stats.pages_total = total_pages

            if total_pages == 0:
                logger.warning("No pages found in PDF!")
                return stats

            # Build rotation dicts for all pages (metadata only)
            all_rotation_dicts = []
            for rot in page_rotations:
                all_rotation_dicts.append(
                    {
                        "rotation": rot.original_pdf_rotation,
                        "mediabox": rot.mediabox,
                        "page_rotation": rot,
                    }
                )

            # 2. Setup persistent OCR subprocess (single model instance, ~400 MB)
            from reportlab.pdfgen import canvas

            from bigocrpdf.services.rapidocr_service.page_worker import process_page

            # Start persistent OCR subprocess (model loaded once, ~400 MB)
            ocr_proc = self._start_ocr_subprocess()

            logger.info(
                f"Chunked processing: {total_pages} pages in chunks of "
                f"{CHUNK_SIZE}, sequential preprocessing, "
                f"1 persistent OCR subprocess"
            )

            # 3. Chunked extraction + preprocessing + OCR loop
            c = canvas.Canvas(str(text_layer_pdf))
            page_standalone_flags: list[bool] = []
            total_confidence = 0.0
            num_chunks = (total_pages + CHUNK_SIZE - 1) // CHUNK_SIZE

            if progress_callback:
                progress_callback(5, 100, _("Starting OCR..."))

            try:
                for chunk_idx in range(num_chunks):
                    chunk_start = chunk_idx * CHUNK_SIZE  # 0-based
                    chunk_end = min(chunk_start + CHUNK_SIZE, total_pages)
                    chunk_page_range = (chunk_start + 1, chunk_end)  # 1-based

                    # Clean images dir for this chunk
                    for f in images_dir.glob("*"):
                        try:
                            f.unlink()
                        except OSError:
                            pass

                    # Extract only this chunk's images
                    if self.config.render_full_pages:
                        chunk_images = self.extractor.render_pages(
                            input_pdf,
                            output_dir=images_dir,
                            page_range=chunk_page_range,
                        )
                    else:
                        chunk_images = self.extractor.extract(
                            input_pdf,
                            output_dir=images_dir,
                            page_range=chunk_page_range,
                        )

                    # Process each page sequentially: preprocess → OCR → render
                    # Sequential avoids heap fragmentation from concurrent threads
                    for i, img_path in enumerate(chunk_images):
                        if hasattr(self, "cancel_event") and self.cancel_event.is_set():
                            raise InterruptedError("Processing cancelled by user")

                        abs_idx = chunk_start + i
                        page_num = abs_idx + 1
                        rot = page_rotations[abs_idx]
                        effective_path = None if rot.deleted else img_path

                        work_item = {
                            "page_num": page_num,
                            "img_path": (str(effective_path) if effective_path else None),
                            "config": self.config,
                            "pdf_rotation": rot.original_pdf_rotation,
                            "run_ocr": False,
                        }

                        # Step 1: Preprocess image
                        result = process_page(work_item)

                        # Step 2: OCR via persistent subprocess
                        if result.get("success") and result.get("temp_out_path"):
                            ocr_raw = self._ocr_image_via_subprocess(
                                ocr_proc, result["temp_out_path"]
                            )
                            result["ocr_raw"] = ocr_raw

                        # Step 3: Render text layer
                        progress_pct = 5 + int((page_num / total_pages) * 75)
                        if progress_callback:
                            progress_callback(
                                progress_pct,
                                100,
                                _("Processing page {0}/{1}...").format(page_num, total_pages),
                            )

                        try:
                            confidence, needs_standalone = self._process_page_result(
                                c,
                                result,
                                work_item,
                                all_rotation_dicts,
                                page_num,
                                stats,
                            )
                            total_confidence += confidence
                            page_standalone_flags.append(needs_standalone)
                        except Exception as page_err:
                            logger.error(f"Error processing page {page_num}: {page_err}")
                            stats.warnings.append(f"Page {page_num} failed: {page_err}")
                            c.setPageSize((595, 842))
                            c.showPage()
                            page_standalone_flags.append(False)

                        # Free page data immediately
                        del result
                        gc.collect()
                        # Force glibc to return freed pages to OS
                        try:
                            import ctypes

                            ctypes.CDLL("libc.so.6").malloc_trim(0)
                        except Exception:
                            pass

                        # Restart OCR subprocess every 3 pages to limit memory growth
                        if (abs_idx + 1) % 3 == 0 and abs_idx < total_pages - 1:
                            self._stop_ocr_subprocess(ocr_proc)
                            ocr_proc = self._start_ocr_subprocess()

                    logger.info(
                        f"Chunk {chunk_idx + 1}/{num_chunks} done "
                        f"(pages {chunk_start + 1}-{chunk_end})"
                    )

                    # Free accumulated memory between chunks
                    gc.collect()
                    try:
                        import ctypes

                        ctypes.CDLL("libc.so.6").malloc_trim(0)
                    except Exception:
                        pass

            finally:
                self._stop_ocr_subprocess(ocr_proc)

            c.save()
            stats.average_confidence = total_confidence
            self._page_standalone_flags = page_standalone_flags

            # 4. Merge text layer with original
            if progress_callback:
                progress_callback(85, 100, _("Merging text layer..."))

            any_standalone = any(page_standalone_flags) if page_standalone_flags else False
            format_changed = self.config.image_export_format not in ("original", "")

            if format_changed:
                import shutil

                shutil.copy2(text_layer_pdf, merged_pdf)
            elif any_standalone:
                smart_merge_pdfs(input_pdf, text_layer_pdf, merged_pdf, page_standalone_flags)
            else:
                self._overlay_text_on_original(input_pdf, text_layer_pdf, merged_pdf)

            # 5. Apply editor modifications
            start_page = 1
            if self.config.page_range:
                start_page = self.config.page_range[0]

            if self.config.page_modifications:
                apply_final_rotation_to_pdf(merged_pdf, page_rotations, start_page)

            split_parts = self._finalize_output(merged_pdf, output_pdf, progress_callback)
            if split_parts:
                stats.split_output_files = [str(p) for p in split_parts]

        except InterruptedError:
            logger.info("Processing cancelled by user")
            raise
        except Exception as e:
            logger.error(f"Error in image-only PDF processing: {e}")
            stats.error = str(e)
            raise
        finally:
            if "images_temp_dir" in locals():
                images_temp_dir.cleanup()
            # Clean up temporary PDFs in the output directory
            for temp_pdf in (merged_pdf, text_layer_pdf):
                try:
                    if temp_pdf.exists():
                        temp_pdf.unlink()
                except OSError:
                    pass

        self._calculate_final_stats(stats, start_time)

        if progress_callback:
            progress_callback(100, 100, _("Done!"))

        logger.info(f"Processing complete in {stats.processing_time_seconds:.1f}s")
        logger.info(
            f"Pages: {stats.pages_processed}, "
            f"Text regions: {stats.total_text_regions}, "
            f"Avg confidence: {stats.average_confidence:.2%}"
        )

        return stats

    def _ocr_image_in_page(
        self,
        img_path: Path,
        img_pos,
        pdf: "pikepdf.Pdf",
        page: "pikepdf.Page",
        page_num: int,
        page_width: float,
        page_height: float,
        stats: ProcessingStats,
    ) -> list[str]:
        """OCR a single image within a page and overlay invisible text.

        Returns:
            List of OCR text strings found in the image.
        """
        # Skip very small images (likely icons or decorations)
        if img_pos.width < 50 or img_pos.height < 50:
            logger.debug(f"Skipping small image: {img_pos.width}x{img_pos.height}")
            return []

        img = load_image_with_exif_rotation(img_path)
        if img is None:
            logger.warning(f"Could not load image: {img_path}")
            return []

        processed_img = self.preprocessor.process(img)
        ocr_results = self._run_ocr(processed_img)

        if not ocr_results:
            logger.debug(f"No text found in image at ({img_pos.x}, {img_pos.y})")
            return []

        img_h, img_w = processed_img.shape[:2]
        scale_x = img_pos.width / img_w
        scale_y = img_pos.height / img_h

        text_commands = self._create_text_layer_commands(
            ocr_results,
            img_pos.x,
            img_pos.y,
            img_pos.width,
            img_pos.height,
            scale_x,
            scale_y,
        )
        self._append_text_to_page(pdf, page, text_commands)

        stats.total_text_regions += len(ocr_results)

        ocr_texts = [r.text for r in ocr_results]

        # Collect OCR boxes for ODF export
        for r in ocr_results:
            xs = [p[0] for p in r.box]
            ys = [p[1] for p in r.box]
            box_x = min(xs) * scale_x + img_pos.x
            box_y = min(ys) * scale_y + img_pos.y
            box_w = (max(xs) - min(xs)) * scale_x
            box_h = (max(ys) - min(ys)) * scale_y

            x_pct = (box_x / page_width) * 100 if page_width > 0 else 0
            y_pct = (box_y / page_height) * 100 if page_height > 0 else 0
            w_pct = (box_w / page_width) * 100 if page_width > 0 else 0

            stats.ocr_boxes.append(
                OCRBoxData(
                    text=r.text,
                    x=x_pct,
                    y=y_pct,
                    width=w_pct,
                    height=box_h,
                    confidence=r.confidence,
                    page_num=page_num,
                )
            )

        logger.debug(
            f"Added {len(ocr_results)} text regions for image at ({img_pos.x:.1f}, {img_pos.y:.1f})"
        )

        return ocr_texts

    @staticmethod
    def _extract_native_text(input_pdf: Path) -> str:
        """Extract existing text from PDF using pdftotext."""
        try:
            result = subprocess.run(
                ["pdftotext", "-layout", str(input_pdf), "-"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                return result.stdout
        except Exception as e:
            logger.warning(f"Could not extract native text: {e}")
        return ""

    @staticmethod
    def _extract_and_filter_images(input_pdf: Path, images_dir: Path) -> list[Path]:
        """Extract images from PDF and filter out masks/small icons."""
        cmd = ["pdfimages", "-all", str(input_pdf), str(images_dir / "img")]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"pdfimages failed: {e.stderr}")
            raise RuntimeError(f"Failed to extract images: {e}") from e

        extracted = sorted(images_dir.glob("img-*"))
        filtered = [
            img
            for img in extracted
            if not (img.stat().st_size < 5000 and img.suffix.lower() == ".png")
        ]
        logger.info(f"Extracted {len(filtered)} images (after filtering masks)")
        return filtered

    def _process_mixed_content_pdf(
        self,
        input_pdf: Path,
        output_pdf: Path,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> ProcessingStats:
        """Process mixed content PDF (text + images).

        Preserves original PDF structure and adds invisible OCR text layers
        at image positions.
        """
        import shutil

        start_time = time.time()
        stats = ProcessingStats()

        if progress_callback:
            progress_callback(0, 100, _("Analyzing PDF structure..."))

        image_positions = extract_image_positions(input_pdf)

        if not image_positions:
            logger.info("No images found in PDF. Copying original.")
            shutil.copy2(input_pdf, output_pdf)
            stats.warnings.append("No images found to OCR in mixed content PDF")
            self._calculate_final_stats(stats, start_time)
            return stats

        total_images = sum(len(imgs) for imgs in image_positions.values())
        logger.info(f"Found {total_images} image(s) across {len(image_positions)} page(s)")
        stats.pages_total = len(image_positions)

        native_text = self._extract_native_text(input_pdf)
        ocr_texts = []

        with tempfile.TemporaryDirectory() as temp_dir:
            images_dir = Path(temp_dir) / "images"
            images_dir.mkdir()

            if progress_callback:
                progress_callback(5, 100, _("Extracting images..."))

            extracted_images = self._extract_and_filter_images(input_pdf, images_dir)

            with pikepdf.open(input_pdf, allow_overwriting_input=True) as pdf:
                processed_images = 0
                current_img_idx = 0

                for page_num in sorted(image_positions.keys()):
                    # Check for cancellation between pages
                    if hasattr(self, "cancel_event") and self.cancel_event.is_set():
                        logger.info("Processing cancelled by user in mixed content mode")
                        raise InterruptedError("Processing cancelled by user")

                    page_imgs = image_positions[page_num]
                    page = pdf.pages[page_num - 1]
                    mediabox = page.mediabox
                    page_height = float(mediabox[3]) - float(mediabox[1])
                    page_width = float(mediabox[2]) - float(mediabox[0])

                    if progress_callback:
                        pct = 10 + int(80 * processed_images / total_images)
                        progress_callback(pct, 100, _("OCR page {0}...").format(page_num))

                    for img_pos in page_imgs:
                        if current_img_idx >= len(extracted_images):
                            logger.warning(
                                f"Image index {current_img_idx} exceeds extracted images"
                            )
                            break

                        img_path = extracted_images[current_img_idx]
                        current_img_idx += 1

                        try:
                            texts = self._ocr_image_in_page(
                                img_path,
                                img_pos,
                                pdf,
                                page,
                                page_num,
                                page_width,
                                page_height,
                                stats,
                            )
                            ocr_texts.extend(texts)
                            if texts:
                                processed_images += 1
                        except Exception as e:
                            logger.error(f"Error processing image {img_path}: {e}")
                            stats.warnings.append(f"Failed to OCR image: {e}")
                            continue

                if progress_callback:
                    progress_callback(90, 100, _("Saving PDF..."))

                stats.pages_processed = len(image_positions)
                pdf.save(output_pdf)

        # Split by file size if limit is configured
        max_mb = self.config.max_file_size_mb
        if max_mb > 0:
            file_size_mb = output_pdf.stat().st_size / (1024 * 1024)
            if file_size_mb > max_mb:
                if progress_callback:
                    progress_callback(92, 100, _("Splitting PDF by size limit..."))
                split_parts = self._split_pdf_by_size(output_pdf, max_mb)
                if split_parts:
                    stats.split_output_files = [str(p) for p in split_parts]

        # Combine native text with OCR text
        if native_text:
            stats.full_text = native_text.strip()
            if ocr_texts:
                stats.full_text += "\n\n--- " + _("Text extracted by OCR from images") + " ---\n\n"
                stats.full_text += "\n".join(ocr_texts)
        else:
            stats.full_text = "\n".join(ocr_texts)

        self._calculate_final_stats(stats, start_time)

        if progress_callback:
            progress_callback(100, 100, _("Done!"))

        logger.info(f"Mixed content processing complete in {stats.processing_time_seconds:.1f}s")
        logger.info(f"Pages: {stats.pages_processed}, Text regions: {stats.total_text_regions}")

        return stats

    def _finalize_output(
        self,
        merged_pdf: Path,
        output_pdf: Path,
        progress_callback: Callable[[int, int, str], None] | None,
    ) -> list[Path]:
        """Finalize output PDF, optionally converting to PDF/A and splitting by size.

        Uses shutil.move when source and destination are on the same
        filesystem for near-instant transfer (rename), avoiding
        large temporary copies in /tmp.

        Args:
            merged_pdf: Path to merged PDF
            output_pdf: Path for final output PDF
            progress_callback: Optional progress callback

        Returns:
            List of split part paths if splitting occurred, empty list otherwise
        """
        import shutil

        if self.config.convert_to_pdfa:
            if progress_callback:
                progress_callback(90, 100, _("Converting to PDF/A..."))
            self._convert_to_pdfa(merged_pdf, output_pdf)
        else:
            # Use move instead of copy — instant rename on same filesystem
            shutil.move(str(merged_pdf), str(output_pdf))

        # Split by file size if limit is configured
        max_mb = self.config.max_file_size_mb
        if max_mb > 0:
            file_size_mb = output_pdf.stat().st_size / (1024 * 1024)
            if file_size_mb > max_mb:
                if progress_callback:
                    progress_callback(92, 100, _("Splitting PDF by size limit..."))
                return self._split_pdf_by_size(output_pdf, max_mb)

        return []

    def _split_pdf_by_size(self, output_pdf: Path, max_mb: int) -> list[Path]:
        """Split a PDF into multiple parts that each stay under the size limit.

        Output files are named: {stem}-01.pdf, {stem}-02.pdf, etc.
        The original file is replaced by the split parts.

        Args:
            output_pdf: Path to the PDF to split
            max_mb: Maximum file size in MB per part

        Returns:
            List of split part paths, or empty list if no split was needed
        """
        import io

        import pikepdf

        max_bytes = max_mb * 1024 * 1024

        with pikepdf.open(output_pdf) as source_pdf:
            total_pages = len(source_pdf.pages)

            if total_pages <= 1:
                logger.info(
                    f"PDF has only 1 page ({output_pdf.stat().st_size / 1024 / 1024:.1f} MB), "
                    f"cannot split further"
                )
                return []

            stem = output_pdf.stem
            suffix = output_pdf.suffix
            parent = output_pdf.parent
            parts: list[Path] = []
            current_pages: list[int] = []

            for page_idx in range(total_pages):
                current_pages.append(page_idx)

                # Estimate size by writing to memory buffer
                test_pdf = pikepdf.new()
                for idx in current_pages:
                    test_pdf.pages.append(source_pdf.pages[idx])

                buf = io.BytesIO()
                test_pdf.save(buf)
                current_size = buf.tell()
                test_pdf.close()
                buf.close()

                if current_size > max_bytes and len(current_pages) > 1:
                    # Remove the page that pushed us over the limit
                    current_pages.pop()

                    # Save this part
                    part_num = len(parts) + 1
                    part_path = parent / f"{stem}-{part_num:02d}{suffix}"
                    part_pdf = pikepdf.new()
                    for idx in current_pages:
                        part_pdf.pages.append(source_pdf.pages[idx])
                    part_pdf.save(str(part_path))
                    part_pdf.close()
                    parts.append(part_path)

                    logger.info(
                        f"Split part {part_num}: {len(current_pages)} pages, "
                        f"{part_path.stat().st_size / 1024 / 1024:.1f} MB"
                    )

                    # Start new part with the page that didn't fit
                    current_pages = [page_idx]

            # Save the last part
            if current_pages:
                part_num = len(parts) + 1
                part_path = parent / f"{stem}-{part_num:02d}{suffix}"
                part_pdf = pikepdf.new()
                for idx in current_pages:
                    part_pdf.pages.append(source_pdf.pages[idx])
                part_pdf.save(str(part_path))
                part_pdf.close()
                parts.append(part_path)

                logger.info(
                    f"Split part {part_num}: {len(current_pages)} pages, "
                    f"{part_path.stat().st_size / 1024 / 1024:.1f} MB"
                )

        # Remove the original oversized file
        if parts:
            output_pdf.unlink()
            logger.info(
                f"PDF split into {len(parts)} parts "
                f"(max {max_mb} MB each): {[p.name for p in parts]}"
            )
            return parts

        return []

    def _calculate_final_stats(self, stats: ProcessingStats, start_time: float) -> None:
        """Calculate final processing statistics.

        Args:
            stats: Stats object to update
            start_time: Processing start time
        """
        stats.processing_time_seconds = time.time() - start_time
        if stats.total_text_regions > 0:
            stats.average_confidence = stats.average_confidence / stats.total_text_regions
