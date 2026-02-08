"""Conclusion Page ODF Export Mixin."""

import os

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gtk

from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger
from bigocrpdf.utils.odf_exporter import OCRTextData, ODFExporter


class ConclusionExportMixin:
    """Mixin providing ODF export functionality for the conclusion page."""

    def _show_odf_export_options_dialog(self, file_path: str) -> None:
        """Show dialog with ODF export options.

        Args:
            file_path: Path to the source PDF file
        """
        dialog = Adw.MessageDialog(
            transient_for=self.window,
            heading=_("ODF Export Options"),
            body=_(
                "Choose the export format:\n\n"
                "• Formatted + Images: Page images with structured text layout below\n"
                "• Images + Simple Text: Page images with plain text below\n"
                "• Formatted Text: Structured text layout only (no images)\n"
                "• Plain Text: Simple text without formatting"
            ),
        )

        dialog.add_response("formatted_images", _("Formatted + Images"))
        dialog.add_response("with_images", _("Images + Simple Text"))
        dialog.add_response("formatted", _("Formatted Text"))
        dialog.add_response("plain", _("Plain Text"))
        dialog.add_response("cancel", _("Cancel"))

        # Determine default based on settings
        settings = self.window.settings
        include_images = getattr(settings, "odf_include_images", True)
        use_formatting = getattr(settings, "odf_use_formatting", True)

        if include_images and use_formatting:
            default_response = "formatted_images"
        elif include_images and not use_formatting:
            default_response = "with_images"
        elif not include_images and use_formatting:
            default_response = "formatted"
        else:
            default_response = "plain"

        dialog.set_response_appearance(default_response, Adw.ResponseAppearance.SUGGESTED)
        dialog.set_default_response(default_response)
        dialog.set_close_response("cancel")

        dialog.connect(
            "response",
            lambda d, r: self._on_odf_options_response(d, r, file_path),
        )
        dialog.present()

    def _on_odf_options_response(
        self, dialog: Adw.MessageDialog, response: str, file_path: str
    ) -> None:
        """Handle ODF export options dialog response.

        Args:
            dialog: The options dialog
            response: The selected response
            file_path: Source PDF file path
        """
        if response == "cancel":
            return

        # Store the selected export mode
        self._odf_export_mode = response

        # Get extracted text
        extracted_text = self._get_extracted_text_for_file(file_path)
        if not extracted_text:
            self._show_toast(_("No text to export"))
            return

        self._show_odf_file_dialog(file_path, extracted_text)

    def _show_odf_file_dialog(self, file_path: str, extracted_text: str) -> None:
        """Show file save dialog for ODF export.

        Args:
            file_path: Source PDF file path
            extracted_text: Text to export
        """
        from gi.repository import Gio

        # Create file save dialog
        save_dialog = Gtk.FileDialog.new()
        save_dialog.set_title(_("Export to OpenDocument"))
        save_dialog.set_modal(True)

        # Default filename based on source file
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_dialog.set_initial_name(f"{base_name}.odt")

        # Add ODF filter
        filters = Gio.ListStore.new(Gtk.FileFilter)
        odf_filter = Gtk.FileFilter()
        odf_filter.set_name("OpenDocument Text (*.odt)")
        odf_filter.add_pattern("*.odt")
        odf_filter.add_mime_type("application/vnd.oasis.opendocument.text")
        filters.append(odf_filter)
        save_dialog.set_filters(filters)
        save_dialog.set_default_filter(odf_filter)

        # Show save dialog
        save_dialog.save(
            parent=self.window,
            cancellable=None,
            callback=lambda d, r: self._on_odf_save_response(d, r, file_path, extracted_text),
        )

    def _on_odf_save_response(
        self, dialog: Gtk.FileDialog, result, file_path: str, extracted_text: str
    ) -> None:
        """Handle ODF save dialog response.

        Args:
            dialog: File dialog
            result: Async result
            file_path: Source PDF file path
            extracted_text: Text to export
        """
        try:
            output_path = self._get_odf_output_path(dialog, result)
            if not output_path:
                return
            self._export_odf_file(output_path, file_path, extracted_text)
        except Exception as e:
            self._handle_odf_export_error(e)

    def _get_odf_output_path(self, dialog: Gtk.FileDialog, result) -> str | None:
        """Get and validate the ODF output path from dialog result.

        Args:
            dialog: File dialog
            result: Async result

        Returns:
            Output path with .odt extension, or None if cancelled
        """
        file = dialog.save_finish(result)
        output_path = file.get_path()
        if not output_path.lower().endswith(".odt"):
            output_path += ".odt"
        return output_path

    def _export_odf_file(self, output_path: str, file_path: str, extracted_text: str) -> None:
        """Export content to ODF file based on selected export mode.

        Args:
            output_path: Destination ODF file path
            file_path: Source PDF file path
            extracted_text: Fallback text if no OCR boxes available
        """
        exporter = ODFExporter()

        # Get export mode (default to formatted)
        export_mode = getattr(self, "_odf_export_mode", "formatted")

        # Plain text mode - no formatting
        if export_mode == "plain":
            logger.info("ODF Export: Using plain text mode")
            success = exporter.export_text(extracted_text, output_path)
            self._report_export_result(success, output_path)
            return

        ocr_boxes = self._get_ocr_boxes_for_file(file_path)
        is_mixed_content = "--- Texto extraído por OCR" in extracted_text

        if is_mixed_content:
            # Mixed content: extract images and use mixed content export
            logger.info(f"ODF Export: Mixed content PDF, mode={export_mode}")

            # Split extracted text into native and OCR portions
            separator = "--- Texto extraído por OCR das imagens ---"
            if separator in extracted_text:
                parts = extracted_text.split(separator, 1)
                native_text = parts[0].strip()
                ocr_text = parts[1].strip() if len(parts) > 1 else ""
            else:
                native_text = extracted_text
                ocr_text = ""

            if export_mode == "with_images":
                # Include images in export
                from bigocrpdf.utils.pdf_utils import extract_images_for_odf

                images, ocr_texts = extract_images_for_odf(file_path, ocr_text)
                if images:
                    success = exporter.export_mixed_content(
                        native_text, images, ocr_texts, output_path
                    )
                else:
                    success = exporter.export_text(extracted_text, output_path)
            else:
                # Formatted text only (no images)
                success = exporter.export_text(extracted_text, output_path)
        elif ocr_boxes:
            # Image-only PDF: use structured OCR data for better formatting
            if export_mode in ("with_images", "formatted_images"):
                # For image-only PDFs, include the images per page
                images = self._extract_images_per_page(file_path)

                if images:
                    if export_mode == "with_images":
                        # Images + simple text (no OCR layout formatting)
                        page_texts = self._group_ocr_by_page(ocr_boxes)
                        success = exporter.export_paged_images(images, page_texts, output_path)
                    else:
                        # Formatted images: images + structured OCR layout formatting
                        ocr_data = [
                            OCRTextData(
                                text=box.text,
                                x=box.x,
                                y=box.y,
                                width=box.width,
                                height=box.height,
                                confidence=getattr(box, "confidence", 1.0),
                                page_num=getattr(box, "page_num", 1),
                            )
                            for box in ocr_boxes
                        ]
                        success = exporter.export_formatted_with_images(
                            ocr_data, images, output_path
                        )
                else:
                    success = exporter.export_text(extracted_text, output_path)
            else:
                # Formatted export using OCR data (no images)
                ocr_data = [
                    OCRTextData(
                        text=box.text,
                        x=box.x,
                        y=box.y,
                        width=box.width,
                        height=box.height,
                        confidence=getattr(box, "confidence", 1.0),
                        page_num=getattr(box, "page_num", 1),
                    )
                    for box in ocr_boxes
                ]
                success = exporter.export_structured_data(ocr_data, output_path)
        else:
            # Fallback to plain text export
            success = exporter.export_text(extracted_text, output_path)

        self._report_export_result(success, output_path)

    def _extract_images_per_page(self, pdf_path: str) -> list[str]:
        """Extract images from PDF, sorted by page order.

        Args:
            pdf_path: Path to the source PDF

        Returns:
            List of image paths sorted by page order
        """
        from bigocrpdf.utils.pdf_utils import extract_images_for_odf

        images, _ = extract_images_for_odf(pdf_path)
        return images

    def _group_ocr_by_page(self, ocr_boxes: list) -> list[str]:
        """Group OCR text by page number.

        Args:
            ocr_boxes: List of OCR boxes with page_num attribute

        Returns:
            List of OCR text strings, one per page
        """
        from collections import defaultdict

        # Group boxes by page number
        pages = defaultdict(list)
        for box in ocr_boxes:
            page_num = getattr(box, "page_num", 1)
            pages[page_num].append(box.text)

        # Find the maximum page number
        if not pages:
            return []

        max_page = max(pages.keys())

        # Build text for each page
        page_texts = []
        for page_num in range(1, max_page + 1):
            if page_num in pages:
                page_text = "\n".join(pages[page_num])
                page_texts.append(page_text)
            else:
                page_texts.append("")

        return page_texts

    def _get_ocr_boxes_for_file(self, file_path: str) -> list | None:
        """Get OCR boxes for a file if available.

        Args:
            file_path: Source PDF file path

        Returns:
            List of OCR boxes or None
        """
        if not hasattr(self.window.settings, "ocr_boxes"):
            logger.info("ODF Export: No ocr_boxes attribute in settings")
            return None

        ocr_boxes = self.window.settings.ocr_boxes.get(file_path)
        logger.info(
            f"ODF Export: Found {len(ocr_boxes) if ocr_boxes else 0} OCR boxes for {file_path}"
        )
        return ocr_boxes

    def _report_export_result(self, success: bool, output_path: str) -> None:
        """Report the export result to user.

        Args:
            success: Whether export succeeded
            output_path: Destination file path
        """
        if success:
            self._show_toast(_("Exported to {}").format(os.path.basename(output_path)))
        else:
            self._show_toast(_("Export failed"))

    def _handle_odf_export_error(self, error: Exception) -> None:
        """Handle errors during ODF export.

        Args:
            error: The exception that occurred
        """
        if "Dismissed" not in str(error):
            logger.error(f"Error exporting to ODF: {error}")
            self._show_toast(_("Export failed"))

    def _show_toast(self, message: str) -> None:
        """Show a toast notification

        Args:
            message: Message to display
        """
        if hasattr(self.window, "toast_overlay"):
            toast = Adw.Toast.new(message)
            self.window.toast_overlay.add_toast(toast)

    def _open_file(self, file_path: str) -> None:
        """Open a file using the default application

        Args:
            file_path: Path to the file to open
        """
        from bigocrpdf.utils.pdf_utils import open_file_with_default_app

        open_file_with_default_app(file_path)

    def _open_in_browser(self, file_path: str) -> None:
        """Open a file in the default web browser.

        Args:
            file_path: Path to the file to open
        """
        import subprocess

        if not file_path or not os.path.exists(file_path):
            logger.warning(f"Cannot open file in browser: {file_path}")
            return

        try:
            # Use xdg-open with the file URI for browser
            file_uri = f"file://{os.path.abspath(file_path)}"
            # Try to detect default browser
            result = subprocess.run(
                ["xdg-settings", "get", "default-web-browser"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            browser_desktop = result.stdout.strip() if result.returncode == 0 else ""

            if browser_desktop:
                # Open with the detected browser via gtk-launch
                browser_name = browser_desktop.replace(".desktop", "")
                subprocess.Popen(
                    ["gtk-launch", browser_desktop, file_uri],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info(f"Opened {file_path} in browser: {browser_name}")
            else:
                # Fallback to xdg-open
                subprocess.Popen(
                    ["xdg-open", file_uri],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info(f"Opened {file_path} with xdg-open")
        except Exception as e:
            logger.error(f"Failed to open file in browser: {e}")

    def _show_extracted_text(self, file_path: str) -> None:
        """Show extracted text dialog

        Args:
            file_path: Path to the PDF file
        """
        # Use the public interface via ui_manager
        if hasattr(self.window, "ui") and hasattr(self.window.ui, "show_extracted_text"):
            self.window.ui.show_extracted_text(file_path)
        else:
            logger.warning("Text viewer dialog not available")
            # Simple fallback - show a basic dialog
            self._show_simple_text_dialog(file_path)

    def _show_simple_text_dialog(self, file_path: str) -> None:
        """Show a simple text dialog as fallback

        Args:
            file_path: Path to the PDF file
        """
        # Get extracted text
        extracted_text = self._get_extracted_text_for_file(file_path)

        # Create simple dialog
        dialog = Adw.MessageDialog(transient_for=self.window)
        dialog.set_heading(_("Extracted Text"))
        dialog.set_body(
            extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
        )
        dialog.add_response("ok", _("OK"))
        dialog.present()

    def _get_extracted_text_for_file(self, file_path: str) -> str:
        """Get extracted text for a file

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text or placeholder message
        """
        # Check if we have text for this file
        if (
            hasattr(self.window.settings, "extracted_text")
            and file_path in self.window.settings.extracted_text
        ):
            return self.window.settings.extracted_text[file_path]

        # Try to read from sidecar file
        sidecar_file = os.path.splitext(file_path)[0] + ".txt"
        if os.path.exists(sidecar_file):
            try:
                with open(sidecar_file, encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading sidecar file: {e}")

        # Return placeholder text
        return _("No text content was detected in this file during OCR processing.")

    def reset_page(self) -> None:
        """Reset the conclusion page to initial state"""
        if self.result_file_count:
            self.result_file_count.set_text("0")
        if self.result_page_count:
            self.result_page_count.set_text("0")
        if self.result_time:
            self.result_time.set_text("00:00")
        if self.result_file_size:
            self.result_file_size.set_text("0 KB")
        if self.result_size_change:
            self.result_size_change.set_text("--")
            self.result_size_change.remove_css_class("success")
            self.result_size_change.remove_css_class("warning")

        # Clear file list
        if self.output_list_box:
            self._clear_output_list()
