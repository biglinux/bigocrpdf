"""
BigOcrPdf - Window Module

This module contains the main application window implementation.
"""

import gi
import threading

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Gtk, Adw, Gio, GLib

import os
import time

from config import APP_ICON_NAME, CONFIG_DIR
from utils.logger import logger
from services.settings import OcrSettings
from services.processor import OcrProcessor
from ui.ui_manager import BigOcrPdfUI
from utils.i18n import _
from utils.timer import safe_remove_source


class BigOcrPdfWindow(Adw.ApplicationWindow):
    """Main application window for BigOcrPdf with stable progress tracking"""

    # Configuration dictionaries for dropdowns
    LANGUAGES_CONFIG = "LANGUAGES"  # This will be populated dynamically
    
    QUALITY_CONFIG = [
        ("normal", _("Normal")),
        ("economic", _("Economic")),
        ("economicplus", _("More economic")),
    ]
    
    ALIGNMENT_CONFIG = [
        ("none", _("Don't change")),
        ("align", _("Align")),
        ("rotate", _("Auto rotate")),
        ("alignrotate", _("Align + rotate")),
    ]
    
    ALIGNMENT_TOOLTIPS = [
        _("Keep pages as they are without any modifications"),
        _("Align page content to improve readability"),
        _("Automatically rotate pages to correct orientation"),
        _("Align page content and automatically rotate pages to correct orientation"),
    ]

    # Configuration file to store welcome dialog preference
    WELCOME_DIALOG_CONFIG = os.path.join(CONFIG_DIR, "show_welcome_dialog")

    def __init__(self, app: Adw.Application):
        """Initialize application window

        Args:
            app: The parent Adw.Application instance
        """
        super().__init__(application=app, title=_("Big OCR PDF"))
        self.set_default_size(820, 600)

        # Set up the window icon
        self.set_icon_name(APP_ICON_NAME)

        # Initialize components
        self.settings = OcrSettings()
        self.ocr_processor = OcrProcessor(self.settings)
        self.ui = BigOcrPdfUI(self)
        
        # Initialize state variables
        self.process_pid = None
        self.processed_files = []
        self.process_start_time = 0
        self.conclusion_timer_id = None
        
        # Remove old progress tracking variables - now handled by terminal_page_manager
        # No more: progress_timer_id, _animation_progress, _animation_direction

        # Initialize UI components
        self.stack = None
        self.toast_overlay = None
        self.step_label = None
        self.back_button = None
        self.next_button = None
        self.header_bar = None

        # Signal handler tracking
        self.signal_handlers = {}

        # Create the main layout
        self.setup_ui()

    def setup_ui(self) -> None:
        """Set up the main user interface"""
        self.toast_overlay = Adw.ToastOverlay()

        self.setup_stack()
        self.setup_header_bar()
        self.setup_content_area()
        self.setup_action_bar()
        
        self.setup_pages()
        
        self.stack.connect("notify::visible-child", self._on_page_changed)

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        main_box.append(self.toolbar_view)
        main_box.append(self.action_bar)

        self.toast_overlay.set_child(main_box)
        self.set_content(self.toast_overlay)
        
    def do_realize(self):
        """Called when the window is being realized"""
        Adw.ApplicationWindow.do_realize(self)
        
        if hasattr(self, 'stack') and self.stack:
            self.stack.set_visible_child_name("settings")

    def setup_stack(self) -> None:
        """Set up the main view stack"""
        self.stack = Adw.ViewStack()
        self.stack.set_vexpand(True)
        self.stack.set_transition_duration(300)

    def setup_header_bar(self) -> None:
        """Set up the application header bar"""
        self.header_bar = Adw.HeaderBar()
        self.header_bar.set_show_end_title_buttons(True)
        self.header_bar.set_show_start_title_buttons(True)
        self.header_bar.add_css_class("flat")

        title_label = Gtk.Label(label=_("BigOcrPdf - Scanned PDFs with search support"))
        self.header_bar.set_title_widget(title_label)

        menu_button = Gtk.MenuButton()
        menu_button.set_icon_name("open-menu-symbolic")
        menu_button.set_tooltip_text(_("Menu"))

        menu = Gio.Menu()
        menu.append(_("Help"), "win.help")
        menu.append(_("About the application"), "app.about")
        menu_button.set_menu_model(menu)
        self.header_bar.pack_end(menu_button)

        help_action = Gio.SimpleAction.new("help", None)
        help_action.connect("activate", lambda a, p: self.show_welcome_dialog())
        self.add_action(help_action)

    def setup_content_area(self) -> None:
        """Set up the main content area with scrolling"""
        content_scroll = Gtk.ScrolledWindow()
        content_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        content_scroll.set_propagate_natural_height(True)
        content_scroll.set_min_content_height(450)
        content_scroll.set_vexpand(True)
        content_scroll.set_child(self.stack)

        self.toolbar_view = Adw.ToolbarView()
        self.toolbar_view.add_top_bar(self.header_bar)
        self.toolbar_view.set_content(content_scroll)

    def setup_action_bar(self) -> None:
        """Set up the bottom action bar for navigation"""
        self.action_bar = Gtk.ActionBar()

        self.step_label = Gtk.Label()
        self.step_label.set_markup(
            "<span font_size='small'>" + _("Step 1/3: Settings") + "</span>"
        )
        self.step_label.add_css_class("dim-label")
        self.step_label.set_margin_start(12)
        self.step_label.set_hexpand(True)
        self.step_label.set_halign(Gtk.Align.CENTER)
        self.action_bar.set_center_widget(self.step_label)

        self.back_button = Gtk.Button()
        self.back_button.set_label(_("Back"))
        self.back_button.set_icon_name("go-previous-symbolic")
        self.connect_signal(self.back_button, "clicked", self.on_back_clicked)
        self.back_button.set_sensitive(False)
        self.back_button.set_visible(False)

        self.next_button = Gtk.Button()
        self.next_button.set_label(_("Start"))
        self.next_button.add_css_class("suggested-action")
        self.connect_signal(self.next_button, "clicked", self.on_next_clicked)

        self.action_bar.pack_start(self.back_button)
        self.action_bar.pack_end(self.next_button)

    def setup_pages(self) -> None:
        """Set up the application pages"""
        settings_page = self.ui.create_settings_page()
        self.stack.add_named(settings_page, "settings")

        terminal_page = self.ui.create_terminal_page()
        self.stack.add_named(terminal_page, "terminal")

        conclusion_page = self.ui.create_conclusion_page()
        self.stack.add_named(conclusion_page, "conclusion")

    def connect_signal(self, widget, signal, callback, *args) -> int:
        """Connect a signal and store the handler ID"""
        handler_id = widget.connect(signal, callback, *args)
        
        if widget not in self.signal_handlers:
            self.signal_handlers[widget] = {}
        
        self.signal_handlers[widget][signal] = handler_id
        
        return handler_id

    def disconnect_signal(self, widget, signal) -> bool:
        """Disconnect a signal if it exists"""
        if widget in self.signal_handlers and signal in self.signal_handlers[widget]:
            widget.disconnect(self.signal_handlers[widget][signal])
            del self.signal_handlers[widget][signal]
            return True
        return False

    def should_show_welcome_dialog(self) -> bool:
        """Check if the welcome dialog should be shown at startup"""
        if not os.path.exists(self.WELCOME_DIALOG_CONFIG):
            try:
                with open(self.WELCOME_DIALOG_CONFIG, "w") as f:
                    f.write("true")
                return True
            except Exception as e:
                logger.error(f"Error creating welcome dialog config: {e}")
                return True
        
        try:
            with open(self.WELCOME_DIALOG_CONFIG, "r") as f:
                content = f.read().strip()
                return content.lower() == "true"
        except Exception as e:
            logger.error(f"Error reading welcome dialog config: {e}")
            return True

    def set_show_welcome_dialog(self, show: bool) -> None:
        """Set whether to show the welcome dialog at startup"""
        try:
            with open(self.WELCOME_DIALOG_CONFIG, "w") as f:
                f.write("true" if show else "false")
            logger.info(f"Set show welcome dialog: {show}")
        except Exception as e:
            logger.error(f"Error setting welcome dialog config: {e}")

    def show_welcome_dialog(self) -> None:
        """Show the welcome dialog as a centered modal"""
        dialog = Adw.Window()
        dialog.set_default_size(650, 380)
        dialog.set_modal(True)
        dialog.set_transient_for(self)
        dialog.set_resizable(False)
        dialog.set_hide_on_close(True)
        
        overlay = Gtk.Overlay()
        
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        content_box.set_margin_top(16)
        content_box.set_margin_bottom(20)
        content_box.set_margin_start(36)
        content_box.set_margin_end(36)
        
        icon = Gtk.Image.new_from_icon_name(APP_ICON_NAME)
        icon.set_pixel_size(64)
        icon.set_margin_bottom(16)
        icon.set_halign(Gtk.Align.CENTER)
        content_box.append(icon)
        
        what_is = Gtk.Label()
        what_is.set_markup(f"<span size='large' weight='bold'>{_('What is')} Big OCR PDF?</span>")
        what_is.set_halign(Gtk.Align.CENTER)
        what_is.set_margin_bottom(14)
        content_box.append(what_is)
        
        what_is_desc = Gtk.Label()
        what_is_desc.set_markup(_(
            "Big OCR PDF adds optical character recognition to your PDF files, "
            "making them searchable and allowing you to select and copy text "
            "from scanned documents."
        ))
        what_is_desc.set_wrap(True)
        what_is_desc.set_justify(Gtk.Justification.LEFT)
        what_is_desc.set_halign(Gtk.Align.START)
        what_is_desc.set_margin_bottom(16)
        what_is_desc.set_max_width_chars(65)
        content_box.append(what_is_desc)
        
        benefits = Gtk.Label()
        benefits.set_markup("<span weight='bold'>" + _("Benefits of using Big OCR PDF:") + "</span>")
        benefits.set_halign(Gtk.Align.START)
        benefits.set_margin_bottom(8)
        content_box.append(benefits)
        
        benefits_list = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        benefits_list.set_halign(Gtk.Align.START)
        benefits_list.set_margin_bottom(16)
        
        benefit_items = [
            (_("Search"), _("Search through your scanned PDF documents")),
            (_("Copy text"), _("Copy text from images and scanned documents")),
            (_("Batch processing"), _("Process multiple files at once")),
            (_("Auto-correction"), _("Automatically correct page alignment and rotation"))
        ]
        
        for title, description in benefit_items:
            benefit_label = Gtk.Label()
            benefit_label.set_markup(f"<span>• <b>{title}:</b> {description}</span>")
            benefit_label.set_wrap(True)
            benefit_label.set_halign(Gtk.Align.START)
            benefit_label.set_xalign(0)
            benefit_label.set_margin_bottom(3)
            
            benefits_list.append(benefit_label)
        
        content_box.append(benefits_list)
        
        separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        separator.set_margin_top(4)
        separator.set_margin_bottom(16)
        content_box.append(separator)
        
        bottom_section = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        
        show_at_startup_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        show_at_startup_box.set_halign(Gtk.Align.FILL)
        
        show_at_startup_label = Gtk.Label(label=_("Show this dialog at startup"))
        show_at_startup_label.set_halign(Gtk.Align.START)
        show_at_startup_label.set_hexpand(True)
        
        show_at_startup_switch = Gtk.Switch()
        show_at_startup_switch.set_active(self.should_show_welcome_dialog())
        show_at_startup_switch.set_valign(Gtk.Align.CENTER)
        show_at_startup_switch.set_halign(Gtk.Align.END)
        
        show_at_startup_box.append(show_at_startup_label)
        show_at_startup_box.append(show_at_startup_switch)
        bottom_section.append(show_at_startup_box)
        
        start_button = Gtk.Button()
        start_button.set_label(_("Let's Get Started"))
        start_button.add_css_class("suggested-action")
        start_button.set_size_request(160, 36)
        start_button.set_halign(Gtk.Align.CENTER)
        bottom_section.append(start_button)
        
        content_box.append(bottom_section)
        
        close_button = Gtk.Button()
        close_button.set_icon_name("window-close-symbolic")
        close_button.add_css_class("circular")
        close_button.add_css_class("flat")
        close_button.set_tooltip_text(_("Close"))
        close_button.set_halign(Gtk.Align.END)
        close_button.set_valign(Gtk.Align.START)
        close_button.set_margin_top(8)
        close_button.set_margin_end(8)
        close_button.connect("clicked", lambda _: dialog.close())
        
        overlay.set_child(content_box)
        overlay.add_overlay(close_button)
        
        dialog.set_content(overlay)
        
        def on_switch_toggle(switch, _param):
            self.set_show_welcome_dialog(switch.get_active())
        
        show_at_startup_switch.connect("notify::active", on_switch_toggle)
        start_button.connect("clicked", lambda _: dialog.close())
        
        dialog.add_css_class("welcome-modal")
        dialog.present()

    def on_back_clicked(self, button: Gtk.Button) -> None:
        """Handle back button navigation"""
        current_page = self.stack.get_visible_child_name()

        if current_page == "terminal":
            self.stack.set_visible_child_name("settings")
            self.step_label.set_markup(
                "<span font_size='small'>" + _("Step 1/3: Settings") + "</span>"
            )
            self.back_button.set_sensitive(False)
            self.back_button.set_visible(False)
            self.next_button.set_label(_("Start Processing"))
            self.next_button.set_sensitive(True)
            self.next_button.set_visible(True)

            self.update_file_info()

        elif current_page == "conclusion":
            self.reset_and_go_to_settings()

    def on_next_clicked(self, button: Gtk.Button) -> None:
        """Handle next button navigation"""
        current_page = self.stack.get_visible_child_name()

        if current_page == "settings":
            self.on_apply_clicked(button)
        elif current_page == "conclusion":
            self.reset_and_go_to_settings()

    def show_help(self, button: Gtk.Button = None) -> None:
        """Show help dialog with usage instructions"""
        self.show_welcome_dialog()

    def show_about_dialog(self) -> None:
        """Show about dialog with application information"""
        app = self.get_application()
        if app:
            app.on_about_action(None, None)

    def on_add_file_clicked(self, button: Gtk.Button) -> None:
        """Handle add file button click"""
        file_chooser = Gtk.FileDialog.new()
        file_chooser.set_title(_("Select PDF Files"))
        file_chooser.set_modal(True)

        pdf_filter = Gtk.FileFilter()
        pdf_filter.set_name(_("PDF Files"))
        pdf_filter.add_mime_type("application/pdf")
        pdf_filter.add_pattern("*.pdf")

        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(pdf_filter)
        file_chooser.set_filters(filters)

        file_chooser.open_multiple(
            parent=self, cancellable=None, callback=self._on_open_multiple_callback
        )

    def _on_open_multiple_callback(
        self, dialog: Gtk.FileDialog, result: Gio.AsyncResult
    ) -> None:
        """Handle file chooser completion for multiple files"""
        try:
            files = dialog.open_multiple_finish(result)

            if files and files.get_n_items() > 0:
                file_paths = []
                for i in range(files.get_n_items()):
                    file = files.get_item(i)
                    if isinstance(file, Gio.File):
                        file_paths.append(file.get_path())

                added = self.settings.add_files(file_paths)

                if added > 0:
                    self.update_file_info()
                else:
                    logger.warning(_("No valid files were selected"))
                    self.show_toast(_("No valid PDF files were selected"))
        except Exception as e:
            logger.error(f"Error adding files: {e}")
            self.show_toast(_("Error adding files"))

    def show_toast(self, message: str, timeout: int = 3) -> None:
        """Show a toast notification"""
        toast = Adw.Toast.new(message)
        toast.set_timeout(timeout)
        self.toast_overlay.add_toast(toast)

    def on_browse_clicked(self, button: Gtk.Button) -> None:
        """Handle the browse button click for selecting a destination folder"""
        folder_chooser = Gtk.FileDialog.new()
        folder_chooser.set_title(_("Select destination folder"))
        folder_chooser.set_modal(True)

        self.set_initial_folder(folder_chooser)

        folder_chooser.select_folder(
            parent=self, cancellable=None, callback=self._on_select_folder_callback
        )

    def set_initial_folder(self, folder_chooser: Gtk.FileDialog) -> None:
        """Set the initial folder for a folder chooser dialog"""
        if self.settings.destination_folder:
            if os.path.isdir(self.settings.destination_folder):
                initial_folder = self.settings.destination_folder
            else:
                initial_folder = os.path.dirname(self.settings.destination_folder)

            if os.path.exists(initial_folder):
                folder = Gio.File.new_for_path(initial_folder)
                folder_chooser.set_initial_folder(folder)

    def _on_select_folder_callback(
        self, dialog: Gtk.FileDialog, result: Gio.AsyncResult
    ) -> None:
        """Handle the folder selection dialog response"""
        try:
            folder = dialog.select_folder_finish(result)
            if folder:
                path = folder.get_path()

                self.settings.destination_folder = path
                if hasattr(self.ui, "dest_entry") and self.ui.dest_entry:
                    self.ui.dest_entry.set_text(path)

                logger.info(f"Destination folder selected: {path}")
                self.show_toast(_("Destination folder selected"))

        except Exception as e:
            logger.error(f"Error selecting save location: {e}")
            self.show_toast(_("Error selecting destination folder"))

    def _validate_ocr_settings(self) -> bool:
        """Validate OCR settings before processing"""
        if not self.settings.selected_files:
            logger.warning(_("No files selected for processing"))
            self.show_toast(_("No files selected for processing"))
            return False

        save_in_same_folder = self.get_save_in_same_folder()

        if not save_in_same_folder and not self.settings.destination_folder:
            logger.warning(_("No destination folder selected"))
            self.show_toast(_("Please select a destination folder"))
            return False

        return True

    def get_save_in_same_folder(self) -> bool:
        """Get the value of the save in same folder switch"""
        if hasattr(self.ui, "same_folder_switch_row") and self.ui.same_folder_switch_row:
            return self.ui.same_folder_switch_row.get_active()
        return False

    def _get_settings_from_ui(self) -> None:
        """Get settings from UI components"""
        if hasattr(self.ui, "lang_dropdown") and self.ui.lang_dropdown is not None:
            lang_index = self.ui.lang_dropdown.get_selected()
            languages = self.ocr_processor.get_available_ocr_languages()
            if 0 <= lang_index < len(languages):
                self.settings.lang = languages[lang_index][0]

        if hasattr(self.ui, "quality_dropdown") and self.ui.quality_dropdown is not None:
            quality_index = self.ui.quality_dropdown.get_selected()
            if 0 <= quality_index < len(self.QUALITY_CONFIG):
                self.settings.quality = self.QUALITY_CONFIG[quality_index][0]

        if hasattr(self.ui, "alignment_dropdown") and self.ui.alignment_dropdown is not None:
            align_index = self.ui.alignment_dropdown.get_selected()
            if 0 <= align_index < len(self.ALIGNMENT_CONFIG):
                self.settings.align = self.ALIGNMENT_CONFIG[align_index][0]

        save_in_same_folder = self.get_save_in_same_folder()

        if hasattr(self.ui, "dest_entry") and self.ui.dest_entry is not None:
            dest_folder = self.ui.dest_entry.get_text()
            if dest_folder:
                self.settings.destination_folder = dest_folder

        self.settings.save_settings(
            self.settings.lang,
            self.settings.quality,
            self.settings.align,
            self.settings.destination_folder,
            save_in_same_folder,
        )

    def on_apply_clicked(self, button: Gtk.Button) -> None:
        """Process the selected files with OCR"""
        if not self._validate_ocr_settings():
            return

        # Clean up any previous processing state
        if hasattr(self, 'ocr_processor') and self.ocr_processor:
            if hasattr(self.ocr_processor, 'force_cleanup'):
                self.ocr_processor.force_cleanup()
        self._cleanup_ocr_processor()
        self.settings.reset_processing_state()

        self._get_settings_from_ui()

        self.processed_files = []
        self.process_start_time = time.time()

        # Register callbacks for OCR processing events
        self.ocr_processor.register_callbacks(
            on_file_complete=self._on_file_processed,
            on_all_complete=self._on_processing_complete,
        )

        # Start OCR processing using Python API
        success = self.ocr_processor.process_with_api()
        if not success:
            logger.error(_("Failed to start OCR processing"))
            self.show_toast(_("Failed to start OCR processing"))
            return

        # Switch to terminal page and update UI
        self.stack.set_visible_child_name("terminal")
        self.step_label.set_markup(
            "<span font_size='small'>" + _("Step 2/3: Processing") + "</span>"
        )
        self.back_button.set_visible(False)
        self.next_button.set_visible(False)

        # Start progress updates - DELEGATE TO TERMINAL PAGE MANAGER
        self.ui.start_progress_monitor()

        logger.info(_("OCR processing started using Python API"))

    def reset_and_go_to_settings(self) -> None:
        """Reset the application state and return to the settings page"""
        # Stop all timers first - delegate to UI manager
        if hasattr(self.ui, 'stop_progress_monitor'):
            self.ui.stop_progress_monitor()
            
        if self.conclusion_timer_id is not None:
            safe_remove_source(self.conclusion_timer_id)
            self.conclusion_timer_id = None

        # Force cleanup of OCR processor and queue
        if hasattr(self, 'ocr_processor') and self.ocr_processor:
            if hasattr(self.ocr_processor, 'force_cleanup'):
                self.ocr_processor.force_cleanup()
            self._cleanup_ocr_processor()

        self.processed_files = []
        
        if hasattr(self.settings, 'reset_processing_state'):
            self.settings.reset_processing_state()
        
        self.next_button.set_label(_("Start"))
        self.next_button.set_sensitive(True)
        self.next_button.set_visible(True)

        self.disconnect_signal(self.next_button, "clicked")
        self.connect_signal(self.next_button, "clicked", self.on_next_clicked)

        self.stack.set_visible_child_name("settings")
        self.step_label.set_markup(
            "<span font_size='small'>" + _("Step 1/3: Settings") + "</span>"
        )

        self.update_file_info()

    def _cleanup_ocr_processor(self) -> None:
        """Force aggressive cleanup of OCR processor resources"""
        if not self.ocr_processor:
            return
            
        try:
            if hasattr(self.ocr_processor, 'ocr_queue') and self.ocr_processor.ocr_queue:
                self.ocr_processor.ocr_queue.stop()
                
                import time
                time.sleep(0.5)
                
                self.ocr_processor.ocr_queue = None
                
            # Reset processor state completely  
            if hasattr(self.ocr_processor, '_progress'):
                self.ocr_processor._progress = 0.0
            if hasattr(self.ocr_processor, '_processed_files'):
                self.ocr_processor._processed_files = 0
            if hasattr(self.ocr_processor, '_total_files'):
                self.ocr_processor._total_files = 0
            self.ocr_processor.process_pid = None
            
            self.ocr_processor.on_file_complete = None
            self.ocr_processor.on_all_complete = None
            
            import gc
            gc.collect()
            
            logger.info(_("OCR processor cleaned up aggressively"))
            
        except Exception as e:
            logger.error(f"Error in aggressive OCR processor cleanup: {e}")

    def show_conclusion_page(self) -> None:
        """Show the conclusion page after OCR processing completes"""
        self.stack.set_visible_child_name("conclusion")
        self.step_label.set_markup(
            "<span font_size='small'>" + _("Step 3/3: Completed") + "</span>"
        )

        self.back_button.set_visible(False)

        self.next_button.set_label(_("Back"))
        self.next_button.set_visible(True)
        self.next_button.set_sensitive(True)

        self.disconnect_signal(self.next_button, "clicked")
        self.connect_signal(self.next_button, "clicked", self.on_next_clicked)
        
    def _safe_show_conclusion_page(self) -> bool:
        """Safely show conclusion page with allocation check"""
        try:
            if self.stack.get_visible_child_name() != "terminal":
                return False
                
            if not self.get_allocated_width() or not self.get_allocated_height():
                GLib.timeout_add(100, self._safe_show_conclusion_page)
                return False
                
            self.show_conclusion_page()
            return False
            
        except Exception as e:
            logger.error(f"Error showing conclusion page safely: {e}")
            self.show_conclusion_page()
            return False

    def update_file_info(self) -> None:
        """Update the file information UI after files have been added or removed"""
        current_page = self.stack.get_visible_child_name()
        
        if current_page != "settings":
            return
            
        if hasattr(self.ui, "refresh_queue_status"):
            self.ui.refresh_queue_status()
            logger.info(f"Queue status refreshed with {len(self.settings.selected_files)} files")
            return
            
        old_page = self.stack.get_visible_child()
        new_page = self.ui.create_settings_page()
        self.stack.remove(old_page)
        self.stack.add_named(new_page, "settings")
        self.stack.set_visible_child_name("settings")
        logger.info(f"UI updated with {len(self.settings.selected_files)} files in queue")

    def _on_page_changed(self, stack, _param) -> None:
        """Handle page change events to update the headerbar styling"""
        current_page = stack.get_visible_child_name()

        if current_page == "settings":
            self._apply_headerbar_sidebar_style()
        else:
            self._remove_headerbar_sidebar_style()

    def _apply_headerbar_sidebar_style(self) -> None:
        """Apply a style to the headerbar for the settings page"""
        if self.header_bar:
            self.header_bar.add_css_class("settings-page-header")
            self.header_bar.remove_css_class("default-header")

    def _remove_headerbar_sidebar_style(self) -> None:
        """Remove style from the headerbar for non-settings pages"""
        if self.header_bar:
            self.header_bar.remove_css_class("settings-page-header")
            self.header_bar.add_css_class("default-header")

    def _on_file_processed(
        self, input_file: str, output_file: str, extracted_text: str = ""
    ) -> None:
        """Callback when a file is processed with OCR"""
        def process_in_main_thread():
            self.ocr_processor.remove_processed_file(input_file)

            logger.info(
                f"Received extracted_text in _on_file_processed: length={len(extracted_text)}"
            )

            if output_file not in self.settings.processed_files:
                self.settings.processed_files.append(output_file)
                logger.info(f"Added {output_file} to processed_files list")

            if not hasattr(self.settings, "extracted_text"):
                self.settings.extracted_text = {}

            self.settings.extracted_text[output_file] = extracted_text
            logger.info(
                f"Stored {len(extracted_text)} characters of extracted text for {os.path.basename(output_file)}"
            )
            
            # Update the status bar - delegate to UI manager
            if hasattr(self.ui, 'update_processing_status'):
                self.ui.update_processing_status(input_file)
            
            return False
        
        GLib.idle_add(process_in_main_thread)

    def _on_processing_complete(self) -> None:
        """Callback when all files are processed with OCR"""
        def complete_in_main_thread():
            logger.info(_("OCR processing complete callback triggered"))

            if self.stack.get_visible_child_name() != "terminal":
                logger.info(
                    _("Processing complete but no longer on terminal page, likely cancelled")
                )
                return False

            # Update the progress display to show 100% - delegate to UI manager
            if hasattr(self.ui, 'show_completion_ui'):
                self.ui.show_completion_ui()

            # Clean up temporary files
            if hasattr(self.settings, 'processed_files') and self.settings.processed_files:
                self.settings.cleanup_temp_files(self.settings.processed_files)

            # Update the conclusion page with actual values
            if hasattr(self.ui, "update_conclusion_page"):
                self.ui.update_conclusion_page()

            # Show conclusion page with a delay
            self.conclusion_timer_id = GLib.timeout_add(
                2000, lambda: self._safe_show_conclusion_page()
            )

            logger.info(
                _("OCR processing completed successfully for {count} files").format(
                    count=self.ocr_processor.get_processed_count()
                )
            )

            self.show_toast(_("OCR processing complete"))
            
            return False
        
        GLib.idle_add(complete_in_main_thread)

    def on_cancel_clicked(self) -> None:
        """Handle cancel button click during OCR processing"""
        if self.ocr_processor.ocr_queue:
            self.ocr_processor.ocr_queue.stop()
            logger.info(_("OCR processing cancelled by user"))

            # Stop the progress monitor - delegate to UI manager
            if hasattr(self.ui, 'stop_progress_monitor'):
                self.ui.stop_progress_monitor()

            self.show_toast(_("OCR processing cancelled"))

            # Reset the OCR processor status
            if hasattr(self.ocr_processor, '_processed_files'):
                self.ocr_processor._processed_files = 0
            if hasattr(self.ocr_processor, '_progress'):
                self.ocr_processor._progress = 0.0
            self.ocr_processor.ocr_queue = None

            self.reset_and_go_to_settings()