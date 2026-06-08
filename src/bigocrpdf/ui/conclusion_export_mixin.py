"""Conclusion Page ODF Export Mixin."""

import os

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gtk

from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.i18n import _
from bigocrpdf.utils.logger import logger

_EXPORT_FAILED_MSG = _("Export failed")
_NOTIFY_ACTIVE = "notify::active"


class ConclusionExportMixin:
    """Mixin providing ODF export functionality for the conclusion page."""

    _MODE_FILENAME_SUFFIX = {
        "formatted_images": "_fmt_img",
        "formatted": "_fmt",
    }

    def _show_odf_export_options_dialog(self, file_path: str) -> None:
        """Show export options dialog with a single Export button and option switches.

        Args:
            file_path: Path to the source PDF file
        """
        dialog = Adw.Dialog()
        dialog.set_title(_("Export to OpenDocument"))
        dialog.set_content_width(380)

        toolbar_view = Adw.ToolbarView()
        header_bar = Adw.HeaderBar()
        toolbar_view.add_top_bar(header_bar)

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=24)
        content_box.set_margin_start(24)
        content_box.set_margin_end(24)
        content_box.set_margin_top(12)
        content_box.set_margin_bottom(24)

        # --- Options group ---
        options_group = Adw.PreferencesGroup()

        settings = self.window.settings
        init_images = getattr(settings, "odf_include_images", True)
        init_open = getattr(settings, "odf_open_after_export", False)
        switch_state = {"images": init_images, "open_after": init_open}

        images_row = Adw.SwitchRow()
        images_row.set_title(_("Include images"))
        images_row.set_subtitle(_("Embed page images alongside text"))
        images_row.set_active(init_images)
        images_row.connect(
            _NOTIFY_ACTIVE,
            lambda row, _p: self._update_odf_setting(
                "odf_include_images", row.get_active(), switch_state, "images"
            ),
        )
        options_group.add(images_row)

        open_row = Adw.SwitchRow()
        open_row.set_title(_("Open after export"))
        open_row.set_subtitle(_("Open file in the default application"))
        open_row.set_active(init_open)
        open_row.connect(
            _NOTIFY_ACTIVE,
            lambda row, _p: self._update_odf_setting(
                "odf_open_after_export", row.get_active(), switch_state, "open_after"
            ),
        )
        options_group.add(open_row)

        content_box.append(options_group)

        # --- Export button ---
        btn_content = Adw.ButtonContent()
        btn_content.set_icon_name("document-save-symbolic")
        btn_content.set_label(_("Export"))

        export_btn = Gtk.Button()
        export_btn.set_child(btn_content)
        export_btn.add_css_class("suggested-action")
        export_btn.add_css_class("pill")
        export_btn.set_halign(Gtk.Align.CENTER)
        set_a11y_label(export_btn, _("Export"))
        export_btn.connect(
            "clicked",
            lambda _b: self._on_export_clicked(
                switch_state["images"],
                switch_state["open_after"],
                file_path,
                dialog,
            ),
        )
        content_box.append(export_btn)

        toolbar_view.set_content(content_box)
        dialog.set_child(toolbar_view)

        dialog.present(self.window)

    def _update_odf_setting(self, attr: str, value: bool, switch_state: dict, key: str) -> None:
        """Persist an ODF export setting and update local switch state."""
        switch_state[key] = value
        settings = self.window.settings
        setattr(settings, attr, value)
        settings._save_odf_settings()
        settings._config.save()

    def _on_export_clicked(
        self,
        include_images: bool,
        open_after: bool,
        file_path: str,
        options_dialog: Adw.Dialog,
    ) -> None:
        """Handle click on the Export button.

        Args:
            include_images: Whether the Include images switch is active
            open_after: Whether to open the file after export
            file_path: Source PDF file path
            options_dialog: The options dialog to close
        """
        export_mode = "formatted_images" if include_images else "formatted"
        logger.info(f"ODF Export: mode={export_mode}, open_after={open_after}")

        self._odf_export_mode = export_mode
        self._odf_open_after = open_after

        options_dialog.force_close()
        self._show_odf_file_dialog(file_path)

    def _show_odf_file_dialog(self, file_path: str) -> None:
        """Show file save dialog for ODF export.

        Args:
            file_path: Source PDF file path
        """
        from gi.repository import Gio

        save_dialog = Gtk.FileDialog.new()
        save_dialog.set_title(_("Export to OpenDocument"))
        save_dialog.set_modal(True)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        suffix = self._MODE_FILENAME_SUFFIX.get(
            getattr(self, "_odf_export_mode", "formatted"), "_fmt"
        )
        save_dialog.set_initial_name(f"{base_name}{suffix}.odt")

        filters = Gio.ListStore.new(Gtk.FileFilter)
        odf_filter = Gtk.FileFilter()
        odf_filter.set_name(_("OpenDocument Text (*.odt)"))
        odf_filter.add_pattern("*.odt")
        odf_filter.add_mime_type("application/vnd.oasis.opendocument.text")
        filters.append(odf_filter)
        save_dialog.set_filters(filters)
        save_dialog.set_default_filter(odf_filter)

        save_dialog.save(
            parent=self.window,
            cancellable=None,
            callback=lambda d, r: self._on_odf_save_response(d, r, file_path),
        )

    def _on_odf_save_response(self, dialog: Gtk.FileDialog, result, file_path: str) -> None:
        """Handle ODF save dialog response.

        Args:
            dialog: File dialog
            result: Async result
            file_path: Source PDF file path
        """
        try:
            file = dialog.save_finish(result)
            output_path = file.get_path()
            if not output_path.lower().endswith(".odt"):
                output_path += ".odt"
            self._export_odf_file(output_path, file_path)
        except Exception as e:
            if "Dismissed" not in str(e):
                logger.error(f"Error exporting to ODF: {e}")
                self.window.show_toast(_EXPORT_FAILED_MSG)

    def _export_odf_file(self, output_path: str, file_path: str) -> None:
        """Export content to ODF using the TSV-based formatted converter.

        Runs the conversion in a background thread while showing a loading dialog.

        Args:
            output_path: Destination ODF file path
            file_path: Source PDF file path (with OCR text layer)
        """
        import threading

        from gi.repository import GLib

        cancel_event = threading.Event()

        # Build and show the loading dialog
        loading_dialog = Adw.Dialog()
        loading_dialog.set_title(_("Exporting…"))
        loading_dialog.set_content_width(300)
        loading_dialog.set_can_close(False)

        toolbar_view = Adw.ToolbarView()
        header = Adw.HeaderBar()
        header.set_show_start_title_buttons(False)
        header.set_show_end_title_buttons(False)
        toolbar_view.add_top_bar(header)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        box.set_margin_start(32)
        box.set_margin_end(32)
        box.set_margin_top(24)
        box.set_margin_bottom(32)
        box.set_halign(Gtk.Align.CENTER)
        box.set_valign(Gtk.Align.CENTER)

        spinner = Gtk.Spinner()
        spinner.set_size_request(48, 48)
        spinner.start()
        spinner.set_halign(Gtk.Align.CENTER)
        box.append(spinner)

        label = Gtk.Label(label=_("Exporting to OpenDocument…"))
        label.add_css_class("title-4")
        box.append(label)

        cancel_btn = Gtk.Button(label=_("Cancel"))
        cancel_btn.add_css_class("destructive-action")
        cancel_btn.add_css_class("pill")
        cancel_btn.set_halign(Gtk.Align.CENTER)
        cancel_btn.set_margin_top(8)
        set_a11y_label(cancel_btn, _("Cancel"))
        cancel_btn.connect("clicked", lambda _b: cancel_event.set())
        box.append(cancel_btn)

        toolbar_view.set_content(box)
        loading_dialog.set_child(toolbar_view)
        loading_dialog.present(self.window)

        include_images = getattr(self, "_odf_export_mode", "formatted") == "formatted_images"

        def _do_export() -> None:
            from bigocrpdf.utils.odf_builder import ExportCancelled

            success = False
            cancelled = False
            try:
                from bigocrpdf.utils.tsv_odf_converter import convert_pdf_to_odf

                convert_pdf_to_odf(
                    file_path,
                    output_path,
                    include_images=include_images,
                    cancel_event=cancel_event,
                )
                success = True
            except ExportCancelled:
                cancelled = True
                logger.info("ODF export cancelled by user")
            except Exception as e:
                logger.error(f"ODF conversion failed: {e}")

            # Clean up partial file on cancel or failure
            if not success and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass

            GLib.idle_add(self._on_export_finished, loading_dialog, success, cancelled, output_path)

        threading.Thread(target=_do_export, daemon=True).start()

    def _on_export_finished(
        self, dialog: Adw.Dialog, success: bool, cancelled: bool, output_path: str
    ) -> None:
        """Handle export thread completion on the main thread."""
        dialog.force_close()
        if cancelled:
            self.window.show_toast(_("Export cancelled"))
        else:
            self._report_export_result(success, output_path)
        return False

    def _report_export_result(self, success: bool, output_path: str) -> None:
        """Report the export result to user and optionally open the file.

        Args:
            success: Whether export succeeded
            output_path: Destination file path
        """
        if success:
            self.window.show_toast(_("Exported to {}").format(os.path.basename(output_path)))
            if getattr(self, "_odf_open_after", False):
                from bigocrpdf.utils.pdf_utils import open_file_with_default_app

                open_file_with_default_app(output_path)
        else:
            self.window.show_toast(_EXPORT_FAILED_MSG)

    # ── Shared export helpers ─────────────────────────────────────────

    @staticmethod
    def _is_user_dismissed(exc: Exception) -> bool:
        """Tell whether a Gtk.FileDialog error came from the user closing it."""
        # FileDialog raises a GError whose message starts with "Dismissed by user".
        # There is no public symbolic constant in the introspected bindings.
        return "Dismissed" in str(exc)

    @staticmethod
    def _unique_path(path: str) -> str:
        """Return *path*, or ``path (1)``, ``path (2)``, … until it doesn't exist.

        Used by bulk export to avoid silently overwriting a file at the
        destination — single-file flows already get FileDialog's native
        overwrite confirmation.
        """
        if not os.path.exists(path):
            return path
        stem, ext = os.path.splitext(path)
        for n in range(1, 1000):
            candidate = f"{stem} ({n}){ext}"
            if not os.path.exists(candidate):
                return candidate
        # Extremely unlikely; fall back to overwrite rather than loop forever.
        return path

    def _build_progress_dialog(
        self,
        title_text: str,
        subtitle_text: str,
        total: int | None = None,
    ):
        """Build a standard cancellable progress dialog.

        Returns ``(dialog, update_progress, cancel_event)`` where
        ``update_progress(done, name)`` is safe to invoke via ``GLib.idle_add``
        and ``cancel_event`` is a :class:`threading.Event` set when the user
        clicks Cancel.
        """
        import threading

        cancel_event = threading.Event()

        dialog = Adw.Dialog()
        dialog.set_title(title_text)
        dialog.set_content_width(360)
        dialog.set_can_close(False)

        toolbar_view = Adw.ToolbarView()
        header = Adw.HeaderBar()
        header.set_show_start_title_buttons(False)
        header.set_show_end_title_buttons(False)
        toolbar_view.add_top_bar(header)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        box.set_margin_start(32)
        box.set_margin_end(32)
        box.set_margin_top(24)
        box.set_margin_bottom(32)
        box.set_halign(Gtk.Align.CENTER)
        box.set_valign(Gtk.Align.CENTER)

        spinner = Gtk.Spinner()
        spinner.set_size_request(40, 40)
        spinner.start()
        spinner.set_halign(Gtk.Align.CENTER)
        box.append(spinner)

        title_label = Gtk.Label(label=title_text)
        title_label.add_css_class("title-4")
        title_label.set_halign(Gtk.Align.CENTER)
        box.append(title_label)

        subtitle_label = Gtk.Label(label=subtitle_text)
        subtitle_label.add_css_class("dim-label")
        subtitle_label.set_halign(Gtk.Align.CENTER)
        box.append(subtitle_label)

        progress_bar: Gtk.ProgressBar | None = None
        if total is not None and total > 0:
            progress_bar = Gtk.ProgressBar()
            progress_bar.set_fraction(0.0)
            box.append(progress_bar)

        cancel_btn = Gtk.Button(label=_("Cancel"))
        cancel_btn.add_css_class("destructive-action")
        cancel_btn.add_css_class("pill")
        cancel_btn.set_halign(Gtk.Align.CENTER)
        cancel_btn.set_margin_top(8)
        set_a11y_label(cancel_btn, _("Cancel"))

        def _on_cancel(_b: Gtk.Button) -> None:
            # Give immediate feedback: the worker may stay inside a long
            # parse step before it next polls the cancel event, so we
            # update the dialog UI (label + disabled button) on the main
            # thread instead of leaving the user staring at an unchanged
            # spinner.
            cancel_event.set()
            cancel_btn.set_sensitive(False)
            cancel_btn.set_label(_("Cancelling…"))
            set_a11y_label(cancel_btn, _("Cancelling…"))
            subtitle_label.set_text(_("Finishing current step…"))

        cancel_btn.connect("clicked", _on_cancel)
        box.append(cancel_btn)

        toolbar_view.set_content(box)
        dialog.set_child(toolbar_view)
        dialog.present(self.window)

        def update_progress(done: int, name: str) -> bool:
            # Once the user clicks Cancel we keep the "Finishing current
            # step…" message and stop overwriting it with per-file progress
            # so they don't see a fresh filename after asking to stop.
            if cancel_event.is_set():
                return False
            if total:
                subtitle_label.set_text(f"{done}/{total} — {name}")
                if progress_bar is not None:
                    progress_bar.set_fraction(done / total)
            else:
                subtitle_label.set_text(name)
            return False

        return dialog, update_progress, cancel_event

    # ── Markdown export ────────────────────────────────────────────────

    def _show_markdown_export_options_dialog(self, file_path: str) -> None:
        """Show export options dialog for Markdown export."""
        dialog = Adw.Dialog()
        dialog.set_title(_("Export to Markdown"))
        dialog.set_content_width(380)

        toolbar_view = Adw.ToolbarView()
        toolbar_view.add_top_bar(Adw.HeaderBar())

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=24)
        content_box.set_margin_start(24)
        content_box.set_margin_end(24)
        content_box.set_margin_top(12)
        content_box.set_margin_bottom(24)

        options_group = Adw.PreferencesGroup()

        settings = self.window.settings
        init_fm = getattr(settings, "md_include_front_matter", False)
        init_open = getattr(settings, "md_open_after_export", False)
        state = {"front_matter": init_fm, "open_after": init_open}

        fm_row = Adw.SwitchRow()
        fm_row.set_title(_("Include YAML front-matter"))
        fm_row.set_subtitle(_("Adds title, source path, page count and date"))
        fm_row.set_active(init_fm)
        fm_row.connect(
            _NOTIFY_ACTIVE,
            lambda row, _p: self._update_md_setting(
                "md_include_front_matter", row.get_active(), state, "front_matter"
            ),
        )
        options_group.add(fm_row)

        open_row = Adw.SwitchRow()
        open_row.set_title(_("Open after export"))
        open_row.set_subtitle(_("Open file in the default application"))
        open_row.set_active(init_open)
        open_row.connect(
            _NOTIFY_ACTIVE,
            lambda row, _p: self._update_md_setting(
                "md_open_after_export", row.get_active(), state, "open_after"
            ),
        )
        options_group.add(open_row)

        content_box.append(options_group)

        btn_content = Adw.ButtonContent()
        btn_content.set_icon_name("document-save-symbolic")
        btn_content.set_label(_("Export"))

        export_btn = Gtk.Button()
        export_btn.set_child(btn_content)
        export_btn.add_css_class("suggested-action")
        export_btn.add_css_class("pill")
        export_btn.set_halign(Gtk.Align.CENTER)
        set_a11y_label(export_btn, _("Export"))
        export_btn.connect(
            "clicked",
            lambda _b: self._on_md_export_clicked(
                state["front_matter"], state["open_after"], file_path, dialog
            ),
        )
        content_box.append(export_btn)

        toolbar_view.set_content(content_box)
        dialog.set_child(toolbar_view)
        dialog.present(self.window)

    def _update_md_setting(self, attr: str, value: bool, state: dict, key: str) -> None:
        """Persist a Markdown export setting and update local state."""
        state[key] = value
        settings = self.window.settings
        setattr(settings, attr, value)
        save_md = getattr(settings, "_save_md_settings", None)
        if callable(save_md):
            save_md()
        config = getattr(settings, "_config", None)
        if config is not None and hasattr(config, "save"):
            config.save()

    def _on_md_export_clicked(
        self,
        include_front_matter: bool,
        open_after: bool,
        file_path: str,
        options_dialog: Adw.Dialog,
    ) -> None:
        """Handle the Export button click for Markdown.

        The selected option values are pinned to the closure passed to the
        file picker so overlapping per-row exports don't clobber each other's
        settings via shared self attributes.
        """
        options_dialog.force_close()
        self._show_markdown_file_dialog(file_path, include_front_matter, open_after)

    def _show_markdown_file_dialog(
        self, file_path: str, include_front_matter: bool, open_after: bool
    ) -> None:
        """Show file save dialog for Markdown export."""
        from gi.repository import Gio

        save_dialog = Gtk.FileDialog.new()
        save_dialog.set_title(_("Export to Markdown"))
        save_dialog.set_modal(True)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        save_dialog.set_initial_name(f"{base_name}.md")

        filters = Gio.ListStore.new(Gtk.FileFilter)
        md_filter = Gtk.FileFilter()
        md_filter.set_name(_("Markdown (*.md)"))
        md_filter.add_pattern("*.md")
        md_filter.add_pattern("*.markdown")
        md_filter.add_mime_type("text/markdown")
        filters.append(md_filter)
        save_dialog.set_filters(filters)
        save_dialog.set_default_filter(md_filter)

        save_dialog.save(
            parent=self.window,
            cancellable=None,
            callback=lambda d, r: self._on_md_save_response(
                d, r, file_path, include_front_matter, open_after
            ),
        )

    def _on_md_save_response(
        self,
        dialog: Gtk.FileDialog,
        result,
        file_path: str,
        include_front_matter: bool,
        open_after: bool,
    ) -> None:
        """Handle the Markdown save dialog response."""
        try:
            file = dialog.save_finish(result)
            output_path = file.get_path()
            if output_path is None:
                logger.error("Markdown export destination has no local path (remote URI)")
                self.window.show_toast(_("Remote locations are not supported"))
                return
            if not output_path.lower().endswith((".md", ".markdown")):
                output_path += ".md"
            self._export_markdown_file(output_path, file_path, include_front_matter, open_after)
        except Exception as e:
            if not self._is_user_dismissed(e):
                logger.error(f"Error exporting to Markdown: {e}")
                self.window.show_toast(_EXPORT_FAILED_MSG)

    def _export_markdown_file(
        self,
        output_path: str,
        file_path: str,
        include_front_matter: bool,
        open_after: bool,
    ) -> None:
        """Convert PDF to Markdown in a background thread.

        Mirrors the ODF flow: a cancellable progress dialog stays on screen
        until the conversion finishes (or the user cancels) so large PDFs
        don't appear to freeze the app. Writes go through a sibling
        ``.tmp`` file and ``os.replace`` so an existing target file is
        preserved if conversion fails or the user cancels.
        """
        import threading

        from gi.repository import GLib

        from bigocrpdf.utils.odf_builder import ExportCancelled

        loading_dialog, _update, cancel_event = self._build_progress_dialog(
            _("Exporting to Markdown…"),
            os.path.basename(file_path),
        )

        def _do_export() -> None:
            from bigocrpdf.utils.tsv_odf_converter import convert_pdf_to_markdown

            success = False
            cancelled = False
            tmp_path = output_path + ".tmp"
            try:
                text = convert_pdf_to_markdown(
                    file_path,
                    include_front_matter=include_front_matter,
                    cancel_event=cancel_event,
                )
                if cancel_event.is_set():
                    raise ExportCancelled
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                with open(tmp_path, "w", encoding="utf-8") as fh:
                    fh.write(text)
                os.replace(tmp_path, output_path)
                success = True
            except ExportCancelled:
                cancelled = True
                logger.info("Markdown export cancelled by user")
            except Exception as e:
                logger.error(f"Markdown conversion failed: {e}")

            self._safe_remove(tmp_path)

            GLib.idle_add(
                self._on_md_export_finished,
                loading_dialog,
                success,
                cancelled,
                output_path,
                open_after,
            )

        threading.Thread(target=_do_export, daemon=True).start()

    def _on_md_export_finished(
        self,
        dialog: Adw.Dialog,
        success: bool,
        cancelled: bool,
        output_path: str,
        open_after: bool = False,
    ) -> bool:
        """Report Markdown export result on the main thread."""
        dialog.force_close()
        if cancelled:
            self.window.show_toast(_("Export cancelled"))
        elif success:
            self.window.show_toast(_("Exported to {}").format(os.path.basename(output_path)))
            if open_after:
                from bigocrpdf.utils.pdf_utils import open_file_with_default_app

                open_file_with_default_app(output_path)
        else:
            self.window.show_toast(_EXPORT_FAILED_MSG)
        return False

    # ── Bulk export ────────────────────────────────────────────────────

    def _create_bulk_export_menu_button(self) -> Gtk.MenuButton:
        """Build the export menu shown inside the selection action bar.

        Uses ``Gio.Menu`` + ``Gtk.PopoverMenu`` for native keyboard nav
        and accessibility — matches the per-row export button's pattern.
        """
        from gi.repository import Gio

        menu_model = Gio.Menu()
        menu_model.append(_("OpenDocument (.odt)"), "bulk.odt")
        menu_model.append(_("Markdown (.md)"), "bulk.md")

        button = Gtk.MenuButton()
        button.set_icon_name("document-save-as-symbolic")
        button.set_tooltip_text(_("Export selected files"))
        button.add_css_class("suggested-action")
        button.set_sensitive(False)
        button.set_menu_model(menu_model)

        group = Gio.SimpleActionGroup()
        odt_action = Gio.SimpleAction.new("odt", None)
        odt_action.connect("activate", lambda *_a: self._bulk_export_selected("odf"))
        group.add_action(odt_action)
        md_action = Gio.SimpleAction.new("md", None)
        md_action.connect("activate", lambda *_a: self._bulk_export_selected("md"))
        group.add_action(md_action)
        button.insert_action_group("bulk", group)
        return button

    def _bulk_export_selected(self, fmt: str) -> None:
        """Capture the current selection and pick a destination folder."""
        files = sorted(self._selected_files)
        if not files:
            return

        from gi.repository import Gio

        dialog = Gtk.FileDialog.new()
        dialog.set_title(_("Choose destination folder"))
        dialog.set_modal(True)

        def _on_folder_chosen(d: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
            try:
                folder = d.select_folder_finish(result)
            except Exception as e:
                if not self._is_user_dismissed(e):
                    logger.error(f"Folder picker failed: {e}")
                    self.window.show_toast(_EXPORT_FAILED_MSG)
                return
            folder_path = folder.get_path()
            if folder_path is None:
                self.window.show_toast(_("Remote locations are not supported"))
                return
            self._run_bulk_export(files, folder_path, fmt)

        dialog.select_folder(parent=self.window, cancellable=None, callback=_on_folder_chosen)

    def _run_bulk_export(self, files: list[str], dest_folder: str, fmt: str) -> None:
        """Bulk export entry point — validates the destination and spawns the worker.

        Per-format settings are snapshotted here at batch start so a mid-batch
        toggle from another dialog can't make some files honour different
        options than others.
        """
        import threading

        # Cheap early checks so the user gets a clear error instead of
        # discovering after every file fails individually.
        if not os.path.isdir(dest_folder):
            self.window.show_toast(_("Destination folder not found"))
            return
        if not os.access(dest_folder, os.W_OK):
            self.window.show_toast(_("Destination folder is not writable"))
            return

        settings = self.window.settings
        options = {
            "include_front_matter": getattr(settings, "md_include_front_matter", False),
            "include_images": getattr(settings, "odf_include_images", True),
        }

        total = len(files)
        loading_dialog, update_progress, cancel_event = self._build_progress_dialog(
            _("Exporting selected files…"),
            f"0/{total}",
            total=total,
        )

        threading.Thread(
            target=self._bulk_export_worker,
            args=(
                files,
                dest_folder,
                fmt,
                options,
                cancel_event,
                update_progress,
                loading_dialog,
            ),
            daemon=True,
        ).start()

    _BULK_EXTENSIONS = {"md": ".md", "odf": ".odt"}

    @staticmethod
    def _safe_remove(path: str) -> None:
        """Best-effort removal of a partial output file."""
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    def _bulk_convert_one(
        self,
        pdf_path: str,
        out_path: str,
        fmt: str,
        options: dict,
        cancel_event,
    ) -> None:
        """Convert *pdf_path* into *out_path* using the requested *fmt*.

        ``options`` carries the per-format flags snapshotted at batch start
        (``include_front_matter`` for Markdown, ``include_images`` for ODF).

        Raises ``ExportCancelled`` if the user cancels mid-file, or any other
        converter exception on failure — the caller is responsible for
        recording the outcome and cleaning up the partial file.
        """
        if fmt == "md":
            from bigocrpdf.utils.tsv_odf_converter import convert_pdf_to_markdown

            text = convert_pdf_to_markdown(
                pdf_path,
                include_front_matter=options.get("include_front_matter", False),
                cancel_event=cancel_event,
            )
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp_path, out_path)
            return

        from bigocrpdf.utils.tsv_odf_converter import convert_pdf_to_odf

        convert_pdf_to_odf(
            pdf_path,
            out_path,
            include_images=options.get("include_images", True),
            cancel_event=cancel_event,
        )

    def _bulk_export_worker(
        self,
        files: list[str],
        dest_folder: str,
        fmt: str,
        options: dict,
        cancel_event,
        update_progress,
        loading_dialog: Adw.Dialog,
    ) -> None:
        """Thread body: convert each file and report aggregate results."""
        from gi.repository import GLib

        from bigocrpdf.utils.odf_builder import ExportCancelled

        if fmt not in self._BULK_EXTENSIONS:
            logger.error("Unknown bulk export format: %s", fmt)
            GLib.idle_add(loading_dialog.force_close)
            GLib.idle_add(self.window.show_toast, _EXPORT_FAILED_MSG)
            return
        ext = self._BULK_EXTENSIONS[fmt]
        results: dict = {"ok": 0, "failed": [], "saved_paths": []}

        for idx, pdf_path in enumerate(files, start=1):
            if cancel_event.is_set():
                break

            basename = os.path.splitext(os.path.basename(pdf_path))[0] + ext
            out_path = self._unique_path(os.path.join(dest_folder, basename))
            GLib.idle_add(update_progress, idx, os.path.basename(out_path))

            try:
                self._bulk_convert_one(pdf_path, out_path, fmt, options, cancel_event)
            except ExportCancelled:
                # User pressed Cancel mid-file — bail out without recording
                # this file as a failure, and clean up the partial output.
                self._safe_remove(out_path)
                break
            except Exception:
                logger.exception("Bulk export failed for %s", pdf_path)
                results["failed"].append(os.path.basename(pdf_path))
                self._safe_remove(out_path)
            else:
                results["ok"] += 1
                results["saved_paths"].append(out_path)

        GLib.idle_add(
            self._on_bulk_export_finished,
            loading_dialog,
            results,
            cancel_event.is_set(),
            dest_folder,
        )

    def _on_bulk_export_finished(
        self,
        dialog: Adw.Dialog,
        results: dict,
        cancelled: bool,
        dest_folder: str,
    ) -> bool:
        """Close the progress dialog and report the outcome to the user."""
        dialog.force_close()
        ok = results["ok"]
        failed_count = len(results["failed"])
        folder_name = os.path.basename(dest_folder) or dest_folder

        if cancelled:
            self.window.show_toast(
                _("Cancelled — saved {ok} of {total}").format(ok=ok, total=ok + failed_count)
            )
        elif failed_count:
            self.window.show_toast(_("Saved {ok}; {n} failed").format(ok=ok, n=failed_count))
        else:
            self.window.show_toast(
                _("Saved {ok} files to {folder}").format(ok=ok, folder=folder_name)
            )
        return False

    def _open_file(self, file_path: str) -> None:
        """Open a file using the default application.

        Args:
            file_path: Path to the file to open
        """
        from bigocrpdf.utils.pdf_utils import open_file_with_default_app

        open_file_with_default_app(file_path)

    def _reveal_in_file_manager(self, file_path: str) -> None:
        """Open the system file manager with the given file selected.

        Uses the freedesktop.org FileManager1 D-Bus interface (ShowItems),
        which is supported by Dolphin, Nautilus, Thunar, Nemo, Caja, etc.
        Falls back to opening the parent directory if D-Bus is unavailable.
        """
        import subprocess

        file_uri = f"file://{os.path.abspath(file_path)}"
        try:
            subprocess.Popen(
                [
                    "dbus-send",
                    "--session",
                    "--dest=org.freedesktop.FileManager1",
                    "--type=method_call",
                    "/org/freedesktop/FileManager1",
                    "org.freedesktop.FileManager1.ShowItems",
                    f"array:string:{file_uri}",
                    "string:",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            # dbus-send not available; open parent folder instead
            from bigocrpdf.utils.pdf_utils import open_file_with_default_app

            open_file_with_default_app(os.path.dirname(file_path))

    def _show_extracted_text(self, file_path: str) -> None:
        """Show extracted text dialog.

        Args:
            file_path: Path to the PDF file
        """
        if hasattr(self.window, "ui") and hasattr(self.window.ui, "show_extracted_text"):
            self.window.ui.show_extracted_text(file_path)
        else:
            logger.warning("Text viewer dialog not available")
            self._show_simple_text_dialog(file_path)

    def _show_simple_text_dialog(self, file_path: str) -> None:
        """Show a simple text dialog as fallback.

        Args:
            file_path: Path to the PDF file
        """
        extracted_text = self._get_extracted_text_for_file(file_path)
        dialog = Adw.AlertDialog(
            heading=_("Extracted Text"),
            body=extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
        )
        dialog.add_response("ok", _("OK"))
        dialog.present(self.window)

    def _get_extracted_text_for_file(self, file_path: str) -> str:
        """Get extracted text for a file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text or placeholder message
        """
        if (
            hasattr(self.window.settings, "extracted_text")
            and file_path in self.window.settings.extracted_text
        ):
            return self.window.settings.extracted_text[file_path]

        sidecar_file = os.path.splitext(file_path)[0] + ".txt"
        if os.path.exists(sidecar_file):
            try:
                with open(sidecar_file, encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading sidecar file: {e}")

        return _("No text content was detected in this file during OCR processing.")

    def reset_page(self) -> None:
        """Reset the conclusion page to initial state."""
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

        if self.output_list_box:
            self._clear_output_list()
