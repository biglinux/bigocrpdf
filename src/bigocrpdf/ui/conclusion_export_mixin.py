"""Conclusion page export actions."""
# Host attributes are supplied by the conclusion page's explicit mixin composition.
# pyright: reportAttributeAccessIssue=false

import os
import threading
from collections.abc import Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk

from bigocrpdf.utils.a11y import set_a11y_label
from bigocrpdf.utils.adw_compat import create_spinner
from bigocrpdf.utils.durable_writes import write_text_atomically
from bigocrpdf.utils.i18n import _, ngettext
from bigocrpdf.utils.logger import logger


class ConclusionExportMixin:
    """Provide single-file and bulk exports from the conclusion page."""

    def _show_odf_export_options_dialog(self, file_path: str) -> None:
        """Show OpenDocument export options."""
        settings = self.window.settings
        init_images = settings.odf_include_images
        init_open = settings.odf_open_after_export
        switch_state = {"images": init_images, "open_after": init_open}
        self._show_export_options_dialog(
            _("Export to OpenDocument"),
            (
                (
                    _("Preserve Text Layout"),
                    _("Turn off for easier paragraph, table, and column editing"),
                    init_images,
                    lambda value: self._update_export_setting(
                        "odf_include_images",
                        value,
                        switch_state,
                        "images",
                        settings._save_odf_settings,
                    ),
                ),
                (
                    _("Open after export"),
                    _("Open file in the default application"),
                    init_open,
                    lambda value: self._update_export_setting(
                        "odf_open_after_export",
                        value,
                        switch_state,
                        "open_after",
                        settings._save_odf_settings,
                    ),
                ),
            ),
            lambda dialog: self._on_export_clicked(
                switch_state["images"],
                switch_state["open_after"],
                file_path,
                dialog,
            ),
        )

    def _on_export_clicked(
        self,
        include_images: bool,
        open_after: bool,
        file_path: str,
        options_dialog: Adw.Dialog,
    ) -> None:
        """Handle click on the Export button.

        Args:
            include_images: Whether fixed-layout editable text mode is active
            open_after: Whether to open the file after export
            file_path: Source PDF file path
            options_dialog: The options dialog to close
        """
        export_mode = "positioned" if include_images else "structured"
        logger.info("ODF export: mode=%s, open_after=%s", export_mode, open_after)

        options_dialog.force_close()
        self._show_odf_file_dialog(file_path, include_images, open_after)

    def _show_odf_file_dialog(self, file_path: str, include_images: bool, open_after: bool) -> None:
        """Show the OpenDocument destination picker."""
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        suffix = "_fmt_img" if include_images else "_fmt"
        self._show_export_file_dialog(
            title=_("Export to OpenDocument"),
            initial_name=f"{base_name}{suffix}.odt",
            filter_name=_("OpenDocument Text (*.odt)"),
            patterns=("*.odt",),
            mime_type="application/vnd.oasis.opendocument.text",
            callback=lambda output_path: self._export_odf_file(
                output_path, file_path, include_images, open_after
            ),
            error_context="ODF",
            extensions=(".odt",),
            default_extension=".odt",
        )

    def _export_odf_file(
        self,
        output_path: str,
        file_path: str,
        include_images: bool,
        open_after: bool,
    ) -> None:
        """Export content to ODF in the shared cancellable worker flow."""
        self._run_single_export(
            output_path,
            file_path,
            _("Exporting to OpenDocument…"),
            "ODF",
            open_after,
            lambda cancel_event: self._bulk_convert_one(
                file_path,
                output_path,
                "odf",
                {"include_images": include_images},
                cancel_event,
            ),
        )

    # ── Shared export helpers ─────────────────────────────────────────

    def _show_export_options_dialog(
        self,
        title: str,
        rows: tuple[tuple[str, str, bool, Callable[[bool], None]], ...],
        on_export: Callable[[Adw.Dialog], None],
    ) -> None:
        """Show the common two-switch export options dialog."""
        dialog = Adw.Dialog(title=title, content_width=380)
        toolbar_view = Adw.ToolbarView()
        toolbar_view.add_top_bar(Adw.HeaderBar())
        content = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=24,
            margin_start=24,
            margin_end=24,
            margin_top=12,
            margin_bottom=24,
        )
        options = Adw.PreferencesGroup()
        for row_title, subtitle, active, on_changed in rows:
            row = Adw.SwitchRow(title=row_title, subtitle=subtitle, active=active)
            row.connect(
                "notify::active",
                lambda switch, _param, callback=on_changed: callback(switch.get_active()),
            )
            options.add(row)
        content.append(options)

        button_content = Adw.ButtonContent(icon_name="document-save-symbolic", label=_("Export"))
        export_button = Gtk.Button(
            child=button_content,
            halign=Gtk.Align.CENTER,
            css_classes=["suggested-action", "pill"],
        )
        set_a11y_label(export_button, _("Export"))
        export_button.connect("clicked", lambda _button: on_export(dialog))
        content.append(export_button)
        toolbar_view.set_content(content)
        dialog.set_child(toolbar_view)
        dialog.present(self.window)

    def _update_export_setting(
        self,
        attr: str,
        value: bool,
        state: dict[str, bool],
        key: str,
        save_settings: Callable[[], None],
    ) -> None:
        """Persist one export option and update its dialog-local snapshot."""
        state[key] = value
        setattr(self.window.settings, attr, value)
        save_settings()
        self.window.settings._config.save()

    def _show_export_file_dialog(
        self,
        *,
        title: str,
        initial_name: str,
        filter_name: str,
        patterns: tuple[str, ...],
        mime_type: str,
        callback: Callable[[str], None],
        error_context: str,
        extensions: tuple[str, ...],
        default_extension: str,
    ) -> None:
        """Show a native save dialog and normalize its local destination."""
        dialog = Gtk.FileDialog(title=title, modal=True, initial_name=initial_name)
        file_filter = Gtk.FileFilter(name=filter_name)
        for pattern in patterns:
            file_filter.add_pattern(pattern)
        file_filter.add_mime_type(mime_type)
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(file_filter)
        dialog.set_filters(filters)
        dialog.set_default_filter(file_filter)

        def on_saved(save_dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
            try:
                file = save_dialog.save_finish(result)
            except GLib.Error as error:
                if not self._is_user_dismissed(error):
                    logger.error("Error choosing %s export destination: %s", error_context, error)
                    self.window.ui.show_toast(_("Export failed"))
                return

            output_path = file.get_path()
            if output_path is None:
                logger.error("%s export destination has no local path", error_context)
                self.window.ui.show_toast(_("Remote locations are not supported"))
                return
            if not output_path.lower().endswith(extensions):
                output_path += default_extension
            callback(output_path)

        dialog.save(parent=self.window, cancellable=None, callback=on_saved)

    @staticmethod
    def _is_user_dismissed(error: GLib.Error) -> bool:
        """Return whether the user dismissed a GTK asynchronous dialog."""
        return error.matches(Gtk.DialogError.quark(), Gtk.DialogError.DISMISSED)

    @staticmethod
    def _reserve_unique_path(path: str) -> str:
        """Reserve a non-existing bulk-export destination without races."""
        stem, ext = os.path.splitext(path)
        candidate = path
        suffix = 1
        while True:
            try:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
                descriptor = os.open(candidate, flags, 0o600)
            except FileExistsError:
                candidate = f"{stem} ({suffix}){ext}"
                suffix += 1
                continue
            os.close(descriptor)
            return candidate

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

        box.append(create_spinner(40))

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
        settings = self.window.settings
        init_fm = settings.md_include_front_matter
        init_open = settings.md_open_after_export
        state = {"front_matter": init_fm, "open_after": init_open}
        self._show_export_options_dialog(
            _("Export to Markdown"),
            (
                (
                    _("Include YAML front-matter"),
                    _("Adds title, source path, page count and date"),
                    init_fm,
                    lambda value: self._update_export_setting(
                        "md_include_front_matter",
                        value,
                        state,
                        "front_matter",
                        settings._save_md_settings,
                    ),
                ),
                (
                    _("Open after export"),
                    _("Open file in the default application"),
                    init_open,
                    lambda value: self._update_export_setting(
                        "md_open_after_export",
                        value,
                        state,
                        "open_after",
                        settings._save_md_settings,
                    ),
                ),
            ),
            lambda dialog: self._on_md_export_clicked(
                state["front_matter"], state["open_after"], file_path, dialog
            ),
        )

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
        """Show the Markdown destination picker."""
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        self._show_export_file_dialog(
            title=_("Export to Markdown"),
            initial_name=f"{base_name}.md",
            filter_name=_("Markdown (*.md)"),
            patterns=("*.md", "*.markdown"),
            mime_type="text/markdown",
            callback=lambda output_path: self._export_markdown_file(
                output_path, file_path, include_front_matter, open_after
            ),
            error_context="Markdown",
            extensions=(".md", ".markdown"),
            default_extension=".md",
        )

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
        don't appear to freeze the app. The completed text is published
        atomically so an existing target survives conversion or write failure.
        """
        self._run_single_export(
            output_path,
            file_path,
            _("Exporting to Markdown…"),
            "Markdown",
            open_after,
            lambda cancel_event: self._bulk_convert_one(
                file_path,
                output_path,
                "md",
                {"include_front_matter": include_front_matter},
                cancel_event,
            ),
        )

    def _run_single_export(
        self,
        output_path: str,
        file_path: str,
        title: str,
        format_name: str,
        open_after: bool,
        convert: Callable[[threading.Event], None],
    ) -> None:
        """Run one cancellable export and publish its result on the GTK thread."""
        from bigocrpdf.utils.odf_builder import ExportCancelled

        dialog, _update, cancel_event = self._build_progress_dialog(
            title, os.path.basename(file_path)
        )

        def worker() -> None:
            success = False
            cancelled = False
            try:
                convert(cancel_event)
                success = True
            except ExportCancelled:
                cancelled = True
                logger.info("%s export cancelled by user", format_name)
            except Exception:
                logger.exception("%s conversion failed", format_name)

            GLib.idle_add(
                self._on_single_export_finished,
                dialog,
                success,
                cancelled,
                output_path,
                open_after,
            )

        threading.Thread(target=worker, daemon=True).start()

    def _on_single_export_finished(
        self,
        dialog: Adw.Dialog,
        success: bool,
        cancelled: bool,
        output_path: str,
        open_after: bool = False,
    ) -> bool:
        """Report a single-file export result on the main thread."""
        dialog.force_close()
        if cancelled:
            self.window.ui.show_toast(_("Export cancelled"))
        elif success:
            self.window.ui.show_toast(_("Exported to {}").format(os.path.basename(output_path)))
            if open_after:
                self._open_file(output_path)
        else:
            self.window.ui.show_toast(_("Export failed"))
        return False

    # ── Bulk export ────────────────────────────────────────────────────

    def _create_bulk_export_menu_button(self) -> Gtk.MenuButton:
        """Build the export menu shown inside the selection action bar.

        Uses ``Gio.Menu`` + ``Gtk.PopoverMenu`` for native keyboard nav
        and accessibility — matches the per-row export button's pattern.
        """
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

        dialog = Gtk.FileDialog.new()
        dialog.set_title(_("Choose destination folder"))
        dialog.set_modal(True)

        def _on_folder_chosen(d: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
            try:
                folder = d.select_folder_finish(result)
            except GLib.Error as error:
                if not self._is_user_dismissed(error):
                    logger.error("Folder picker failed: %s", error)
                    self.window.ui.show_toast(_("Export failed"))
                return

            folder_path = folder.get_path()
            if folder_path is None:
                self.window.ui.show_toast(_("Remote locations are not supported"))
                return
            self._run_bulk_export(files, folder_path, fmt)

        dialog.select_folder(parent=self.window, cancellable=None, callback=_on_folder_chosen)

    def _run_bulk_export(self, files: list[str], dest_folder: str, fmt: str) -> None:
        """Bulk export entry point — validates the destination and spawns the worker.

        Per-format settings are snapshotted here at batch start so a mid-batch
        toggle from another dialog can't make some files honour different
        options than others.
        """
        # Cheap early checks so the user gets a clear error instead of
        # discovering after every file fails individually.
        if not os.path.isdir(dest_folder):
            self.window.ui.show_toast(_("Destination folder not found"))
            return
        if not os.access(dest_folder, os.W_OK):
            self.window.ui.show_toast(_("Destination folder is not writable"))
            return

        settings = self.window.settings
        if fmt == "md":
            options = {"include_front_matter": settings.md_include_front_matter}
        elif fmt == "odf":
            options = {"include_images": settings.odf_include_images}
        else:
            logger.error("Unknown bulk export format: %s", fmt)
            self.window.ui.show_toast(_("Export failed"))
            return

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
        """Remove a partial output without hiding cleanup failures."""
        try:
            os.remove(path)
        except FileNotFoundError:
            return
        except OSError as error:
            logger.warning("Could not remove partial export %s: %s", path, error)

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
            write_text_atomically(out_path, text)
            return
        if fmt == "odf":
            from bigocrpdf.utils.tsv_odf_converter import convert_pdf_to_odf

            convert_pdf_to_odf(
                pdf_path,
                out_path,
                include_images=options.get("include_images", True),
                cancel_event=cancel_event,
            )
            return
        raise ValueError(f"unsupported export format: {fmt}")

    def _bulk_export_worker(
        self,
        files: list[str],
        dest_folder: str,
        fmt: str,
        options: dict[str, bool],
        cancel_event: threading.Event,
        update_progress: Callable[[int, str], bool],
        loading_dialog: Adw.Dialog,
    ) -> None:
        """Convert each selected file and publish the aggregate result."""
        from bigocrpdf.utils.odf_builder import ExportCancelled

        extension = self._BULK_EXTENSIONS[fmt]
        saved = 0
        failed = 0
        cancelled = False

        for index, pdf_path in enumerate(files, start=1):
            if cancel_event.is_set():
                cancelled = True
                break

            basename = os.path.splitext(os.path.basename(pdf_path))[0] + extension
            try:
                output_path = self._reserve_unique_path(os.path.join(dest_folder, basename))
            except OSError:
                logger.exception("Could not reserve bulk export destination for %s", pdf_path)
                failed += 1
                continue

            GLib.idle_add(update_progress, index, os.path.basename(output_path))
            try:
                self._bulk_convert_one(pdf_path, output_path, fmt, options, cancel_event)
            except ExportCancelled:
                self._safe_remove(output_path)
                cancelled = True
                break
            except Exception:
                logger.exception("Bulk export failed for %s", pdf_path)
                failed += 1
                self._safe_remove(output_path)
            else:
                saved += 1

        GLib.idle_add(
            self._on_bulk_export_finished,
            loading_dialog,
            saved,
            failed,
            cancelled,
            len(files),
            dest_folder,
        )

    def _on_bulk_export_finished(
        self,
        dialog: Adw.Dialog,
        saved: int,
        failed: int,
        cancelled: bool,
        total: int,
        dest_folder: str,
    ) -> bool:
        """Close the progress dialog and report the batch outcome."""
        dialog.force_close()
        folder_name = os.path.basename(dest_folder) or dest_folder

        if cancelled:
            self.window.ui.show_toast(
                _("Cancelled — saved {ok} of {total}").format(ok=saved, total=total)
            )
        elif failed:
            self.window.ui.show_toast(
                _("Saved {ok}; {n} failed").format(
                    ok=saved,
                    n=failed,
                )
            )
        else:
            self.window.ui.show_toast(
                ngettext(
                    "Saved {count} file to {folder}",
                    "Saved {count} files to {folder}",
                    saved,
                ).format(count=saved, folder=folder_name)
            )
        return False

    def _open_file(self, file_path: str) -> None:
        """Open a file using the default application.

        Args:
            file_path: Path to the file to open
        """
        from bigocrpdf.utils.pdf_utils import open_file_with_default_app

        if not open_file_with_default_app(file_path):
            self.window.ui.show_toast(_("Failed to open file"))

    def _reveal_in_file_manager(self, file_path: str) -> None:
        """Open the system file manager with the given file selected.

        Uses the freedesktop.org FileManager1 D-Bus interface (ShowItems),
        which is supported by Dolphin, Nautilus, Thunar, Nemo, Caja, etc.
        Falls back to opening the parent directory if D-Bus is unavailable.
        """
        import subprocess

        file_uri = Gio.File.new_for_path(file_path).get_uri()
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
            self._open_file(os.path.dirname(file_path))

    def _show_extracted_text(self, file_path: str) -> None:
        """Show extracted text using the main UI's dialog owner."""
        self.window.ui.dialogs_manager.show_extracted_text(file_path)
