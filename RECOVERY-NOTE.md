# Recovered working snapshot

This archive was reconstructed on 2026-08-03 from the originally uploaded
`bigocrpdf.tar.gz` because the temporary working tree used during earlier review
steps was no longer present in the active runtime.

Only changes explicitly recoverable from the conversation and persisted reports
were reapplied:

- generated `egg-info` and Python bytecode removed;
- empty npm manifests removed;
- benchmark programs and benchmark-specific tests moved to `benchmarks/`;
- PDF/A-2b enabled by default, with `--no-pdfa` for explicit opt-out;
- incorrect tagged-PDF `MarkInfo` declaration removed;
- keyboard shortcuts migrated from deprecated `Gtk.ShortcutsWindow` to
  `Adw.ShortcutsDialog`;
- persisted documentation and changelog included.

This is not claimed to be byte-for-byte identical to the lost temporary tree.
