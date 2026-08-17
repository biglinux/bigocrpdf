# BigOCRPDF maintenance guide

This document records durable implementation contracts. Temporary plans, audit
checklists, command logs, screenshots, and generated benchmark exports do not
belong in the source tree.

## Ownership map

- `src/bigocrpdf/application.py`, `window.py`, and `image_application.py` own
  application lifecycle and top-level GTK state.
- `src/bigocrpdf/ui/` owns presentation, focus, actions, and main-loop delivery.
- `src/bigocrpdf/services/settings.py` owns the flat settings API consumed by the UI.
- `src/bigocrpdf/services/processor.py` owns queue and OCR orchestration.
- `src/bigocrpdf/services/rapidocr_service/` owns OCR, PDF inspection, geometry,
  text-layer generation, compression, and worker protocols.
- `src/bigocrpdf/services/export_service.py` and the ODF/Markdown owners implement
  document exports.
- `src/bigocrpdf/utils/config_manager.py`, `durable_writes.py`, history, and
  checkpoint modules own durable application state.
- `usr/share/`, `tools/stage-data.sh`, `pkgbuild/PKGBUILD`, and `default.nix` own
  installed desktop integration.

Keep ownership names domain-specific. Do not add generic `manager`, `helper`, or
`util` modules to avoid choosing an owner.

## Runtime contract

BigOCRPDF is a Linux GTK 4.22+/libadwaita application on Python 3.12+. Python remains responsible for
GTK state, asynchronous delivery, OCR orchestration, and PDF/document models.
Shell is limited to validation, staging, and packaging.

The base runtime includes GTK4, libadwaita, RapidOCR with the PP-OCRv6 small
model, OpenVINO, Poppler tools, pikepdf, Pillow, OpenCV, NumPy, SciPy, ReportLab,
and odfpy. ONNX Runtime is an optional inference backend. `jbig2enc` is optional;
when it is absent, bilevel output must continue through CCITT Group 4.

Subprocess calls use direct argv. A zero exit status is authoritative success;
nonzero status, timeout, or signal is an explicit error/cancellation result.

## OCR and PDF invariants

- CPU OCR remains fully functional.
- The supported recognition model is PP-OCRv6 small with its unified language
  coverage. Do not reconstruct legacy language-model selection.
- OCR configuration, including engine, DPI, thresholds, batch size, and render
  limits, travels through `OCRConfig`; do not rebuild a partial settings object.
- Unicode searchable PDFs use real extractable text. Validation must include
  `pdftotext`, not only inspection of internal strings.
- The text layer preserves page geometry and keeps OCR/native-text detection
  working after crop, deskew, dewarp, rotation, and mixed-content processing.
- PDF resource limits and hostile-input checks remain enforced.
- An OCR run publishes exactly one file: the PDF. Nothing else may appear beside
  a user's document. Structured OCR is written only where the caller names it —
  `ocr --sidecar-json [FILE]` — and read back only when the caller names it —
  `export-* --from-json FILE`. Without it, TXT, Markdown, and ODT export read
  the PDF's own text layer, which agreed with the structured path within 92–98%
  on real documents.
- Structured OCR JSON stays bound to the PDF by byte size and SHA-256, so a file
  that describes a different PDF loads as nothing rather than as stale text.
  Version 1, and the `unavailable` marker written by older versions, decode only
  as "no structured OCR"; version 1 needs the explicit `allow_unverified_legacy`
  opt-in and is never authoritative. The payload is compact by contract:
  indentation was 62% of the bytes of an 18-page document.
- Split output parts record their family in the PDF's private XMP namespace
  (`splitFamilyRoot`, `splitPartIndex`, `splitPartCount`). This is what makes
  retiring superseded parts safe: a numbered file name is not proof of
  membership, and a user's own `contract-02.pdf` must never be retired.
- Positioned ODT keeps editable text at page coordinates; reflowable ODT keeps
  paragraphs, tables, and columns; empty pages remain present.

OCR or geometry changes require a focused regression plus a representative
benchmark. File-size or line-count pressure alone is not a reason to rewrite an
algorithm.

## Durable state

Configuration, history, checkpoints, and published output are user data.

- Writes use a temporary file on the destination filesystem, flush and fsync the
  file, replace atomically, fsync the directory, and clean up on failure.
- Published files use private reflink/copy snapshots, so a staged path that cannot
  be removed can never remain a mutable hardlink to user-visible output.
- Multi-file publication is restricted to one canonical destination directory.
  Its private `PREPARING` -> `PREPARED` -> `COMMITTED`/`ROLLED_BACK` journal makes
  interrupted batches converge to the complete old or new set. Recovery runs
  before the next publication in that directory and is also available through
  `recover_pending_publications()`.
- Every OCR PDF is published in the same transaction as its versioned sidecar.
  A split output publishes all PDF/sidecar pairs as one set. Sidecar generation
  failure fails the input rather than exposing a PDF with stale metadata.
- Non-overwriting multi-file publication applies one collision counter to the
  complete batch. Domain callers that derive companion names provide their
  candidate family under the destination-directory lock.
- POSIX cannot make several destination names visible as one atomic operation.
  Readers may observe a mixed set during the interruption window, but journal
  recovery never accepts that mixed set as the final state.
- Rollback currently restores destination names, bytes, and regular access mode.
  Extended attributes, ACLs, and timestamps are not part of that recovery
  contract; callers must not use publication to preserve those metadata.
- Existing configuration keys and migrations remain readable unless a documented
  migration with rollback replaces them.
- Corrupt or partial input must not destroy the last readable state.
- Destructive operations are idempotent and scoped to product-owned temporary
  paths or an explicitly confirmed user target.

## UI contract

The main journey is queue -> settings -> processing -> results. The image journey
is source selection/capture -> processing -> copyable result. Visible state comes
from the queue, dependency probe, worker outcome, and editor model; the UI must not
announce readiness or success before those sources resolve.

Render information already held by the caller immediately. File page counts,
Poppler metadata, thumbnails, and other I/O must not block the GTK main loop.
Worker results may update only a still-live row/document.

Required controls are keyboard reachable and have stable AT-SPI names. Icon-only
actions have a matching accessible name and native tooltip. Status changes need a
real accessible event; an invisible label is not proof of an announcement.

Use native GTK/libadwaita components and theme tokens. Do not add visual decoration
or custom transient widgets when the toolkit already owns the interaction.

The supported floor is GTK 4.14 with libadwaita 1.5, which is what an AppImage
built on Ubuntu 24.04 carries; it is not the version we develop against. Widgets
introduced after that floor go through `utils/adw_compat.py`, which probes with
`hasattr` -- a symbol can be missing because the library is old or because its
typelib was bundled incompletely -- and falls back to an equivalent that renders
the same. `tests/test_adw_compat.py` exercises both branches and fails if such a
widget is used directly anywhere else.

Every symbolic icon the interface references is bundled under
`src/bigocrpdf/resources/icontheme/` and registered as a private icon theme by
`utils/icons.py` during application startup, so the interface renders identically
on any host and inside an AppImage. Adding an `icon-name` means adding the
matching SVG; `tests/test_icons.py` enforces this in both directions. The theme
inherits Adwaita and hicolor, so unbundled names still resolve.

Changed user-visible journeys require all three forms of evidence:

1. a rendered 1280x720 screenshot review;
2. AT-SPI roles, names, states, and actions;
3. the real action's observable side effect.

Measure 25, 100, and 500 queue/editor objects before considering list
virtualization. Keep the current structure when it meets the measured budget.

## Internationalization and desktop metadata

`locale/bigocrpdf.pot` and `locale/*.po` are the gettext text sources. Compiled
MO files are build artifacts produced by `tools/stage-data.sh` and are not
versioned.

Translations embedded in `.desktop`, `.desktop.in`, and Nemo metadata remain in
those files. They are not extracted to or generated from PO catalogs. Staging
copies this content unchanged, then renames KIO `.desktop.in` files in the
destination only.

Every PO must validate without fuzzy, obsolete, or untranslated entries. Preserve
placeholders, plural forms, Pango/XML markup, and accelerators.

`utils/i18n.py` resolves the catalog directory at runtime, in this order:
`BIGOCRPDF_LOCALE_DIR`, `$APPDIR/usr/share/locale`, `$TEXTDOMAINDIR` (exported by
the AppRun that appimage-creator generates), directories walked up from the
installed package, then `sys.prefix` and `/usr/share/locale`. A candidate only
counts when it actually holds `*/LC_MESSAGES/bigocrpdf.mo`, so a relocatable
build never silently falls through to a stale host catalog. The domain is bound
on both `gettext` and the C library, the latter covering strings GLib translates
on our behalf.

## Benchmarks

Benchmark sources and commands live in `benchmarks/prepare_benchmark_datasets.py`,
`benchmarks/ocr_benchmark.py`, `benchmarks/validate_text_layer.py`, and
`benchmarks/compare_benchmarks.py`. Raw datasets and generated PDF/ODT/TXT/Markdown
exports stay outside the repository. Keep only compact READMEs, manifests, JSONL,
CSV, and JSON summaries needed to interpret or reproduce a run.

The retained July 2026 baselines establish the following reference points:

- all 36 RapidOCR v6 CPU matrix runs completed and produced text layers;
- ONNX Runtime small was fastest in that six-page run, but startup/model loading
  was included and the sample is not a universal engine ranking;
- complex FUNSD pages exposed recognition and layout limits;
- structured sidecars improved export token recall and restored table structures
  in two of three dense forms;
- the form separator refinement preserved recall and table counts.

Do not report a roadmap, read document, or generated artifact as implemented
behavior. A new performance claim includes command, dataset manifest, environment,
before/after metrics, and failure count.

## Validation and release

Run the canonical local gate:

```bash
bash tools/validate.sh
```

The gate covers Python compilation, Ruff, formatting, Pyright, gettext freshness
and completeness, staged desktop/AppStream data, ShellCheck, shfmt, pytest, and the
frozen CLI/settings compatibility snapshot in `tools/compatibility-baseline.json`.
CI runs that gate in Arch Linux so its Python and GTK versions satisfy the declared
Python 3.12 and GTK 4.22 runtime floors.

Before release, also build and inspect the Arch package, run `nix flake check` and
`nix build` in a working Nix environment, inspect installed file ownership and
permissions, and exercise the installed package in a disposable VM. Verify KIO,
Nemo, Nautilus, desktop launchers, OCR without `jbig2enc`, and a real Unicode
`pdftotext` round trip.

## Reduction policy

Prefer deletion, direct imports, and existing owners. A new file or abstraction is
acceptable only when it removes more duplicated behavior and shortens callers.
Tests cover distinct observable regressions, not getters, constants, wrappers,
private call order, or toolkit behavior. A compiler, type checker, authoritative
metadata validator, or existing public-flow test is sufficient for mechanical
changes it completely covers.
