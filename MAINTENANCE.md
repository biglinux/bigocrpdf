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
- `src/bigocrpdf/services/ocr/` owns OCR, PDF inspection, geometry,
  text-layer generation, compression, and worker protocols.
- `src/bigocrpdf/services/export_service.py` and the ODF/Markdown owners implement
  document exports.
- `src/bigocrpdf/utils/config_manager.py`, `durable_writes.py`, history, and
  checkpoint modules own durable application state.
- `usr/share/`, `tools/stage-data.sh`, `pkgbuild/PKGBUILD`, and `default.nix` own
  installed desktop integration.

Name a new module after the domain concept it owns. `utils/` and the
`*_manager.py` modules are not a place for new code.

## Dependency boundaries

| Dependency | Delegate directly | BigOCRPDF must retain |
| --- | --- | --- |
| ppocr-native | PP-OCRv6 detection and recognition, the vertical re-read, the detector's boxes for deskew and its probability map for dewarp | PDF policy, preprocessing policy, retries, process protocol, coordinate mapping, text layer, exports |
| pikepdf/qpdf | PDF object model, page copying, overlays, image access, save options | editor model, page selection semantics, geometry decisions, resource limits, atomic publication |
| GTK 4 / libadwaita | widgets, models, list factories, actions, dialogs, accessibility primitives | application state, callbacks, async lifetime checks, workflow and accessible names |
| OpenCV / SciPy | image transforms, thresholding, contours, interpolation, remapping | parameter selection, document detection, quality gates, fallback order, OCR-coordinate preservation |
| ReportLab | PDF drawing and text primitives | invisible Unicode text placement and mapping OCR coordinates to PDF coordinates |
| odfpy | valid OpenDocument XML and package writing | layout inference, tables/columns, positioned/reflowable export policy |

## Runtime contract

BigOCRPDF is a Linux GTK 4.14+/libadwaita 1.5+ application on Python 3.12+. Python remains responsible for
GTK state, asynchronous delivery, OCR orchestration, and PDF/document models.
Shell is limited to validation, staging, and packaging.

The base runtime includes GTK4, libadwaita, ppocr-native, the text-layer font
Noto Sans from `noto-fonts`, Poppler tools, pikepdf, Pillow, OpenCV, NumPy, SciPy,
ReportLab, and odfpy. `jbig2enc` is optional; when it is absent, bilevel output
must continue through CCITT Group 4.

`services/ocr/ppocr_native.py` owns the OCR engine: it finds the `ppocr-native`
executable and the plans in `PPOCR_NATIVE_PLANS` (default
`/usr/share/ppocr-native`), builds the worker's command line, and holds the one
client of the worker's stdin/stdout JSON-lines protocol: `Worker` for a job's
persistent OCR worker, and `Detector` for deskew and dewarp, which asks that
worker or, in the preprocessing pool, a detect-only worker of its own. Image
OCR runs one worker per request at normal priority. Each job owns its workers, so one job finishing or cancelling never
touches another's. The worker reads only binary PPM and PGM, so OCR inputs are
written as PPM. Its recognisers run one per physical core, capped by the
resource manager's OCR thread count. A request has a 300 s deadline, and a
cancelled job's waiting request returns within 0.1 s and kills its worker.

Subprocess calls use direct argv. A zero exit status is authoritative success;
nonzero status, timeout, or signal is an explicit error/cancellation result.

## OCR and PDF invariants

- The supported recognition model is PP-OCRv6 small with its unified language
  coverage. Do not reconstruct legacy language-model selection.
- OCR configuration, including DPI, thresholds, detection resolution, and render
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
  the PDF's own text layer.
- Structured OCR JSON stays bound to the PDF by byte size and SHA-256, so a file
  that describes a different PDF loads as nothing rather than as stale text.
  Version 1, and the `unavailable` marker written by older versions, decode only
  as "no structured OCR"; version 1 needs the explicit `allow_unverified_legacy`
  opt-in and is never authoritative. The payload is written without
  indentation by contract.
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

Published documents are user data; configuration, the queue, history,
checkpoints and dialog flags are program state.

- Every write goes to a temporary file in the destination directory and is
  renamed over the target, so a reader sees the old or the complete new file.
- Documents are fsynced before the rename and their directory after it, so a
  power cut also leaves one of the two. Program state is never fsynced: losing
  its last seconds costs nothing, and its loaders fall back to defaults.
- Without overwrite an existing name is never replaced: a file gets the first
  free `name-N`, and a split output's parts all share one suffix. Where the
  filesystem lacks rename flags (FUSE, NFS) or hard links (vfat) the no-replace
  install falls back to a link or an exclusive placeholder.
- A split output publishes its parts one by one. An interrupted run can leave a
  mixed family; the next overwrite replaces it, and stale parts are removed only
  when their own XMP metadata names this family and part.
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

Every symbolic icon the interface references is bundled as a `hicolor` theme
under `src/bigocrpdf/resources/icontheme/`, which `utils/icons.py` appends to the
icon search path during application startup. Every theme's lookup chain ends at
hicolor, so the bundle fills names the host theme lacks, on any host and inside
an AppImage, while the user's theme keeps priority. Adding an `icon-name` means
adding the matching SVG; `tests/test_icons.py` enforces this in both directions.

Changed user-visible journeys require all three forms of evidence:

1. a rendered 1280x720 screenshot review;
2. AT-SPI roles, names, states, and actions;
3. the real action's observable side effect.

Measure 25, 100, and 500 queue/editor objects before considering list
virtualization. Keep the current structure when it meets the measured budget.

## Internationalization and desktop metadata

`locale/bigocrpdf.pot` and `locale/*.po` are the gettext text sources. Compiled
MO files are build artifacts produced by `tools/stage-data.sh` and are not
versioned. A source run therefore finds no catalog in the checkout and uses the
installed one; to check a catalog change, stage it and point the application at
the result:

```bash
bash tools/stage-data.sh "$TMPDIR/stage"
BIGOCRPDF_LOCALE_DIR="$TMPDIR/stage/usr/share/locale" LANGUAGE=pt_BR PYTHONPATH=src python3 -m bigocrpdf
```

Translations embedded in `.desktop` files (applications and KIO service menus)
and Nemo actions remain in those files. They are not extracted to or generated
from PO catalogs, and staging copies them unchanged.

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
CSV, and JSON summaries needed to interpret or reproduce a run. Recorded
reference points are in `benchmarks/baselines/README.md`.

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
It runs locally before a release: the CI workflow only triggers the package
builds, and the PKGBUILD has no `check()`. Arch's GTK is newer than the 4.14/1.5
floor; `tests/test_adw_compat.py` exercises the fallbacks that floor needs.

Before release, also run `bash tools/quality.sh` with ppocr-native and its plans
installed, or with `PATH` and `PPOCR_NATIVE_PLANS` pointing at a build: it holds
the one test that drives the real worker through the protocol. Build and inspect
the Arch package, run `nix flake check` and `nix build` in a working Nix
environment, inspect installed file ownership and permissions, and exercise the
installed package in a disposable VM. Verify KIO, Nemo, Nautilus, desktop
launchers, OCR without `jbig2enc`, and a real Unicode `pdftotext` round trip.

## Reduction policy

Prefer deletion, direct imports, and existing owners. A new file or abstraction is
acceptable only when it removes more duplicated behavior and shortens callers.
Tests cover distinct observable regressions, not getters, constants, wrappers,
private call order, or toolkit behavior. A compiler, type checker, authoritative
metadata validator, or existing public-flow test is sufficient for mechanical
changes it completely covers.
