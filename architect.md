# BigOCRPDF refactoring architecture

## Objective

Reduce incidental code and make ownership explicit without changing the public
CLI, persisted configuration, desktop entry points, OCR/PDF output, cancellation,
error behavior, or user-visible journeys. Line count is a diagnostic, not an
acceptance criterion. A deletion is valid only when an official dependency owns
the same contract or a compatibility verifier proves the behavior redundant.

## Current system

BigOCRPDF is not merely a graphical launcher around RapidOCR. The dependency
provides text detection, line classification, recognition, and coordinates for
an image. BigOCRPDF additionally owns:

- PDF inspection, hostile-input limits, mixed/native-text policy, page rendering,
  text-layer geometry, Unicode font embedding, overlays, PDF/A metadata, and
  output publication;
- queue orchestration, worker processes, cancellation, checkpoints, retries, and
  resource-dependent chunking;
- page editing, undo state, thumbnails, split/merge/compression workflows, and
  file-manager/desktop integration;
- document structure inference and TXT, Markdown, and ODT exports;
- perspective, skew, illumination, orientation, border, and curvature policies
  composed from lower-level OpenCV and SciPy primitives;
- GTK/libadwaita application state, accessibility, navigation, and persistence.

The complexity concern is still valid. The initial main sources were implicit
host state across 25 mixins, compatibility forwarding in the UI, flat settings
that combine four kinds of state, and repeated persistence declarations. The
main-window UI, actions, and processing mixins have since become composed owners;
remaining mixins are removed only after their host contracts are captured. Exact
duplicate functions account for only 34 source lines, so clone deletion cannot
produce the desired reduction by itself.

## Dependency boundaries

| Dependency | Delegate directly | BigOCRPDF must retain |
| --- | --- | --- |
| RapidOCR 3.x | model loading, detection, classification, recognition, word boxes | PDF policy, preprocessing policy, retries, process protocol, coordinate mapping, text layer, exports |
| pikepdf/qpdf | PDF object model, page copying, overlays, image access, save options | editor model, page selection semantics, geometry decisions, resource limits, atomic publication |
| GTK 4 / libadwaita | widgets, models, list factories, actions, dialogs, accessibility primitives | application state, callbacks, async lifetime checks, workflow and accessible names |
| OpenCV / SciPy | image transforms, thresholding, contours, interpolation, remapping | parameter selection, document detection, quality gates, fallback order, OCR-coordinate preservation |
| ReportLab | PDF drawing and text primitives | invisible Unicode text placement and mapping OCR coordinates to PDF coordinates |
| odfpy | valid OpenDocument XML and package writing | layout inference, tables/columns, positioned/reflowable export policy |

Runtime requirements and validation gates are recorded in `MAINTENANCE.md`.

## Stable contracts

The following are compatibility boundaries, not implementation details:

1. `bigocrpdf`, `bigocrimage`, and `bigocrpdf-cli` entry points and all CLI
   options, defaults, stdout, stderr, status codes, and exceptions.
2. Existing JSON keys, selected-file migration, corrupt-state recovery, atomic
   replacement, checkpoint/history formats, and destination naming.
3. `OCRConfig`, OCR worker line protocol, cancellation, timeouts, retry behavior,
   and CPU operation without optional GPU/JBIG2 dependencies.
4. Searchable Unicode output verified with `pdftotext`, original page geometry,
   mixed-content preservation, editor operations, and structured exports.
5. Queue -> settings -> processing -> results and image/capture workflows,
   including GTK main-loop safety, keyboard actions, and AT-SPI state.

`tools/compatibility_verifier.py` freezes the machine-readable CLI, import,
settings, queue, persistence, stdout/stderr, and exit-status subset. The original
tests, focused PDF fixtures, benchmark manifests, and UI evidence cover contracts
that cannot be represented by that verifier.

## Target ownership

### Application and UI

- Application objects own lifecycle and windows only.
- `BigOcrPdfWindow` retains GTK virtual methods and lifecycle hooks; composed UI,
  action/session, and processing controllers own product behavior.
- Each page owns its widgets and actions. Callers access the page owner directly;
  `BigOcrPdfUI` must not duplicate child references for compatibility.
- GTK models and factories own repeated queue/editor rows when measured object
  counts justify migration. Do not replace current widgets merely to reduce lines.
- Controllers receive explicit collaborators or typed callbacks. New mixins are
  prohibited; existing mixins are removed incrementally after their host contract
  is captured.

### State

- Persistent preferences are described once by a typed setting specification and
  remain exposed through the current flat `OcrSettings` attributes during migration.
- Queue inputs, editor modifications, and processing results become distinct state
  owners only when existing callers and persistence behavior are covered.
- `ConfigManager` remains the durable JSON boundary and atomic writer.

### OCR and PDF

- `OCRConfig` is the complete immutable-at-dispatch processing request.
- Pipeline phases use explicit context/result objects instead of attributes that
  appear on a host through multiple inheritance.
- pikepdf, RapidOCR, OpenCV, ReportLab, and odfpy are called through their public
  documented APIs. Product policy stays in domain-named modules.
- Geometric and OCR algorithms are not rewritten for line-count reasons. A change
  requires a focused regression and representative before/after benchmark.

## Migration sequence

1. Freeze contracts with the differential verifier and the canonical gate.
2. Replace repeated settings load/save code with one declarative schema while
   preserving the flat API and every persisted key.
3. Remove UI forwarding aliases and dynamic method injection; make ownership and
   inheritance explicit without changing rendered journeys.
4. Replace window/editor mixins with controllers using explicit collaborators.
5. Replace backend mixins with pipeline phase services and typed context objects.
6. Reassess algorithms against official dependency APIs and delete only behavior
   now genuinely supplied by a dependency.

Each stage must pass compilation, Ruff, formatting, Pyright, gettext/metadata,
all tests, the compatibility verifier, relevant end-to-end PDF checks, and any UI
evidence required by `MAINTENANCE.md`.

## Completion criteria

- No runtime behavior or documented capability is lost.
- No dynamic method injection remains.
- Cross-owner access is explicit and type checked; mixin host attributes are gone.
- Persistent keys have one declaration and round-trip tests.
- The canonical validation gate and compatibility verifier pass.
- OCR/geometry changes include benchmark and real `pdftotext` evidence.
- The final reduction is reported by product lines and substantive lines, with no
  target achieved by moving code to generated or untested artifacts.
