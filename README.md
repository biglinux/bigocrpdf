<div align="center">

# BigOcrPDF

**OCR for scanned PDFs and images, with PDF page editing and native Linux interfaces.**

[![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB.svg)](pyproject.toml)
[![GTK4 + Libadwaita](https://img.shields.io/badge/GTK4-Libadwaita-4A86CF.svg)](https://gnome.org)

</div>

---

BigOcrPDF is a GTK4/libadwaita application for adding searchable text layers to scanned
PDFs, extracting text from images, and editing PDF pages. The project also provides a
dedicated image OCR application and a separate command-line PDF toolbox.

## Main capabilities

- **Searchable PDF output** using PP-OCRv6 on [ppocr-native](https://github.com/biglinux/ppocr-native)
- **Mixed-content handling** that distinguishes image-only pages from pages with native text
- **PDF page editing** for reordering, rotating, flipping, deleting, merging, splitting, and
  compressing pages
- **Image preprocessing** for perspective, skew, curvature, orientation, contrast, brightness,
  noise, and border corrections
- **Structured exports** to plain text, Markdown, and ODT
- **Batch processing** with a persistent checkpoint and an option to resume an interrupted run
- **Image and screen-capture OCR** in the standalone `bigocrimage` interface
- **File-manager actions** for KDE Dolphin, GNOME Files (Nautilus), and Nemo when the
  corresponding integration files are installed

---

## Installation

On BigLinux, install the `bigocrpdf` package from the BigLinux repository:

```bash
sudo pacman -S bigocrpdf
```

The package pulls in ppocr-native, the OCR engine, and Noto Sans, the font the text layer is
written with. On other Arch-based systems, build [`pkgbuild/PKGBUILD`](pkgbuild/PKGBUILD) with
`makepkg -si` after installing ppocr-native from its own repository. Nix users can build
[`default.nix`](default.nix); nixpkgs has no ppocr-native package, so `default.nix` takes one
as its `ppocr-native` argument, and a build without it reports the engine missing at startup.

### Requirements

Package names vary by distribution. Outside the packages above, provide:

- Python 3.12 or newer
- GTK 4.14 or newer, libadwaita 1.5 or newer, PyGObject, Pycairo, and their GObject
  Introspection data, including Poppler's (`poppler-glib` on Arch)
- Poppler command-line tools, including `pdfinfo`, `pdfimages`, `pdftotext`, `pdftoppm`,
  `pdffonts`, and `pdfdetach`
- [ppocr-native](https://github.com/biglinux/ppocr-native) and its plans in
  `/usr/share/ppocr-native` (`PPOCR_NATIVE_PLANS` names another directory)
- Noto Sans (`/usr/share/fonts/noto/NotoSans-Regular.ttf`; `BIGOCRPDF_TEXT_FONT` names
  another file)
- The Python dependencies declared in [`pyproject.toml`](pyproject.toml)

Optional runtime components:

- `jbig2enc` for JBIG2 compression (CCITT Group 4 is used when it is absent)
- WenQuanYi Micro Hei (`/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc`;
  `BIGOCRPDF_CJK_FONT` names another file), which keeps Chinese and Japanese text in the
  text layer; without it those lines are left out
- An sRGB ICC profile from colord or Ghostscript for the PDF/A-2b option
- `python-psutil` for accurate available-memory detection
- XDG Desktop Portal, Spectacle, GNOME Screenshot, or Flameshot for region capture
- `nautilus-python` or `nemo-python` for the GNOME Files and Nemo context-menu actions

---

## Usage

### Graphical applications

```bash
bigocrpdf [FILES...]          # PDF OCR interface
bigocrpdf --edit FILE.pdf     # Open the PDF page editor
bigocrimage [IMAGE]           # Standalone image OCR interface
```

Use `bigocrpdf --help` for graphical-launch options such as `--debug`, `--verbose`,
`--image-mode`, and `--edit`.

### Command-line toolbox

The non-graphical command is `bigocrpdf-cli`, not `bigocrpdf`:

```bash
bigocrpdf-cli --help
bigocrpdf-cli ocr input.pdf -o searchable.pdf
bigocrpdf-cli split input.pdf -o parts --pages 10
bigocrpdf-cli merge first.pdf second.pdf -o combined.pdf
bigocrpdf-cli export-txt searchable.pdf
```

Available subcommands are `ocr`, `split`, `merge`, `compress`, `rotate`, `delete`, `extract`,
`insert`, `reorder`, `info`, `export-odf`, `export-txt`, `export-md`, and `edit`. Use
`export-odf --preserve-text-layout` for editable text fixed at the PDF positions;
without that option, ODT paragraphs, tables, and columns remain reflowable. Run
`bigocrpdf-cli COMMAND --help` for the current options of a specific operation.

### File-manager integration

The files under [`usr/share`](usr/share/) provide:

- KDE KIO/Dolphin service menus for PDF OCR, image OCR, PDF editing, and PDF creation
- A Nautilus extension for PDF/image actions
- Nemo actions for the same workflows

These integrations are available only when installed by the system package and when the
file manager's required extension package is present.

### Screen capture

Open `bigocrimage` and select **Screen Capture**. The application first tries the desktop
portal and then supported screenshot tools installed on the system.

---

## Features

### PDF editor

- Drag-and-drop page reordering with thumbnail previews
- Left/right rotation and horizontal/vertical flipping
- Page deletion and multi-page selection
- Merging PDFs and supported raster images into one PDF
- JPEG, PNG, WebP, TIFF, and BMP image import; the main queue also accepts AVIF
- EXIF orientation handling when images are converted to PDF
- Thumbnail zoom presets from 50% through 400%
- Per-page inclusion or exclusion from OCR
- Context actions to save a page as an image or a PDF
- Compression with configurable image quality and target DPI
- Splitting by page count or target file size
- Undo for page operations with <kbd>Ctrl</kbd>+<kbd>Z</kbd>

### OCR pipeline

- PP-OCRv6 small, using one unified model for supported languages
- [ppocr-native](https://github.com/biglinux/ppocr-native) as the engine: PP-OCRv6 in plain
  Rust on the CPU, reading a page's lines on one thread per physical core
- Automatic worker selection for parallel OCR
- Four detection-threshold presets in the graphical settings
- Invisible Unicode text layers for searchable and selectable output
- Automatic handling of image-only and mixed-content PDF pages
- Optional replacement of an existing OCR text layer
- Optional geometric corrections and image enhancements
- Resource limits for page count, image dimensions, and rendered page size

PP-OCRv6 recognizes Chinese, English, Japanese, and 46 Latin-script languages with one
unified model, so the graphical applications do not ask users to choose a language. The
application requires ppocr-native and its plans at startup.

### Image preprocessing

- Perspective correction for photographed documents
- Deskew and baseline dewarping
- 90°/180°/270° orientation detection
- Contrast and brightness correction
- Denoising and dark-border cleanup
- Background whitening through the scanner-effect option
- Optional correction of embedded images on mixed-content pages

Geometric corrections are enabled by default. Most color enhancements remain opt-in so the
user can choose between preserving the source appearance and applying stronger cleanup.
The scanner effect only changes page images the program rewrites into the PDF; OCR always
reads the page without it, because whitening textured paper costs the recognizer more text
than it gains.

### Output and export

| Format | Behavior |
|--------|----------|
| **Searchable PDF** | Adds an invisible text layer; pages that require geometric or appearance changes may be rendered into the output |
| **PDF/A-2b option** | Adds PDF/A metadata and an sRGB output intent when a supported ICC profile is available; otherwise processing falls back to a normal PDF |
| **Image quality presets** | Preserve original images or select lossy quality presets from 30% through 95% |
| **Black-and-white output** | Uses JBIG2 when `jbig2enc` is installed and CCITT Group 4 as the fallback; forcing this mode removes color |
| **Plain text (`.txt`)** | Exports structured text when positional PDF text data is available |
| **Markdown (`.md`)** | Exports detected document structure, optionally with YAML front matter |
| **ODT (`.odt`)** | Choose editable text positioned by PDF coordinates or a reflowable structured export of detected paragraphs, headings, tables, columns, and page breaks |

ODT layout is inferred from OCR/PDF coordinates and remains experimental. Review the result
when the source contains complex layouts, unusual fonts, or ambiguous tables.

### Image and screen-capture OCR

- Open an image, drop one onto the window, or paste image data with
  <kbd>Ctrl</kbd>+<kbd>V</kbd>
- Capture a region through the **Screen Capture** action
- Use the XDG Desktop Portal when available, with Spectacle, GNOME Screenshot, and Flameshot
  as command-line fallbacks
- Edit the recognized text and copy it to the clipboard
- Run `bigocrimage` as a separate image OCR application

JPEG, PNG, WebP, TIFF, and BMP are the common supported inputs. The standalone interface
recognizes additional filename extensions, including GIF and AVIF, but successful decoding
of those formats depends on the image loaders available on the system.

### Batch processing and recovery

- Multi-file queue with grid and list presentations
- PDF metadata, font, image, and attachment information in the queue
- Persistent per-run checkpoints; the application offers to resume pending files after an
  interrupted session
- Processing history with file, page, duration, and outcome data
- Cooperative cancellation and temporary-output cleanup
- Optional maximum output size that publishes numbered PDF parts when the result is too large
- A results page with per-file outcomes, extracted text, and export actions

---

## Interface

- GTK4 and libadwaita application windows
- Settings → Processing → Results workflow for PDF OCR
- Focusable settings and explicit labels for primary controls
- Grid/list queue presentations and contextual file/page actions
- Toast notifications for non-blocking feedback
- Before/after file-size comparison after processing
- Persisted window dimensions
- Keyboard shortcuts for common queue, editor, image, and application actions

Gettext catalogs are maintained for `be`, `bg`, `cs`, `da`, `de`, `el`, `en`, `es`, `et`,
`fi`, `fr`, `he`, `hr`, `hu`, `is`, `it`, `ja`, `ko`, `nl`, `no`, `pl`, `pt`, `pt_BR`,
`ro`, `ru`, `sk`, `sv`, `tr`, `uk`, `zh`, `zh_CN`, and `zh_TW`.

---

## Development

With the requirements above installed, run the applications from a checkout:

```bash
git clone https://github.com/biglinux/bigocrpdf.git
cd bigocrpdf
PYTHONPATH=src python3 -m bigocrpdf          # PDF OCR interface
PYTHONPATH=src python3 -m bigocrpdf.cli --help
```

Desktop files, icons, compiled translations and file-manager actions live under
[`usr/share`](usr/share/) and are installed by the package. A source run therefore uses the
installed translations; [`MAINTENANCE.md`](MAINTENANCE.md) shows how to test a catalog change.

### Layout

- `src/bigocrpdf/`: the applications and the CLI; `services/ocr/` holds the OCR pipeline,
  `ui/` the GTK interface, `ui/pdf_editor/` the page editor
- `usr/share/`: desktop entries, AppStream metadata, icons and file-manager actions
- `locale/`: the gettext template and catalogs
- `tests/`: the test suite; `benchmarks/`: the OCR benchmark harness and its baselines
- `tools/`: the validation gate, data staging for packages, and the benchmark driver

[`MAINTENANCE.md`](MAINTENANCE.md) maps each module to what it owns and records the contracts a
change must keep.

### Checks

```bash
bash tools/validate.sh
```

The gate runs Ruff, Pyright, the gettext checks, the desktop and AppStream validators,
ShellCheck and the test suite. `bash tools/quality.sh` adds the slow tests that drive the real
OCR engine; it needs ppocr-native and its plans installed.

### Translations

`locale/bigocrpdf.pot` is the template and `locale/<language>.po` the catalogs. The gate fails
when the template no longer matches the source: regenerate it with the `xgettext` command in
[`tools/validate.sh`](tools/validate.sh), then merge each catalog:

```bash
for po in locale/*.po; do msgmerge --update --backup=none "$po" locale/bigocrpdf.pot; done
```

Every catalog must stay free of fuzzy, obsolete and untranslated entries.

---

## License

[GPL-3.0-or-later](LICENSE)
