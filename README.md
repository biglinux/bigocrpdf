<div align="center">

# BigOcrPDF

**The complete OCR toolkit for Linux — turn scanned PDFs and images into searchable, editable documents.**

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://python.org)
[![GTK4 + Libadwaita](https://img.shields.io/badge/GTK4-Libadwaita-4A86CF.svg)](https://gnome.org)

</div>

---

BigOcrPDF is a powerful, all-in-one OCR application that adds searchable text layers to scanned PDFs, extracts text from images, and provides a full-featured PDF editor — all from a modern, native Linux interface.

## Why BigOcrPDF?

- **AI-Powered OCR** — Uses **RapidOCR PP-OCRv5** with OpenVINO hardware acceleration for fast, accurate text recognition across **130+ languages**
- **Edit, Merge & Organize PDFs** — Reorder pages, rotate, delete, and combine multiple PDFs and images into a single document
- **Smart Preprocessing** — Automatic perspective correction, deskew, dewarping, and illumination normalization — even photos of documents come out clean
- **Multiple Export Formats** — Searchable PDF, PDF/A-2b archival, plain text, and ODF/ODT with layout-aware formatting
- **Screen Capture OCR** — Select any region on screen and instantly extract text
- **Batch Processing** — Process dozens of files at once with checkpoint/resume support
- **File Manager Integration** — Right-click any PDF or image to OCR it directly

---

## Key Features

### PDF Editor

Manage your documents before and after OCR — no need for a separate tool.

- **Drag-and-drop page reordering** with thumbnail previews
- **Rotate pages** left or right in 90° increments
- **Delete pages** you don't need
- **Merge files** — combine pages from multiple PDFs and images into one document
- **Create PDFs from images** — import JPEG, PNG, TIFF, WebP, RAW photos, and more
- **EXIF-aware import** — automatically applies correct orientation from camera metadata
- **Zoom control** — 50% to 200% thumbnail scaling
- **Select pages for OCR** — choose exactly which pages to process

### OCR Engine

State-of-the-art text recognition powered by deep learning.

- **RapidOCR PP-OCRv5** models with OpenVINO inference (ONNX fallback)
- **130+ languages** across 12 script families: Latin, Chinese, Japanese, Korean, Arabic, Cyrillic, Greek, Devanagari, Tamil, Telugu, Thai, and more
- **4 precision levels** — from fast to very precise, tunable per job
- **Parallel processing** — multi-core batch OCR with automatic worker scaling
- **Invisible text layer** — preserves original page appearance while adding searchable text
- **Smart detection** — auto-identifies image-only vs. mixed-content PDFs
- **Re-OCR support** — replace existing text layers with improved recognition
- **Right-to-left text** — full BiDi support for Arabic and Hebrew via `fribidi`

### Image Preprocessing

Automatically clean up scans and photos before OCR for maximum accuracy.

- **Perspective correction** — 6-mode cascade that straightens photographed documents
- **Auto deskew** — fixes tilted scans using morphological analysis + Hough transform
- **Baseline dewarp** — per-line polynomial fitting to flatten curved text
- **Orientation detection** — auto-correct 90°/180°/270° rotations
- **Illumination normalization** — even out uneven lighting
- **Scanner effect** — LAB-space background normalization
- **Denoising** — bilateral filter and Non-Local Means
- **All toggles individually controllable** from the settings page

### Export Options

Get your text out in the format you need.

| Format | Description |
|--------|-------------|
| **Searchable PDF** | Original pages with invisible OCR text layer |
| **PDF/A-2b** | ISO archival standard with JPEG 2000 compression |
| **Custom Quality PDF** | Choose JPEG quality: 30%, 50%, 70%, 85%, or 95% |
| **Plain Text (.txt)** | Extracted text from all pages |
| **ODF/ODT** | 4 modes: formatted + images, images + simple text, formatted text only, or plain text |

ODF export includes **layout analysis**: automatic paragraph/heading detection, table detection, image embedding, and proper page breaks.

### Screen Capture & Image OCR

Extract text from anything on your screen.

- **Region capture** — select an area and get the text instantly
- **Works with**: Spectacle (KDE), GNOME Screenshot, Flameshot
- **Open any image** — JPEG, PNG, WebP, TIFF, RAW formats (CR2, DNG, NEF, ARW, and more)
- **Copy to clipboard** with one click
- **Standalone mode** — run `bigocrimage` for a dedicated image OCR window

### Batch Processing & Session Management

Handle large workloads efficiently.

- **Multi-file queue** — add files via drag-and-drop or file chooser
- **Checkpoint/resume** — interrupted sessions automatically resume on next launch
- **Processing history** — tracks file sizes, page counts, processing time, and success/failure
- **Cancel anytime** with clean cleanup
- **Auto-split output** — configurable maximum file size (10MB–100MB)
- **Results page** with per-file statistics, text viewer, and export actions

---

## Installation

### Arch Linux / BigLinux

```bash
pacman -S bigocrpdf
```

### From Source

```bash
git clone https://github.com/biglinux/bigocrpdf.git
cd bigocrpdf
pip install -e .
```

#### Dependencies

| Package | Purpose |
|---------|---------|
| `python >= 3.10` | Runtime |
| `gtk4`, `libadwaita` | User interface |
| `python-rapidocr-pp-ocrv5` | OCR engine |
| `python-rapidocr-openvino` | Hardware-accelerated inference |
| `poppler-utils` | PDF image extraction (`pdfimages`, `pdftoppm`, `pdfinfo`) |
| `ghostscript` | PDF/A-2b conversion |
| `python-opencv` | Image preprocessing |
| `python-numpy` | Array operations |
| `python-pillow` | Image format support |
| `python-odfpy` | ODF/ODT export |
| `fribidi` | BiDi text reordering (Arabic, Hebrew) |

---

## Usage

### GUI

```bash
bigocrpdf                     # PDF OCR interface
bigocrimage                   # Image OCR window
```

### Command Line

```
bigocrpdf [OPTIONS] [FILES...]

Options:
  -v, --version     Show version and exit
  -d, --debug       Enable debug logging
  --verbose         Verbose output
  --image-mode      Launch in image OCR mode
  FILES             PDF or image files to open
```

### File Manager Integration

- **Right-click a PDF** → *Recognize text in scanned PDF (OCR)*
- **Right-click an image** → *Extract text from image (OCR)*
- **KDE Dolphin** context menu integration included

### Screen Capture

Press **Print Screen** → select a region → export to **Extract text from image (OCR)**.

---

## Interface

### UI Highlights

- **GTK4 + Libadwaita** — clean, modern design following GNOME Human Interface Guidelines
- **Multi-page wizard** — Settings → Processing → Results
- **Toast notifications** — non-intrusive status feedback
- **Before/After comparison** — track file size changes after OCR
- **Window size persistence** — remembers your preferred dimensions
- **28 UI languages** — Bulgarian, Chinese, Czech, Danish, Dutch, English, Estonian, Finnish, French, German, Greek, Hebrew, Croatian, Hungarian, Icelandic, Italian, Japanese, Korean, Norwegian, Polish, Portuguese, Romanian, Russian, Slovak, Spanish, Swedish, Turkish, Ukrainian

---

## Architecture

```
src/bigocrpdf/
├── application.py              # Adw.Application entry point
├── window.py                   # Main PDF OCR window
├── config.py                   # Constants and configuration
├── services/
│   ├── processor.py            # OCR engine interface
│   ├── screen_capture.py       # Screen capture + image OCR
│   ├── export_service.py       # PDF/text/ODF export
│   ├── contour_analysis.py     # Document contour detection
│   ├── perspective_correction.py
│   └── rapidocr_service/      # RapidOCR PP-OCRv5 integration
│       ├── engine.py           # Singleton OCR engine
│       ├── ocr_worker.py       # Subprocess OCR worker
│       ├── preprocessor.py     # Image preprocessing pipeline
│       ├── rotation.py         # Orientation detection
│       └── ...
├── ui/
│   ├── image_ocr_window.py     # Standalone image OCR
│   ├── settings_page.py        # OCR settings
│   ├── conclusion_page.py      # Results & export
│   ├── pdf_editor/             # PDF page editor
│   └── ...
└── utils/
    ├── odf_exporter.py         # ODF document generation
    ├── layout_analyzer.py      # Document structure detection
    ├── checkpoint_manager.py   # Session resume support
    └── ...
```

---

## License

[GPL-3.0-or-later](LICENSE)
