# BigOcrPDF

Add OCR to your PDF documents to make them searchable — powered by **RapidOCR PP-OCRv5**.  
Modern GTK4 + Libadwaita interface for BigLinux and Arch-based distributions.

## Features

### OCR Engine
- **RapidOCR PP-OCRv5** AI models for state-of-the-art text recognition
- **27 languages** including Latin, Chinese, Japanese, Korean, Arabic, Cyrillic, Devanagari, and more
- **Parallel processing** with multi-core CPU utilization for batch jobs
- **BiDi text support** via `fribidi` for right-to-left scripts (Arabic, Hebrew)

### Image Processing
- **Auto deskew** — automatic rotation correction for scanned pages
- **Orientation detection** — auto-detect and fix 90°/180°/270° rotations
- **Perspective correction** — straighten photographed documents
- **Quality preservation** — auto-detect original JPEG quality to avoid recompression

### Output Formats
- **PDF with OCR layer** — searchable PDF preserving the original layout
- **PDF/A-2b** — archival format with JPEG 2000 compression
- **Text export** — auto-save extracted text to `.txt` files
- **ODF export** — export to LibreOffice/OpenDocument format

### Image OCR Mode (bigocrimage)
- **Screen capture** — select a region to extract text instantly
- **Image file OCR** — open any image and extract text
- **Drag and drop** support

### User Interface
- **GTK4 + Libadwaita** — clean, accessible design following GNOME HIG
- **Adw.StatusPage** welcome and loading screens
- **Toast notifications** for non-intrusive feedback
- **Before/After comparison** — track file size changes
- **Processing history** — view statistics of processed files
- **20+ languages** for the UI (translations via gettext)

## System Requirements

- **Python** 3.10+
- **GTK4** and **Libadwaita**
- **poppler-utils** — `pdfimages`, `pdftoppm`, `pdfinfo` for PDF image extraction
- **ghostscript** — PDF/A-2b conversion
- **fribidi** — BiDi text reordering for Arabic/Hebrew OCR

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

## Usage

### GUI Application

```bash
bigocrpdf                     # Start the main PDF OCR interface
bigocrimage                   # Start the Image OCR window
```

### Command Line

```
bigocrpdf [OPTIONS] [FILES...]

Options:
  -v, --version     Print version information and exit
  -d, --debug       Enable debug mode
  --verbose         Enable verbose output
  --image-mode      Start in image OCR mode
  FILES             PDF or image files to process
```

### Context Menu Integration

Right-click on PDF files in your file manager and select **OCR PDF**.  
Right-click on image files and select **Extract text from image (OCR)**.

### Screen Capture OCR

Press **Print Screen**, select a region, then export to **Extract text from image (OCR)**.

## Project Structure

```
src/bigocrpdf/
├── application.py          # Adw.Application entry point
├── window.py               # Main PDF OCR window
├── config.py               # Constants and configuration
├── services/               # Business logic (OCR, capture, export)
│   ├── processor.py        # OCR engine interface
│   ├── screen_capture.py   # Screen capture + image OCR
│   ├── export_service.py   # PDF/text/ODF export
│   └── rapidocr_service/   # RapidOCR PP-OCRv5 integration
├── ui/                     # Presentation layer (GTK4 widgets)
│   ├── image_ocr_window.py # Standalone image OCR window
│   ├── settings_page.py    # Settings page
│   └── pdf_editor/         # PDF page editor
└── utils/                  # Pure Python helpers
    ├── i18n.py             # Internationalization
    ├── odf_exporter.py     # ODF document generation
    └── pdf_utils.py        # PDF manipulation utilities
```

## License

GPL-3.0-or-later

---

# PT-BR

OCR para PDF e arquivos de imagem integrado no sistema.

Arquivos em formato PDF que foram digitalizados não possuem a opção de efetuar buscas ou copiar o texto. No BigLinux, basta clicar com o botão direito no arquivo e utilizar a opção de OCR — será criado um novo arquivo com esses recursos.

Se for necessário efetuar o procedimento em vários arquivos PDF, basta selecionar todos e utilizar a opção de OCR uma vez.

Também é possível extrair o texto de um arquivo de imagem, basta clicar com o botão direito e utilizar a opção: **"Extrair texto da imagem (OCR)"**.

E ainda é possível utilizar diretamente da ferramenta de captura de tela: aperte **Print Screen**, use a ferramenta de **"Região Retangular"**, selecione a região com o texto e depois clique em **"Exportar"** → **"Extrair o texto da imagem (OCR)"**.
