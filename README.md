**BigOCR PDF** is a powerful utility integrated into the Linux desktop environment (specifically optimized for BigLinux) that brings Optical Character Recognition (OCR) capabilities to your PDF documents and image files. It seamlessly transforms scanned documents into searchable PDFs and allows for easy text extraction from images or screen regions.

![BigOCR PDF Screenshot](https://user-images.githubusercontent.com/6098501/178167560-d00d941c-8ba7-46b9-9c64-4bd1a8a47a13.jpeg)

## ✨ Features

-   **Make PDFs Searchable**: Convert scanned non-searchable PDFs into files where you can search, select, and copy text.
-   **Image OCR**: Extract text directly from standard image files (JPG, PNG, etc.).
-   **Screen Capture Integration**: Extract text from anywhere on your screen—perfect for grabbing text from videos, protected websites, or UI elements—by selecting a rectangular region.
-   **Batch Processing**: Efficiently process multiple files at once directly from your file manager.

## 🚀 Usage

### 1. Processing PDF Files
Scanned PDFs often lack a text layer. To fix this:
1.  Open your file manager.
2.  Select one or more PDF files.
3.  Right-click and select the **"OCR"** option.
4.  A new, searchable version of the file will be generated.

### 2. Extracting Text from Images
1.  Right-click on any image file.
2.  Select **"Extract text from image (OCR)"**.
3.  The extracted text will be available for use.

### 3. Screen Text Extraction
For text that cannot be selected normally (e.g., inside a video or image on a website):
1.  Launch your screenshot tool (e.g., press `Print Screen`).
2.  Select the **"Rectangular Region"** tool.
3.  Highlight the area containing the text you want to copy.
4.  Click **"Export"** and choose **"Extract text from image (OCR)"**.

## 🛠️ Installation & Development

### Prerequisites

Ensure you have the following system dependencies installed:
-   Python 3.10 or higher
-   GTK4 and Libadwaita
-   OCRmyPDF (the core OCR engine)
-   Tesseract OCR
-   Ghostscript

### Building from Source

To install the latest version from the repository:

```bash
# Clone the repository
git clone https://github.com/biglinux/bigocrpdf.git
cd bigocrpdf

# Install the package
pip install .
```

