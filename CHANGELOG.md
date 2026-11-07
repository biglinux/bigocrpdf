# Changelog

All notable user-visible and release-engineering changes are documented here.

## Unreleased

- Open images from PDFs, screenshots and other files only in the formats OCR reads, so a crafted image cannot reach a decoder that runs an external program such as Ghostscript. OpenCV refuses images over the same pixel limit as Pillow.
- Show file names and PDF metadata as plain text: a name or title holding `<` or `&` was shown as markup or not at all. A file whose name starts with a dash reaches the PDF tools as a file, never as an option.
- Read PDFs in a sandbox: the Poppler tools that parse a PDF for page images, renders, text and metadata run under bubblewrap, without network, home directory or write access outside their working directory, and the window no longer renders PDFs with Poppler in its own process. Without a usable bwrap they run as before and the OCR diagnostics say so. Depends on `bubblewrap` instead of `poppler-glib`.
- Read text on the graphics card by default, through Vulkan, and on the processor when there is no usable one. A 10-page 300 dpi scan takes 6.9 s instead of 11.8 s on a Core i5-13400 with a GeForce GTX 1660 Ti, with the same text. Turning off Use the Graphics Card in the advanced settings, or passing `bigocrpdf-cli ocr --no-gpu`, keeps OCR on the processor, and a worker that crashes on the graphics card is started again on the processor. Needs ppocr-native 26.11.06-1615 or newer.
- Extract and preprocess the next pages while the current ones are read, so the OCR engine no longer waits between batches of pages. A 177-page scanned book takes about 115 s instead of 145 s with a GeForce GTX 1660 Ti, with the same text. Machines with a single preprocessing worker, which budget memory for one batch, still prepare one batch at a time.
- Delete each page's OCR input as soon as the page is read and each batch's extracted images once they are processed, so temporary disk use stays near two batches of pages however large the PDF: about 230 MB instead of 630 MB on a 177-page scan.
- Quit the application on SIGTERM or SIGHUP, at logout or shutdown, the way closing its window does, and stop `bigocrpdf-cli` on them the way Ctrl+C does: a running OCR job is cancelled and its workers and temporary files are cleaned up instead of being left behind. A job's temporary folder now carries its process ID, so the next job removes one left by a run that was killed or crashed.
- Keep scanned pages the way the PDF shows them when it draws the page image turned or mirrored, also from inside a form XObject. Such pages came out upside down or mirrored after OCR.
- OCR PDFs that hold an image over 128 MP instead of refusing the whole document. A JPEG photo of that size, as 200 MP phone cameras take, gets the same corrections as any other page; an image of that size in another format gets no geometric correction. Image OCR reads such photos instead of refusing them.
- Render a page too large for the 45 MP render budget at whatever resolution fits it, instead of stopping the whole document: a photo saved at 72 dpi or none makes a page metres wide. A page that still yields no image is reported as unread and the following pages are read as usual.
- Publish the output when some pages cannot be read, with those pages kept without their text, instead of failing the whole document. The command line warns which pages and why; the window marks the file and lists the pages and causes when processing ends. A document none of whose pages could be read still fails.

## 4.0.0 - 2026-10-13

### OCR engine

- Read pages with [ppocr-native](https://github.com/biglinux/ppocr-native), PP-OCRv6 in plain Rust, instead of RapidOCR on OpenVINO, and straighten pages with its detector. A 30-page scanned PDF takes 110–116 s instead of 216 s on an 8-core laptop, at a peak of 2.5–2.8 GB instead of 17.3 GB of memory, with the same character error rate.
- Remove RapidOCR, OpenVINO, ONNX Runtime and the GPU backends, and with them the `bigocrpdf-cli ocr` options `--engine-type`, `--model-type`, `--rec-batch-num`, `--use-textline-cls`, `--gpu-backend`, `--gpu-device-id`, `--gpu-fp16` and `--no-gpu-fallback`. Scripts that pass them must drop them.
- Remove the language selection: PP-OCRv6 reads Chinese, English, Japanese and Latin-script languages with one model.
- Add `--full-resolution` and `--no-full-resolution` to `bigocrpdf-cli ocr`; full-resolution detection stays off by default.
- Stop lines on low-contrast scans from being read as blank, and stop one vertical line from disabling detection on the pages after it.
- Run OCR on the page without the scanner effect, which only changes how the page looks, and make the effect turn paper texture white instead of into black speckle.
- Place OCR text of skewed lines on mixed-content and embedded-image pages on the line's own baseline and glyph height, as on scanned pages.
- Restore perspective correction, which did nothing with NumPy 2.

### Searchable text

- Write the text layer in Noto Sans from `noto-fonts`, so Greek, Cyrillic and extended Latin words stay searchable and copyable. A missing font fails the job instead of falling back to Helvetica, which dropped every character outside Latin-1. `BIGOCRPDF_TEXT_FONT` names another font file.
- Write lines with Chinese or Japanese characters in WenQuanYi Micro Hei when `wqy-microhei` is installed (`BIGOCRPDF_CJK_FONT` names another file).
- Publish one file per OCR run: the structured OCR JSON is written only with `ocr --sidecar-json [FILE]` and read only with `export-* --from-json FILE`.
- Fix inverted pages in Black & White output when `jbig2enc` is absent.

### Exports

- Keep double-spaced pages as paragraphs in text, Markdown and ODT exports instead of one paragraph per line.
- Export detected headings as ODT headings, which gives the document an outline and a table of contents.
- Stop the fixed-layout ODT from overlapping its own lines, and give it the page's real size.

### Interface

- Write the PDF editor's merged or saved PDF in the background: the window keeps responding and shows Saving…, closing it waits for the save, and a failed save says why. The standalone editor saves unsaved changes through Save As.
- Keep `--edit` working when BigOCR PDF is already running, so Create PDF from the file manager opens the editor.
- Accept `--debug` and `--verbose` in the graphical application.
- Clear the finished queue when returning to the start screen.
- Show file sizes in translated units with the decimal multiples the file manager uses, and counts with the locale's separators.
- Give the image OCR window a main menu with Keyboard Shortcuts (Ctrl+?) and About.
- Keep the text order of a screen capture, and open Spectacle's region selector directly on KDE Plasma.
- Use the icon theme the user chose, with bundled icons only for names it lacks.
- Name editor pages and other controls for screen readers.
- Complete the translations in every catalog and fix plural forms that showed both forms at once.

### Files and packaging

- Replace files with a plain atomic rename, which keeps settings after a crash during a save and works on sshfs, NFS and OverlayFS. An interrupted save in an earlier version may have left a hidden `.bigocr-publish-*` file next to an output: it holds the previous version and can be renamed back.
- Require `ppocr-native` and `noto-fonts` in the Arch package, and drop the RapidOCR and OpenVINO packages from its dependencies.
- Build the Nix package against GTK 4.14 and libadwaita 1.5, the floor the application checks at startup.

## 3.0.0 - 2026-08-02

- Use the current libadwaita shortcuts dialog instead of the deprecated GTK widget.
- Validate GTK 4.22 and libadwaita 1.8 before importing the graphical applications.
- Use OpenVINO's public Python API for backend availability checks.
- Fix AVIF pages being accepted by the editor but treated as PDF input during save.
- Align desktop, KIO, Nemo, and Nautilus MIME types with the formats each workflow accepts.
- Make the Arch package version deterministic and pin the fallback Git source to the release tag.
- Correct Arch and Nix Poppler dependencies, Nix data paths, runtime command lookup, and ICC profile discovery.
- Enable PDF/A metadata preparation by default, with an explicit regular-PDF opt-out, and document that formal conformance requires external validation.
- Require a Pillow version that can decode AVIF when built with libavif support.
- Normalize gettext revision headers and validate all catalogs against the template.
- Add release hygiene checks for bytecode, compiled translations, and generated package metadata.
- Remove generated Python metadata, bytecode, and unused npm placeholder files from the source release.
- Replace the refactoring plan with current architecture and maintenance documentation.
