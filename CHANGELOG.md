# Changelog

All notable user-visible and release-engineering changes are documented here.

## Unreleased

- Write the text layer in Noto Sans from `noto-fonts`, so Greek, Cyrillic and extended Latin words stay searchable and copyable; a missing font now fails the job instead of falling back to Helvetica, which dropped every character outside Latin-1. `BIGOCRPDF_TEXT_FONT` names another font file. Lines with Chinese or Japanese characters use WenQuanYi Micro Hei from `wqy-microhei` when it is installed (`BIGOCRPDF_CJK_FONT` names another file).
- Feed the text detector PaddleOCR's input normalization and tighter line boxes, which stops whole lines on low-contrast scans from being read as blanks; the box expansion ratio is no longer stored in the settings file.
- Run OCR on the page without the scanner effect, which only changes how the page image looks, and rebuild the effect so paper texture turns white instead of into black speckle.
- Place OCR text of skewed lines on mixed-content and embedded-image pages the way scanned pages do, on the line's own baseline and glyph height instead of its bounding box.
- Show file sizes in the queue, the file details and the results page in translated units with the decimal multiples the file manager uses, and size changes and text counts with the locale's separators.
- Start the graphical application with `--debug` or `--verbose`, which it refused as unknown options.
- Let the Nix package build against GTK 4.14 and libadwaita 1.5, the floor the application checks at startup.
- Replace files with a plain atomic rename, which keeps settings after a crash during a save, works on sshfs and NFS, and needs no free space beyond the new file; settings saves no longer sync the disk. An interrupted older build may have left a hidden `.bigocr-publish-*` file next to an output: it holds the previous version and can be renamed back.
- Write the PDF editor's merged or saved PDF in the background: the window keeps responding and shows Saving…, editing waits until the file is written, closing the window waits for the save, and a failed save says why. Unsaved changes in the standalone editor are saved through Save As instead of an Apply that did nothing.
- Give the image OCR window a main menu with Keyboard Shortcuts (Ctrl+?) and About; F1 no longer opens About there.
- Read pages with ppocr-native, PP-OCRv6 in plain Rust, instead of RapidOCR on OpenVINO, and straighten pages with its detector. A 30-page scanned PDF took 110–116 s instead of 216 s on an 8-core laptop, at a peak of 2.5–2.8 GB instead of 17.3 GB of memory (PSS summed over all processes), with the same text on 26 of the pages and the same character error rate. RapidOCR, OpenVINO, ONNX Runtime and the GPU backends are gone, and with them the `--engine-type`, `--model-type`, `--rec-batch-num`, `--use-textline-cls`, `--gpu-backend`, `--gpu-device-id`, `--gpu-fp16` and `--no-gpu-fallback` options. The benchmark profiles are `fast`, `balanced` and `quality`.
- Hand pages to the OCR worker as uncompressed PPM, which ppocr-native requires: a 12-megapixel page is written in 29 ms instead of 438 ms as PNG and read in 43 ms instead of 195 ms; the 30-page PDF above peaked at 288 MiB of temporary files instead of 162 MiB.

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
