# Changelog

All notable user-visible and release-engineering changes are documented here.

## Unreleased

- Feed the text detector PaddleOCR's input normalization and tighter line boxes, which stops whole lines on low-contrast scans from being read as blanks; the box expansion ratio is no longer stored in the settings file.
- Run OCR on the page without the scanner effect, which only changes how the page image looks, and rebuild the effect so paper texture turns white instead of into black speckle.
- Place OCR text of skewed lines on mixed-content and embedded-image pages the way scanned pages do, on the line's own baseline and glyph height instead of its bounding box.
- Show file sizes on the results page in translated units with the decimal multiples the file manager uses.
- Start the OCR worker without importing the rest of the application, which cuts its startup imports from about half a second to a few hundredths.
- Let the Nix package build against GTK 4.14 and libadwaita 1.5, the floor the application checks at startup.
- Replace files with a plain atomic rename, which keeps settings after a crash during a save, works on sshfs and NFS, and needs no free space beyond the new file; settings saves no longer sync the disk. An interrupted older build may have left a hidden `.bigocr-publish-*` file next to an output: it holds the previous version and can be renamed back.

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
