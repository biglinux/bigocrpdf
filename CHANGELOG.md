# Changelog

All notable user-visible and release-engineering changes are documented here.

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
