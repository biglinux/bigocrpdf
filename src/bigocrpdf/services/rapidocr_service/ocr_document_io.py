"""Publish OCR PDFs, and read or write structured OCR JSON on request.

An OCR run produces one file: the PDF. Structured OCR data is written only
where the caller asks for it, because a machine-readable copy of the page
text is a second copy of the document, and nobody asked for it to appear
next to their files.

What must outlive the run travels inside the PDF instead. Split outputs
carry their family in a private XMP namespace, which is what lets a later
overwrite retire the parts it replaced without guessing from file names.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bigocrpdf.services.rapidocr_service.config import (
    OcrDocument,
    OcrLayoutBlock,
    OcrLine,
    OcrPage,
    OCRResult,
    OcrWord,
)
from bigocrpdf.utils.durable_writes import (
    RetirementTarget,
    write_text_atomically,
)

OCR_JSON_VERSION = 2
_LEGACY_JSON_VERSION = 1
_HASH_CHUNK_BYTES = 1024 * 1024

# Private XMP schema for the few facts that must survive inside the PDF.
# XMP is the PDF's own metadata container, so these survive the PDF/A step
# and every tool that preserves metadata; a custom prefix needs its URI
# spelled out, or pikepdf writes an element with no namespace at all.
_XMP_NS = "http://bigocrpdf.biglinux.com.br/ns/ocr/1.0/"
_XMP_FAMILY_ROOT = f"{{{_XMP_NS}}}splitFamilyRoot"
_XMP_PART_INDEX = f"{{{_XMP_NS}}}splitPartIndex"
_XMP_PART_COUNT = f"{{{_XMP_NS}}}splitPartCount"


@dataclass(frozen=True)
class _SplitFamily:
    """Which part of which split output family one PDF is."""

    family_root: str
    part_index: int
    part_count: int


def ocr_document_json_path(pdf_path: str | Path) -> Path:
    """Return the default structured OCR JSON name for a PDF path.

    Only used when the caller asks for structured JSON without naming a
    destination; nothing writes here on its own.
    """
    path = Path(pdf_path)
    return path.with_suffix(".bigocr.json")


def complete_ocr_document(
    document: OcrDocument,
    *,
    pages_total: int,
    pages_processed: int,
) -> OcrDocument | None:
    """Return structured OCR only when it covers every output page exactly."""
    if pages_total <= 0 or pages_processed != pages_total:
        return None
    if not _document_covers_pages(document, pages_total):
        return None
    return document


def render_ocr_document_json(
    document: OcrDocument,
    pdf_path: str | Path,
    *,
    pdf_fingerprint: tuple[str, int] | None = None,
) -> str:
    """Render structured OCR bound to the PDF it describes.

    Written compact: this is a machine contract, and indentation was 62% of
    the bytes on an 18-page document.
    """
    from bigocrpdf.services.rapidocr_service.ocr_document_export import (
        enrich_ocr_document_layout,
    )

    enrich_ocr_document_layout(document)
    pdf_hash, pdf_size = pdf_fingerprint or _pdf_fingerprint(Path(pdf_path))
    payload = {
        "version": OCR_JSON_VERSION,
        "state": "document",
        "pdf": {
            "sha256": pdf_hash,
            "size_bytes": pdf_size,
        },
        "document": _document_to_dict(document),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"


def write_ocr_document_json(
    document: OcrDocument,
    pdf_path: str | Path,
    json_path: str | Path,
) -> Path:
    """Write structured OCR for *pdf_path* to the requested *json_path*."""
    destination = Path(json_path)
    write_text_atomically(
        destination,
        render_ocr_document_json(document, pdf_path),
    )
    return destination


def publish_ocr_pdfs(
    publications: Sequence[tuple[Path, Path]],
    *,
    overwrite: bool,
    family_root: Path | None = None,
) -> list[Path]:
    """Publish staged OCR PDFs to their requested paths as one recoverable set.

    Each publication is ``(staged_pdf, requested_pdf)``. More than one
    publication is a split output family: each part records its family in the
    PDF's own metadata, so a later overwrite can retire the parts it replaces.
    """
    items = [(Path(staged), Path(requested)) for staged, requested in publications]
    if not items:
        return []
    parents = {requested.parent.resolve(strict=True) for _staged, requested in items}
    if len(parents) != 1:
        raise ValueError("OCR PDF outputs must share one directory")
    directory = next(iter(parents))
    canonical_family_root = None
    if family_root is not None:
        canonical_family_root = family_root.parent.resolve(strict=True) / family_root.name
        if canonical_family_root.parent != directory:
            raise ValueError("OCR output family must use the publication directory")

    from bigocrpdf.utils.durable_writes import (
        copy_file_atomically,
        publish_files_transactionally,
    )

    with tempfile.TemporaryDirectory(
        prefix=".bigocr_ocr_bundle_",
        dir=directory,
    ) as staging_name:
        staging_dir = Path(staging_name)
        flattened: list[tuple[Path, Path]] = []
        expected_source_content: dict[str | Path, tuple[str, int]] = {}
        for index, (staged_pdf, requested_pdf) in enumerate(items):
            snapshot_pdf = staging_dir / f"{index}-payload.pdf"
            copy_file_atomically(
                staged_pdf,
                snapshot_pdf,
                overwrite=True,
            )
            _validated_pdf_page_count(snapshot_pdf)
            if canonical_family_root is not None and len(items) > 1:
                _write_split_family(
                    snapshot_pdf,
                    _SplitFamily(
                        family_root=canonical_family_root.name,
                        part_index=index + 1,
                        part_count=len(items),
                    ),
                )
            expected_source_content[snapshot_pdf] = _pdf_fingerprint(snapshot_pdf)
            flattened.append((snapshot_pdf, requested_pdf))

        def target_candidates(counter: int) -> list[Path]:
            return [
                (
                    requested_pdf
                    if counter == 0
                    else requested_pdf.with_name(
                        f"{requested_pdf.stem}-{counter}{requested_pdf.suffix}"
                    )
                )
                for _staged, requested_pdf in items
            ]

        return publish_files_transactionally(
            flattened,
            overwrite=overwrite,
            target_candidates=target_candidates,
            retire_candidates=(
                _ocr_output_family_retire_candidates(canonical_family_root)
                if overwrite and canonical_family_root is not None
                else None
            ),
            expected_source_content=expected_source_content,
        )


def _write_split_family(pdf_path: Path, family: _SplitFamily) -> None:
    """Record which split family part this PDF is, inside the PDF."""
    import pikepdf

    with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        with pdf.open_metadata(set_pikepdf_as_editor=False) as meta:
            meta[_XMP_FAMILY_ROOT] = family.family_root
            meta[_XMP_PART_INDEX] = str(family.part_index)
            meta[_XMP_PART_COUNT] = str(family.part_count)
        pdf.save(
            pdf_path,
            object_stream_mode=pikepdf.ObjectStreamMode.preserve,
            stream_decode_level=pikepdf.StreamDecodeLevel.none,
            compress_streams=True,
            force_version="1.7",
        )


def _read_split_family(pdf_path: Path) -> _SplitFamily | None:
    """Return the split family this PDF declares, if it declares a valid one."""
    import pikepdf

    try:
        with pikepdf.open(pdf_path) as pdf:
            meta = pdf.open_metadata()
            family_root = meta.get(_XMP_FAMILY_ROOT)
            raw_index = meta.get(_XMP_PART_INDEX)
            raw_count = meta.get(_XMP_PART_COUNT)
    except (pikepdf.PdfError, OSError, ValueError):
        return None
    if not isinstance(family_root, str) or not family_root:
        return None
    try:
        part_index = int(str(raw_index))
        part_count = int(str(raw_count))
    except (TypeError, ValueError):
        return None
    if part_count <= 1 or part_index < 1 or part_index > part_count:
        return None
    return _SplitFamily(
        family_root=family_root,
        part_index=part_index,
        part_count=part_count,
    )


def _ocr_output_family_retire_candidates(
    family_root: Path,
) -> Callable[[Sequence[Path]], Sequence[RetirementTarget]]:
    """Build a locked callback that retires inactive canonical OCR outputs."""
    root = family_root.parent.resolve(strict=True) / family_root.name
    numbered_pdf = re.compile(rf"^{re.escape(root.stem)}-([0-9]{{2,}}){re.escape(root.suffix)}$")

    def candidates(active_targets: Sequence[Path]) -> list[RetirementTarget]:
        active = set(active_targets)
        existing_family: list[RetirementTarget] = []
        if root not in active and os.path.lexists(root):
            existing_family.append(RetirementTarget.capture(root))
        for entry in root.parent.iterdir():
            match = numbered_pdf.fullmatch(entry.name)
            if match is None or entry in active:
                continue
            part = _verified_split_family_part(entry, root, int(match.group(1)))
            if part is not None:
                existing_family.append(part)
        return sorted(existing_family, key=lambda target: target.path)

    return candidates


def _verified_split_family_part(
    pdf_path: Path,
    family_root: Path,
    part_index: int,
) -> RetirementTarget | None:
    """Return *pdf_path* as a retirement target only if it is really our part.

    The name alone never decides: a file called ``contract-02.pdf`` may be the
    user's own. Only a PDF that declares this family and this part index in its
    own metadata is retired.
    """
    try:
        first = RetirementTarget.capture(pdf_path)
        if not stat.S_ISREG(pdf_path.lstat().st_mode):
            return None
        family = _read_split_family(pdf_path)
        if (
            family is None
            or family.family_root != family_root.name
            or family.part_index != part_index
        ):
            return None
        second = RetirementTarget.capture(pdf_path)
    except (OSError, UnicodeError, ValueError):
        return None
    if first != second:
        return None
    return second


def load_ocr_document_json(
    json_path: str | Path,
    pdf_path: str | Path,
    *,
    allow_unverified_legacy: bool = False,
) -> OcrDocument | None:
    """Load structured OCR from *json_path*, if it still describes *pdf_path*.

    Returns ``None`` when the file is absent or describes a different PDF, and
    raises ``ValueError`` when it is present but not readable as our contract.
    """
    sidecar_path = Path(json_path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(sidecar_path, flags)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"Invalid OCR JSON {sidecar_path}: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"Invalid OCR JSON {sidecar_path}: not a regular file")
        with os.fdopen(descriptor, encoding="utf-8") as stream:
            descriptor = -1
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid OCR JSON {sidecar_path}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid OCR JSON {sidecar_path}: payload must be an object")
    version = payload.get("version")
    if type(version) is not int or version not in {
        _LEGACY_JSON_VERSION,
        OCR_JSON_VERSION,
    }:
        raise ValueError(f"Unsupported OCR JSON version: {version!r}")
    if version == OCR_JSON_VERSION:
        pdf_payload = payload.get("pdf")
        if not isinstance(pdf_payload, dict):
            raise ValueError(f"Invalid OCR JSON {sidecar_path}: missing PDF metadata")
        expected_hash = pdf_payload.get("sha256")
        expected_size = pdf_payload.get("size_bytes")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError(f"Invalid OCR JSON {sidecar_path}: invalid PDF fingerprint")
        if type(expected_size) is not int or expected_size < 0:
            raise ValueError(f"Invalid OCR JSON {sidecar_path}: invalid PDF size")
        try:
            current_hash, current_size = _pdf_fingerprint(Path(pdf_path))
            if current_size != expected_size:
                return None
        except OSError:
            return None
        if not hmac.compare_digest(expected_hash, current_hash):
            return None
        state = payload.get("state")
        if state == "unavailable":
            reason = payload.get("reason")
            if not isinstance(reason, str) or not reason:
                raise ValueError(f"Invalid OCR JSON {sidecar_path}: missing unavailable reason")
            return None
        if state != "document":
            raise ValueError(f"Invalid OCR JSON {sidecar_path}: invalid state")
    if "document" not in payload:
        raise ValueError(f"Invalid OCR JSON {sidecar_path}: missing document payload")
    document_payload = payload["document"]
    if not isinstance(document_payload, dict):
        raise ValueError(f"Invalid OCR JSON {sidecar_path}: document payload must be an object")
    try:
        document = _document_from_dict(document_payload)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid OCR JSON {sidecar_path}: {exc}") from exc
    if version == _LEGACY_JSON_VERSION and not allow_unverified_legacy:
        return None
    return document


def _document_covers_pages(document: OcrDocument, page_count: int) -> bool:
    if len(document.pages) != page_count:
        return False
    page_indices = sorted(page.page_index for page in document.pages)
    return page_indices == list(range(1, page_count + 1))


def _validated_pdf_page_count(path: Path) -> int:
    import pikepdf

    path_stat = path.lstat()
    if not stat.S_ISREG(path_stat.st_mode):
        raise ValueError(f"Staged OCR output is not a regular PDF: {path}")
    expected_identity = (path_stat.st_dev, path_stat.st_ino)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        opened_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_stat.st_mode)
            or (opened_stat.st_dev, opened_stat.st_ino) != expected_identity
        ):
            raise OSError(f"Staged OCR output changed before validation: {path}")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            try:
                with pikepdf.open(stream) as pdf:
                    page_count = len(pdf.pages)
            except pikepdf.PdfError as error:
                raise ValueError(f"Invalid staged OCR PDF {path}: {error}") from error
        if page_count <= 0:
            raise ValueError(f"Staged OCR PDF has no pages: {path}")
        final_stat = os.fstat(descriptor)
        if (
            final_stat.st_dev,
            final_stat.st_ino,
            final_stat.st_size,
            final_stat.st_mtime_ns,
            final_stat.st_ctime_ns,
        ) != (
            opened_stat.st_dev,
            opened_stat.st_ino,
            opened_stat.st_size,
            opened_stat.st_mtime_ns,
            opened_stat.st_ctime_ns,
        ):
            raise OSError(f"Staged OCR output changed while validating: {path}")
    finally:
        os.close(descriptor)
    return page_count


def _pdf_fingerprint(path: Path) -> tuple[str, int]:
    return _regular_file_fingerprint(path, description="OCR PDF")


def _regular_file_fingerprint(
    path: Path,
    *,
    description: str,
) -> tuple[str, int]:
    path_stat = path.lstat()
    if not stat.S_ISREG(path_stat.st_mode):
        raise ValueError(f"{description} is not a regular file: {path}")
    expected_identity = (path_stat.st_dev, path_stat.st_ino)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        opened_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_stat.st_mode)
            or (opened_stat.st_dev, opened_stat.st_ino) != expected_identity
        ):
            raise OSError(f"{description} changed before fingerprinting: {path}")
        while chunk := os.read(descriptor, _HASH_CHUNK_BYTES):
            digest.update(chunk)
        final_stat = os.fstat(descriptor)
        if (
            final_stat.st_dev,
            final_stat.st_ino,
            final_stat.st_size,
            final_stat.st_mtime_ns,
            final_stat.st_ctime_ns,
        ) != (
            opened_stat.st_dev,
            opened_stat.st_ino,
            opened_stat.st_size,
            opened_stat.st_mtime_ns,
            opened_stat.st_ctime_ns,
        ):
            raise OSError(f"{description} changed while fingerprinting: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest(), opened_stat.st_size


def _document_to_dict(document: OcrDocument) -> dict[str, Any]:
    return {
        "diagnostics": _json_safe(document.diagnostics),
        "pages": [_page_to_dict(page) for page in document.pages],
    }


def _page_to_dict(page: OcrPage) -> dict[str, Any]:
    return {
        "page_index": page.page_index,
        "width_px": page.width_px,
        "height_px": page.height_px,
        "dpi": page.dpi,
        "native_text": page.native_text,
        "text_layer_quality": page.text_layer_quality,
        "retry_level": page.retry_level,
        "diagnostics": _json_safe(page.diagnostics),
        "text_results": [_ocr_result_to_dict(result) for result in page.text_results],
        "lines": [_ocr_line_to_dict(line) for line in page.lines],
        "layout_blocks": [_layout_block_to_dict(block) for block in page.layout_blocks],
    }


def _ocr_result_to_dict(result: OCRResult) -> dict[str, Any]:
    return {
        "text": result.text,
        "box": result.box,
        "confidence": result.confidence,
    }


def _ocr_line_to_dict(line: OcrLine) -> dict[str, Any]:
    return {
        "text": line.text,
        "bbox": line.bbox,
        "reading_order": line.reading_order,
        "source": line.source,
        "words": [_ocr_word_to_dict(word) for word in line.words],
    }


def _ocr_word_to_dict(word: OcrWord) -> dict[str, Any]:
    return {
        "text": word.text,
        "bbox": word.bbox,
        "confidence": word.confidence,
    }


def _layout_block_to_dict(block: OcrLayoutBlock) -> dict[str, Any]:
    return {
        "kind": block.kind,
        "text": block.text,
        "rows": block.rows,
        "raw_lines": block.raw_lines,
        "indent_chars": block.indent_chars,
        "y_top": block.y_top,
        "reading_order": block.reading_order,
    }


def _document_from_dict(payload: dict[str, Any]) -> OcrDocument:
    document = OcrDocument(diagnostics=dict(payload.get("diagnostics") or {}))
    for page_payload in payload.get("pages") or []:
        document.append_page(_page_from_dict(page_payload))
    return document


def _page_from_dict(payload: dict[str, Any]) -> OcrPage:
    return OcrPage(
        page_index=int(payload["page_index"]),
        width_px=int(payload["width_px"]),
        height_px=int(payload["height_px"]),
        dpi=int(payload["dpi"]),
        text_results=[
            OCRResult(
                text=str(result.get("text") or ""),
                box=result.get("box") or [],
                confidence=float(result.get("confidence") or 0.0),
            )
            for result in payload.get("text_results") or []
        ],
        lines=[_line_from_dict(line) for line in payload.get("lines") or []],
        layout_blocks=[
            _layout_block_from_dict(block) for block in payload.get("layout_blocks") or []
        ],
        native_text=str(payload.get("native_text") or ""),
        text_layer_quality=str(payload.get("text_layer_quality") or "absent"),
        retry_level=int(payload.get("retry_level") or 0),
        diagnostics=dict(payload.get("diagnostics") or {}),
    )


def _line_from_dict(payload: dict[str, Any]) -> OcrLine:
    return OcrLine(
        text=str(payload.get("text") or ""),
        bbox=[float(value) for value in payload.get("bbox") or []],
        words=[_word_from_dict(word) for word in payload.get("words") or []],
        reading_order=int(payload.get("reading_order") or 0),
        source=str(payload.get("source") or "ocr"),
    )


def _word_from_dict(payload: dict[str, Any]) -> OcrWord:
    return OcrWord(
        text=str(payload.get("text") or ""),
        bbox=[float(value) for value in payload.get("bbox") or []],
        confidence=float(payload.get("confidence") or 0.0),
    )


def _layout_block_from_dict(payload: dict[str, Any]) -> OcrLayoutBlock:
    return OcrLayoutBlock(
        kind=str(payload.get("kind") or "paragraph"),
        text=str(payload.get("text") or ""),
        rows=[
            [str(cell) for cell in row]
            for row in payload.get("rows") or []
            if isinstance(row, (list, tuple))
        ],
        raw_lines=[str(line) for line in payload.get("raw_lines") or []],
        indent_chars=int(payload.get("indent_chars") or 0),
        y_top=float(payload.get("y_top") or 0.0),
        reading_order=int(payload.get("reading_order") or 0),
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    return str(value)
