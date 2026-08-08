#!/usr/bin/env python3
"""Prepare normalized benchmark manifests for BigOCRPDF."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

FUNSD_DATASET_URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
FUNSD_DATASET_SHA256 = "c31735649e4f441bcbb4fd0f379574f7520b42286e80b01d80b445649d54761f"
MAX_FUNSD_DOWNLOAD_BYTES = 64 * 1024 * 1024
MAX_ZIP_MEMBERS = 5_000
MAX_ZIP_MEMBER_BYTES = 128 * 1024 * 1024
MAX_ZIP_TOTAL_BYTES = 512 * 1024 * 1024
MAX_ZIP_COMPRESSION_RATIO = 200
COPY_CHUNK_BYTES = 1024 * 1024
MANIFEST_VERSION = 1
SAFE_SAMPLE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
DETERMINISTIC_PDF_TIMESTAMP = time.gmtime(946684800)
SUPPORTED_DATASETS = {
    "dharmaocr",
    "funsd",
    "portuguese_ocr",
    "portuguese_synthetic",
    "sroie",
    "synthetic",
}


@dataclass(frozen=True)
class HuggingFaceDatasetSpec:
    hf_id: str
    revision: str
    dataset_label: str
    language: str
    tags: list[str]
    text_fields: list[str]


HF_DATASET_SPECS = {
    "sroie": HuggingFaceDatasetSpec(
        hf_id="rth/sroie-2019-v2",
        revision="ede4bd3cd7e687bd5580d0e3f52cda1b4e9bac1c",
        dataset_label="SROIE",
        language="en",
        tags=["receipt", "small_text", "english", "numbers"],
        text_fields=["words", "text", "ocr_words", "transcription", "ground_truth"],
    ),
    "dharmaocr": HuggingFaceDatasetSpec(
        hf_id="Dharma-AI/DharmaOCR-Benchmark",
        revision="e8f4bb516839c1a0a32ee0234b3cbcd5aa5c10d3",
        dataset_label="DharmaOCR",
        language="pt",
        tags=["pt_br", "legal", "administrative"],
        text_fields=["text", "transcription", "ground_truth", "gt_text", "answer"],
    ),
    "portuguese_ocr": HuggingFaceDatasetSpec(
        hf_id="mazafard/portuguese-ocr-dataset",
        revision="b94db451b02f53105c0ad540705a865b13d06109",
        dataset_label="Portuguese OCR",
        language="pt",
        tags=["pt_br", "synthetic_text", "unicode"],
        text_fields=["text", "transcription", "ground_truth", "gt_text", "label"],
    ),
}


def _relative(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def load_manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    """Load JSONL benchmark rows from a manifest."""
    rows: list[dict[str, Any]] = []
    if not manifest_path.exists():
        return rows
    with manifest_path.open(encoding="utf-8") as manifest_file:
        for line in manifest_file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_manifest(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """Validate and atomically write a content-addressed JSONL benchmark manifest."""
    normalized_rows: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"Benchmark row {row_number} has no stable sample id")
        if not SAFE_SAMPLE_ID_PATTERN.fullmatch(sample_id):
            raise ValueError(f"Benchmark row {row_number} has an unsafe sample id: {sample_id!r}")
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate benchmark sample id: {sample_id}")
        sample_ids.add(sample_id)

        normalized = dict(row)
        normalized["manifest_version"] = MANIFEST_VERSION
        file_sha256: dict[str, str] = {}
        file_bytes: dict[str, int] = {}
        for field in ("image", "pdf", "gt_text"):
            raw_path = normalized.get(field)
            if raw_path is None:
                if field in {"pdf", "gt_text"}:
                    raise ValueError(f"Benchmark sample {sample_id} has no {field} file")
                continue
            materialized_path = _resolve_manifest_material(out_dir, str(raw_path), sample_id)
            file_sha256[field] = _file_sha256(materialized_path)
            file_bytes[field] = materialized_path.stat().st_size
        normalized["file_sha256"] = file_sha256
        normalized["file_bytes"] = file_bytes
        normalized_rows.append(normalized)

    manifest_path = out_dir / "manifest.jsonl"
    payload = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) for row in normalized_rows
    )
    if payload:
        payload += "\n"
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=".manifest.",
        suffix=".jsonl",
        dir=out_dir,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as manifest_file:
            manifest_file.write(payload)
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
        os.replace(temp_path, manifest_path)
        _fsync_directory(out_dir)
    except Exception:
        try:
            os.close(temp_fd)
        except OSError:
            pass
        temp_path.unlink(missing_ok=True)
        raise
    return manifest_path


def _resolve_manifest_material(out_dir: Path, raw_path: str, sample_id: str) -> Path:
    relative_path = Path(raw_path)
    if relative_path.is_absolute():
        raise ValueError(f"Benchmark sample {sample_id} uses an absolute material path")
    root = out_dir.resolve()
    try:
        resolved = (out_dir / relative_path).resolve(strict=True)
    except OSError as error:
        raise ValueError(
            f"Benchmark sample {sample_id} material does not exist: {raw_path}"
        ) from error
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"Benchmark sample {sample_id} has an unsafe material path: {raw_path}")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def write_readme(out_dir: Path, rows: list[dict[str, Any]], requested: set[str]) -> None:
    """Write a short generated dataset summary."""
    dataset_counts: dict[str, int] = {}
    for row in rows:
        dataset = str(row.get("dataset") or "unknown")
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

    lines = [
        "# BigOCRPDF benchmark datasets",
        "",
        "Generated by `benchmarks/prepare_benchmark_datasets.py`.",
        "",
        f"Requested datasets: {', '.join(sorted(requested))}",
        "",
        "| dataset | samples |",
        "|---|---:|",
    ]
    for dataset, count in sorted(dataset_counts.items()):
        lines.append(f"| {dataset} | {count} |")
    lines.append("")
    _write_text_atomically(
        out_dir / "README.generated.md",
        "\n".join(lines),
    )


def _write_text_atomically(path: Path, payload: str) -> None:
    """Durably replace one generated text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            os.close(temp_fd)
        except OSError:
            pass
        temp_path.unlink(missing_ok=True)
        raise


def save_deterministic_image_pdf(
    image: Image.Image,
    pdf_path: Path,
    *,
    resolution: float,
) -> None:
    """Encode a Pillow image PDF with fixed metadata timestamps."""
    converted = image.convert("RGB")
    try:
        converted.save(
            pdf_path,
            "PDF",
            resolution=resolution,
            title="BigOCRPDF benchmark fixture",
            creationDate=DETERMINISTIC_PDF_TIMESTAMP,
            modDate=DETERMINISTIC_PDF_TIMESTAMP,
        )
    finally:
        converted.close()


def _publish_staged_rows(
    source_dir: Path,
    out_dir: Path,
    rows: list[dict[str, Any]],
    namespace: str,
) -> list[dict[str, Any]]:
    """Publish selected materials under one immutable content-addressed directory."""
    if not rows:
        return []
    if not SAFE_SAMPLE_ID_PATTERN.fullmatch(namespace):
        raise ValueError(f"Unsafe benchmark generation namespace: {namespace!r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".generation-",
        dir=out_dir,
    ) as temp_name:
        payload_dir = Path(temp_name) / "payload"
        payload_dir.mkdir()
        digest_rows: list[dict[str, Any]] = []
        for row in rows:
            sample_id = str(row.get("id") or "")
            if not SAFE_SAMPLE_ID_PATTERN.fullmatch(sample_id):
                raise ValueError(f"Benchmark row has an unsafe sample id: {sample_id!r}")
            digest_row = {
                key: value
                for key, value in row.items()
                if key not in {"file_sha256", "file_bytes", "manifest_version"}
            }
            material_hashes: dict[str, str] = {}
            for field in ("image", "pdf", "gt_text"):
                raw_path = row.get(field)
                if raw_path is None:
                    continue
                _copy_staged_material(
                    source_dir,
                    payload_dir,
                    str(raw_path),
                    sample_id,
                )
                material_hashes[field] = _file_sha256(
                    _resolve_manifest_material(
                        payload_dir,
                        str(raw_path),
                        sample_id,
                    )
                )
            digest_row["material_sha256"] = material_hashes
            digest_rows.append(digest_row)

        generation_digest = hashlib.sha256(
            json.dumps(
                digest_rows,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        generation_name = f"{namespace}-{generation_digest[:20]}"
        generations_fd = _open_generations_directory(out_dir)
        try:
            fcntl.flock(generations_fd, fcntl.LOCK_EX)
            try:
                published_fd = os.open(
                    generation_name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=generations_fd,
                )
            except FileNotFoundError:
                _fsync_staged_tree(payload_dir)
                os.rename(
                    payload_dir,
                    generation_name,
                    dst_dir_fd=generations_fd,
                )
                os.fsync(generations_fd)
            except OSError as error:
                raise ValueError(
                    f"Benchmark generation path is unsafe: generations/{generation_name}"
                ) from error
            else:
                try:
                    published_dir = Path(f"/proc/self/fd/{published_fd}")
                    _verify_published_generation(payload_dir, published_dir, rows)
                finally:
                    os.close(published_fd)
        finally:
            os.close(generations_fd)

    prefix = Path("generations") / generation_name
    published_rows: list[dict[str, Any]] = []
    for row in rows:
        published = dict(row)
        for field in ("image", "pdf", "gt_text"):
            if raw_path := row.get(field):
                published[field] = (prefix / str(raw_path)).as_posix()
        published_rows.append(published)
    return published_rows


def _open_generations_directory(out_dir: Path) -> int:
    """Open the publication directory by fd without following its final component."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.is_symlink() or not out_dir.is_dir():
        raise ValueError(f"Benchmark output directory is unsafe: {out_dir}")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    out_fd = os.open(out_dir, directory_flags)
    try:
        try:
            os.mkdir("generations", mode=0o700, dir_fd=out_fd)
            os.fsync(out_fd)
        except FileExistsError:
            pass
        try:
            return os.open("generations", directory_flags, dir_fd=out_fd)
        except OSError as error:
            raise ValueError(
                f"Benchmark generations directory is unsafe: {out_dir / 'generations'}"
            ) from error
    finally:
        os.close(out_fd)


def _verify_published_generation(
    staged_dir: Path,
    published_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    for row in rows:
        sample_id = str(row["id"])
        for field in ("image", "pdf", "gt_text"):
            raw_path = row.get(field)
            if raw_path is None:
                continue
            staged_path = _resolve_manifest_material(staged_dir, str(raw_path), sample_id)
            published_path = _resolve_manifest_material(
                published_dir,
                str(raw_path),
                sample_id,
            )
            if _file_sha256(staged_path) != _file_sha256(published_path):
                raise ValueError(f"Published benchmark generation is corrupted: {published_dir}")


def _fsync_staged_tree(root: Path) -> None:
    directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        _fsync_directory(directory)


def prepare_synthetic_dataset(out_dir: Path, max_samples: int) -> list[dict[str, Any]]:
    """Generate synthetic fixtures privately without replacing the published manifest."""
    generator = Path(__file__).with_name("make_synthetic_ocr_fixtures.py")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".bigocrpdf-synthetic-",
        dir=out_dir.parent,
    ) as stage_name:
        stage_dir = Path(stage_name)
        subprocess.run(
            [
                sys.executable,
                str(generator),
                "--out",
                str(stage_dir),
                "--private-stage",
            ],
            check=True,
            timeout=120,
        )
        rows = [
            row
            for row in load_manifest_rows(stage_dir / "manifest.jsonl")
            if row.get("dataset") == "synthetic"
        ][:max_samples]
        return _publish_staged_rows(
            stage_dir,
            out_dir,
            rows,
            "synthetic",
        )


def _copy_staged_material(
    stage_dir: Path,
    out_dir: Path,
    raw_path: str,
    sample_id: str,
) -> None:
    source = _resolve_manifest_material(stage_dir, raw_path, sample_id)
    relative_path = Path(raw_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Benchmark sample {sample_id} has an unsafe material path: {raw_path}")
    destination = out_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_root = out_dir.resolve()
    destination_parent = destination.parent.resolve()
    if not destination_parent.is_relative_to(output_root):
        raise ValueError(f"Benchmark sample {sample_id} has an unsafe material path: {raw_path}")

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        destination_fd = os.open(destination, flags, 0o600)
        with source.open("rb") as input_file, os.fdopen(destination_fd, "wb") as output_file:
            shutil.copyfileobj(input_file, output_file, COPY_CHUNK_BYTES)
            output_file.flush()
            os.fsync(output_file.fileno())
        _fsync_directory(destination_parent)
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def ensure_funsd_raw_dataset(raw_root: Path, *, download: bool) -> Path:
    """Ensure the FUNSD dataset exists under the raw data root."""
    raw_dir = raw_root / "funsd"
    if raw_dir.is_symlink():
        raise ValueError(f"FUNSD dataset root cannot be a symbolic link: {raw_dir}")
    zip_path = raw_dir / "dataset.zip"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if raw_dir.is_symlink():
        raise ValueError(f"FUNSD dataset root cannot be a symbolic link: {raw_dir}")

    if not any(_iter_funsd_split_dirs(raw_dir)):
        if not zip_path.exists():
            if not download:
                raise FileNotFoundError(
                    f"FUNSD is not available under {raw_dir}; rerun without --no-download"
                )
            if not FUNSD_DATASET_URL.startswith("https://"):
                raise ValueError(f"Unsupported FUNSD download URL: {FUNSD_DATASET_URL}")
            _download_verified_file(
                FUNSD_DATASET_URL,
                zip_path,
                expected_sha256=FUNSD_DATASET_SHA256,
                max_bytes=MAX_FUNSD_DOWNLOAD_BYTES,
            )
        _verify_file_sha256(zip_path, FUNSD_DATASET_SHA256)
        with zipfile.ZipFile(zip_path) as archive:
            _extract_zip_safely(archive, raw_dir)
        _write_funsd_verification_marker(raw_dir)

    if not any(_iter_funsd_split_dirs(raw_dir)):
        raise FileNotFoundError(f"FUNSD images/annotations were not found under {raw_dir}")
    return raw_dir


def _funsd_tree_sha256(raw_dir: Path) -> str:
    digest = hashlib.sha256()
    material_paths: list[Path] = []
    for split_dir in _iter_funsd_split_dirs(raw_dir):
        for child_dir in (split_dir / "images", split_dir / "annotations"):
            for path in child_dir.iterdir():
                if path.name.startswith("."):
                    continue
                _validate_funsd_path(raw_dir, path, expected_directory=False)
                if path.is_file():
                    material_paths.append(path)
    for path in sorted(material_paths):
        relative_path = path.relative_to(raw_dir).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(4, "big"))
        digest.update(relative_path)
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def _funsd_archive_tree_sha256(zip_path: Path) -> str:
    digest = hashlib.sha256()
    with zipfile.ZipFile(zip_path) as archive:
        members = sorted(
            (
                member
                for member in archive.infolist()
                if not member.is_dir()
                and not Path(member.filename).name.startswith(".")
                and any(
                    part.endswith("_data")
                    and index + 1 < len(Path(member.filename).parts)
                    and Path(member.filename).parts[index + 1] in {"images", "annotations"}
                    for index, part in enumerate(Path(member.filename).parts)
                )
            ),
            key=lambda member: member.filename,
        )
        for member in members:
            relative_path = member.filename.encode("utf-8")
            member_digest = hashlib.sha256()
            with archive.open(member) as source:
                while chunk := source.read(COPY_CHUNK_BYTES):
                    member_digest.update(chunk)
            digest.update(len(relative_path).to_bytes(4, "big"))
            digest.update(relative_path)
            digest.update(member_digest.digest())
    return digest.hexdigest()


def _write_funsd_verification_marker(raw_dir: Path) -> None:
    marker = {
        "schema_version": 1,
        "archive_url": FUNSD_DATASET_URL,
        "archive_sha256": FUNSD_DATASET_SHA256,
        "tree_sha256": _funsd_tree_sha256(raw_dir),
    }
    _write_text_atomically(
        raw_dir / ".bigocrpdf-funsd-source.json",
        json.dumps(marker, sort_keys=True) + "\n",
    )


def _funsd_source_metadata(raw_dir: Path) -> dict[str, str]:
    tree_sha256 = _funsd_tree_sha256(raw_dir)
    zip_path = raw_dir / "dataset.zip"
    try:
        _verify_file_sha256(zip_path, FUNSD_DATASET_SHA256)
        archive_tree_sha256 = _funsd_archive_tree_sha256(zip_path)
    except (OSError, ValueError, zipfile.BadZipFile):
        archive_tree_sha256 = None
    if archive_tree_sha256 == tree_sha256:
        return {
            "kind": "funsd",
            "verification": "verified_official_archive",
            "url": FUNSD_DATASET_URL,
            "sha256": FUNSD_DATASET_SHA256,
            "tree_sha256": tree_sha256,
        }
    return {
        "kind": "funsd_local",
        "verification": "unverified_local_tree",
        "tree_sha256": tree_sha256,
    }


def _snapshot_funsd_source(raw_dir: Path, snapshot_dir: Path) -> Path:
    """Copy one concrete FUNSD source tree without following symbolic links."""
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    file_flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        root_fd = os.open(raw_dir, directory_flags)
    except OSError as error:
        raise ValueError(f"FUNSD dataset root is not safe: {raw_dir}") from error
    snapshot_dir.mkdir(mode=0o700)
    copied_files = 0
    copied_bytes = 0

    def copy_directory(source_fd: int, destination: Path) -> None:
        nonlocal copied_files, copied_bytes
        for name in sorted(os.listdir(source_fd)):
            if name.startswith("."):
                continue
            try:
                file_stat = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
            except OSError as error:
                raise ValueError(f"FUNSD source changed while snapshotting: {name}") from error
            target = destination / name
            if stat.S_ISDIR(file_stat.st_mode):
                try:
                    child_fd = os.open(name, directory_flags, dir_fd=source_fd)
                except OSError as error:
                    raise ValueError(
                        f"FUNSD directory is not safe while snapshotting: {name}"
                    ) from error
                target.mkdir(mode=0o700)
                try:
                    copy_directory(child_fd, target)
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError(f"FUNSD source contains a non-regular material: {name}")

            copied_files += 1
            if copied_files > MAX_ZIP_MEMBERS:
                raise ValueError("FUNSD source contains too many files")
            try:
                input_fd = os.open(name, file_flags, dir_fd=source_fd)
            except OSError as error:
                raise ValueError(
                    f"FUNSD material is not safe while snapshotting: {name}"
                ) from error
            output_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            output_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            output_fd = os.open(target, output_flags, 0o600)
            file_bytes = 0
            try:
                with os.fdopen(output_fd, "wb") as output:
                    while chunk := os.read(input_fd, COPY_CHUNK_BYTES):
                        file_bytes += len(chunk)
                        copied_bytes += len(chunk)
                        if file_bytes > MAX_ZIP_MEMBER_BYTES:
                            raise ValueError(f"FUNSD material is too large: {name}")
                        if copied_bytes > MAX_ZIP_TOTAL_BYTES:
                            raise ValueError("FUNSD source exceeds the snapshot size limit")
                        output.write(chunk)
            except Exception:
                target.unlink(missing_ok=True)
                raise
            finally:
                os.close(input_fd)

    try:
        copy_directory(root_fd, snapshot_dir)
    finally:
        os.close(root_fd)
    return snapshot_dir


def prepare_funsd_dataset(
    out_dir: Path,
    raw_root: Path,
    max_samples: int,
    *,
    download: bool,
) -> list[dict[str, Any]]:
    """Prepare FUNSD images, one-page PDFs, ground truth, and manifest rows."""
    raw_dir = ensure_funsd_raw_dataset(raw_root, download=download)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".bigocrpdf-funsd-",
        dir=out_dir.parent,
    ) as stage_name:
        stage_dir = Path(stage_name)
        snapshot_dir = _snapshot_funsd_source(raw_dir, stage_dir / "_source")
        source_metadata = _funsd_source_metadata(snapshot_dir)
        images_dir = stage_dir / "images"
        pdfs_dir = stage_dir / "pdfs"
        ground_truth_dir = stage_dir / "ground_truth"
        for directory in [images_dir, pdfs_dir, ground_truth_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        for split_dir in _iter_funsd_split_dirs(snapshot_dir):
            split_name = split_dir.name.replace("_data", "")
            annotations_dir = split_dir / "annotations"
            images_source_dir = split_dir / "images"
            for annotation_path in sorted(annotations_dir.glob("*.json")):
                if len(rows) >= max_samples:
                    break
                if annotation_path.name.startswith("."):
                    continue
                _validate_funsd_path(
                    snapshot_dir,
                    annotation_path,
                    expected_directory=False,
                )
                image_source = _find_matching_image(images_source_dir, annotation_path.stem)
                if image_source is None:
                    continue
                _validate_funsd_path(
                    snapshot_dir,
                    image_source,
                    expected_directory=False,
                )
                sample_id = f"funsd_{split_name}_{annotation_path.stem}"
                image_path = images_dir / f"{sample_id}{image_source.suffix.lower()}"
                pdf_path = pdfs_dir / f"{sample_id}.pdf"
                gt_path = ground_truth_dir / f"{sample_id}.txt"

                shutil.copyfile(image_source, image_path)
                if not _write_image_pdf(image_path, pdf_path):
                    image_path.unlink(missing_ok=True)
                    continue
                words = extract_funsd_words(json.loads(annotation_path.read_text(encoding="utf-8")))
                gt_path.write_text(" ".join(words), encoding="utf-8")
                rows.append(
                    {
                        "id": sample_id,
                        "dataset": "FUNSD",
                        "image": _relative(image_path, stage_dir),
                        "pdf": _relative(pdf_path, stage_dir),
                        "gt_text": _relative(gt_path, stage_dir),
                        "language": "en",
                        "tags": ["form", "noisy_scan", "english"],
                        "source": {
                            **source_metadata,
                            "split": split_name,
                        },
                    }
                )
            if len(rows) >= max_samples:
                break
        return _publish_staged_rows(
            stage_dir,
            out_dir,
            rows,
            "funsd",
        )


def prepare_huggingface_dataset(
    out_dir: Path,
    dataset_key: str,
    max_samples: int,
    *,
    download: bool,
) -> list[dict[str, Any]]:
    """Prepare an optional Hugging Face image/text dataset."""
    spec = HF_DATASET_SPECS[dataset_key]
    dataset = _load_huggingface_dataset(spec.hf_id, spec.revision, download=download)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".bigocrpdf-{dataset_key}-",
        dir=out_dir.parent,
    ) as stage_name:
        stage_dir = Path(stage_name)
        images_dir = stage_dir / "images"
        pdfs_dir = stage_dir / "pdfs"
        ground_truth_dir = stage_dir / "ground_truth"
        for directory in [images_dir, pdfs_dir, ground_truth_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        for split_name, split_rows in _iter_huggingface_splits(dataset):
            for row_index, row in enumerate(split_rows):
                if len(rows) >= max_samples:
                    break
                text = _extract_huggingface_text(row, spec.text_fields)
                if not text:
                    continue
                sample_id = f"{dataset_key}_{split_name}_{row_index:04d}"
                image_path = images_dir / f"{sample_id}.png"
                pdf_path = pdfs_dir / f"{sample_id}.pdf"
                gt_path = ground_truth_dir / f"{sample_id}.txt"
                if not _write_huggingface_image(row.get("image"), image_path):
                    continue
                if not _write_image_pdf(image_path, pdf_path):
                    image_path.unlink(missing_ok=True)
                    continue
                gt_path.write_text(text, encoding="utf-8")
                rows.append(
                    {
                        "id": sample_id,
                        "dataset": spec.dataset_label,
                        "image": _relative(image_path, stage_dir),
                        "pdf": _relative(pdf_path, stage_dir),
                        "gt_text": _relative(gt_path, stage_dir),
                        "language": spec.language,
                        "tags": spec.tags,
                        "source": {
                            "kind": "huggingface",
                            "dataset_id": spec.hf_id,
                            "revision": spec.revision,
                            "split": split_name,
                            "row_index": row_index,
                        },
                    }
                )
            if len(rows) >= max_samples:
                break
        return _publish_staged_rows(
            stage_dir,
            out_dir,
            rows,
            dataset_key,
        )


def extract_funsd_words(annotation: dict[str, Any]) -> list[str]:
    """Extract FUNSD ground-truth words in annotation order."""
    words: list[str] = []
    for form_entry in annotation.get("form") or []:
        for word in form_entry.get("words") or []:
            text = str(word.get("text") or "").strip()
            if text:
                words.append(text)
        if not form_entry.get("words"):
            text = str(form_entry.get("text") or "").strip()
            if text:
                words.extend(text.split())
    return words


def _iter_funsd_split_dirs(raw_dir: Path) -> list[Path]:
    split_dirs: list[Path] = []
    for path in raw_dir.rglob("*_data"):
        images_dir = path / "images"
        annotations_dir = path / "annotations"
        if not images_dir.is_dir() or not annotations_dir.is_dir():
            continue
        _validate_funsd_path(raw_dir, path, expected_directory=True)
        _validate_funsd_path(raw_dir, images_dir, expected_directory=True)
        _validate_funsd_path(raw_dir, annotations_dir, expected_directory=True)
        split_dirs.append(path)
    return sorted(split_dirs)


def _validate_funsd_path(
    raw_dir: Path,
    path: Path,
    *,
    expected_directory: bool,
) -> None:
    """Reject links and escapes anywhere below the designated FUNSD root."""
    try:
        relative_path = path.relative_to(raw_dir)
    except ValueError as error:
        raise ValueError(f"FUNSD path is outside its raw tree: {path}") from error
    current = raw_dir
    for component in relative_path.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f"FUNSD path cannot contain a symbolic link: {current}")
    try:
        resolved_root = raw_dir.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"FUNSD path does not exist: {path}") from error
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"FUNSD path is outside its raw tree: {path}")
    if expected_directory:
        if not path.is_dir():
            raise ValueError(f"FUNSD directory is invalid: {path}")
    elif not path.is_file():
        raise ValueError(f"FUNSD material is invalid: {path}")


def _find_matching_image(images_dir: Path, stem: str) -> Path | None:
    if stem.startswith("."):
        return None
    for suffix in [".png", ".jpg", ".jpeg"]:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.is_symlink():
            raise ValueError(f"FUNSD material cannot be a symbolic link: {candidate}")
        if candidate.is_file() and not candidate.name.startswith("."):
            return candidate
    return None


def _extract_zip_safely(  # noqa: C901 - security checks remain adjacent to extraction
    archive: zipfile.ZipFile, destination: Path
) -> None:
    members = archive.infolist()
    if len(members) > MAX_ZIP_MEMBERS:
        raise ValueError(f"Archive contains too many members: {len(members)} > {MAX_ZIP_MEMBERS}")

    destination_root = destination.resolve()
    total_uncompressed_bytes = 0
    member_paths: set[Path] = set()
    for member in members:
        member_path = destination / member.filename
        resolved_member_path = member_path.resolve()
        if not resolved_member_path.is_relative_to(destination_root):
            raise ValueError(f"Unsafe archive member path: {member.filename}")
        if resolved_member_path in member_paths:
            raise ValueError(f"Duplicate archive member path: {member.filename}")
        member_paths.add(resolved_member_path)

        unix_mode = member.external_attr >> 16
        file_type = stat.S_IFMT(unix_mode)
        if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise ValueError(f"Unsupported archive member type: {member.filename}")
        if member.flag_bits & 0x1:
            raise ValueError(f"Encrypted archive member is not supported: {member.filename}")
        if member.file_size > MAX_ZIP_MEMBER_BYTES:
            raise ValueError(f"Archive member is too large: {member.filename}")
        if member.file_size > max(member.compress_size, 1) * MAX_ZIP_COMPRESSION_RATIO:
            raise ValueError(f"Archive member compression ratio is too high: {member.filename}")

        total_uncompressed_bytes += member.file_size
        if total_uncompressed_bytes > MAX_ZIP_TOTAL_BYTES:
            raise ValueError("Archive uncompressed size exceeds the safety limit")

    destination.mkdir(parents=True, exist_ok=True)
    extracted_bytes = 0
    for member in members:
        target = destination / member.filename
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            target_fd = os.open(target, flags, 0o600)
        except FileExistsError as exc:
            raise ValueError(f"Archive member would overwrite a file: {member.filename}") from exc

        member_bytes = 0
        try:
            with archive.open(member) as source, os.fdopen(target_fd, "wb") as output:
                while chunk := source.read(COPY_CHUNK_BYTES):
                    member_bytes += len(chunk)
                    extracted_bytes += len(chunk)
                    if member_bytes > member.file_size or member_bytes > MAX_ZIP_MEMBER_BYTES:
                        raise ValueError(
                            f"Archive member exceeded its declared size: {member.filename}"
                        )
                    if extracted_bytes > MAX_ZIP_TOTAL_BYTES:
                        raise ValueError("Archive extraction exceeded the safety limit")
                    output.write(chunk)
            if member_bytes != member.file_size:
                raise ValueError(f"Archive member size mismatch: {member.filename}")
        except Exception:
            target.unlink(missing_ok=True)
            raise


def _download_verified_file(
    url: str,
    destination: Path,
    *,
    expected_sha256: str,
    max_bytes: int,
) -> None:
    """Download one HTTPS artifact atomically after size and digest verification."""
    if not url.startswith("https://"):
        raise ValueError(f"Unsupported download URL: {url}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".download",
        dir=destination.parent,
    )
    temp_path = Path(temp_name)
    digest = hashlib.sha256()
    downloaded_bytes = 0
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "BigOCRPDF benchmark tool"})
        # The only caller passes the pinned HTTPS FUNSD URL; redirects remain HTTPS
        # and the complete body is verified against FUNSD_DATASET_SHA256.
        with urllib.request.urlopen(  # nosec B310  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
            request,
            timeout=30,
        ) as response:
            final_url = response.geturl()
            if not final_url.startswith("https://"):
                raise ValueError(f"Download redirected to an unsupported URL: {final_url}")
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > max_bytes:
                raise ValueError("Download exceeds the configured size limit")

            with os.fdopen(temp_fd, "wb") as output:
                while chunk := response.read(COPY_CHUNK_BYTES):
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > max_bytes:
                        raise ValueError("Download exceeded the configured size limit")
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        if digest.hexdigest() != expected_sha256:
            raise ValueError("Downloaded artifact failed SHA-256 verification")
        os.replace(temp_path, destination)
        _fsync_directory(destination.parent)
    except Exception:
        try:
            os.close(temp_fd)
        except OSError:
            pass
        temp_path.unlink(missing_ok=True)
        raise


def _verify_file_sha256(path: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"Dataset archive failed SHA-256 verification: {path}")


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    directory_fd = os.open(directory, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_image_pdf(image_path: Path, pdf_path: Path) -> bool:
    try:
        with Image.open(image_path) as image:
            save_deterministic_image_pdf(
                image,
                pdf_path,
                resolution=300.0,
            )
    except OSError:
        return False
    return True


def _load_huggingface_dataset(hf_id: str, revision: str, *, download: bool) -> Any:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Optional Hugging Face dataset support requires the `datasets` package. "
            "Install it in a development environment or use --datasets synthetic,funsd."
        ) from exc

    if download:
        return load_dataset(hf_id, revision=revision)

    previous_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        return load_dataset(hf_id, revision=revision)
    finally:
        if previous_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous_offline


def _iter_huggingface_splits(dataset: Any) -> list[tuple[str, Any]]:
    if hasattr(dataset, "items"):
        return [(str(split_name), split_rows) for split_name, split_rows in dataset.items()]
    return [("default", dataset)]


def _extract_huggingface_text(row: dict[str, Any], text_fields: list[str]) -> str:
    for field in text_fields:
        text = _flatten_text(row.get(field))
        if text:
            return text
    return ""


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list | tuple):
        return " ".join(text for item in value if (text := _flatten_text(item)))
    if isinstance(value, dict):
        return " ".join(text for item in value.values() if (text := _flatten_text(item)))
    return ""


def _write_huggingface_image(image_value: Any, image_path: Path) -> bool:
    if image_value is None:
        return False
    if hasattr(image_value, "save"):
        try:
            image_value.convert("RGB").save(image_path)
            return True
        except OSError:
            return False
    image_source = _image_path_from_huggingface_value(image_value)
    if image_source is None or not image_source.exists():
        return False
    shutil.copyfile(image_source, image_path)
    return True


def _image_path_from_huggingface_value(image_value: Any) -> Path | None:
    if isinstance(image_value, str):
        return Path(image_value)
    if isinstance(image_value, dict):
        raw_path = image_value.get("path")
        return Path(str(raw_path)) if raw_path else None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--datasets",
        default="portuguese_synthetic",
        help=(
            "Comma-separated datasets: synthetic, portuguese_synthetic, funsd, "
            "sroie, dharmaocr, portuguese_ocr."
        ),
    )
    parser.add_argument("--max-samples-per-dataset", type=int, default=50)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use only datasets already present under --raw-dir.",
    )
    args = parser.parse_args()

    requested = {name.strip() for name in args.datasets.split(",") if name.strip()}
    unsupported = requested - SUPPORTED_DATASETS
    if unsupported:
        sys.stderr.write(f"Unsupported datasets: {', '.join(sorted(unsupported))}\n")
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if requested & {"synthetic", "portuguese_synthetic"}:
        rows.extend(prepare_synthetic_dataset(args.out, args.max_samples_per_dataset))
    if "funsd" in requested:
        rows.extend(
            prepare_funsd_dataset(
                args.out,
                args.raw_dir,
                args.max_samples_per_dataset,
                download=not args.no_download,
            )
        )
    for dataset_key in sorted(requested & set(HF_DATASET_SPECS)):
        rows.extend(
            prepare_huggingface_dataset(
                args.out,
                dataset_key,
                args.max_samples_per_dataset,
                download=not args.no_download,
            )
        )

    manifest_path = write_manifest(args.out, rows)
    write_readme(args.out, rows, requested)
    sys.stdout.write(f"{manifest_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
