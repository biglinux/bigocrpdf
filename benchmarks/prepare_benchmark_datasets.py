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
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

COPY_CHUNK_BYTES = 1024 * 1024
MANIFEST_VERSION = 1
SAFE_SAMPLE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
DETERMINISTIC_PDF_TIMESTAMP = time.gmtime(946684800)
SUPPORTED_DATASETS = {
    "dharmaocr",
    "portuguese_synthetic",
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
    "dharmaocr": HuggingFaceDatasetSpec(
        hf_id="Dharma-AI/DharmaOCR-Benchmark",
        revision="e8f4bb516839c1a0a32ee0234b3cbcd5aa5c10d3",
        dataset_label="DharmaOCR",
        language="pt",
        tags=["pt_br", "legal", "administrative"],
        # The transcription this benchmark publishes is the assistant turn.
        # The plain variant comes first because the other wraps the same text
        # in JSON, and a comparison against braces measures nothing.
        text_fields=["assistant_without_json", "text", "transcription", "ground_truth"],
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

    # Degraded variants share their clean sample's ground truth on purpose --
    # the degradation moved pixels, not text -- so the same file is published
    # once per row that cites it. Re-publishing identical bytes is that case
    # and is fine; different bytes at the same path is a real collision and
    # must still fail, which is why this compares rather than just overwriting.
    if destination.exists() and _file_sha256(destination) == _file_sha256(source):
        return

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    # Opened before the try so that only a file this call created can be
    # cleaned up by it. Inside, a refused O_EXCL would send the cleanup after
    # whatever was already sitting at that path.
    destination_fd = os.open(destination, flags, 0o600)
    try:
        with source.open("rb") as input_file, os.fdopen(destination_fd, "wb") as output_file:
            shutil.copyfileobj(input_file, output_file, COPY_CHUNK_BYTES)
            output_file.flush()
            os.fsync(output_file.fileno())
        _fsync_directory(destination_parent)
    except Exception:
        destination.unlink(missing_ok=True)
        raise


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
            "Install it in a development environment or use --datasets synthetic."
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
    parser.add_argument(
        "--datasets",
        default="portuguese_synthetic",
        help="Comma-separated datasets: synthetic, portuguese_synthetic, dharmaocr.",
    )
    parser.add_argument("--max-samples-per-dataset", type=int, default=50)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use only datasets already cached locally.",
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
    for dataset_key in sorted(requested & set(HF_DATASET_SPECS)):
        rows.extend(
            prepare_huggingface_dataset(
                args.out,
                dataset_key,
                args.max_samples_per_dataset,
                download=not args.no_download,
            )
        )

    if not rows:
        # A dataset can publish images and no transcription -- one of the
        # Portuguese sets on the Hub does exactly that. Every row is then
        # dropped for having no ground truth, and printing a manifest path
        # afterwards reports success for a corpus that can measure nothing.
        sys.stderr.write(
            f"No usable samples from: {', '.join(sorted(requested))}. "
            "Every row lacked ground-truth text; nothing was written.\n"
        )
        return 1

    manifest_path = write_manifest(args.out, rows)
    write_readme(args.out, rows, requested)
    sys.stdout.write(f"{manifest_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
