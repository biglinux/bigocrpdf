import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from benchmarks import prepare_benchmark_datasets as dataset_tools
from benchmarks.prepare_benchmark_datasets import (
    MAX_ZIP_COMPRESSION_RATIO,
    _download_verified_file,
    _extract_huggingface_text,
    _extract_zip_safely,
    _publish_staged_rows,
    extract_funsd_words,
    prepare_funsd_dataset,
    prepare_huggingface_dataset,
    prepare_synthetic_dataset,
    save_deterministic_image_pdf,
    write_manifest,
    write_readme,
)


class _DownloadResponse:
    def __init__(self, payload: bytes, *, content_length: int | None = None) -> None:
        self._payload = payload
        self._offset = 0
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def geturl(self) -> str:
        return "https://guillaumejaume.github.io/FUNSD/dataset.zip"

    def read(self, size: int) -> bytes:
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def test_extract_funsd_words_prefers_word_annotations() -> None:
    words = extract_funsd_words(
        {
            "form": [
                {"words": [{"text": "Invoice"}, {"text": "123"}]},
                {"text": "Fallback text"},
            ]
        }
    )

    assert words == ["Invoice", "123", "Fallback", "text"]


def test_prepare_synthetic_dataset_does_not_replace_published_manifest(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "benchmarks"
    out_dir.mkdir()
    manifest_path = out_dir / "manifest.jsonl"
    manifest_path.write_text("published manifest\n", encoding="utf-8")

    rows = prepare_synthetic_dataset(out_dir, 1)

    assert len(rows) == 1
    assert manifest_path.read_text(encoding="utf-8") == "published manifest\n"
    assert (out_dir / rows[0]["pdf"]).is_file()
    assert Path(rows[0]["pdf"]).parts[0] == "generations"


def test_prepare_funsd_dataset_from_existing_raw_tree(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    split_dir = raw_root / "funsd" / "dataset" / "training_data"
    images_dir = split_dir / "images"
    annotations_dir = split_dir / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)
    Image.new("RGB", (100, 60), "white").save(images_dir / "sample.png")
    (annotations_dir / "sample.json").write_text(
        json.dumps({"form": [{"words": [{"text": "Name"}, {"text": "Alice"}]}]}),
        encoding="utf-8",
    )
    out_dir = tmp_path / "benchmarks"

    rows = prepare_funsd_dataset(out_dir, raw_root, 5, download=False)

    assert len(rows) == 1
    assert {
        key: value for key, value in rows[0].items() if key not in {"image", "pdf", "gt_text"}
    } == {
        "id": "funsd_training_sample",
        "dataset": "FUNSD",
        "language": "en",
        "tags": ["form", "noisy_scan", "english"],
        "source": {
            "kind": "funsd_local",
            "verification": "unverified_local_tree",
            "tree_sha256": rows[0]["source"]["tree_sha256"],
            "split": "training",
        },
    }
    assert Path(rows[0]["pdf"]).parts[:1] == ("generations",)
    assert (out_dir / rows[0]["pdf"]).exists()
    assert (out_dir / rows[0]["gt_text"]).read_text(encoding="utf-8") == "Name Alice"
    assert "url" not in rows[0]["source"]
    assert "sha256" not in rows[0]["source"]


def test_prepare_funsd_uses_one_private_source_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_root = tmp_path / "raw"
    split_dir = raw_root / "funsd" / "dataset" / "training_data"
    images_dir = split_dir / "images"
    annotations_dir = split_dir / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)
    source_image = images_dir / "sample.png"
    source_annotation = annotations_dir / "sample.json"
    Image.new("RGB", (40, 20), "white").save(source_image)
    source_annotation.write_text(
        json.dumps({"form": [{"words": [{"text": "Original"}]}]}),
        encoding="utf-8",
    )
    captured_metadata = {}
    real_metadata = dataset_tools._funsd_source_metadata

    def mutate_raw_after_snapshot(snapshot_dir: Path) -> dict[str, str]:
        metadata = real_metadata(snapshot_dir)
        captured_metadata.update(metadata)
        Image.new("RGB", (40, 20), "black").save(source_image)
        source_annotation.write_text(
            json.dumps({"form": [{"words": [{"text": "Changed"}]}]}),
            encoding="utf-8",
        )
        return metadata

    monkeypatch.setattr(
        dataset_tools,
        "_funsd_source_metadata",
        mutate_raw_after_snapshot,
    )

    out_dir = tmp_path / "benchmarks"
    rows = prepare_funsd_dataset(out_dir, raw_root, 1, download=False)

    assert (out_dir / rows[0]["gt_text"]).read_text(encoding="utf-8") == "Original"
    with Image.open(out_dir / rows[0]["image"]) as published:
        assert published.getpixel((0, 0)) == (255, 255, 255)
    assert rows[0]["source"]["tree_sha256"] == captured_metadata["tree_sha256"]


def test_prepare_funsd_snapshot_rejects_non_regular_source(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    split_dir = raw_root / "funsd" / "dataset" / "training_data"
    images_dir = split_dir / "images"
    annotations_dir = split_dir / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)
    Image.new("RGB", (40, 20), "white").save(images_dir / "sample.png")
    (annotations_dir / "sample.json").write_text("{}", encoding="utf-8")
    os.mkfifo(raw_root / "funsd" / "unexpected.pipe")

    with pytest.raises(ValueError, match="non-regular"):
        prepare_funsd_dataset(
            tmp_path / "benchmarks",
            raw_root,
            1,
            download=False,
        )


def test_prepare_funsd_dataset_skips_hidden_resource_fork_files(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    split_dir = raw_root / "funsd" / "dataset" / "testing_data"
    images_dir = split_dir / "images"
    annotations_dir = split_dir / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)
    (images_dir / "._sample.png").write_bytes(b"not an image")
    (annotations_dir / "._sample.json").write_text("{}", encoding="utf-8")

    rows = prepare_funsd_dataset(tmp_path / "benchmarks", raw_root, 5, download=False)

    assert rows == []


def test_prepare_funsd_dataset_rejects_unsafe_zip_member(tmp_path: Path) -> None:
    zip_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../escape.txt", "bad")

    with zipfile.ZipFile(zip_path) as archive:
        try:
            _extract_zip_safely(archive, tmp_path / "raw")
        except ValueError as exc:
            assert "Unsafe archive member path" in str(exc)
        else:
            raise AssertionError("Unsafe FUNSD archive member was accepted")


def test_extract_zip_safely_rejects_compression_bomb(tmp_path: Path) -> None:
    zip_path = tmp_path / "dataset.zip"
    payload = b"0" * (MAX_ZIP_COMPRESSION_RATIO * 1024)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bomb.txt", payload)

    with zipfile.ZipFile(zip_path) as archive:
        try:
            _extract_zip_safely(archive, tmp_path / "raw")
        except ValueError as exc:
            assert "compression ratio" in str(exc)
        else:
            raise AssertionError("Compressed archive bomb was accepted")


def test_extract_zip_safely_rejects_symbolic_link(tmp_path: Path) -> None:
    zip_path = tmp_path / "dataset.zip"
    link = zipfile.ZipInfo("dataset/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(link, "../../outside")

    with zipfile.ZipFile(zip_path) as archive:
        try:
            _extract_zip_safely(archive, tmp_path / "raw")
        except ValueError as exc:
            assert "Unsupported archive member type" in str(exc)
        else:
            raise AssertionError("Symbolic link archive member was accepted")


def test_extract_zip_safely_writes_regular_files_without_overwrite(tmp_path: Path) -> None:
    zip_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("dataset/training_data/sample.txt", "safe")
    destination = tmp_path / "raw"

    with zipfile.ZipFile(zip_path) as archive:
        _extract_zip_safely(archive, destination)

    extracted = destination / "dataset/training_data/sample.txt"
    assert extracted.read_text(encoding="utf-8") == "safe"
    with zipfile.ZipFile(zip_path) as archive:
        try:
            _extract_zip_safely(archive, destination)
        except ValueError as exc:
            assert "overwrite" in str(exc)
        else:
            raise AssertionError("Archive extraction overwrote an existing file")


def test_verified_download_replaces_symlink_without_touching_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = b"verified dataset"
    protected = tmp_path / "protected.zip"
    protected.write_bytes(b"KEEP")
    destination = tmp_path / "dataset.zip"
    destination.symlink_to(protected)
    monkeypatch.setattr(
        "benchmarks.prepare_benchmark_datasets.urllib.request.urlopen",
        lambda *_args, **_kwargs: _DownloadResponse(payload, content_length=len(payload)),
    )

    _download_verified_file(
        "https://guillaumejaume.github.io/FUNSD/dataset.zip",
        destination,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        max_bytes=1024,
    )

    assert protected.read_bytes() == b"KEEP"
    assert not destination.is_symlink()
    assert destination.read_bytes() == payload


def test_verified_download_rejects_digest_mismatch_and_cleans_temp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / "dataset.zip"
    monkeypatch.setattr(
        "benchmarks.prepare_benchmark_datasets.urllib.request.urlopen",
        lambda *_args, **_kwargs: _DownloadResponse(b"tampered"),
    )

    try:
        _download_verified_file(
            "https://guillaumejaume.github.io/FUNSD/dataset.zip",
            destination,
            expected_sha256=hashlib.sha256(b"expected").hexdigest(),
            max_bytes=1024,
        )
    except ValueError as exc:
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("Download with the wrong digest was accepted")

    assert not destination.exists()
    assert list(tmp_path.glob(".dataset.zip.*.download")) == []


def test_verified_download_rejects_declared_oversize(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "dataset.zip"
    monkeypatch.setattr(
        "benchmarks.prepare_benchmark_datasets.urllib.request.urlopen",
        lambda *_args, **_kwargs: _DownloadResponse(b"small", content_length=2048),
    )

    try:
        _download_verified_file(
            "https://guillaumejaume.github.io/FUNSD/dataset.zip",
            destination,
            expected_sha256=hashlib.sha256(b"small").hexdigest(),
            max_bytes=1024,
        )
    except ValueError as exc:
        assert "size limit" in str(exc)
    else:
        raise AssertionError("Oversized download was accepted")

    assert not destination.exists()


def test_prepare_huggingface_dataset_from_mocked_dataset(tmp_path: Path, monkeypatch) -> None:
    import benchmarks.prepare_benchmark_datasets as benchmark_datasets

    def fake_load_dataset(
        _hf_id: str,
        _revision: str,
        *,
        download: bool,
    ) -> dict[str, list[dict[str, object]]]:
        assert not download
        return {
            "train": [
                {
                    "image": Image.new("RGB", (100, 60), "white"),
                    "text": "Olá mundo",
                }
            ]
        }

    monkeypatch.setattr(benchmark_datasets, "_load_huggingface_dataset", fake_load_dataset)
    out_dir = tmp_path / "benchmarks"

    rows = prepare_huggingface_dataset(out_dir, "portuguese_ocr", 5, download=False)

    assert len(rows) == 1
    assert {
        key: value for key, value in rows[0].items() if key not in {"image", "pdf", "gt_text"}
    } == {
        "id": "portuguese_ocr_train_0000",
        "dataset": "Portuguese OCR",
        "language": "pt",
        "tags": ["pt_br", "synthetic_text", "unicode"],
        "source": {
            "kind": "huggingface",
            "dataset_id": "mazafard/portuguese-ocr-dataset",
            "revision": "b94db451b02f53105c0ad540705a865b13d06109",
            "split": "train",
            "row_index": 0,
        },
    }
    assert Path(rows[0]["pdf"]).parts[:1] == ("generations",)
    assert (out_dir / rows[0]["image"]).exists()
    assert (out_dir / rows[0]["pdf"]).exists()
    assert (out_dir / rows[0]["gt_text"]).read_text(encoding="utf-8") == "Olá mundo"


def test_extract_huggingface_text_flattens_nested_values() -> None:
    text = _extract_huggingface_text(
        {
            "text": "",
            "ground_truth": {
                "lines": ["Invoice", {"value": "123"}],
                "ignored": None,
            },
        },
        ["text", "ground_truth"],
    )

    assert text == "Invoice 123"


def test_write_manifest_and_readme(tmp_path: Path) -> None:
    for relative_path, payload in [
        ("images/sample.png", b"image"),
        ("pdfs/sample.pdf", b"%PDF-1.7\n"),
        ("ground_truth/sample.txt", b"expected"),
    ]:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    rows = [
        {
            "id": "sample",
            "dataset": "FUNSD",
            "image": "images/sample.png",
            "pdf": "pdfs/sample.pdf",
            "gt_text": "ground_truth/sample.txt",
            "language": "en",
            "tags": ["form"],
        }
    ]

    manifest_path = write_manifest(tmp_path, rows)
    write_readme(tmp_path, rows, {"funsd"})

    manifest_row = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_row["id"] == "sample"
    assert manifest_row["manifest_version"] == 1
    assert manifest_row["file_sha256"]["pdf"] == hashlib.sha256(b"%PDF-1.7\n").hexdigest()
    assert manifest_row["file_bytes"]["gt_text"] == len(b"expected")
    readme = (tmp_path / "README.generated.md").read_text(encoding="utf-8")
    assert "| FUNSD | 1 |" in readme


def test_write_manifest_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    ground_truth = tmp_path / "sample.txt"
    pdf.write_bytes(b"%PDF")
    ground_truth.write_text("text", encoding="utf-8")
    row = {
        "id": "duplicate",
        "dataset": "synthetic",
        "pdf": pdf.name,
        "gt_text": ground_truth.name,
        "language": "en",
        "tags": [],
    }

    with pytest.raises(ValueError, match="Duplicate benchmark sample id"):
        write_manifest(tmp_path, [row, row])


def test_deterministic_pillow_pdf_has_stable_hash(tmp_path: Path) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    image = Image.new("RGB", (100, 60), "white")

    save_deterministic_image_pdf(image, first, resolution=300.0)
    save_deterministic_image_pdf(image, second, resolution=300.0)

    assert (
        hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    )


def test_new_dataset_generation_cannot_invalidate_published_manifest(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "benchmarks"
    old_pdf = out_dir / "generations" / "old" / "old.pdf"
    old_text = out_dir / "generations" / "old" / "old.txt"
    old_pdf.parent.mkdir(parents=True)
    old_pdf.write_bytes(b"%PDF-1.7\nold")
    old_text.write_text("old", encoding="utf-8")
    manifest_path = write_manifest(
        out_dir,
        [
            {
                "id": "old",
                "dataset": "synthetic",
                "pdf": "generations/old/old.pdf",
                "gt_text": "generations/old/old.txt",
                "language": "en",
                "tags": [],
            }
        ],
    )
    published_manifest = manifest_path.read_bytes()
    published_pdf = old_pdf.read_bytes()

    rows = prepare_synthetic_dataset(out_dir, 1)

    assert manifest_path.read_bytes() == published_manifest
    assert old_pdf.read_bytes() == published_pdf
    assert Path(rows[0]["pdf"]).parts[0] == "generations"


def test_generation_publication_rejects_symlinked_generations_directory(
    tmp_path: Path,
) -> None:
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    stage_dir.joinpath("sample.pdf").write_bytes(b"%PDF-1.7\n")
    stage_dir.joinpath("sample.txt").write_text("text", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    out_dir.joinpath("generations").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="generations"):
        _publish_staged_rows(
            stage_dir,
            out_dir,
            [
                {
                    "id": "sample",
                    "dataset": "synthetic",
                    "pdf": "sample.pdf",
                    "gt_text": "sample.txt",
                    "language": "en",
                    "tags": [],
                }
            ],
            "synthetic",
        )

    assert list(outside.iterdir()) == []


def test_prepare_funsd_rejects_symlinked_split_outside_raw_tree(tmp_path: Path) -> None:
    external_split = tmp_path / "external" / "training_data"
    images_dir = external_split / "images"
    annotations_dir = external_split / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir()
    Image.new("RGB", (100, 60), "white").save(images_dir / "sample.png")
    annotations_dir.joinpath("sample.json").write_text(
        json.dumps({"form": [{"words": [{"text": "outside"}]}]}),
        encoding="utf-8",
    )
    raw_dir = tmp_path / "raw" / "funsd" / "dataset"
    raw_dir.mkdir(parents=True)
    raw_dir.joinpath("training_data").symlink_to(
        external_split,
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="symbolic link|outside"):
        prepare_funsd_dataset(
            tmp_path / "out",
            tmp_path / "raw",
            1,
            download=False,
        )


def test_prepare_funsd_rejects_symlinked_dataset_root(tmp_path: Path) -> None:
    external = tmp_path / "external"
    images_dir = external / "training_data" / "images"
    annotations_dir = external / "training_data" / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir()
    Image.new("RGB", (100, 60), "white").save(images_dir / "sample.png")
    annotations_dir.joinpath("sample.json").write_text(
        json.dumps({"form": [{"words": [{"text": "outside"}]}]}),
        encoding="utf-8",
    )
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw_root.joinpath("funsd").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="root|safe|symbolic link"):
        prepare_funsd_dataset(
            tmp_path / "out",
            raw_root,
            1,
            download=False,
        )
