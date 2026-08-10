import hashlib
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from benchmarks import make_synthetic_ocr_fixtures as fixture_tools
from benchmarks import prepare_benchmark_datasets as dataset_tools
from benchmarks.prepare_benchmark_datasets import (
    _copy_staged_material,
    _extract_huggingface_text,
    _publish_staged_rows,
    prepare_huggingface_dataset,
    prepare_synthetic_dataset,
    save_deterministic_image_pdf,
    write_manifest,
    write_readme,
)


def _unrenderable_samples() -> list[str]:
    """Sample ids whose script no host font covers.

    The generator refuses to draw tofu, so on a host without CJK or Arabic
    fonts every synthetic run raises.  Asking it the same question up front
    turns that into an honest skip instead of a failure that looks like a code
    defect -- and keeps the check truthful, since it is the generator's own
    coverage test rather than a guess about which packages are installed.
    """
    unrenderable = []
    for sample_id, language, _tags, text in fixture_tools.SAMPLES:
        font = fixture_tools.load_font(36, language)
        if fixture_tools._missing_font_characters(font, text):
            unrenderable.append(sample_id)
    return unrenderable


requires_sample_fonts = pytest.mark.skipif(
    bool(_unrenderable_samples()),
    reason=f"host fonts cannot render synthetic samples: {', '.join(_unrenderable_samples())}",
)


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

    rows = prepare_huggingface_dataset(out_dir, "dharmaocr", 5, download=False)

    assert len(rows) == 1
    assert {
        key: value for key, value in rows[0].items() if key not in {"image", "pdf", "gt_text"}
    } == {
        "id": "dharmaocr_train_0000",
        "dataset": "DharmaOCR",
        "language": "pt",
        "tags": ["pt_br", "legal", "administrative"],
        "source": {
            "kind": "huggingface",
            "dataset_id": "Dharma-AI/DharmaOCR-Benchmark",
            "revision": "e8f4bb516839c1a0a32ee0234b3cbcd5aa5c10d3",
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
            "dataset": "DharmaOCR",
            "image": "images/sample.png",
            "pdf": "pdfs/sample.pdf",
            "gt_text": "ground_truth/sample.txt",
            "language": "en",
            "tags": ["form"],
        }
    ]

    manifest_path = write_manifest(tmp_path, rows)
    write_readme(tmp_path, rows, {"dharmaocr"})

    manifest_row = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_row["id"] == "sample"
    assert manifest_row["manifest_version"] == 1
    assert manifest_row["file_sha256"]["pdf"] == hashlib.sha256(b"%PDF-1.7\n").hexdigest()
    assert manifest_row["file_bytes"]["gt_text"] == len(b"expected")
    readme = (tmp_path / "README.generated.md").read_text(encoding="utf-8")
    assert "| DharmaOCR | 1 |" in readme


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


@requires_sample_fonts
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


def test_rows_may_share_one_ground_truth_file(tmp_path: Path) -> None:
    """A degraded variant cites the clean sample's ground truth, not a copy of it.

    The degradation moves pixels and leaves the text alone, which is what keeps
    the corpus exact however severe the damage -- so the same path is published
    once per row citing it. Refusing the second row made every tier that
    generates variants unbuildable.
    """
    stage = tmp_path / "stage"
    (stage / "ground_truth").mkdir(parents=True)
    (stage / "ground_truth" / "sample.txt").write_text("Ação nº 1", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()

    for _ in range(2):
        _copy_staged_material(stage, out, "ground_truth/sample.txt", "sample")

    assert (out / "ground_truth" / "sample.txt").read_text(encoding="utf-8") == "Ação nº 1"


def test_a_different_file_at_the_same_path_is_still_refused(tmp_path: Path) -> None:
    """Sharing is only safe while the bytes agree; disagreement is a collision."""
    stage = tmp_path / "stage"
    (stage / "ground_truth").mkdir(parents=True)
    (stage / "ground_truth" / "sample.txt").write_text("segundo", encoding="utf-8")
    out = tmp_path / "out"
    (out / "ground_truth").mkdir(parents=True)
    (out / "ground_truth" / "sample.txt").write_text("primeiro", encoding="utf-8")

    with pytest.raises(FileExistsError):
        _copy_staged_material(stage, out, "ground_truth/sample.txt", "sample")

    assert (out / "ground_truth" / "sample.txt").read_text(encoding="utf-8") == "primeiro"


def test_preparing_a_dataset_without_ground_truth_fails_loudly(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Images with no transcription measure nothing, so they are not a corpus.

    One of the Portuguese sets on the Hub publishes exactly that. Every row was
    dropped for lacking ground truth and the run still printed a manifest path
    and exited zero, which reads as success.
    """
    monkeypatch.setattr(sys, "argv", ["prepare", "--datasets", "dharmaocr", "--out", str(tmp_path)])
    monkeypatch.setattr(dataset_tools, "prepare_huggingface_dataset", lambda *a, **k: [])

    assert dataset_tools.main() == 1
    assert not (tmp_path / "manifest.jsonl").exists()
    assert "ground-truth" in capsys.readouterr().err
