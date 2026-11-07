"""Tests for pdf_utils module."""

import os
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from bigocrpdf.utils import pdf_utils
from bigocrpdf.utils.pdf_utils import (
    confined,
    get_pdf_info,
    images_to_pdf,
    is_image_file,
    is_regular_file,
    sandbox_state,
)

needs_sandbox = pytest.mark.skipif(sandbox_state()[0] is None, reason="no usable bwrap")


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False, timeout=30)


def test_pdf_info_uses_one_snapshot_for_pages_and_metadata(tmp_path) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"pdf")
    result = SimpleNamespace(stdout="Pages: 3\nTitle: Report\nAuthor: Ada\n")

    with patch("bigocrpdf.utils.pdf_utils.subprocess.run", return_value=result) as run:
        info = get_pdf_info(str(pdf_path))

    run.assert_called_once()
    assert info["pages"] == 3
    assert info["title"] == "Report"
    assert info["author"] == "Ada"


def test_pdf_info_keeps_metadata_when_page_count_is_invalid(tmp_path) -> None:
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"pdf")
    result = SimpleNamespace(stdout="Pages: unknown\nTitle: Report\nAuthor: Ada\n")

    with patch("bigocrpdf.utils.pdf_utils.subprocess.run", return_value=result):
        info = get_pdf_info(str(pdf_path))

    assert info["pages"] == 0
    assert info["title"] == "Report"
    assert info["author"] == "Ada"


def test_images_to_pdf_removes_owned_output_when_save_fails(tmp_path) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (10, 10), "white").save(image_path)
    generated_path = ""

    def create_temp(*, prefix: str, suffix: str) -> tuple[int, str]:
        nonlocal generated_path
        fd, generated_path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=tmp_path)
        return fd, generated_path

    with (
        patch("bigocrpdf.utils.temp_manager.mkstemp", side_effect=create_temp),
        patch.object(Image.Image, "save", side_effect=OSError("disk write failed")),
        pytest.raises(RuntimeError, match="Failed to create PDF"),
    ):
        images_to_pdf([str(image_path)])

    assert generated_path
    assert not os.path.exists(generated_path)


class TestIsImageFile:
    def test_png_is_image(self):
        assert is_image_file("photo.png") is True

    def test_jpg_is_image(self):
        assert is_image_file("photo.jpg") is True

    def test_jpeg_is_image(self):
        assert is_image_file("photo.jpeg") is True

    def test_tiff_is_image(self):
        assert is_image_file("scan.tiff") is True

    def test_bmp_is_image(self):
        assert is_image_file("image.bmp") is True

    def test_webp_is_image(self):
        assert is_image_file("image.webp") is True

    def test_pdf_is_not_image(self):
        assert is_image_file("document.pdf") is False

    def test_txt_is_not_image(self):
        assert is_image_file("notes.txt") is False

    def test_case_insensitive(self):
        assert is_image_file("PHOTO.PNG") is True
        assert is_image_file("photo.JPG") is True

    def test_empty_string(self):
        assert is_image_file("") is False

    def test_no_extension(self):
        assert is_image_file("myfile") is False

    def test_path_with_directories(self):
        assert is_image_file("/home/user/photo.png") is True


@needs_sandbox
def test_sandbox_sees_only_given_files(tmp_path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("kept out")

    assert _run(confined(["cat", str(secret)])).returncode != 0
    assert _run(confined(["cat", str(secret)], read=[secret])).stdout == "kept out"
    assert _run(confined(["ls", os.path.expanduser("~")])).returncode != 0


@needs_sandbox
def test_sandbox_writes_only_its_directory(tmp_path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    source = tmp_path / "in.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    assert _run(confined(["touch", str(out / "made")], write=[out])).returncode == 0
    assert (out / "made").exists()
    assert _run(confined(["touch", str(source)], read=[source], write=[out])).returncode != 0
    # Outside its directory a write lands in the sandbox's own tmpfs.
    _run(confined(["touch", str(tmp_path / "beside")], write=[out]))
    assert not (tmp_path / "beside").exists()
    # A file read from inside the writable directory stays read-only.
    inside = out / "in.pdf"
    inside.write_text("kept")
    overwrite = ["sh", "-c", f"echo changed > {inside}"]
    assert _run(confined(overwrite, read=[inside], write=[out])).returncode != 0
    assert inside.read_text() == "kept"


@needs_sandbox
def test_tool_outside_usr_runs_confined(tmp_path, monkeypatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    tool = bin_dir / "pdfinfo"
    shutil.copy2(shutil.which("pdfinfo"), tool)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")

    command = confined(["pdfinfo", "-v"])

    assert str(tool) in command
    assert _run(command).returncode == 0


@needs_sandbox
def test_sandbox_has_no_network() -> None:
    interfaces = _run(confined(["cat", "/proc/self/net/dev"])).stdout.splitlines()[2:]

    assert [line.split(":")[0].strip() for line in interfaces] == ["lo"]


def test_missing_bwrap_runs_tools_as_given(monkeypatch) -> None:
    real_which = pdf_utils.shutil.which
    monkeypatch.setattr(
        pdf_utils.shutil, "which", lambda name: None if name == "bwrap" else real_which(name)
    )
    monkeypatch.delenv("FLATPAK_ID", raising=False)
    monkeypatch.setattr(pdf_utils, "_sandbox", None)

    assert sandbox_state() == (None, "none (bwrap not installed)")
    assert confined(["pdfinfo", "/x.pdf"], read=["/x.pdf"]) == ["pdfinfo", "/x.pdf"]


def test_planted_link_is_rejected(tmp_path) -> None:
    private = tmp_path / "private.jpg"
    private.write_bytes(b"\xff\xd8\xff")
    planted = tmp_path / "obj-000.jpg"
    planted.symlink_to(private)

    assert is_regular_file(private)
    assert not is_regular_file(planted)
    assert not is_regular_file(tmp_path / "missing.jpg")
