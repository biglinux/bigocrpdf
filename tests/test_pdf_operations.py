"""Tests for pdf_operations module."""

import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pikepdf
import pytest

from bigocrpdf.services import pdf_operations
from bigocrpdf.services.pdf_operations import (
    PDFInfo,
    get_pdf_info,
    insert_pages,
    rotate_pages,
)


def _create_test_pdf(path: str, num_pages: int = 3) -> str:
    """Create a simple test PDF with the given number of pages."""
    pdf = pikepdf.Pdf.new()
    for i in range(num_pages):
        page = pikepdf.Page(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Page,
                MediaBox=[0, 0, 612, 792],
                Resources=pikepdf.Dictionary(),
                Contents=pdf.make_stream(f"BT /F1 12 Tf 100 700 Td (Page {i + 1}) Tj ET".encode()),
            )
        )
        pdf.pages.append(page)
    pdf.save(path)
    return path


class TestGetPdfInfo:
    def test_basic_info(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            _create_test_pdf(path, 5)
            info = get_pdf_info(path)
            assert isinstance(info, PDFInfo)
            assert info.page_count == 5
            assert info.file_size_bytes > 0
            assert info.path == path
        finally:
            os.unlink(path)

    def test_nonexistent_file_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            get_pdf_info("/nonexistent/file.pdf")


def test_atomic_pdf_save_preserves_existing_destination_on_failure(tmp_path):
    destination = tmp_path / "existing.pdf"
    destination.write_bytes(b"existing")
    pdf = MagicMock()

    def fail_after_partial_write(path, **_options):
        Path(path).write_bytes(b"partial")
        raise OSError("simulated save failure")

    pdf.save.side_effect = fail_after_partial_write

    with pytest.raises(OSError, match="simulated save failure"):
        pdf_operations._save_pdf_atomically(pdf, destination)

    assert destination.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [destination]


def test_atomic_pdf_save_publishes_the_pdf_and_nothing_else(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "output.pdf"
    _create_test_pdf(str(destination), 1)
    replacement = pikepdf.Pdf.new()
    replacement.add_blank_page()

    try:
        pdf_operations._save_pdf_atomically(replacement, destination)
    finally:
        replacement.close()

    assert [entry.name for entry in tmp_path.iterdir()] == ["output.pdf"]


def test_atomic_new_pdf_save_closes_document_on_failure(tmp_path):
    pdf = MagicMock()

    with (
        patch.object(pdf_operations, "_save_pdf_atomically", side_effect=OSError("failure")),
        pytest.raises(OSError, match="failure"),
    ):
        pdf_operations._save_and_close_pdf_atomically(pdf, tmp_path / "output.pdf")

    pdf.close.assert_called_once_with()


def test_atomic_pdf_save_preserves_existing_destination_mode(tmp_path: Path) -> None:
    destination = tmp_path / "existing.pdf"
    _create_test_pdf(str(destination), 1)
    destination.chmod(0o640)
    replacement = pikepdf.Pdf.new()
    replacement.add_blank_page()

    try:
        pdf_operations._save_pdf_atomically(replacement, destination)
    finally:
        replacement.close()

    assert stat.S_IMODE(destination.stat().st_mode) == 0o640


def test_rotate_duplicate_page_only_once(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    output = tmp_path / "rotated.pdf"
    _create_test_pdf(str(source), 1)

    result = rotate_pages(source, output, [1, 1], 90)

    assert result.success is True
    assert result.pages_affected == 1
    with pikepdf.open(output) as pdf:
        assert int(pdf.pages[0].get("/Rotate", 0)) == 90


def test_insert_empty_source_page_list_inserts_nothing(tmp_path: Path) -> None:
    target = tmp_path / "target.pdf"
    source = tmp_path / "source.pdf"
    output = tmp_path / "output.pdf"
    _create_test_pdf(str(target), 1)
    _create_test_pdf(str(source), 2)

    result = insert_pages(target, source, output, source_pages=[])

    assert result.success is False
    assert result.pages_affected == 0
    assert not output.exists()


def test_atomic_pdf_save_cleans_staged_file_when_mode_copy_fails(tmp_path: Path) -> None:
    destination = tmp_path / "existing.pdf"
    destination.write_bytes(b"existing")
    destination.chmod(0o640)
    pdf = MagicMock()

    with (
        patch.object(pdf_operations.os, "fchmod", side_effect=OSError("chmod failed")),
        pytest.raises(OSError, match="chmod failed"),
    ):
        pdf_operations._save_pdf_atomically(pdf, destination)

    pdf.save.assert_not_called()
    assert list(tmp_path.iterdir()) == [destination]


def _pdf_with_image(path: Path, **fields) -> Path:
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(400, 400))
    stream = pdf.make_stream(fields.pop("data"), **fields)
    pdf.pages[0].Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=stream))
    pdf.pages[0].Contents = pdf.make_stream(b"q 400 0 0 400 0 0 cm /Im0 Do Q")
    pdf.save(path)
    return path


def test_compress_keeps_an_undecodable_image(tmp_path: Path) -> None:
    source = _pdf_with_image(
        tmp_path / "in.pdf",
        data=b"\0" * (64 * 64 // 8),
        Type=pikepdf.Name.XObject,
        Subtype=pikepdf.Name.Image,
        Width=64,
        Height=64,
        ColorSpace=pikepdf.Name.DeviceRGB,
        BitsPerComponent=1,
    )

    result = pdf_operations.compress_pdf(source, tmp_path / "out.pdf")

    assert result.success
    assert result.pages_affected == 0


def test_compress_skips_a_non_integer_width(tmp_path: Path) -> None:
    source = _pdf_with_image(
        tmp_path / "in.pdf",
        data=b"\0" * (64 * 64),
        Type=pikepdf.Name.XObject,
        Subtype=pikepdf.Name.Image,
        Width=pikepdf.Name("/W"),
        Height=64,
        ColorSpace=pikepdf.Name.DeviceGray,
        BitsPerComponent=8,
    )

    result = pdf_operations.compress_pdf(source, tmp_path / "out.pdf")

    assert result.success
    assert result.pages_affected == 0


def test_compress_skips_a_jpeg_larger_than_declared(tmp_path: Path) -> None:
    import io

    from PIL import Image

    jpeg = io.BytesIO()
    Image.new("L", (1200, 1000), 255).save(jpeg, "JPEG")
    source = _pdf_with_image(
        tmp_path / "in.pdf",
        data=jpeg.getvalue(),
        Type=pikepdf.Name.XObject,
        Subtype=pikepdf.Name.Image,
        Width=400,
        Height=400,
        ColorSpace=pikepdf.Name.DeviceGray,
        BitsPerComponent=8,
        Filter=pikepdf.Name.DCTDecode,
    )

    with (
        patch.object(pdf_operations, "_MAX_COMPRESS_PIXELS", 1_000_000),
        patch.object(pdf_operations, "_encode_pdf_image_as_jpeg") as encode,
    ):
        result = pdf_operations.compress_pdf(source, tmp_path / "out.pdf")

    assert result.success
    encode.assert_not_called()


def _grained_ramp(channels: int) -> np.ndarray:
    """A 200x200 ramp with grain, which Flate stores poorly and JPEG well."""
    grain = np.random.default_rng(0).integers(0, 8, (200, 200, channels))
    return (np.arange(200)[None, :, None] + grain).astype(np.uint8)


def test_compress_writes_cmyk_as_cmyk(tmp_path: Path) -> None:
    import io

    from PIL import Image

    samples = _grained_ramp(4)
    source = _pdf_with_image(
        tmp_path / "in.pdf",
        data=samples.tobytes(),
        Type=pikepdf.Name.XObject,
        Subtype=pikepdf.Name.Image,
        Width=200,
        Height=200,
        ColorSpace=pikepdf.Name.DeviceCMYK,
        BitsPerComponent=8,
    )

    result = pdf_operations.compress_pdf(source, tmp_path / "out.pdf", image_dpi=600)

    assert result.pages_affected == 1
    with pikepdf.open(tmp_path / "out.pdf") as pdf:
        image = pdf.pages[0].Resources.XObject.Im0
        assert image.ColorSpace == pikepdf.Name.DeviceCMYK
        assert list(image.Decode) == [1, 0] * 4
        with Image.open(io.BytesIO(image.read_raw_bytes())) as jpeg:
            assert jpeg.mode == "CMYK"
            cyan = np.asarray(jpeg)[:, 140:160, 0].mean()
    assert cyan == pytest.approx(samples[:, 140:160, 0].mean(), abs=4)


def test_compress_keeps_masked_images(tmp_path: Path) -> None:
    fields = {
        "Type": pikepdf.Name.XObject,
        "Subtype": pikepdf.Name.Image,
        "Width": 200,
        "Height": 200,
        "BitsPerComponent": 8,
    }
    with pikepdf.Pdf.new() as pdf:
        pdf.add_blank_page(page_size=(400, 400))
        mask = pdf.make_stream(b"\0" * (200 * 200), ColorSpace=pikepdf.Name.DeviceGray, **fields)
        image = pdf.make_stream(
            _grained_ramp(3).tobytes(), ColorSpace=pikepdf.Name.DeviceRGB, SMask=mask, **fields
        )
        pdf.pages[0].Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=image))
        pdf.pages[0].Contents = pdf.make_stream(b"q 400 0 0 400 0 0 cm /Im0 Do Q")
        pdf.save(tmp_path / "in.pdf")

    result = pdf_operations.compress_pdf(tmp_path / "in.pdf", tmp_path / "out.pdf", image_dpi=600)

    assert result.pages_affected == 0
    with pikepdf.open(tmp_path / "out.pdf") as pdf:
        assert "/SMask" in pdf.pages[0].Resources.XObject.Im0


def test_compress_skips_flate_bombs(tmp_path: Path) -> None:
    import zlib

    source = _pdf_with_image(
        tmp_path / "in.pdf",
        data=zlib.compress(b"\0" * (64 << 20)),
        Type=pikepdf.Name.XObject,
        Subtype=pikepdf.Name.Image,
        Width=100,
        Height=100,
        ColorSpace=pikepdf.Name.DeviceGray,
        BitsPerComponent=8,
        Filter=pikepdf.Name.FlateDecode,
    )

    with patch.object(pikepdf.PdfImage, "as_pil_image") as decode:
        result = pdf_operations.compress_pdf(source, tmp_path / "out.pdf")

    assert result.success
    decode.assert_not_called()
