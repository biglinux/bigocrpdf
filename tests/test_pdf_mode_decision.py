from pathlib import Path

from bigocrpdf.services.rapidocr_service import backend
from bigocrpdf.services.rapidocr_service.config import OCRConfig


def test_pdf_mode_ocr_forces_image_only_pipeline(monkeypatch):
    calls = {"native": 0, "trusted": 0}

    def has_native(_path):
        calls["native"] = 1
        return True

    def has_trusted(_path):
        calls["trusted"] = 1
        return True

    monkeypatch.setattr(backend, "has_native_text", has_native)
    monkeypatch.setattr(backend, "has_trusted_native_text", has_trusted)

    config = OCRConfig(pdf_mode="ocr")

    assert backend.should_use_mixed_content_pipeline(config, Path("doc.pdf")) is False
    assert calls == {"native": 0, "trusted": 0}


def test_auto_mode_uses_mixed_pipeline_for_trusted_native_text(monkeypatch):
    monkeypatch.setattr(backend, "has_native_text", lambda _path: True)
    monkeypatch.setattr(backend, "has_trusted_native_text", lambda _path: True)

    config = OCRConfig(pdf_mode="auto")

    assert backend.should_use_mixed_content_pipeline(config, Path("doc.pdf")) is True


def test_auto_mode_rejects_untrusted_native_text(monkeypatch):
    monkeypatch.setattr(backend, "has_native_text", lambda _path: True)
    monkeypatch.setattr(backend, "has_trusted_native_text", lambda _path: False)

    config = OCRConfig(pdf_mode="auto")

    assert backend.should_use_mixed_content_pipeline(config, Path("doc.pdf")) is False


def test_auto_verified_does_not_skip_ocr_without_box_verifier(monkeypatch):
    calls = {"native": 0, "trusted": 0}

    def has_native(_path):
        calls["native"] = 1
        return True

    def has_trusted(_path):
        calls["trusted"] = 1
        return True

    monkeypatch.setattr(backend, "has_native_text", has_native)
    monkeypatch.setattr(backend, "has_trusted_native_text", has_trusted)

    config = OCRConfig(pdf_mode="auto_verified")

    assert backend.should_use_mixed_content_pipeline(config, Path("doc.pdf")) is False
    assert calls == {"native": 0, "trusted": 0}


def test_force_full_ocr_overrides_pdf_mode(monkeypatch):
    monkeypatch.setattr(backend, "has_native_text", lambda _path: True)
    monkeypatch.setattr(backend, "has_trusted_native_text", lambda _path: True)

    config = OCRConfig(pdf_mode="geometric", force_full_ocr=True)

    assert backend.should_use_mixed_content_pipeline(config, Path("doc.pdf")) is False
