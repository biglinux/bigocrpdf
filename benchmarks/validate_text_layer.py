#!/usr/bin/env python3
"""Validate extracted searchable text from a PDF.

allow-noisy-log: validation reports are user-facing CLI output.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks.ocr_metrics import (
    char_error_rate,
    levenshtein,
    levenshtein_ratio,
    normalize_for_ocr_metric,
    word_error_rate,
)
from bigocrpdf.services.rapidocr_service.pdf_validation import validate_searchable_pdf_text


def extract_text_with_pdftotext(pdf_path: Path) -> str | None:
    if shutil.which("pdftotext") is None:
        return None
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def extract_text_with_pypdf(pdf_path: Path) -> str | None:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]

        reader = PdfReader(str(pdf_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        return None


def extract_text_with_pdfminer(pdf_path: Path) -> str | None:
    try:
        from pdfminer.high_level import extract_text

        return extract_text(str(pdf_path))
    except Exception:
        return None


def extract_text_with_python_pdf_lib(pdf_path: Path) -> str | None:
    """Compatibility helper that tries the supported Python extractors in order."""
    return extract_text_with_pypdf(pdf_path) or extract_text_with_pdfminer(pdf_path)


def extract_pdf_text_with_method(pdf_path: Path) -> tuple[str, str, str]:
    """Extract PDF text and identify the exact implementation and version."""
    text = extract_text_with_pdftotext(pdf_path)
    if text is not None:
        return text, "pdftotext", _extractor_version("pdftotext")
    text = extract_text_with_pypdf(pdf_path)
    if text is not None:
        return text, "pypdf", _extractor_version("pypdf")
    text = extract_text_with_pdfminer(pdf_path)
    if text is not None:
        return text, "pdfminer.six", _extractor_version("pdfminer.six")
    raise RuntimeError("No PDF text extractor available (pdftotext, pypdf or pdfminer.six)")


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text while preserving the original string-only helper API."""
    return extract_pdf_text_with_method(pdf_path)[0]


@lru_cache(maxsize=3)
def _extractor_version(method: str) -> str:
    if method == "pdftotext":
        try:
            result = subprocess.run(
                ["pdftotext", "-v"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            first_line = (result.stderr or result.stdout).splitlines()[0].strip()
            return first_line or "unknown"
        except (OSError, subprocess.SubprocessError, IndexError):
            return "unknown"
    try:
        return importlib.metadata.version(method)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def build_report(pdf_path: Path, expected_path: Path | None) -> dict[str, Any]:
    extracted, extractor, extractor_version = extract_pdf_text_with_method(pdf_path)
    validation = validate_searchable_pdf_text(extracted)
    report = {
        "pdf": str(pdf_path),
        "text_extractor": extractor,
        "text_extractor_version": extractor_version,
        "extracted_pdf_text_chars": validation.extracted_chars,
        "suspicious_ratio": validation.suspicious_ratio,
        "text_layer_ok": validation.ok,
        "failure_reason": validation.reason,
    }
    if expected_path:
        expected = expected_path.read_text(encoding="utf-8")
        normalized_extracted = normalize_for_ocr_metric(extracted)
        normalized_expected = normalize_for_ocr_metric(expected)
        extracted_words = normalized_extracted.split()
        expected_words = normalized_expected.split()
        report["char_error_rate"] = char_error_rate(extracted, expected)
        report["word_error_rate"] = word_error_rate(extracted, expected)
        report["levenshtein_ratio"] = levenshtein_ratio(extracted, expected)
        report["char_edit_distance"] = levenshtein(normalized_extracted, normalized_expected)
        report["expected_char_count"] = len(normalized_expected)
        report["word_edit_distance"] = levenshtein(extracted_words, expected_words)
        report["expected_word_count"] = len(expected_words)
        report["unicode_loss_count"] = sum(extracted.count(char) for char in "�□")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    args = parser.parse_args()

    report = build_report(args.pdf, args.expected)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for key, value in report.items():
            print(f"{key}: {value}")
    return 0 if report["text_layer_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
