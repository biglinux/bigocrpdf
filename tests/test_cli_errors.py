"""End-to-end CLI error-boundary contracts."""

from pathlib import Path
from unittest.mock import patch

from bigocrpdf.cli import main


def test_invalid_split_range_returns_error_without_traceback(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "input.pdf"
    source.write_bytes(b"present")

    with patch("bigocrpdf.cli.logging.basicConfig"):
        status = main(
            [
                "split",
                str(source),
                "--output",
                str(tmp_path / "parts"),
                "--ranges",
                "not-a-range",
            ]
        )

    captured = capsys.readouterr()
    assert status == 1
    assert captured.out == ""
    assert "Error: Invalid range specification" in captured.err
    assert "Traceback" not in captured.err


def test_child_interpreters_ignore_modules_planted_in_cwd(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    import subprocess
    import sys

    marker = tmp_path / "planted-ran"
    (tmp_path / "json.py").write_text(f"open({str(marker)!r}, 'w').close()\n", encoding="utf-8")
    monkeypatch.delenv("PYTHONSAFEPATH", raising=False)

    assert main([]) == 0
    capsys.readouterr()
    subprocess.run([sys.executable, "-c", "import json"], cwd=tmp_path, check=True, timeout=30)

    assert not marker.exists()
