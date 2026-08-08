#!/usr/bin/env python3
"""Generate and compare stable BigOCRPDF compatibility snapshots."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_MARKER = "BIGOCR_COMPAT="

_PROBE = r"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path

from bigocrpdf import __version__
from bigocrpdf.cli import main
from bigocrpdf.cli_parser import _parse_page_list, _parse_ranges, build_parser
from bigocrpdf.config import SELECTED_FILE_PATH
from bigocrpdf.services.settings import OcrSettings
from bigocrpdf.utils.config_manager import CONFIG_FILE_PATH

HOME = str(Path.home())


def normalize(value):
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        return value.replace(HOME, "$HOME")
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, set):
        return [normalize(item) for item in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return repr(value).replace(HOME, "$HOME")


def action_schema(action):
    default = action.default
    if default is argparse.SUPPRESS:
        default = "<SUPPRESS>"
    value_type = getattr(action.type, "__name__", None)
    return {
        "class": type(action).__name__,
        "dest": action.dest,
        "options": list(action.option_strings),
        "required": action.required,
        "nargs": action.nargs,
        "default": normalize(default),
        "choices": normalize(list(action.choices) if action.choices is not None else None),
        "type": value_type,
    }


def parser_schema():
    parser = build_parser()
    subcommands = {}
    root_actions = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, child in sorted(action.choices.items()):
                subcommands[name] = [action_schema(item) for item in child._actions]
        else:
            root_actions.append(action_schema(action))
    return {"prog": parser.prog, "root": root_actions, "subcommands": subcommands}


def invoke(argv):
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main(argv)
    except SystemExit as exc:
        status = exc.code
    return {
        "status": status,
        "stdout": normalize(stdout.getvalue()),
        "stderr": normalize(stderr.getvalue()),
    }


SETTING_ATTRIBUTES = (
    "lang", "ocr_language", "replace_existing_ocr", "enhance_embedded_images",
    "destination_folder", "save_in_same_folder", "pdf_suffix",
    "use_original_filename", "overwrite_existing", "include_date", "include_year",
    "include_month", "include_day", "include_time", "date_format_order", "save_txt",
    "separate_txt_folder", "txt_folder", "save_odf", "odf_include_images",
    "odf_open_after_export", "md_include_front_matter", "md_open_after_export",
    "image_export_format", "image_export_quality", "image_export_preserve_original",
    "auto_detect_quality", "convert_to_pdfa", "max_file_size_mb", "page_layout",
    "enable_bilevel_compression", "force_bilevel_compression", "dpi",
    "enable_preprocessing", "enable_deskew", "enable_baseline_dewarp",
    "enable_perspective_correction", "enable_orientation_detection",
    "enable_auto_contrast", "enable_auto_brightness", "enable_denoise",
    "enable_scanner_effect", "scanner_effect_strength", "enable_border_clean",
    "enable_vintage_look", "vintage_bw", "text_score_threshold", "box_thresh",
    "unclip_ratio", "ocr_profile", "detection_full_resolution", "parallel_workers",
    "quick_start_mode",
)


def settings_values(settings):
    return {name: normalize(getattr(settings, name)) for name in SETTING_ATTRIBUTES}


def replacement(name, value):
    if name in {"lang", "ocr_language"}:
        return "english"
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 7
    if isinstance(value, float):
        return value + 0.125
    if isinstance(value, dict):
        return {"year": 3, "month": 1, "day": 2}
    return "compat-" + name


def settings_schema():
    settings = OcrSettings()
    defaults = settings_values(settings)
    for name in SETTING_ATTRIBUTES:
        setattr(settings, name, replacement(name, getattr(settings, name)))
    output = Path.home() / "output"
    settings.save_settings("english", str(output), save_in_same_folder=False)

    queued = Path.home() / "line one\nline two.pdf"
    queued.write_bytes(b"%PDF-1.4\n")
    unsupported = Path.home() / "unsupported.xyz"
    unsupported.write_text("x", encoding="utf-8")
    queue_results = [
        settings.add_files([str(queued), str(unsupported)]),
        settings.add_files([str(queued)]),
    ]

    reloaded = OcrSettings()
    config_path = Path(CONFIG_FILE_PATH)
    selected_path = Path(SELECTED_FILE_PATH)
    return {
        "attributes": list(SETTING_ATTRIBUTES),
        "public_methods": sorted(
            name
            for name in dir(OcrSettings)
            if not name.startswith("_") and callable(getattr(OcrSettings, name))
        ),
        "defaults": defaults,
        "reloaded": settings_values(reloaded),
        "config": normalize(json.loads(config_path.read_text(encoding="utf-8"))),
        "selected_payload": normalize(json.loads(selected_path.read_text(encoding="utf-8"))),
        "queue_results": queue_results,
        "display_name": reloaded.display_name(str(queued)),
    }


def parse_contracts():
    cases = ["", "1", "1-3", "3,1,3", "0,-2", "bad", "4-2"]
    result = {}
    for case in cases:
        for name, parser in (("pages", _parse_page_list), ("ranges", _parse_ranges)):
            try:
                value = parser(case)
            except Exception as exc:
                value = {"exception": type(exc).__name__, "message": str(exc)}
            result[f"{name}:{case}"] = normalize(value)
    return result


snapshot = {
    "version": __version__,
    "imports": [
        "bigocrpdf", "bigocrpdf.config", "bigocrpdf.cli", "bigocrpdf.services.settings",
        "bigocrpdf.services.pdf_operations", "bigocrpdf.services.rapidocr_service.config",
    ],
    "parser": parser_schema(),
    "parsers": parse_contracts(),
    "invocations": {
        "no_args": invoke([]),
        "help": invoke(["--help"]),
        "version": invoke(["--version"]),
        "missing_info": invoke(["info", "/definitely/missing-bigocrpdf.pdf"]),
    },
    "settings": settings_schema(),
}
print("BIGOCR_COMPAT=" + json.dumps(normalize(snapshot), ensure_ascii=False, sort_keys=True))
"""


def _source_dir(root: Path) -> Path:
    source = root.resolve() / "src"
    if not (source / "bigocrpdf").is_dir():
        raise ValueError(f"No src/bigocrpdf package under {root}")
    return source


def generate_snapshot(root: Path) -> dict[str, Any]:
    """Run the compatibility probe in a clean process and isolated home."""
    with tempfile.TemporaryDirectory(prefix="bigocr-compat-") as home:
        environment = dict(os.environ)
        environment.update(
            {
                "HOME": home,
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PYTHONPATH": str(_source_dir(root)),
                "PYTHONNOUSERSITE": "1",
            }
        )
        result = subprocess.run(
            [sys.executable, "-c", _PROBE],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            env=environment,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Compatibility probe failed for {root} with status {result.returncode}:\n{result.stderr}"
        )
    marker_lines = [line for line in result.stdout.splitlines() if line.startswith(_MARKER)]
    if len(marker_lines) != 1:
        raise RuntimeError(f"Compatibility probe emitted no unique snapshot:\n{result.stdout}")
    return json.loads(marker_lines[0][len(_MARKER) :])


def _serialized(snapshot: dict[str, Any]) -> str:
    return json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def compare(reference: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Print a unified diff and return whether snapshots match."""
    before = _serialized(reference).splitlines(keepends=True)
    after = _serialized(candidate).splitlines(keepends=True)
    diff = list(difflib.unified_diff(before, after, fromfile="reference", tofile="candidate"))
    if diff:
        sys.stdout.writelines(diff)
        return False
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="Record a frozen snapshot")
    record.add_argument("--source-root", type=Path, default=Path.cwd())
    record.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="Compare a source tree with a frozen snapshot")
    verify.add_argument("--source-root", type=Path, default=Path.cwd())
    verify.add_argument("--baseline", type=Path, required=True)

    differential = subparsers.add_parser("compare", help="Compare two source trees directly")
    differential.add_argument("--reference-root", type=Path, required=True)
    differential.add_argument("--candidate-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "record":
        snapshot = generate_snapshot(args.source_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(_serialized(snapshot), encoding="utf-8")
        print(f"Recorded compatibility snapshot: {args.output}")
        return 0
    if args.command == "verify":
        reference = json.loads(args.baseline.read_text(encoding="utf-8"))
        candidate = generate_snapshot(args.source_root)
    else:
        reference = generate_snapshot(args.reference_root)
        candidate = generate_snapshot(args.candidate_root)
    if not compare(reference, candidate):
        print("Compatibility verification failed.", file=sys.stderr)
        return 1
    print("Compatibility verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
