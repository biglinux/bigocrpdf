"""Clean-process import contracts for entrypoint dependency order."""

import os
import subprocess
import sys
from pathlib import Path


def test_config_import_has_no_utils_cycle() -> None:
    source_dir = Path(__file__).parents[1] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(source_dir), environment.get("PYTHONPATH", "")) if path
    )
    result = subprocess.run(
        [sys.executable, "-c", "import bigocrpdf.config"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
