#!/usr/bin/env bash
# Run the OCR quality layer: the tests that drive the real engine against a
# ground-truth corpus. Slow and model-dependent, so tools/validate.sh excludes
# them and this script opts back in.
#
# Usage: tools/quality.sh [extra pytest args...]
set -euo pipefail
cd "$(dirname "$0")/.."

# Marker selection has to be explicit: it overrides the -m in pyproject addopts.
exec python3 -m pytest -q -p no:cacheprovider -m 'real_ocr or slow' "$@"
