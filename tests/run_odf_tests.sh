#!/bin/bash
#
# Automated ODF Export Quality Testing Script
# Usage: ./run_odf_tests.sh [pdf_file] [--open]
#
# Tests the ODF export quality and optionally opens the result.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
SRC_DIR="$REPO_ROOT/src"

# Default test file
DEFAULT_PDF="$SRC_DIR/bigocrpdf/avaliacao-neuro-completa-ocr-1.pdf"

# Parse arguments
PDF_FILE="${1:-$DEFAULT_PDF}"
OPEN_RESULT=false

for arg in "$@"; do
    case $arg in
        --open)
            OPEN_RESULT=true
            shift
            ;;
    esac
done

# Check if file exists
if [[ ! -f "$PDF_FILE" ]]; then
    echo "Error: File not found: $PDF_FILE"
    exit 1
fi

# Generate output path
OUTPUT_ODF="/tmp/$(basename "${PDF_FILE%.*}").odt"

echo "=========================================="
echo "ODF Export Quality Test"
echo "=========================================="
echo "Input PDF:  $PDF_FILE"
echo "Output ODF: $OUTPUT_ODF"
echo "------------------------------------------"
echo ""

# Run the export test
cd "$SRC_DIR"
python -m bigocrpdf.scripts.test_odf_export "$PDF_FILE" "$OUTPUT_ODF" -v

# Optionally open the result
if $OPEN_RESULT; then
    echo ""
    echo "Opening result..."
    xdg-open "$OUTPUT_ODF" 2>/dev/null &
fi

echo ""
echo "Done! Output saved to: $OUTPUT_ODF"
