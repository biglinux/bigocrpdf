#!/usr/bin/env bash
# Run and gate the OCR quality benchmark.
#
# The pieces this drives -- ocr_benchmark.py, compare_benchmarks.py,
# prepare_benchmark_datasets.py -- were already complete and already tested.
# Nothing invoked them, so no regression was ever actually caught. This is that
# invocation, and nothing more.
#
#   tools/benchmark.sh prepare
#   tools/benchmark.sh run    [--profile NAME] [--out FILE]
#   tools/benchmark.sh gate   --host-class SLUG [--metrics quality|performance|all]
#   tools/benchmark.sh accept --host-class SLUG
#
# Datasets and model weights stay out of the repository; the baseline JSONL
# does not, because it is text and because compare_benchmarks fails closed when
# a baseline is not comparable with the candidate.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR=${BIGOCRPDF_BENCH_DATA:-data/benchmarks}
BUILD_DIR=build/bench
BASELINE_DIR=benchmarks/baselines
MANIFEST="$DATA_DIR/manifest.jsonl"

usage() {
	sed -n '3,15p' "$0" >&2
	exit 2
}

require_manifest() {
	[ -f "$MANIFEST" ] || {
		printf 'No manifest at %s. Run: tools/benchmark.sh prepare\n' "$MANIFEST" >&2
		exit 1
	}
}

command=${1:-}
[ -n "$command" ] || usage
shift || true

case $command in
prepare)
	mkdir -p "$DATA_DIR"
	python3 benchmarks/make_synthetic_ocr_fixtures.py --out "$DATA_DIR"
	printf 'Synthetic fixtures staged in %s\n' "$DATA_DIR"
	printf 'For the real datasets:\n'
	printf '  python3 benchmarks/prepare_benchmark_datasets.py --datasets dharmaocr --out %s\n' "$DATA_DIR"
	;;

run)
	profile=balanced_cpu
	out="$BUILD_DIR/candidate.jsonl"
	while [ $# -gt 0 ]; do
		case $1 in
		--profile)
			profile=$2
			shift 2
			;;
		--out)
			out=$2
			shift 2
			;;
		*) usage ;;
		esac
	done
	require_manifest
	mkdir -p "$(dirname "$out")"
	python3 benchmarks/ocr_benchmark.py \
		--manifest "$MANIFEST" \
		--profile "$profile" \
		--repeats 3 \
		--warmup-runs 1 \
		--out "$out"
	printf 'Candidate written to %s\n' "$out"
	;;

gate)
	host_class=
	metrics=quality
	candidate="$BUILD_DIR/candidate.jsonl"
	while [ $# -gt 0 ]; do
		case $1 in
		--host-class)
			host_class=$2
			shift 2
			;;
		--metrics)
			metrics=$2
			shift 2
			;;
		--candidate)
			candidate=$2
			shift 2
			;;
		*) usage ;;
		esac
	done
	[ -n "$host_class" ] || usage
	baseline="$BASELINE_DIR/$host_class/balanced_cpu.jsonl"
	[ -f "$baseline" ] || {
		printf 'No baseline at %s. Record one with: tools/benchmark.sh accept --host-class %s\n' \
			"$baseline" "$host_class" >&2
		exit 1
	}
	[ -f "$candidate" ] || {
		printf 'No candidate at %s. Run: tools/benchmark.sh run\n' "$candidate" >&2
		exit 1
	}
	mkdir -p "$BUILD_DIR"
	# Exits non-zero on regression, which is the whole point.
	python3 benchmarks/compare_benchmarks.py "$baseline" "$candidate" \
		--out "$BUILD_DIR/report.md"
	printf 'Gate passed (%s metrics). Report: %s/report.md\n' "$metrics" "$BUILD_DIR"
	;;

accept)
	host_class=
	candidate="$BUILD_DIR/candidate.jsonl"
	while [ $# -gt 0 ]; do
		case $1 in
		--host-class)
			host_class=$2
			shift 2
			;;
		--candidate)
			candidate=$2
			shift 2
			;;
		*) usage ;;
		esac
	done
	[ -n "$host_class" ] || usage
	[ -f "$candidate" ] || {
		printf 'No candidate at %s. Run: tools/benchmark.sh run\n' "$candidate" >&2
		exit 1
	}
	target="$BASELINE_DIR/$host_class/balanced_cpu.jsonl"
	mkdir -p "$(dirname "$target")"
	# Deliberately a copy and not a move: accepting a baseline is a reviewed
	# commit, so the candidate stays where it is for inspection.
	cp "$candidate" "$target"
	printf 'Baseline updated: %s\n' "$target"
	printf 'Commit it with the gate report in the message.\n'
	;;

*) usage ;;
esac
