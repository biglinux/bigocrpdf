#!/usr/bin/env bash
# Run and gate the OCR quality benchmark through ocr_benchmark.py,
# compare_benchmarks.py and prepare_benchmark_datasets.py.
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

data_dir=${BIGOCRPDF_BENCH_DATA:-data/benchmarks}
build_dir=build/bench
baseline_dir=benchmarks/baselines
manifest="$data_dir/manifest.jsonl"

usage() {
	sed -n '3,15p' "$0" >&2
	exit 2
}

require_manifest() {
	[ -f "$manifest" ] || {
		printf 'No manifest at %s. Run: tools/benchmark.sh prepare\n' "$manifest" >&2
		exit 1
	}
}

command=${1:-}
[ -n "$command" ] || usage
shift || true

case $command in
prepare)
	mkdir -p "$data_dir"
	python3 benchmarks/make_synthetic_ocr_fixtures.py --out "$data_dir"
	printf 'Synthetic fixtures staged in %s\n' "$data_dir"
	printf 'For the real datasets:\n'
	printf '  python3 benchmarks/prepare_benchmark_datasets.py --datasets dharmaocr --out %s\n' "$data_dir"
	;;

run)
	profile=balanced_cpu
	out="$build_dir/candidate.jsonl"
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
		--manifest "$manifest" \
		--profile "$profile" \
		--repeats 3 \
		--warmup-runs 1 \
		--out "$out"
	printf 'Candidate written to %s\n' "$out"
	;;

gate)
	host_class=
	metrics=quality
	candidate="$build_dir/candidate.jsonl"
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
	baseline="$baseline_dir/$host_class/balanced_cpu.jsonl"
	[ -f "$baseline" ] || {
		printf 'No baseline at %s. Record one with: tools/benchmark.sh accept --host-class %s\n' \
			"$baseline" "$host_class" >&2
		exit 1
	}
	[ -f "$candidate" ] || {
		printf 'No candidate at %s. Run: tools/benchmark.sh run\n' "$candidate" >&2
		exit 1
	}
	mkdir -p "$build_dir"
	# Exits non-zero on regression, which is the whole point.
	python3 benchmarks/compare_benchmarks.py "$baseline" "$candidate" \
		--out "$build_dir/report.md"
	printf 'Gate passed (%s metrics). Report: %s/report.md\n' "$metrics" "$build_dir"
	;;

accept)
	host_class=
	candidate="$build_dir/candidate.jsonl"
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
	target="$baseline_dir/$host_class/balanced_cpu.jsonl"
	mkdir -p "$(dirname "$target")"
	# Deliberately a copy and not a move: accepting a baseline is a reviewed
	# commit, so the candidate stays where it is for inspection.
	cp "$candidate" "$target"
	printf 'Baseline updated: %s\n' "$target"
	printf 'Commit it with the gate report in the message.\n'
	;;

*) usage ;;
esac
