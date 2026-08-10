# Benchmark baselines

One committed JSONL per host class, recorded with `tools/benchmark.sh accept`.

The results are committed; the data that produced them is not. A baseline is a
few hundred kilobytes of text and diffs usefully, while datasets and model
weights belong outside the repository.

This is safe because `compare_benchmarks.py` enforces a twenty-field
comparability contract -- model SHA-256s, engine, DPI, environment -- and fails
closed. A baseline recorded on a different machine or against different model
files produces an explicit incomparability error rather than a bogus pass, so a
stale baseline cannot quietly approve a regression.

Layout:

    benchmarks/baselines/<host-class>/balanced_cpu.jsonl

`<host-class>` is a short slug you choose for a machine profile, for example
`x86_64-openvino-cpu`. Quality metrics are comparable across machines;
timing and peak RSS are not, so compare those only within one host class.
