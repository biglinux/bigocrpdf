# Benchmark baselines

One committed JSONL per host class, recorded with `tools/benchmark.sh accept`.

The results are committed; the data that produced them is not. A baseline is a
few hundred kilobytes of text and diffs usefully, while datasets and model
weights belong outside the repository.

This is safe because `compare_benchmarks.py` enforces a comparability
contract (worker and plan SHA-256s, OCR settings, DPI, environment) and
fails closed. A baseline recorded on a different machine or with a different
ppocr-native build produces an explicit incomparability error rather than a
bogus pass, so a stale baseline cannot quietly approve a regression.

Layout:

    benchmarks/baselines/<host-class>/balanced.jsonl

`<host-class>` is a short slug you choose for a machine profile, for example
`x86_64-ryzen7-5700u`. A baseline compares only against runs on the same
host, with the same packages and the same ppocr-native build: the gate checks
quality, time and peak RSS together. Record it with the packaged worker and
plans (`/usr/bin/ppocr-native`, `/usr/share/ppocr-native`), so that a later
package build of the same source can be gated against it.

## Reference points

- `x86_64-ryzen7-5700u`: the 50 DharmaOCR documents, three runs each, with
  the text layer in Noto Sans and eight recognisers, read with a mean CER of
  0.154 and WER of 0.270, in 6.5 s a document (median 5.6 s) at a peak of
  1.6 GB, with the packaged worker and plans.
- Structured OCR JSON is written compact: indentation was 62% of the bytes of
  an 18-page payload.
