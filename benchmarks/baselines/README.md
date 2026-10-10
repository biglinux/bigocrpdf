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

## Reference points

The July 2026 runs established:

- all 36 RapidOCR v6 CPU matrix runs completed and produced text layers;
- ONNX Runtime small was fastest in that six-page run, but startup/model loading
  was included and the sample is not a universal engine ranking;
- complex FUNSD pages exposed recognition and layout limits;
- structured sidecars improved export token recall and restored table structures
  in two of three dense forms;
- the form separator refinement preserved recall and table counts;
- exports read from the PDF text layer agreed with the structured-OCR path
  within 92-98% on real documents;
- indentation was 62% of the bytes of an 18-page structured OCR JSON payload,
  which is why that payload is written compact.
- full-resolution detection on 20 DharmaOCR documents through the whole
  pipeline moved mean CER 0.1704 -> 0.1517 and mean WER 0.2705 -> 0.2649,
  left 10 documents unchanged, took 13% more time and raised per-worker peak
  RSS 2.5 -> 4.9 GB; the OCR stage alone, over 50 documents, moved CER
  0.1893 -> 0.1465, because preprocessing and dewarp already recover part of
  what the downscaled detector loses.
