# Benchmarks

Two harnesses exist:

1. `nexus/evaluation/benchmarks.py` — deterministic ELF/ProgramRecord analysis
   microbenchmark.
2. `nexus/benchmarks/` — dataset-driven evaluation suite with function-boundary,
   cross-compiler function-retrieval and masked-instruction tasks, deterministic
   baselines, exact denominators and machine-readable reports.

```bash
.venv/bin/nexus benchmark --path /usr/bin/ls --json
.venv/bin/nexus benchmark --dataset artifacts/dataset/realworld \
  --checkpoint models/smoke-test.pt --split test \
  --out artifacts/benchmarks/benchmark_report.json
```

The suite task definitions, baselines and reproduction commands are in
`docs/BENCHMARK_SPECIFICATION.md`. Recorded results are in
`BENCHMARK_RESULTS.md` and `artifacts/benchmarks/benchmark_report.json`.

These are not decompiler-accuracy benchmarks. The checkpoint is a
random-initialization smoke model, so results measure infrastructure and
deterministic analysis, not trained semantic capability. No commercial
decompiler baseline was run, and behavioral execution remains disabled.
