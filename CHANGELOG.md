# Changelog

## 2026-11-08 - v0.1.0 model release

- Published `residual-v15-joint` as a release asset with SHA-256 checksum.
- Added `MODEL_CARD.md`, dataset summaries, CI, and reproducible training and
  evaluation commands.
- Active checkpoint is selected by `artifacts/residual_runs/ACTIVE`.
- Added deterministic JNI declaration canonicalization and `candidate_exact`
  reporting; unvalidated auxiliary heads return `null`.
- Full test suite and Ruff pass.

## 2026-11-07 - Residual model: copy and whitespace

- Fixed the tokenizer dropping whitespace; encoded code round-trips exactly.
- Decoder now attends to the full tool sequence instead of a pooled vector.
- Added pointer-generator copy attention and `copy_labels` supervision.
- Copy labels point to full source positions; all-ignored copy batches no
  longer produce NaN.
- Added teacher-forced and generated exact-match metrics.
- Trained `v13-copy`, `v14-copy-ws` and `v15-joint`; held-out JNI repair token
  accuracy rose from 12.0% (v11) to 29.6% (v14) to 33.0% (v15).
- Candidate/failure negative experiments (`v16`, `v17`) did not learn and are
  inactive.
- Added deterministic JNI canonicalization: `([B)[B` ->
  `public static native byte[] decryptAssetBlob(byte[] arg0);`.
- Runtime and evaluators load legacy and copy checkpoints compatibly.

## 2026-11-07 - JNI semantic reconstruction

- Native `.so` Java reconstruction parses Ghidra-decompiled JNI call sequences
  and emits recovered Java behavior: installed-app list filtering, hidden-
  package guards, string-contains checks, Activity/Window dispatch, lifecycle
  calls, reflection wrappers, fields and constructors.
- JNI symbols decode parameters, arrays and constructors correctly.
- Dynamically registered methods (`JNI_OnLoad` + `RegisterNatives`) are
  recovered from static and stack-built `JNINativeMethod` tables.
- Native Java output is labelled `jni_semantic_reconstruction` or
  `jni_bridge_reconstruction`; evidence includes templates, confidence,
  operations and limitations.

## 2026-11-07 - Java reconstruction quality

- JADX partial runs (non-zero exit with emitted sources) are accepted as
  `completed_with_errors` instead of discarded.
- JADX runs with `--show-bad-code` and 8 threads; output is cleaned and cached
  safely.
- App-package classes are ranked first; the API returns a bounded payload while
  the full JADX tree remains on disk.
- `/v1/java/analyze-binary` returns only the summary by default; the previous
  full APK payload could be hundreds of MB.
- Java compile verification reports `unavailable` when Android/Kotlin
  dependencies are absent instead of a misleading `failed`.

## 2026-11-06 - Tool-first reset

- Ghidra, JADX and CFR are the default reconstruction backends.
- The residual model was reduced to a repair layer.
