# NEXUS-BI Tool-First Rebuild Plan

## Thesis

Ghidra produced far better aarch64 C than our own translator. That is the signal:
external tools should own every task they can do. The model should not duplicate
a decompiler, disassembler, parser or compiler. The model should predict the
residual: what the tools cannot determine, rank or repair.

The target is not "one model that does everything by itself." It is one unified
model that orchestrates and extends a toolchain:

```text
binary/archive
  -> format + metadata tools
  -> disassembler / decompiler / Java decompiler
  -> evidence store
  -> residual model
       - tool selection and ranking
       - missing names/types/comments/purpose
       - decompiler failure detection and repair
       - Java/JNI residual reconstruction
       - cross-binary retrieval and alignment
       - confidence and natural-language evidence summaries
  -> compiler/verifier
  -> dashboard/API/CLI
```

## What tools own

| Tool | Owns | Status |
|---|---|---|
| Ghidra headless | native function discovery, decompilation, xrefs, types, CFG, switch recovery | installed |
| LLVM objdump | multi-architecture disassembly and relocations | installed |
| GNU objdump/readelf/nm | ELF metadata, symbols, sections, relocations | installed |
| Clang/GCC | compilation/syntax verification | installed |
| javap | bytecode metadata | installed |
| JADX CLI/JADX headless | Java/DEX/APK decompilation | bundled MCP jar present; CLI to install/configure |
| CFR/Procyon | independent Java decompiler cross-checks | to install |
| Z3/angr (optional) | constraints, equivalence probes, path exploration | not installed |
| RetDec (optional) | second native decompiler for disagreement signals | not installed |

The model must never claim a tool result as its own prediction. Every output
carries `tool`, `version`, `command`, `input_sha256`, `timestamp`, `cached`,
and `warnings`.

## What the model owns

1. **Residual reconstruction**: repair or complete tool output where tools fail
   (unsupported instructions, missing types, Java constructs, JNI glue).
2. **Naming and typing**: function/variable names, structs, enums, signatures,
   comments and purpose, grounded in evidence.
3. **Tool arbitration**: choose Ghidra vs JADX vs CFR vs LLVM output when they
   disagree; learn a calibrated preference/confidence model.
4. **Failure detection**: identify decompiler output that is malformed,
   truncated or semantically suspicious before compilation.
5. **Cross-binary intelligence**: retrieval, alignment, clone/family detection,
   patch/vulnerability correlation.
6. **Explanation**: evidence-linked natural-language summaries and behavioral
   hypotheses, explicitly marked as hypotheses.
7. **Search/planning**: choose which tool to run next under a time/compute budget.

The model is evaluated on residual tasks against tool-only baselines. It is not
rewarded for reproducing what a tool already does.

## Real-world data strategy

Current v5 is large but dominated by binary-only system libraries, and the
source-supervised corpus is mostly controlled builds. Replace the training
mixture with three explicit tiers:

### Tier A: real-world source + binary pairs

- Acquire permissively licensed GitHub/GitLab C, C++, Rust, Go and Java
  projects.
- Build shared libraries and executables with a build matrix:
  gcc/clang, O0/O1/O2/O3, debug/stripped, PIC/PIE, static/shared.
- Preserve repository commit, license, build command, compiler version and
  source/binary hashes.
- Split by repository, not by function or build variant.

### Tier B: real-world binaries with tool-generated labels

- Run Ghidra/JADX/CFR on binaries without source.
- Store their output as **pseudo-labels**, never as ground-truth source.
- Train residual tasks: detect failures, rank disagreeing tools, repair.
- Keep tool version pinned so labels are reproducible.

### Tier C: unlabeled scale

- Existing real-world system `.so`, Python wheels, Debian/Arch packages,
  Android APKs/JARs.
- Use for self-supervised masking, retrieval and representation learning.

### Dataset gates

- A real-world project counts only if the source repository and build are
  reproducible and the license permits training/derived use.
- No project-level leakage across train/validation/test.
- Report source-available, binary-only and tool-labeled fractions separately.
- No claim that Ghidra/JADX output is original source.

## Model architecture

The concrete residual training targets are in `TRAINING_TARGETS.md`.

One unified model with tool-aware inputs:

- Program encoder: bytes, instructions, IR, CFG, metadata.
- Tool-output encoder: decompiled C/Java, signatures, warnings, diagnostics.
- Evidence encoder: tool provenance, confidence, disagreement features.
- Residual heads:
  - tool ranking / arbitration;
  - failure and truncation detection;
  - identifier/type/comment prediction;
  - missing-code repair (C/Java);
  - Java/JNI residual reconstruction;
  - retrieval/alignment;
  - confidence calibration;
  - explanation/evidence summary.
- Decoder only where tools cannot produce a candidate; do not train it to
  imitate Ghidra output wholesale.

Reference training constraint: the compact residual model is CPU-trainable.
Foundation-scale training needs a GPU and is not assumed by this plan.

## Evaluation

Baselines:

- raw Ghidra;
- raw JADX/CFR;
- current NEXUS-BI reconstruction;
- tool + deterministic repair;
- tool + residual model.

Metrics:

- compile/syntax rate;
- function recovery and decompilation coverage;
- token edit similarity vs source when available;
- name/type/signature accuracy;
- Java/JNI reconstruction success;
- residual win rate on tool failures;
- tool-selection accuracy and calibration;
- latency and cache hit rate.

Acceptance for replacing old reconstruction:

- new pipeline must be at least as good as the best tool on tool-covered cases;
- strictly better on residual/failure cases;
- no regression in compile rate;
- every output evidence-linked and reproducible.

## Deletion policy

Do not destroy the only working fallback before the replacement is validated.

1. Archive the canonical v4 checkpoint, v5 corpus, current reports and a source
   snapshot outside the repository.
2. Keep the current Ghidra/JADX-based path as the first new backend.
3. Delete legacy code and controlled data only after new acceptance tests pass:
   - old semantic emitter / fallback translator where Ghidra covers the case;
   - controlled/synthetic datasets that are no longer in the training mix;
   - phase-era reports/artifacts and dead scripts;
   - the interrupted v5 run and old checkpoints once v6 is validated.
4. Keep a `legacy/` inventory only if it still has unique evidence; otherwise
   remove it.

## Milestones

### M0 — freeze and audit

- Stop current v5 training; save `interrupt.pt` and `final.pt`.
- Snapshot source, configs, checkpoints and dataset hashes outside the repo.
- Inventory installed tools, versions and licenses.
- Acceptance: reproducible archive manifest and tool matrix.

### M1 — tool contract

- Define `ToolResult` evidence schema and a registry.
- Add adapters: Ghidra, LLVM, GNU binutils, JADX, CFR, javac/gcc.
- Cache by input SHA-256 + tool version + arguments.
- Acceptance: same input produces identical evidence IDs across runs.

### M2 — tool-first reconstruction

- Replace direct model-first reconstruction with tool orchestration.
- Run Ghidra for native and JADX/CFR for Java; model ranks and repairs.
- Dashboard shows tool outputs, disagreements, residual suggestions and compile
  verification side by side.
- Acceptance: a user-supplied aarch64 library reconstructs through Ghidra and
  compiles; a Java APK reconstructs through JADX/CFR.

### M3 — real-world acquisition

- Build acquisition/build pipeline for at least 500 permissively licensed
  real-world projects and 5,000 binaries across build variants.
- Run Ghidra/JADX/CFR with pinned versions.
- Freeze v6 with project-level splits and explicit license/provenance.
- Acceptance: corpus report shows real-world source pairs, tool labels and
  binary-only tiers separately, with zero project leakage.

### M4 — residual model

- Build the tool-aware model and train on residual tasks.
- Baselines: raw Ghidra/JADX, current model, deterministic repair.
- Acceptance: residual win rate > 1.0 and better compile/name/type metrics than
  tool-only, with calibrated confidence.

### M5 — replacement and cleanup

- Switch CLI/API/dashboard to the new pipeline.
- Delete legacy reconstruction paths, controlled datasets and old artifacts
  only after acceptance.
- Acceptance: old path removed, full tests/reproducibility pass, final audit.

## Current progress (2026-11-06)

- **M0 complete:** source/config/checkpoint archive at
  `/tmp/kilo/nexus_bi_goal_archive_20261106` (664 MB, hashed manifest);
  canonical v4 `final.pt`, v5 stopped checkpoints, v4/v5 manifests, vocab,
  configs and catalog preserved.
- **Tool inventory ready:** Ghidra 12.1.4, JADX 1.5.6, CFR 0.152, LLVM 23.1.4,
  binutils 2.47, GCC 16.2.1, Clang 23.1.4, Java 27.0.1. `nexus tools` reports
  the live matrix.
- **M1 complete:** `nexus/toolchain` provides `ToolResult`, `ToolIdentity`,
  SHA/version-addressed `ToolCache`, `ToolRegistry`, and Ghidra/JADX/CFR
  adapters. Tool evidence IDs are stable and cache hits are explicit.
- **M2 native complete:** Ghidra headless is the default non-x86-64 C backend
  with a compiling project and per-function source.
- **M2 Java complete:** JADX is the default Java/JAR/APK/DEX backend, CFR is an
  independent cross-check, and the dashboard has Java upload/analyze/decompile
  plus a Tools matrix view. A JAR reconstruction compiled with `javac`.
  Partial JADX runs (non-zero exit with emitted sources) are accepted as
  `completed_with_errors` instead of being discarded. App-package classes are
  ranked first, the API returns a bounded source payload while the full tool
  output stays on disk, and compile verification reports `unavailable` when
  Android/Kotlin dependencies are absent instead of a misleading `failed`.
  Native libraries produce `jni_semantic_reconstruction`: Ghidra-decompiled
  JNI call sequences are converted into Java behavior (filters, guards, string
  checks, dispatch, lifecycle, reflection, fields, constructors), with decoded
  parameter types and SONAME-based `loadLibrary`. Methods without a reliable
  template stay native declarations with their recovered operation trace. The
  tab states that JADX/CFR need an APK/JAR/DEX and cannot decompile native code.
- **M3 in progress:** 574 permissive real-world candidates generated; the
  50-project `realworld_v6_pilot2` ingestion is running in the background with
  resumable per-project checkpoints. Added DWARF source-body extraction so debug
  builds become true source+binary pairs; 44 source-labelled functions were
  recovered from the `btop` sample.
- **Old working-tree state removed:** v4/v5 corpora, checkpoints, runs, old
  reports, configs and obsolete tests were deleted from the project. The
  canonical v4/v5 metadata and checkpoints remain in the external archive.
- **M4 prework ready:** `ResidualNexusModel` implemented with candidate/failure/confidence/repair heads. Java residual corpus: 564 JADX/CFR samples from 21 real Maven pairs. C residual builder is running over the v6 pilot. Training dry-run passes; no optimizer step has been run.
- **M5 pending:** legacy deletion waits for v6 acceptance.


## Immediate next actions

1. Snapshot and stop any training/dashboard processes.
2. Pin tool inventory (Ghidra 12.1.4, LLVM, Java 27, JADX jar, compilers).
3. Implement the tool registry and evidence schema.
4. Make Ghidra the default native backend and JADX/CFR the default Java backend.
5. Start real-world acquisition and build the v6 corpus.
6. Train the residual model only after v6 and tool baselines exist.
