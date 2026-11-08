# NEXUS-BI

[![ci](https://github.com/ArchLinuxAddict/nexus-bi/actions/workflows/ci.yml/badge.svg)](https://github.com/ArchLinuxAddict/nexus-bi/actions/workflows/ci.yml)

NEXUS-BI is a **tool-first binary intelligence system** with one residual model
for the gaps deterministic tools cannot close. Ghidra, JADX, CFR and native JNI
analysis own reconstruction; the residual model repairs and ranks their output.

## Model

- **Active model:** `residual-v15-joint` (compact profile, ~16.4M parameters).
- **Architecture:** tool-aware transformer with full tool-sequence memory,
  pointer-generator copy attention, whitespace-preserving tokenization, and
  optional candidate/failure/confidence heads.
- **Checkpoint:** [`residual-v15-joint-best.pt`](https://github.com/ArchLinuxAddict/nexus-bi/releases/download/v0.1.0/residual-v15-joint-best.pt)
  (61 MB, SHA-256 `66752628870fa8f375af6b1ff26429742f19fc23fe00e1b89504ee593c07b195`).
- **Model card:** [`MODEL_CARD.md`](MODEL_CARD.md).
- **Status:** repair/ranking research layer. It is **not** an autonomous code
  generator and its output is not compile- or behavior-verified. Deterministic
  tool output remains authoritative.

Held-out JNI/Java repair token accuracy improved from **12.0%** (v11) to
**29.6%** (v14) to **33.0%** (v15). Exact long-form generation is not solved;
when a JNI method and descriptor are supplied, the Java declaration is
canonicalized deterministically.

## Install

Python 3.13 is required.

```bash
python3.13 -m venv .venv
.venv/bin/pip install -e ".[model,dev]"
```

On CPU-only machines, install PyTorch from the CPU wheel index first if needed:

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e ".[model,dev]"
```

## Use

```bash
.venv/bin/nexus tools
.venv/bin/nexus inspect binary /path/to/file
.venv/bin/nexus analyze binary /path/to/file --json
.venv/bin/nexus reconstruct c /path/to/file --out artifacts/custom_recon --verify
.venv/bin/nexus java reconstruct /path/to/app.jar --out artifacts/java_recon --verify
.venv/bin/nexus dashboard
```

Open `http://127.0.0.1:8000`. The **Residual** tab loads the active checkpoint,
shows held-out metrics, and runs inference. The **Reconstruction** tab exposes
Ghidra/JNI reconstruction; the **Java/APK** tab exposes JADX/CFR.

## Get the checkpoint

```bash
mkdir -p checkpoints
curl -L -o checkpoints/residual-v15-joint-best.pt \
  https://github.com/ArchLinuxAddict/nexus-bi/releases/download/v0.1.0/residual-v15-joint-best.pt
printf '%s\n' "$PWD/checkpoints/residual-v15-joint-best.pt" > artifacts/residual_runs/ACTIVE
```

The runtime also falls back to `artifacts/residual_runs/residual-v15-joint/best.pt`.

## Train and evaluate

The training corpus is rebuilt from local artifacts; raw datasets and tool
outputs are not stored in Git.

```bash
.venv/bin/python scripts/build_jni_facade_dataset.py --path /path/to/native-libs --out artifacts/dataset/jni_facade_v1
.venv/bin/python scripts/build_jni_method_dataset.py --out artifacts/dataset/jni_method_all
.venv/bin/python scripts/build_jni_repair_dataset.py --out artifacts/dataset/jni_repair_v1
.venv/bin/python scripts/build_jni_semantic_dataset.py --path /path/to/libnative.so --out artifacts/dataset/jni_semantic_v1
.venv/bin/python scripts/combine_residual_datasets.py

.venv/bin/python scripts/train_residual.py \
  --train artifacts/dataset/jni_joint_v1/train.jsonl \
  --validation artifacts/dataset/jni_joint_v1/validation.jsonl \
  --resume checkpoints/residual-v15-joint-best.pt \
  --profile compact --steps 300 --batch-size 2 --grad-accum 2 \
  --lr 5e-5 --c-every 4 --run-name residual-next

.venv/bin/python scripts/evaluate_residual.py \
  --checkpoint artifacts/residual_runs/residual-next/best.pt \
  --test artifacts/dataset/jni_joint_v1/validation.jsonl
```

`scripts/eval_residual_samples.py` prints per-sample token accuracy, teacher-
forced exact match, generation exact match, and optional generated text.

## Documentation

- [`MODEL_CARD.md`](MODEL_CARD.md) — intended use, data, metrics, risks.
- [`RESIDUAL_MODEL.md`](RESIDUAL_MODEL.md) — architecture, datasets, training.
- [`TRAINING_TARGETS.md`](TRAINING_TARGETS.md) — residual training targets.
- [`TOOL_FIRST_PLAN.md`](TOOL_FIRST_PLAN.md) — architecture and milestones.
- [`CUSTOM_FILES.md`](CUSTOM_FILES.md) — running tools on arbitrary files.
- [`REALWORLD_SO_DATASET.md`](REALWORLD_SO_DATASET.md) — real-world catalog.
- [`CHANGELOG.md`](CHANGELOG.md) — changes.

## Hardware

8 vCPU, 31 GB RAM, no GPU in the reference environment. The compact profile is
CPU-trainable; larger profiles are supported by configuration but were not
trained here.

## License

MIT. See [`LICENSE`](LICENSE). Third-party tool outputs, datasets and native
samples retain their original licenses and are not redistributed here. Use the
tools and model only on artifacts you are authorized to analyze.
