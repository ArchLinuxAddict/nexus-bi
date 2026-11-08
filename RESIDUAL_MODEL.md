# Residual Model

The residual model is the learned layer of the tool-first NEXUS-BI architecture.
It does not replace Ghidra, JADX, CFR or native JNI analysis; it repairs and
ranks their output.

## Architecture

`nexus/foundation/residual_model.py`

- **Program encoder:** the unified foundation model encodes optional instruction
  and block evidence.
- **Tool encoder:** a transformer encodes the full tool-output sequence.
- **Candidate encoder:** optional sequence encoder for a candidate repair.
- **Metadata:** tool, language, status, compile status, warning count and cache
  state are projected and fused with the program/tool representations.
- **Repair decoder:** a transformer decoder with cross-attention over the full
  tool sequence.
- **Copy head:** pointer-generator distribution over vocabulary and tool tokens;
  `copy_labels` supervise copy alignment.
- **Auxiliary heads:** candidate score, failure probability and confidence.
  These are experimental; checkpoints without validated weights return `null`.
- **Whitespace:** the tokenizer preserves whitespace through byte tokens, so
  decode round-trips `public static native ...` exactly.

The active checkpoint is `residual-v15-joint` (compact profile, ~16.4M
parameters total). The checkpoint is distributed as a GitHub release asset, not
committed to Git.

`artifacts/residual_runs/ACTIVE` selects the checkpoint used by the dashboard
and `/v1/residual/infer`. If it is absent, the runtime prefers the v15 release
path and then the newest checkpoint under `artifacts/residual_runs`.

## Datasets

Datasets are rebuilt from your own analysis artifacts; raw data is not stored in
Git. See
`docs/dataset_summaries/` for counts and limitations.

- **C/Java residual pairs:** source/binary builds with Ghidra/JADX/CFR tool
  labels and original/reconstructed targets.
- **JNI facade pairs:** exported `Java_*` methods and `JNINativeMethod` tables
  mapped to exact Java native declarations.
- **JNI method pairs:** one native declaration per sample, including protector-
  style libraries with recovered registrations.
- **JNI repair pairs:** synthetic corrupted-to-correct declaration repairs.
- **JNI semantic pairs:** Java bodies recovered from Ghidra-decompiled JNI call
  sequences.
- **Combined residual corpus:** C, Java and JNI samples with deterministic
  train/validation/test splits.

Builders:

```bash
scripts/build_c_residual_dataset.py
scripts/build_java_residual_dataset.py
scripts/build_jni_facade_dataset.py
scripts/build_jni_method_dataset.py
scripts/build_jni_repair_dataset.py
scripts/build_jni_semantic_dataset.py
scripts/combine_residual_datasets.py
```

## Training

```bash
.venv/bin/python scripts/train_residual.py \
  --train artifacts/dataset/jni_joint_v1/train.jsonl \
  --validation artifacts/dataset/jni_joint_v1/validation.jsonl \
  --resume checkpoints/residual-v15-joint-best.pt \
  --profile compact --steps 300 --batch-size 2 --grad-accum 2 \
  --lr 5e-5 --warmup 20 --val-interval 50 --patience 6 \
  --c-every 4 --threads 8 --run-name residual-next
```

`--resume` loads the config and vocabulary from the checkpoint. Missing optional
heads are reported and remain untrained; the core model is still compatible.

Outputs: `last.pt`, `best.pt`, `final.pt`, `metrics.jsonl`.

## Evaluation

```bash
.venv/bin/python scripts/evaluate_residual.py \
  --checkpoint checkpoints/residual-v15-joint-best.pt \
  --test artifacts/dataset/jni_joint_v1/validation.jsonl

.venv/bin/python scripts/eval_residual_samples.py \
  --checkpoint checkpoints/residual-v15-joint-best.pt \
  --records artifacts/dataset/jni_joint_v1/validation.jsonl \
  --filter <sample-id> --generate
```

Metrics: repair token accuracy, teacher-forced exact match, generated exact
match, candidate/failure accuracy when heads are available.

Held-out JNI repair token accuracy: **12.0% (v11) → 29.6% (v14) → 33.0% (v15)**.
Long-form exact generation remains unsolved; deterministic canonicalization
fixes JNI declaration signatures when method/descriptor evidence is supplied.

## Inference

`/v1/residual/infer` accepts `tool_text`, optional `target_text`,
`candidate_text`, `language`, `tool`, and optional `jni_method` /
`jni_descriptor`. It returns:

- `generated_text`: raw model output (experimental);
- `canonical_text`: deterministic JNI declaration when method/descriptor exist;
- `candidate_exact`: candidate/reference comparison when both exist;
- `teacher_forced_token_accuracy` / `teacher_forced_exact` when a reference is
  supplied;
- auxiliary scores or `null`.

The dashboard **Residual** tab exposes the same flow.

## Known limitations

- Repair token accuracy is not compile or behavioral verification.
- Exact long-form generation is not reliable.
- Descriptor-to-Java type mapping is handled deterministically, not by the model.
- Candidate/failure/confidence heads were not validated; ranking experiments
  did not learn.
- C repair is weaker than Java/JNI repair.
- Runtime-decrypted payloads are not in the static corpus and are not claimed.

## License

MIT. See `LICENSE`.
