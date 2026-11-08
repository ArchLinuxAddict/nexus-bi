# NEXUS-BI Residual Model Card

## Model details

| Field | Value |
|---|---|
| Model name | NEXUS-BI Residual Model |
| Version | `residual-v15-joint` |
| Architecture | Tool-aware transformer with pointer-generator copy |
| Parameters | ~16.4M total; ~15.7M core trained in v15 |
| Profile | `compact` |
| Context | 1,024 tool/source text tokens |
| Checkpoint | `residual-v15-joint-best.pt` (61 MB) |
| SHA-256 | `66752628870fa8f375af6b1ff26429742f19fc23fe00e1b89504ee593c07b195` |
| License | MIT (code and weights) |
| Release | https://github.com/ArchLinuxAddict/nexus-bi/releases/tag/v0.1.0 |

### Architecture

The residual model conditions on tool output text, an optional program
representation, and metadata (tool, language, status, compiler state, warnings,
cache state). The decoder attends to the full tool-token sequence rather than a
pooled vector. A pointer-generator copy head mixes a vocabulary distribution
with attention over tool tokens, and `copy_labels` supervise source alignment.
Whitespace is preserved through byte tokens so generated text remains readable
across old and new vocabularies. Optional candidate/failure/confidence heads are
auxiliary and currently disabled unless a checkpoint contains validated weights.

## Intended use

- Repair and rank JADX/CFR/Ghidra/JNI reconstruction output for C, Java, and
  JNI/Java facades.
- Assist human review of reconstructed code by proposing fixes and confidence
  signals.
- Research on tool-aware code repair and copy mechanisms.

**Not intended for:** autonomous code generation, security-critical decisions,
compilation guarantees, or use on artifacts without authorization.

## Training data

The model was trained on locally built residual corpora:

- **C/Java residual pairs:** 1,341 C and 583 Java samples derived from
  permissively licensed source/binary builds. Tool text comes from Ghidra,
  JADX and CFR; targets are the original or reconstructed source.
- **JNI facade data:** 28 classes / 186 methods recovered from exported
  `Java_*` symbols and `JNINativeMethod` tables in authorized native libraries.
- **JNI method and repair sets:** method-level facades and synthetic
  corrupted-to-correct declaration repairs.
- **JNI semantic pairs:** 19 Java bodies recovered from Ghidra-decompiled JNI
  call sequences.
- **Synthetic corruption:** wrong return/parameter types, modifiers and
  declarations for repair training.

No raw APKs, decrypted third-party payloads, runtime memory dumps, credentials,
or private user data are included in this repository. Native samples and tool
outputs are not redistributed.

## Evaluation

Held-out JNI/Java repair results under the whitespace-preserving tokenizer:

| Checkpoint | Held-out repair token accuracy |
|---|---|
| `v11-jni-all` | 12.0% |
| `v14-copy-ws` | 29.6% |
| `v15-joint` | **33.0%** |

Additional observations:

- Teacher-forced exact match on long declarations is 0% in the current
  validation; token accuracy is not correctness.
- The copy mechanism improves identifier and formatting copying but does not
  reliably map JNI descriptors such as `[B` to `byte[]`.
- Deterministic canonicalization fixes this for JNI declarations:
  `([B)[B` → `public static native byte[] decryptAssetBlob(byte[] arg0);`.
- Candidate/failure/confidence heads trained on mostly positive labels were not
  reliable. Ranking experiments (`v16`, `v17`) did not learn and are inactive.
- C repair remains weaker than Java/JNI repair.

## Limitations

- Generated code may be syntactically or semantically wrong.
- Token accuracy, exact match, and teacher-forced metrics do not establish
  compilation, behavioral equivalence, or security.
- The model can overfit to JNI naming/type patterns in the training data.
- It cannot recover encrypted or runtime-decrypted payloads that were never in
  the training data.
- Auxiliary scores are experimental; `candidate_exact` against a reference is
  authoritative, and canonical JNI signatures are deterministic.
- The released model was trained CPU-only with the compact profile; larger
  models or more diverse data may improve results but were not trained for this
  release.

## Ethical and legal use

Use NEXUS-BI only on binaries and applications you own or are authorized to
analyze. Do not use it to bypass access controls, distribute decrypted
third-party code, or violate software licenses. Model outputs are research
artifacts and must be reviewed by a human.

## Citation

```bibtex
@misc{nexusbi2026residual,
  title  = {NEXUS-BI Residual Model v15},
  author = {Pears, Eliza},
  year   = {2026},
  url    = {https://github.com/ArchLinuxAddict/nexus-bi}
}
```
