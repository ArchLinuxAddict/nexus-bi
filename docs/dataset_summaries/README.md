# Dataset summaries

These JSON files record the locally built residual/JNI corpora used for the
released model. Raw samples, binaries and tool outputs are **not** committed.

| Summary | Contents |
|---|---|
| `residual_v6.json` | Combined C/Java/JNI residual corpus counts and splits. |
| `jni_facade_v1.json` | JNI class/method facades recovered from symbols and registration tables. |
| `jni_method_all.json` | Method-level JNI samples with repeats and general residual data. |
| `jni_repair_v1.json` | Synthetic corrupted-to-correct declaration repairs. |
| `jni_semantic_v1.json` | JNI call-sequence to Java semantic pairs. |

Rebuild commands are in `README.md` and `RESIDUAL_MODEL.md`. Counts can change
when local samples are added; compare checkpoints only on the same split.
