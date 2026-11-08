# Real-World Shared-Object Catalog

**Catalog:** `artifacts/dataset/realworld_so/catalog.jsonl`
**Purpose:** Tier C unlabeled real-world data for the new tool-first model.

## Catalog

| Item | Value |
|---|---:|
| Unique shared objects | 7,011 |
| Total bytes | 3,344,200,909 |
| x86-64 / x86 | 7,002 / 9 |
| Stripped / unstripped | 6,476 / 535 |
| System packages | 6,382 |
| Python wheels | 37 |

The catalog is deduplicated by device/inode and SHA-256. It records package,
version, license metadata, architecture, stripped state, symbols, dependencies
and a deterministic split bucket.

## Extracted samples

The catalog has extracted function samples under
`artifacts/dataset/realworld_so/samples` (6,338 sample files, 279,177 function
samples). These were previously folded into the removed v5 corpus; the raw
sample files remain available for the v6 tool-first corpus.

## Limitations

- Binary-only samples; no source reconstruction labels.
- Package license metadata is recorded but not legally re-verified.
- Raw system binaries are analyzed in place and are not redistributed.
- This is Tier C data, not the primary source+binary tier.

The v6 source+binary tier comes from `artifacts/dataset/realworld_v6_pilot2`.
