# Model weights

The active checkpoint is distributed as a GitHub release asset:

- `residual-v15-joint-best.pt` (61 MB)
- SHA-256: `66752628870fa8f375af6b1ff26429742f19fc23fe00e1b89504ee593c07b195`

Download:

```bash
mkdir -p checkpoints
curl -L -o checkpoints/residual-v15-joint-best.pt \
  https://github.com/ArchLinuxAddict/nexus-bi/releases/download/v0.1.0/residual-v15-joint-best.pt
mkdir -p artifacts/residual_runs
printf '%s\n' "$PWD/checkpoints/residual-v15-joint-best.pt" > artifacts/residual_runs/ACTIVE
```

Weights are not committed to Git. See `MODEL_CARD.md` for architecture,
training data, metrics and limitations.
