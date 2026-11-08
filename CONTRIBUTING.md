# Contributing

- Run `ruff check nexus tests scripts` and `pytest tests/unit -q` before a PR.
- Keep deterministic tool output authoritative; the residual model is a repair
  layer and its output must not be presented as verified code.
- Do not commit model weights, datasets, APK/native samples, credentials, logs,
  or third-party decrypted content.
- Report metrics with the dataset and split used; do not compare checkpoints on
  different validation sets.
