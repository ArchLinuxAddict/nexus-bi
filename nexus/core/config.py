import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class NexusPaths:
    root: Path = field(default_factory=lambda: Path(os.environ.get("NEXUS_HOME", Path.cwd())).resolve())
    artifacts: Path | None = None
    cas: Path | None = None
    database: Path | None = None
    models: Path | None = None

    def __post_init__(self) -> None:
        self.artifacts = self.artifacts or self.root / "artifacts"
        self.cas = self.cas or self.artifacts / "cas"
        self.database = self.database or self.artifacts / "nexus.sqlite3"
        self.models = self.models or self.root / "models"

    def ensure(self) -> "NexusPaths":
        for path in (self.artifacts, self.cas, self.models):
            path.mkdir(parents=True, exist_ok=True)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        return self


def default_paths() -> NexusPaths:
    return NexusPaths().ensure()
