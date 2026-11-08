import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger("nexus")
if not _LOGGER.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)


def event(kind: str, **fields: Any) -> dict:
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "kind": kind,
        **fields,
    }
    _LOGGER.info(json.dumps(payload, sort_keys=True, default=str))
    return payload


def audit_log(path: str | Path, kind: str, **fields: Any) -> None:
    payload = event(kind, **fields)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
