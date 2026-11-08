import json
import tarfile
from pathlib import Path

import pytest

from nexus.datasets.dwarf_source import extract_source_labels

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "artifacts/dataset/realworld_v6_pilot/samples/real_woltapp_blurhash_gcc_O0_debug"
ARCHIVE = ROOT / "artifacts/dataset/realworld_v6_pilot/sources/woltapp_blurhash.tar.gz"


def test_dwarf_source_labels_from_real_build(tmp_path):
    if not (SAMPLE / "dwarf_info.txt").exists() or not ARCHIVE.exists():
        pytest.skip("v6 debug sample is not available")
    with tarfile.open(ARCHIVE, "r:*") as handle:
        handle.extractall(tmp_path)
    roots = [path for path in tmp_path.iterdir() if path.is_dir()]
    source_root = roots[0] if len(roots) == 1 else tmp_path
    record = json.loads((SAMPLE / "program_record.json").read_text(encoding="utf-8"))
    labels = extract_source_labels(record, source_root, (SAMPLE / "dwarf_info.txt").read_text(encoding="utf-8", errors="replace"))
    assert labels
    first = next(iter(labels.values()))
    assert first["alignment"] == "dwarf"
    assert first["source_sha256"]
    assert first["text"].strip()
