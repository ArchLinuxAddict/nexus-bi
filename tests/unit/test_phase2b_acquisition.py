import json
import tarfile
from pathlib import Path

import pytest

from nexus.acquisition.adapters import DebianAdapter
from nexus.acquisition.dedup import compare_against_index, signature_from_tree
from nexus.acquisition.licensing import detect_license
from nexus.acquisition.models import SourceCandidate
from nexus.acquisition.pipeline import AcquisitionError, run_pilot, safe_extract_tar

MIT_TEXT = """MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""


def _project(root: Path, name: str = "demo") -> Path:
    project = root / name
    project.mkdir()
    (project / "LICENSE").write_text(MIT_TEXT, encoding="utf-8")
    (project / "main.c").write_text("int add(int a, int b) { return a + b; }\nint main(void) { return add(2, 3); }\n", encoding="utf-8")
    return project


def _archive(root: Path, source: Path, name: str = "demo.tar.gz") -> Path:
    archive = root / name
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname=f"{source.name}-1.0")
    return archive


def test_detect_license_accepts_mit(tmp_path: Path) -> None:
    project = _project(tmp_path)
    evidence = detect_license(project, expected="MIT")
    assert evidence.status == "accepted"
    assert evidence.spdx == "MIT"
    assert evidence.sha256


def test_detect_license_quarantines_gpl_and_unknown(tmp_path: Path) -> None:
    gpl = tmp_path / "gpl"
    gpl.mkdir()
    (gpl / "COPYING").write_text("GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007", encoding="utf-8")
    assert detect_license(gpl).status == "quarantined"
    unknown = tmp_path / "unknown"
    unknown.mkdir()
    (unknown / "LICENSE").write_text("All rights reserved. Contact the author for terms.", encoding="utf-8")
    assert detect_license(unknown).status == "quarantined"


def test_detect_license_reports_conflict(tmp_path: Path) -> None:
    project = tmp_path / "conflict"
    project.mkdir()
    (project / "LICENSE").write_text(MIT_TEXT, encoding="utf-8")
    (project / "COPYING").write_text("GNU GENERAL PUBLIC LICENSE Version 2, June 1991", encoding="utf-8")
    evidence = detect_license(project)
    assert evidence.status == "quarantined"


def test_safe_extract_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz"
    payload = tmp_path / "payload.txt"
    payload.write_text("x", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(payload, arcname="../escape.txt")
    with pytest.raises(AcquisitionError):
        safe_extract_tar(archive, tmp_path / "out")


def test_local_acquisition_is_resumable_and_preserves_quarantine(tmp_path: Path) -> None:
    project = _project(tmp_path)
    archive = _archive(tmp_path, project)
    gpl_project = tmp_path / "gpl"
    gpl_project.mkdir()
    (gpl_project / "COPYING").write_text("GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007", encoding="utf-8")
    (gpl_project / "main.c").write_text("int main(void){return 0;}\n", encoding="utf-8")
    gpl_archive = _archive(tmp_path, gpl_project, "gpl.tar.gz")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "pilot_id": "test_pilot",
                "projects": [
                    {
                        "project_id": "demo_local",
                        "adapter": "local",
                        "repository": "demo/local",
                        "canonical_url": "https://example.invalid/demo",
                        "commit": "0123456789abcdef",
                        "category": "utilities",
                        "expected_license": "MIT",
                        "extra": {"archive_path": str(archive)},
                        "build_recipe": {"kind": "executable", "sources": ["main.c"], "auto_include": False},
                    },
                    {
                        "project_id": "gpl_local",
                        "adapter": "local",
                        "repository": "demo/gpl",
                        "canonical_url": "https://example.invalid/gpl",
                        "commit": "0123456789abcdef",
                        "category": "utilities",
                        "extra": {"archive_path": str(gpl_archive)},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "dataset"
    reports = tmp_path / "reports"
    first = run_pilot(config, out, reports, max_projects=2, historical_dataset=None, historical_index_path=None, swh_check=False)
    second = run_pilot(config, out, reports, max_projects=2, historical_dataset=None, historical_index_path=None, swh_check=False)
    assert first["accepted"] == 1
    assert first["quarantined"] == 1
    assert second["accepted"] == 1
    assert second["quarantined"] == 1
    catalog = json.loads((out / "catalog.json").read_text(encoding="utf-8"))
    assert catalog[0]["build_recipe"]["kind"] == "executable"
    assert (out / "corpus" / "demo_local" / "source.tar.gz").exists()


def test_workspace_marker_refuses_foreign_directory(tmp_path: Path) -> None:
    project = _project(tmp_path)
    archive = _archive(tmp_path, project)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "pilot_id": "test_pilot",
                "projects": [
                    {
                        "project_id": "demo_local",
                        "adapter": "local",
                        "repository": "demo/local",
                        "canonical_url": "https://example.invalid/demo",
                        "commit": "0123456789abcdef",
                        "category": "utilities",
                        "expected_license": "MIT",
                        "extra": {"archive_path": str(archive)},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "dataset"
    out.mkdir()
    (out / "historical.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(AcquisitionError):
        run_pilot(config, out, tmp_path / "reports", historical_dataset=None, historical_index_path=None, swh_check=False)


def test_deduplication_exact_and_unique(tmp_path: Path) -> None:
    first = _project(tmp_path, "first")
    _archive(tmp_path, first, "first.tar.gz")
    signature = signature_from_tree(first)
    index = {"projects": [{"project_id": "first", "name": "first", "archive_sha256": "abc", "signature": signature, "file_hashes": {}}]}
    duplicate = compare_against_index(archive_sha256="abc", signature=signature, file_hashes={}, index=index)
    assert duplicate["status"] == "duplicate"
    unique = compare_against_index(archive_sha256="def", signature=[0] * 64, file_hashes={}, index=index)
    assert unique["status"] in {"unique", "watch"}


def test_debian_adapter_resolves_orig_archive_from_synthetic_index(tmp_path: Path) -> None:
    import lzma

    index = """Package: demo
Version: 1.0-1
Directory: pool/main/d/demo
Files:
 0123456789abcdef0123456789abcdef 10 demo_1.0.orig.tar.gz
 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 5 demo_1.0-1.debian.tar.xz
Checksums-Sha256:
 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef 10 demo_1.0.orig.tar.gz
 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 5 demo_1.0-1.debian.tar.xz
"""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "debian-stable-Sources.xz").write_bytes(lzma.compress(index.encode()))
    adapter = DebianAdapter(cache_dir=cache)
    candidate = SourceCandidate(project_id="demo_debian", adapter="debian", repository="demo", canonical_url="", category="utilities", extra={"package": "demo", "suite": "stable"})
    resolved = adapter.resolve(candidate)
    assert resolved.metadata["orig"]["name"] == "demo_1.0.orig.tar.gz"
    assert resolved.metadata["orig"]["sha256"].startswith("012345")
    assert resolved.commit == "1.0-1"
