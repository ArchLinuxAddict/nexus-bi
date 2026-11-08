import json
import tarfile
from pathlib import Path

from nexus.acquisition.pipeline import run_pilot
from nexus.builds.matrix import run_build_matrix


def _text_sha(path: Path) -> str | None:
    from nexus.elf.parser import parse_elf

    elf = parse_elf(path)
    section = elf.section_by_name(".text")
    if section is None:
        return None
    import hashlib

    return hashlib.sha256(elf.raw[section.offset : section.offset + section.size]).hexdigest()


def test_build_matrix_builds_debug_and_stripped_variants(tmp_path: Path) -> None:
    project = tmp_path / "src"
    project.mkdir()
    (project / "LICENSE").write_text(
        "MIT License\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\n"
        "of this software and associated documentation files (the \"Software\"), to deal\n"
        "in the Software without restriction, including without limitation the rights\n"
        "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
        "copies of the Software, and to permit persons to whom the Software is\n"
        "furnished to do so, subject to the following conditions:\n\n"
        "The above copyright notice and this permission notice shall be included in all\n"
        "copies or substantial portions of the Software.\n",
        encoding="utf-8",
    )
    (project / "main.c").write_text("int add(int a,int b){return a+b;}\nint main(void){return add(2,3);}\n", encoding="utf-8")
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(project, arcname="demo-1.0")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "pilot_id": "test_pilot",
                "projects": [
                    {
                        "project_id": "demo_build",
                        "adapter": "local",
                        "repository": "demo/build",
                        "canonical_url": "https://example.invalid/demo",
                        "commit": "0123456789abcdef",
                        "category": "utilities",
                        "expected_license": "MIT",
                        "extra": {"archive_path": str(archive)},
                        "build_recipe": {"kind": "executable", "sources": ["main.c"], "auto_include": False},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset"
    run_pilot(config, dataset, tmp_path / "reports", historical_dataset=None, historical_index_path=None, swh_check=False)
    report = run_build_matrix(dataset, matrix="pilot", jobs=1, reproduce=False)
    assert report["builds_built"] == 10
    assert report["builds_failed"] == 0
    assert report["by_compiler"] == {"clang": 5, "gcc": 5}
    debug = dataset / "builds" / "demo_build" / "gcc_O2_debug_pie_executable" / "build_metadata.json"
    stripped = dataset / "builds" / "demo_build" / "gcc_O2_stripped_pie_executable" / "build_metadata.json"
    assert debug.exists() and stripped.exists()
    debug_metadata = json.loads(debug.read_text(encoding="utf-8"))
    stripped_metadata = json.loads(stripped.read_text(encoding="utf-8"))
    assert debug_metadata["elf"]["stripped"] is False
    assert stripped_metadata["elf"]["stripped"] is True
    assert debug_metadata["elf"]["text_sha256"] == stripped_metadata["elf"]["text_sha256"]
    assert debug_metadata["extras"]["assembly"]
    assert debug_metadata["extras"]["objects"]
    assert (dataset / "catalog.json").exists()
