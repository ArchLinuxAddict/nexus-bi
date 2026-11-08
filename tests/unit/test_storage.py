from nexus.core.config import NexusPaths
from nexus.storage.store import Storage


def test_cas_and_binary_identity(tmp_path, simple_binary):
    paths = NexusPaths(root=tmp_path).ensure()
    storage = Storage(paths)
    first = storage.add_binary(simple_binary)
    second = storage.add_binary(simple_binary)
    assert first["id"] == second["id"]
    assert first["sha256"] == second["sha256"]
    assert paths.cas.joinpath(first["sha256"][:2], first["sha256"]).exists()
    assert storage.get_binary(first["id"])["sha256"] == first["sha256"]


def test_analysis_and_findings(tmp_path, simple_binary):
    paths = NexusPaths(root=tmp_path).ensure()
    storage = Storage(paths)
    binary = storage.add_binary(simple_binary)
    analysis_id = storage.create_analysis(binary["id"], {"functions": 2}, None)
    analysis = storage.get_analysis(analysis_id)
    assert analysis["id"] == analysis_id
    assert analysis["summary"]["functions"] == 2
    storage.add_findings(binary["id"], [{"finding_id": "sec_1", "title": "test"}])
    findings = storage.list_findings(binary["id"])
    assert findings[0]["title"] == "test"


def test_investigation_storage(tmp_path, simple_binary):
    paths = NexusPaths(root=tmp_path).ensure()
    storage = Storage(paths)
    binary = storage.add_binary(simple_binary)
    report = {"investigation_id": "inv_test", "objective": "test", "status": "completed", "facts": []}
    investigation_id = storage.add_investigation(binary["id"], report)
    assert investigation_id == "inv_test"
    loaded = storage.get_investigation(investigation_id)
    assert loaded["report"]["objective"] == "test"
    assert storage.list_investigations(binary["id"])[0]["id"] == "inv_test"
