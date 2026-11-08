from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nexus.api.app import create_app
from nexus.core.config import NexusPaths
from nexus.core.engine import Engine
from nexus.storage.store import Storage


@pytest.fixture()
def client(tmp_path):
    paths = NexusPaths(root=tmp_path / "api-home").ensure()
    engine = Engine(paths, Storage(paths))
    return TestClient(create_app(engine))


def _upload(client, path):
    with open(path, "rb") as handle:
        response = client.post("/v1/binaries", files={"file": (Path(path).name, handle, "application/octet-stream")})
    assert response.status_code == 200, response.text
    return response.json()


def test_health_and_capabilities(client):
    assert client.get("/healthz").json()["status"] == "ok"
    capabilities = client.get("/v1/capabilities").json()
    assert capabilities["dynamic_binary_execution"]["status"] == "disabled"
    hardware = client.get("/v1/hardware").json()
    assert hardware["large_training_ready"] is False
    tools = client.get("/v1/tools").json()
    assert tools["schema_version"] == "nexus.tool-matrix/v1"
    assert "ghidra_headless" in tools["tools"]
    assert client.get("/v1/jobs").status_code == 200


def test_upload_analyze_and_inspect(client, simple_binary):
    binary = _upload(client, simple_binary)
    assert binary["sha256"]
    response = client.post("/v1/analyses", json={"binary_id": binary["id"]})
    assert response.status_code == 200, response.text
    analysis = response.json()
    assert analysis["analysis_id"]
    assert analysis["job_id"]
    analysis_id = analysis["analysis_id"]
    job = client.get(f"/v1/jobs/{analysis['job_id']}").json()
    assert job["status"] == "completed"
    functions = client.get(f"/v1/analyses/{analysis_id}/functions").json()
    assert any(name.get("value") == "add" for function in functions for name in function.get("names", []))
    record = client.get(f"/v1/analyses/{analysis_id}/program-record").json()
    assert record["program"]["sha256"] == binary["sha256"]
    assert client.get(f"/v1/analyses/{analysis_id}/cfg").status_code == 200
    assert client.get(f"/v1/analyses/{analysis_id}/callgraph").status_code == 200
    assert client.get(f"/v1/analyses/{analysis_id}/disassembly?limit=10").status_code == 200


def test_reconstruction_and_verification(client, simple_binary):
    binary = _upload(client, simple_binary)
    analysis = client.post("/v1/analyses", json={"binary_id": binary["id"]}).json()
    response = client.post("/v1/reconstructions", json={"analysis_id": analysis["analysis_id"], "language": "c", "function_ids": ["add"]})
    assert response.status_code == 200, response.text
    reconstruction = response.json()
    assert reconstruction["reconstruction_id"]
    assert reconstruction["classification"] == "partial_behavioral_reconstruction"
    assert reconstruction["job_id"]
    verification = client.post(f"/v1/reconstructions/{reconstruction['reconstruction_id']}/verify").json()
    assert verification["status"] == "passed"
    assert verification["compile_rate"] == 1.0
    assert client.get(f"/v1/reconstructions/{reconstruction['reconstruction_id']}/files/src/add.c").status_code == 200


def test_investigation(client, simple_binary):
    binary = _upload(client, simple_binary)
    analysis = client.post("/v1/analyses", json={"binary_id": binary["id"]}).json()
    response = client.post("/v1/investigations", json={"analysis_id": analysis["analysis_id"], "objective": "understand behavior", "max_steps": 5})
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["status"] == "completed"
    assert report["hypotheses"]
    assert client.get(f"/v1/investigations/{report['investigation_id']}").status_code == 200


def test_auth_hook(tmp_path, simple_binary):
    paths = NexusPaths(root=tmp_path / "auth-home").ensure()
    engine = Engine(paths, Storage(paths))
    protected = TestClient(create_app(engine, api_key="secret"))
    assert protected.get("/v1/capabilities").status_code == 200
    assert protected.get("/v1/binaries").status_code == 401
    assert protected.get("/v1/binaries", headers={"X-API-Key": "secret"}).status_code == 200
    with open(simple_binary, "rb") as handle:
        response = protected.post("/v1/binaries", files={"file": ("simple", handle, "application/octet-stream")}, headers={"X-API-Key": "secret"})
    assert response.status_code == 200


def test_dashboard_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "NEXUS-BI" in response.text
    assert client.get("/static/app.js").status_code == 200


def test_java_analyze_binary_returns_summary_only(client, tmp_path):
    from nexus.datasets.apk_factory import build_minimal_android_manifest, build_minimal_apk
    from nexus.datasets.dex_factory import build_minimal_dex

    apk = build_minimal_apk(manifest=build_minimal_android_manifest(), dex=build_minimal_dex())
    path = tmp_path / "fixture.apk"
    path.write_bytes(apk)
    binary = _upload(client, path)
    response = client.post("/v1/java/analyze-binary", json={"binary_id": binary["id"]})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert set(payload) == {"format", "summary"}
    assert payload["format"] == "apk"
    assert payload["summary"]["classes"] == 1
    assert "apk" not in payload
