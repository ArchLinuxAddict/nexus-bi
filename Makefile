PY := .venv/bin/python
PIP := .venv/bin/pip
NEXUS := .venv/bin/nexus

.PHONY: setup test tools serve doctor benchmark v6-pilot v6-audit clean

setup:
	python3.13 -m venv .venv
	$(PIP) install -e ".[dev,model]"

test:
	$(PY) -m pytest -q

tools:
	$(NEXUS) tools

serve:
	$(NEXUS) dashboard --host 127.0.0.1 --port 8000

doctor:
	$(NEXUS) doctor

benchmark:
	$(NEXUS) benchmark --path /usr/bin/ls --json

v6-pilot:
	$(PY) scripts/ingest_realworld_v6.py --out artifacts/dataset/realworld_v6_pilot2 --max-projects 50 --variants debug-stripped

v6-audit:
	$(PY) scripts/audit_realworld_v6.py --root artifacts/dataset/realworld_v6_pilot2

clean:
	rm -rf artifacts/cas artifacts/records artifacts/reconstructions artifacts/uploads artifacts/nexus.sqlite3
