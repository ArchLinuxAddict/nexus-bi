import pytest

from nexus.core.errors import SandboxUnavailable
from nexus.sandbox.gate import execute, require, status


def test_sandbox_rejection():
    sandbox = status()
    assert sandbox.available is False
    with pytest.raises(SandboxUnavailable):
        require()
    with pytest.raises(SandboxUnavailable):
        execute("anything")
