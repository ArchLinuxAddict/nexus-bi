import os
import shutil
from dataclasses import dataclass

from nexus.core.errors import SandboxUnavailable


@dataclass
class SandboxStatus:
    available: bool
    reason: str
    kvm: bool
    qemu: str | None
    bubblewrap: str | None

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "reason": self.reason,
            "kvm": self.kvm,
            "qemu": self.qemu,
            "bubblewrap": self.bubblewrap,
            "policy": "dynamic execution is disabled until an isolated, verified KVM worker exists",
        }


def status() -> SandboxStatus:
    kvm = os.path.exists("/dev/kvm")
    qemu = shutil.which("qemu-system-x86_64") or shutil.which("qemu-x86_64")
    bwrap = shutil.which("bwrap")
    reason = "No verified KVM-backed execution worker is implemented; dynamic analysis is disabled."
    if not kvm:
        reason = "KVM is unavailable and no verified isolated execution worker exists; dynamic analysis is disabled."
    return SandboxStatus(available=False, reason=reason, kvm=kvm, qemu=qemu, bubblewrap=bwrap)


def require() -> SandboxStatus:
    sandbox = status()
    if not sandbox.available:
        raise SandboxUnavailable(sandbox.reason, details=sandbox.to_dict())
    return sandbox


def execute(*_args, **_kwargs):
    require()
    raise SandboxUnavailable("Execution backend is intentionally not implemented in this build.")


def differential_test(*_args, **_kwargs) -> dict:
    sandbox = status()
    return {
        "status": "capability_unavailable",
        "reason": sandbox.reason,
        "sandbox": sandbox.to_dict(),
        "executed": False,
    }


def fuzz(*_args, **_kwargs) -> dict:
    sandbox = status()
    return {
        "status": "capability_unavailable",
        "reason": sandbox.reason,
        "sandbox": sandbox.to_dict(),
        "executed": False,
    }
