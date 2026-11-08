import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class ToolInfo:
    name: str
    path: str | None
    version: str | None
    available: bool

    def to_dict(self) -> dict:
        return {"name": self.name, "path": self.path, "version": self.version, "available": self.available}


def _version(command: str, args: list[str]) -> str | None:
    try:
        result = subprocess.run([command, *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = result.stdout.strip() or result.stderr.strip()
    return text.splitlines()[0] if text else None


def detect_toolchains() -> dict[str, ToolInfo]:
    tools = {}
    for name, args in (
        ("gcc", ["--version"]), ("clang", ["--version"]),
        ("javac", ["-version"]), ("java", ["-version"]),
        ("mvn", ["-version"]), ("gradle", ["--version"]),
        ("make", ["--version"]), ("cmake", ["--version"]),
    ):
        path = shutil.which(name)
        tools[name] = ToolInfo(name=name, path=path, version=_version(name, args) if path else None, available=path is not None)
    return tools


def available_compilers() -> list[str]:
    tools = detect_toolchains()
    return [name for name in ("gcc", "clang") if tools[name].available]


def available_java() -> bool:
    return detect_toolchains()["javac"].available


def maven_available() -> bool:
    return detect_toolchains()["mvn"].available


def gradle_available() -> bool:
    return detect_toolchains()["gradle"].available
