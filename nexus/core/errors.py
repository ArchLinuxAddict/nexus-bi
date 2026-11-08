class NexusError(Exception):
    """Base class for NEXUS-BI errors."""

    code = "nexus_error"

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}


class ValidationError(NexusError):
    code = "validation_error"


class ElfError(NexusError):
    code = "elf_error"


class DisassemblyError(NexusError):
    code = "disassembly_error"


class CapabilityUnavailable(NexusError):
    code = "capability_unavailable"


class SandboxUnavailable(CapabilityUnavailable):
    code = "sandbox_unavailable"


class ToolUnavailable(NexusError):
    code = "tool_unavailable"


class NotFoundError(NexusError):
    code = "not_found"


class ConflictError(NexusError):
    code = "conflict"


class BudgetExceeded(NexusError):
    code = "budget_exceeded"


class MissionPolicyError(ValidationError):
    code = "mission_policy_error"
