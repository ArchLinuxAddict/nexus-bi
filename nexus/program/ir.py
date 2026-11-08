import json
from dataclasses import dataclass, field
from typing import Any

from nexus.disassembly.model import Instruction
from nexus.program.bundle import AnalysisBundle
from nexus.program.model import FunctionCandidate

IR_VERSION = "0.1.0"


@dataclass
class IROperand:
    kind: str
    value: str
    access: str | None = None
    width: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "value": self.value, "access": self.access, "width": self.width}


@dataclass
class IRInstruction:
    ir_id: str
    address: int
    block_id: str
    op: str
    destination: list[IROperand]
    sources: list[IROperand]
    targets: list[int]
    memory: list[dict[str, Any]]
    evidence: str
    confidence: float
    raw_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ir_id": self.ir_id,
            "address": f"0x{self.address:x}",
            "block_id": self.block_id,
            "op": self.op,
            "destination": [item.to_dict() for item in self.destination],
            "sources": [item.to_dict() for item in self.sources],
            "targets": [f"0x{target:x}" for target in self.targets],
            "memory": self.memory,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "raw_text": self.raw_text,
        }


@dataclass
class IRBlock:
    block_id: str
    start: int
    end: int
    instruction_ids: list[str]
    successors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "start": f"0x{self.start:x}",
            "end": f"0x{self.end:x}",
            "instruction_ids": self.instruction_ids,
            "successors": self.successors,
        }


@dataclass
class FunctionIR:
    function_id: str
    name: str
    entry: int
    signature: dict[str, Any]
    blocks: list[IRBlock]
    instructions: list[IRInstruction]
    dataflow: dict[str, Any]
    type_ids: list[str]
    evidence: list[str]
    confidence: float
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "function_id": self.function_id,
            "name": self.name,
            "entry": f"0x{self.entry:x}",
            "signature": self.signature,
            "blocks": [block.to_dict() for block in self.blocks],
            "instructions": [instruction.to_dict() for instruction in self.instructions],
            "dataflow": self.dataflow,
            "type_ids": self.type_ids,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "limitations": self.limitations,
        }


def _op_for_instruction(instruction: Instruction) -> str:
    mnemonic = instruction.mnemonic
    if mnemonic in {"mov", "movabs", "movzx", "movsx", "movsxd", "lea", "movbe"}:
        return "move"
    if mnemonic in {"cmp", "test"}:
        return "compare"
    if mnemonic in {"add", "sub", "imul", "and", "or", "xor", "shl", "shr", "sar", "inc", "dec", "neg", "not", "mul", "div", "idiv"}:
        return "arithmetic"
    if instruction.control_flow in {"conditional_branch", "unconditional_branch", "indirect_branch"}:
        return "branch"
    if instruction.control_flow in {"call", "indirect_call"}:
        return "call"
    if instruction.control_flow == "return":
        return "return"
    if instruction.control_flow == "trap":
        return "trap"
    return "other"


def _operand(instruction: Instruction, raw: str, access: str) -> IROperand:
    register = raw.strip()
    from nexus.disassembly.objdump import canonical_register, operand_register_width
    canonical = canonical_register(register)
    if canonical:
        return IROperand(kind="register", value=canonical, access=access, width=operand_register_width(register))
    if "[" in raw or "PTR" in raw.upper():
        width = next((effect.width for effect in instruction.memory if effect.address == raw), None)
        return IROperand(kind="memory", value=raw, access=access, width=width)
    if raw.strip().lstrip("$").lstrip("-").isdigit() or raw.strip().lstrip("$").startswith("0x"):
        return IROperand(kind="immediate", value=raw, access="read", width=None)
    return IROperand(kind="unknown", value=raw, access=access, width=None)


def lower_instruction(instruction: Instruction, block_id: str, binary_sha256: str) -> IRInstruction:
    destination = []
    sources = []
    op = _op_for_instruction(instruction)
    if op in {"compare", "branch", "call", "return", "trap"} or instruction.mnemonic == "push":
        sources.extend(_operand(instruction, operand, "read") for operand in instruction.operands)
    elif instruction.mnemonic == "pop":
        if instruction.operands:
            destination.append(_operand(instruction, instruction.operands[0], "write"))
            sources.extend(_operand(instruction, operand, "read") for operand in instruction.operands[1:])
    else:
        if instruction.operands:
            access = "read_write" if op == "arithmetic" else "write"
            destination.append(_operand(instruction, instruction.operands[0], access))
            sources.extend(_operand(instruction, operand, "read") for operand in instruction.operands[1:])
    return IRInstruction(
        ir_id=f"ir_{instruction.address:x}",
        address=instruction.address,
        block_id=block_id,
        op=op,
        destination=destination,
        sources=sources,
        targets=[target.va for target in instruction.targets if target.va is not None],
        memory=[effect.to_dict() for effect in instruction.memory],
        evidence=f"ev_{binary_sha256[:12]}_ins_{instruction.address:x}",
        confidence=instruction.confidence,
        raw_text=instruction.text,
    )


def build_function_ir(function: FunctionCandidate, dataflow: dict[str, Any], type_candidates: list[dict[str, Any]], binary_sha256: str) -> FunctionIR:
    blocks = []
    instructions = []
    for block in function.blocks:
        block_instructions = [lower_instruction(instruction, block.block_id, binary_sha256) for instruction in block.instructions]
        instructions.extend(block_instructions)
        blocks.append(IRBlock(
            block_id=block.block_id,
            start=block.start,
            end=block.end,
            instruction_ids=[instruction.ir_id for instruction in block_instructions],
            successors=list(block.successors),
        ))
    signature = {}
    if function.entry in {item.entry for item in [function]}:
        signature = {"parameters": len(dataflow.get("argument_candidates", [])), "return_written": dataflow.get("return_written", False)}
    limitations = [
        "IR is a normalized instruction/CFG representation, not a verified compiler IR.",
        "Semantic operations are inferred from mnemonics and operand roles.",
        "Pointer aliasing, indirect memory and flags are not fully modeled.",
    ]
    return FunctionIR(
        function_id=function.function_id,
        name=function.name,
        entry=function.entry,
        signature=signature,
        blocks=blocks,
        instructions=instructions,
        dataflow={
            "def_use_chains": len(dataflow.get("def_use_chains", [])),
            "memory_accesses": len(dataflow.get("memory_accesses", [])),
            "stack_slots": len(dataflow.get("stack_slots", [])),
            "globals": len(dataflow.get("globals", [])),
            "argument_candidates": dataflow.get("argument_candidates", []),
        },
        type_ids=[item["type_id"] for item in type_candidates],
        evidence=[f"ev_{binary_sha256[:12]}_func_{function.entry:x}"],
        confidence=function.confidence,
        limitations=limitations,
    )


def build_program_ir(bundle: AnalysisBundle) -> dict[str, Any]:
    functions = []
    for function in bundle.functions:
        functions.append(build_function_ir(
            function,
            bundle.dataflows.get(function.function_id, {}),
            bundle.type_candidates.get(function.function_id, []),
            bundle.elf.sha256,
        ).to_dict())
    return {
        "schema_version": IR_VERSION,
        "binary_sha256": bundle.elf.sha256,
        "program": {
            "machine": bundle.elf.header.machine_name,
            "elf_class": bundle.elf.header.elf_class,
            "entry": f"0x{bundle.elf.header.entry:x}",
        },
        "functions": functions,
        "limitations": [
            "Normalized binary IR, not LLVM IR or verified source semantics.",
            "Instruction-level relationships are preserved with exact addresses.",
        ],
    }


def serialize_program_ir(ir: dict[str, Any]) -> str:
    return json.dumps(ir, sort_keys=True, indent=2) + "\n"
