from typing import Any

TASKS: dict[str, dict[str, Any]] = {
    "cross_compiler_function_matching": {
        "input": "Two function descriptors from builds of the same project with different compilers and the same optimization level. Descriptors contain binary hash, address range, size, instruction count, block count and mnemonic histogram; function names and source locations are excluded from the input.",
        "target": "Binary match label between the two functions.",
        "label_source": "DWARF function identity (source file + declaration line + name) corroborated by symbol-table boundaries.",
        "required_alignment": "function",
        "minimum_classification": "DEBUG_DERIVED",
        "known_shortcuts": [
            "Compiler-family instruction distributions can identify the build without matching the function.",
            "Instruction count and size alone can produce above-chance pairs on same-project functions.",
            "Shared runtime/helper functions can create repeated labels.",
        ],
        "leakage_risks": [
            "Source file, function name, address and build identity must not be model inputs.",
            "Positive and negative pairs must come from the same project and split.",
        ],
        "metrics": ["Recall@1", "Recall@5", "MRR", "pair accuracy", "random-pair baseline", "instruction-count baseline"],
        "baselines": ["random pair", "instruction-count similarity", "mnemonic-histogram cosine", "name oracle (leaked; diagnostic only)"],
        "eligibility_rule": "At least two compilers and one optimization level with aligned functions in the same project.",
    },
    "cross_optimization_function_matching": {
        "input": "Two function descriptors from builds of the same project and compiler at different optimization levels.",
        "target": "Binary match label between the two functions.",
        "label_source": "DWARF function identity (source file + declaration line + name) corroborated by symbol-table boundaries.",
        "required_alignment": "function",
        "minimum_classification": "DEBUG_DERIVED",
        "known_shortcuts": [
            "Optimization level can be inferred from instruction count and block count.",
            "Unoptimized builds expose stack-frame patterns that make matching easier.",
            "Function size alone can be predictive.",
        ],
        "leakage_risks": [
            "Optimization level and compiler identity must not be inputs.",
            "Paired functions must not share binary hashes or addresses.",
        ],
        "metrics": ["Recall@1", "Recall@5", "MRR", "pair accuracy", "random-pair baseline", "instruction-count baseline"],
        "baselines": ["random pair", "instruction-count similarity", "mnemonic-histogram cosine", "optimization-oracle (diagnostic only)"],
        "eligibility_rule": "Same compiler and at least two optimization levels with aligned functions in the same project.",
    },
    "source_to_binary_alignment": {
        "input": "Binary function address range and binary hash.",
        "target": "Source file, function name and declaration line.",
        "label_source": "DWARF subprogram declaration plus line table; symbol boundaries corroborate function ranges.",
        "required_alignment": "function and instruction",
        "minimum_classification": "DEBUG_DERIVED",
        "known_shortcuts": [
            "Function names can leak source structure.",
            "Source paths can encode project identity.",
            "Instruction-level samples are line-table coverage, not token-level AST mapping.",
        ],
        "leakage_risks": [
            "Source text and source paths must not be model inputs for function alignment.",
            "Do not mix project splits.",
        ],
        "metrics": ["function mapping accuracy", "source-file accuracy", "line-start accuracy", "instruction line precision", "instruction coverage"],
        "baselines": ["symbol-only mapping", "random line within file", "entry-only mapping"],
        "eligibility_rule": "DWARF-derived functions or symbol-derived functions; instruction-level eligibility requires line-table coverage.",
    },
    "function_boundary_detection": {
        "input": "Stripped binary bytes and executable ranges.",
        "target": "Function start and end addresses.",
        "label_source": "Debug build function boundaries transferred only when the .text section is byte-identical to the stripped input.",
        "required_alignment": "function",
        "minimum_classification": "DEBUG_DERIVED",
        "known_shortcuts": [
            "Symbol names in a debug companion would leak labels and are not inputs.",
            "PLT/thunk/startup functions can dominate counts.",
            "Padding and alignment can reveal function starts.",
        ],
        "leakage_risks": [
            "Stripped inputs must not contain a symbol table or DWARF.",
            "Debug companion files must not be available to the model.",
        ],
        "metrics": ["precision", "recall", "F1", "random-address baseline", "entry-only baseline"],
        "baselines": ["random executable addresses", "ELF entry point only", "nexus structural recovery"],
        "eligibility_rule": "A stripped build with a byte-identical debug companion or an explicit symbol table.",
    },
    "basic_block_correspondence": {
        "input": "Function binary range and basic-block instruction sequences.",
        "target": "Source line span for the block.",
        "label_source": "Intersection of recovered CFG blocks with DWARF line-table ranges.",
        "required_alignment": "basic block",
        "minimum_classification": "HEURISTIC",
        "known_shortcuts": [
            "Line-table ranges are compiler-generated and can over-approximate.",
            "Block boundaries depend on the CFG recovery heuristic.",
        ],
        "leakage_risks": [
            "Block ordering can leak source control-flow structure.",
            "Do not use debug files as model input.",
        ],
        "metrics": ["block line precision", "containment rate", "coverage", "random-line baseline"],
        "baselines": ["random line within function", "single-line function mapping"],
        "eligibility_rule": "DWARF line mappings and recovered CFG blocks for the same function.",
    },
    "instruction_semantics": {
        "input": "Instruction token sequence with one masked instruction.",
        "target": "Masked mnemonic/operand token.",
        "label_source": "Self-supervised input tokens; not a semantic category.",
        "required_alignment": "instruction token sequence",
        "minimum_classification": "UNVERIFIED",
        "known_shortcuts": [
            "Instruction sequences are highly regular; adjacent tokens predict the mask.",
            "Compiler peephole idioms can make some masks trivial.",
        ],
        "leakage_risks": [
            "No function/source names in model input.",
            "Do not evaluate on the same masked positions used for training.",
        ],
        "metrics": ["masked-token accuracy", "majority baseline", "per-mnemonic accuracy"],
        "baselines": ["most frequent mnemonic", "previous-token copy", "random mnemonic"],
        "eligibility_rule": "Disassembly of a verified x86-64 build. Source alignment is not required for masked-token self-supervision. Semantic category labels are UNAVAILABLE.",
    },
    "compiler_transformation_analysis": {
        "input": "Aligned function descriptors at two optimization levels or compilers.",
        "target": "Derived transformation descriptor: instruction-count delta, block-count delta, mnemonic multiset delta and code-size delta.",
        "label_source": "Binary diff over functions whose source identity is aligned by DWARF.",
        "required_alignment": "function",
        "minimum_classification": "DEBUG_DERIVED",
        "known_shortcuts": [
            "Optimization level is highly correlated with deltas.",
            "Function size dominates the descriptor.",
        ],
        "leakage_risks": [
            "Optimization/compiler metadata must not be inputs.",
            "Deltas are descriptive, not causal optimization labels.",
        ],
        "metrics": ["descriptor MAE", "mnemonic-delta accuracy", "size-delta correlation"],
        "baselines": ["mean descriptor", "zero-delta", "size-only"],
        "eligibility_rule": "Two aligned builds of the same function with different optimization levels or compilers.",
    },
    "type_dataflow_inference": {
        "input": "Function binary descriptor and parameter positions.",
        "target": "DWARF parameter types and local variable roles.",
        "label_source": "DWARF type DIEs and location expressions.",
        "required_alignment": "variable",
        "minimum_classification": "DEBUG_DERIVED",
        "known_shortcuts": [
            "Parameter count and ABI register width are strong priors.",
            "Type names can be truncated or compiler-specific.",
        ],
        "leakage_risks": [
            "Source text and debug dumps must not be inputs.",
            "Location lists are not recovered; only simple locations are eligible.",
        ],
        "metrics": ["parameter type accuracy", "parameter count accuracy", "variable-role accuracy"],
        "baselines": ["default int", "ABI-width majority", "parameter-count prior"],
        "eligibility_rule": "DWARF formal parameters with resolved type names and simple or unavailable-but-declared locations.",
    },
    "c_reconstruction": {
        "input": "Binary function bytes, disassembly and CFG.",
        "target": "Source function body.",
        "label_source": "Original source text at the aligned function span.",
        "required_alignment": "function",
        "minimum_classification": "DEBUG_DERIVED",
        "known_shortcuts": [
            "Function name and source line can leak the target.",
            "Comments, formatting and identifier names are not semantically required.",
            "No behavioral tests are available in this milestone.",
        ],
        "leakage_risks": [
            "Source text must never appear in the input for training or evaluation.",
            "Project split isolation is mandatory.",
        ],
        "metrics": ["compile rate", "test pass rate", "normalized edit distance"],
        "baselines": ["empty stub", "signature-only stub", "existing pattern-based reconstruction"],
        "eligibility_rule": "Aligned source function span and binary function. Compile/behavioral verification remains PARTIAL or UNAVAILABLE.",
    },
    "function_purpose_inference": {
        "input": "Binary function descriptor, CFG and call context.",
        "target": "Function purpose label.",
        "label_source": "None available with verified ground truth.",
        "required_alignment": "function",
        "minimum_classification": "DEBUG_DERIVED",
        "known_shortcuts": [
            "Function names are a leaked label source and must not be used as ground truth.",
            "Documentation comments are not aligned to binaries.",
        ],
        "leakage_risks": [
            "Name-based pseudo-labels would fabricate ground truth.",
            "No purpose labels are claimed.",
        ],
        "metrics": ["unavailable"],
        "baselines": ["unavailable"],
        "eligibility_rule": "No verified purpose labels exist in this milestone; task is UNAVAILABLE.",
    },
}


def eligibility_summary() -> dict[str, Any]:
    return {
        name: {
            "input": card["input"],
            "target": card["target"],
            "label_source": card["label_source"],
            "required_alignment": card["required_alignment"],
            "minimum_classification": card["minimum_classification"],
            "known_shortcuts": card["known_shortcuts"],
            "leakage_risks": card["leakage_risks"],
            "metrics": card["metrics"],
            "baselines": card["baselines"],
            "eligibility_rule": card["eligibility_rule"],
        }
        for name, card in TASKS.items()
    }
