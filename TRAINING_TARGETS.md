# What the Residual Model Must Learn

The tools already own parsing, disassembly, decompilation, Java/DEX decompilation,
CFG, xrefs, symbol discovery, string extraction, JNI inventory and compilation
checks. Do **not** train the model to reproduce those outputs.

Train only the residual: decisions and repairs that tools cannot make reliably.

## P0 — required for the first useful model

### 1. Tool arbitration and failure detection

Input: binary/function evidence plus candidate outputs from Ghidra, JADX, CFR
and any future tool.

Predict:

- which candidate is most likely correct;
- whether a tool failed, truncated or emitted malformed output;
- when to abstain.

Labels:

- source-backed functions from DWARF/debug builds;
- compile/syntax outcomes;
- perturbation tests: delete tokens, corrupt types, truncate functions;
- real tool disagreements where one candidate matches source.

Metrics: pairwise ranking accuracy, failure-detection AUROC, precision at
abstention, compile-rate lift over the best single tool.

### 2. Names, types and signatures

Input: tool output plus binary evidence (`FUN_...`, `local_...`, `undefined8`,
obfuscated JADX names).

Predict:

- function/variable/field/struct/enum names;
- C/Java types and function signatures;
- struct layout and array/pointer decisions.

Labels: source names, DWARF types, source signatures, JADX/Ghidra pseudo-labels
paired with real source where available.

Metrics: exact match, normalized edit distance, struct field recall,
signature exact match.

### 3. Java/JNI residual reconstruction

Input: JADX/CFR output, DEX/APK metadata, JNI declarations and native evidence.

Predict:

- repairs for JADX failures/obfuscation;
- Java native-method ↔ native-function mapping;
- JNI signature/type recovery and JNI glue semantics.

Labels: decompiled-vs-source pairs from permissively licensed real Java/Android
projects, JNI fixtures with source, JADX/CFR disagreements.

Metrics: repaired compile rate, native-method match F1, JNI type/signature
accuracy, source token similarity.

### 4. Confidence and abstention

Input: tool status, warnings, disagreement, compile result, evidence coverage.

Predict calibrated confidence and explicit abstention. A wrong confident answer
is worse than “tool cannot resolve this.”

Labels: per-task correctness outcomes from held-out source and compile checks.

Metrics: ECE/Brier score, risk–coverage curve.

## P1 — high-value after P0

### 5. Compile-guided repair

Given tool output that fails to compile, predict the minimal repair.

Labels: pairs of failing tool output and corrected source from real projects;
compiler diagnostics; successful recompile traces.

Metrics: repair success rate, edits per repair, no-regression compile rate.

### 6. Cross-binary retrieval and version analysis

Retrieve the same/similar function across builds, compiler variants, versions
and architectures; classify patches as refactor, bug fix, security fix or
feature change.

Labels: project-level variants from v6, version pairs, symbol/source alignment.

Metrics: Recall@k, MRR, patch-classification accuracy.

### 7. Tool planning

Choose the next tool/action under a latency/compute budget: Ghidra, JADX, CFR,
LLVM, compiler, deeper analysis or stop.

Labels: recorded tool traces and outcomes on v6 projects; imitation from
oracle choices.

Metrics: success per budget, tool calls per solved case, latency.

## P2 — only after the residual pipeline proves useful

### 8. Neural residual decompilation

Generate code only when all available tools fail or omit a construct. This is a
fallback, not the primary reconstruction path.

Labels: tool-failure cases with source available.

Metrics: compile rate and token similarity on failure-only cases.

### 9. Security/behavior hypotheses

Generate evidence-linked capability/behavior hypotheses, never confirmed
vulnerabilities without execution or an oracle.

Labels: curated real-world security patches and vulnerability datasets.

Metrics: precision/recall of hypothesis categories, citation/evidence coverage.

## Data required

- Real source+binary pairs: v6 debug builds with DWARF source bodies.
- Tool outputs stored as `ToolResult` evidence for every pair.
- Tool failures and disagreements, not only successful outputs.
- Real Java/Android projects with source and JNI libraries.
- Cross-build/cross-version variants for retrieval and patch tasks.
- Compile traces and diagnostics for repair tasks.

## Explicitly not training targets

- ELF parsing, disassembly, CFG recovery, xrefs or strings.
- Ghidra/JADX/CFR decompilation quality in general.
- Java/DEX decompilation that JADX already performs.
- Behavioral equivalence or vulnerability confirmation without an oracle.
- Any claim that tool output is original source.

## Acceptance gate

The residual model ships only if it beats:

1. the best raw tool;
2. tool + deterministic repair;
3. the current no-model baseline

on residual tasks, while abstaining when it cannot improve the tool answer.

## Full C and Java reconstruction

“Full” reconstruction means producing source that is structurally faithful,
compiles, and passes tests where tests exist. It does not mean proving
behavioral equivalence without execution.

### C curriculum

1. Syntax and statements
   - declarations, expressions, assignments, loops, switch, goto, returns.
2. Types
   - integer widths/signedness, pointers, arrays, structs, unions, enums,
     function pointers, typedefs, const/volatile.
3. Memory and ABI
   - stack frames, calling conventions, argument/return registers, varargs,
     globals, TLS, alignment, endianness, red zones.
4. Control flow
   - if/else, loops, switch/jump tables, short-circuit logic, early returns,
     irreducible control flow.
5. Data structures and idioms
   - linked lists, trees, hash tables, buffers, string operations, bitfields,
     error handling, resource lifetimes.
6. Optimization effects
   - O0/O1/O2/O3, inlining, vectorization, tail calls, branchless code,
     register allocation.
7. Compiler-aware verification
   - generated C must compile; where tests or reference I/O exist, compare
     outputs in a safe sandbox.
8. Residual repair
   - fix Ghidra/LLVM/tool output; predict omitted types, names and constructs.

### Java curriculum

1. Class model
   - classes, interfaces, records, enums, annotations, inner/anonymous classes,
     inheritance, overrides.
2. Bytecode/DEX semantics
   - operand stack, locals, invokes, field access, exceptions, monitors,
     arrays, boxing, string constants.
3. Types and generics
   - erasure, casts, wildcards, type bounds, lambda/method-reference targets.
4. Control flow and exceptions
   - try/catch/finally, synchronized, switch, loops, labeled breaks,
     exception tables.
5. Obfuscation and tool repair
   - JADX/CFR disagreements, synthetic accessors, bridge methods, dead code,
     renaming, decompiler failures.
6. Android/JNI
   - APK/DEX, native methods, JNI signatures, JNIEnv calls, library mapping.
7. Verification
   - generated Java must compile; tests where available. DEX/APK behavior
     requires a sandbox and is not claimed without execution.

## Is assembly needed?

Yes, as an **input and intermediate representation**, not as the final output.

Why:

- C reconstruction needs instruction effects, flags, memory accesses, calling
  conventions and control-flow targets.
- Tool output can be wrong; the model needs assembly/IR evidence to detect and
  repair it.
- Cross-architecture work (x86-64, AArch64, ARM, RISC-V) needs a shared semantic
  representation above raw assembly.
- Assembly/IR is already produced by Ghidra/LLVM/binutils; do not train a
  separate disassembler.

What to train:

- masked instruction/operand modeling;
- register/flag/memory effect prediction;
- CFG and call-target reasoning;
- assembly/IR to C/Java construct alignment;
- tool-output failure detection using assembly evidence.

What not to train:

- a replacement disassembler;
- byte-level assembly generation when objdump/LLVM already decode it;
- assembly as the product unless a user explicitly asks for it.

## Verification and honesty

- Compilation is required but is not behavioral equivalence.
- Tests/reference outputs are the only acceptable behavioral oracle.
- Dynamic execution remains disabled until a verified sandbox exists.
- “Fully learned C/Java” is an asymptotic goal; report coverage by construct,
  architecture, optimization level and tool-failure case.
