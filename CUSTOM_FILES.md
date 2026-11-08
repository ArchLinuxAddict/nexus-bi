# Running NEXUS-BI on custom files

## What works now

- Ghidra headless decompilation for native ELF binaries, including aarch64/ARM/x86-64.
- LLVM/GNU objdump and ELF metadata extraction.
- JADX Java/JAR/APK/DEX decompilation with CFR cross-check.
- Compilation verification for generated C and Java projects.
- Dashboard upload for native and Java/APK artifacts.

No trained residual-model checkpoint is loaded yet. The commands below use the
toolchain directly; model analysis will be re-enabled when the v6 model exists.

## Dashboard

```bash
cd nexus-bi
.venv/bin/nexus dashboard --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`:

- **Binaries** uploads native ELF files.
- **Java/APK** uploads JAR/APK/DEX/class files.
- **Reconstruction** runs Ghidra or JADX on the selected function.
- **Tools** shows the live tool matrix.
- **Residual** shows the residual checkpoint, held-out metrics and lets you run a tool-output sample.

## Native file

```bash
cd nexus-bi
TARGET="/absolute/path/to/your_binary_or_library"
OUT="artifacts/custom_checks/$(basename "$TARGET")"
mkdir -p "$OUT"

.venv/bin/nexus inspect binary "$TARGET" --json > "$OUT/inspect.json"
.venv/bin/nexus functions binary "$TARGET" --json > "$OUT/functions.json"
.venv/bin/nexus analyze binary "$TARGET" --out "$OUT/analysis.json" --json > "$OUT/analyze.json"
.venv/bin/nexus reconstruct c "$TARGET" --out "$OUT/reconstruction" --verify --json > "$OUT/reconstruction.json"
```

For non-x86-64 binaries, the C reconstruction uses Ghidra and caches the
decompilation under `artifacts/reconstruction/ghidra_cache/`. First run can take
30-60 seconds.

## Java/APK

```bash
.venv/bin/nexus java analyze /path/to/app.jar --full --json
.venv/bin/nexus java reconstruct /path/to/app.jar --out artifacts/java_recon --verify --json
.venv/bin/nexus jni correlate /path/to/app.apk /path/to/libnative.so --json
```

JADX produces the source, CFR provides a second opinion, and `javac` verifies
the generated project. These remain decompiler reconstructions, not original
source.

## Tools

```bash
.venv/bin/nexus tools
```

Required tools: Ghidra, JADX, CFR, Java, a C compiler. Optional: Z3 for future
constraint checks.
