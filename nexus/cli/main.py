import argparse
import json
import sys
from pathlib import Path

from nexus.core.capabilities import as_dict as capabilities_dict
from nexus.core.config import default_paths
from nexus.core.engine import Engine
from nexus.core.errors import NexusError
from nexus.core.hardware import hardware_report
from nexus.datasets.factory import generate_dataset, inspect_dataset
from nexus.elf.parser import parse_elf
from nexus.java.service import analyze_java_artifact, reconstruct_java_artifact
from nexus.jni.correlation import correlate_java_native
from nexus.program.bundle import analyze_binary
from nexus.program.record import build_program_record
from nexus.research.knowledge import research_topic
from nexus.storage.store import Storage
from nexus.verification.compiler import compile_c_project, compile_java_project
from nexus.verification.static import verify_program_record


def _print(value, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(value, indent=2, default=str))
    else:
        print(value if isinstance(value, str) else json.dumps(value, indent=2, default=str))


def _add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")


def cmd_version(args) -> int:
    print("NEXUS-BI 0.7.0")
    print("model status: one unified foundation model; trained checkpoint available")
    return 0


def cmd_doctor(args) -> int:
    hardware = hardware_report()
    capabilities = capabilities_dict()
    report = {"hardware": hardware, "capabilities": capabilities}
    if args.json:
        _print(report, True)
        return 0
    host = hardware["host"]
    print(f"cpu: {host['cpu_model']} ({host['logical_cpus']} logical)")
    print(f"memory: {(host['memory_total_bytes'] or 0) / 1073741824:.1f} GiB")
    print(f"disk free: {host['disk_free_bytes'] / 1073741824:.1f} GiB")
    print(f"gpu compute: {hardware['gpu']['compute_available']}")
    print(f"kvm: {hardware['isolation']['kvm']}  qemu: {any(hardware['isolation']['qemu'].values())}")
    print("capabilities:")
    for name, item in capabilities.items():
        print(f"  {name:32s} {item['status']:12s} {item['reason']}")
    return 0


def cmd_inspect(args) -> int:
    elf = parse_elf(args.path)
    result = elf.inspect_dict()
    if args.json:
        _print(result, True)
        return 0
    print(f"file: {elf.path}")
    print(f"sha256: {elf.sha256}")
    print(f"class: {elf.header.elf_class}  machine: {elf.header.machine_name}  type: {elf.header.type_name}")
    print(f"entry: 0x{elf.header.entry:x}  pie: {elf.is_pie}  shared: {elf.is_shared_object}  static: {elf.is_static}  stripped: {elf.is_stripped}")
    print(f"build-id: {elf.build_id}")
    print(f"segments: {len(elf.program_headers)}  sections: {len(elf.sections)}  symbols: {len(elf.symbols)}  dynsym: {len(elf.dynamic_symbols)}  relocations: {len(elf.relocations)}")
    print("dependencies:")
    for dependency in elf.dependencies:
        print(f"  {dependency.get('kind')}: {dependency.get('name')}")
    return 0


def cmd_analyze(args) -> int:
    engine = Engine()
    result = engine.analyze_path(args.path, include_record=False)
    if args.out:
        Path(args.out).write_text(json.dumps(result.get("record", engine.load_record(result["summary"]["analysis_id"])), indent=2) + "\n", encoding="utf-8")
    _print(result["summary"], args.json)
    return 0


def cmd_functions(args) -> int:
    bundle = analyze_binary(args.path)
    result = [function.to_dict() for function in bundle.functions]
    if args.json:
        _print({"binary_sha256": bundle.elf.sha256, "functions": result}, True)
        return 0
    for function in bundle.functions:
        print(f"{function.name:32s} 0x{function.entry:x} size={function.size:5d} confidence={function.confidence:.2f} source={function.source}")
    return 0


def cmd_disassemble(args) -> int:
    bundle = analyze_binary(args.path)
    if bundle.disassembly is None:
        _print({"status": "capability_unavailable", "reason": "disassembly not available for this architecture"}, args.json)
        return 2
    instructions = bundle.disassembly.instructions
    if args.function:
        selected = next((function for function in bundle.functions if function.name == args.function or function.function_id == args.function), None)
        if selected is None:
            _print({"status": "not_found", "function": args.function}, args.json)
            return 1
        ranges = selected.ranges
        instructions = [instruction for instruction in instructions if any(start <= instruction.address < end for start, end in ranges)]
    instructions = instructions[:args.limit]
    if args.json:
        _print([instruction.to_dict() for instruction in instructions], True)
    else:
        for instruction in instructions:
            print(f"0x{instruction.address:x}  {instruction.bytes_hex:20s} {instruction.mnemonic:8s} {' '.join(instruction.operands)}")
    return 0


def cmd_cfg(args) -> int:
    bundle = analyze_binary(args.path)
    record = build_program_record(bundle)
    cfg = record.get("cfg", {})
    if args.function:
        function = next((item for item in record.get("functions", []) if item.get("function_id") == args.function or any(name.get("value") == args.function for name in item.get("names", []))), None)
        if function is None:
            _print({"status": "not_found", "function": args.function}, args.json)
            return 1
        blocks = {block["block_id"] for block in function.get("blocks", [])}
        cfg = {"nodes": [node for node in cfg.get("nodes", []) if node.get("node_id") in blocks or node.get("function_id") == function.get("function_id")], "edges": [edge for edge in cfg.get("edges", []) if edge.get("source") in blocks or edge.get("target") in blocks]}
    _print(cfg, args.json)
    return 0


def cmd_callgraph(args) -> int:
    bundle = analyze_binary(args.path)
    record = build_program_record(bundle)
    _print(record.get("callgraph", {}), args.json)
    return 0


def cmd_ir(args) -> int:
    from nexus.program.ir import build_program_ir, serialize_program_ir
    bundle = analyze_binary(args.path)
    ir = build_program_ir(bundle)
    if args.out:
        Path(args.out).write_text(serialize_program_ir(ir), encoding="utf-8")
    _print(ir, args.json)
    return 0


def cmd_program_record(args) -> int:
    bundle = analyze_binary(args.path)
    record = build_program_record(bundle)
    validation = verify_program_record(record)
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if args.json:
        _print({"validation": validation, "record": record}, True)
    else:
        print(f"program record: {validation['status']}")
        print(json.dumps(validation["counts"], indent=2))
        if args.out:
            print(f"written: {args.out}")
    return 0 if validation["status"] == "passed" else 1


def cmd_reconstruct(args) -> int:
    engine = Engine()
    analysis = engine.analyze_path(args.path)
    analysis_id = analysis["summary"]["analysis_id"]
    result = engine.reconstruct(analysis_id, args.language, args.function, args.out)
    if args.verify:
        verification = engine.verify_reconstruction(result["reconstruction_id"])
        result["verification"] = verification
    if args.json:
        _print(result, True)
    else:
        print(f"analysis: {analysis_id}")
        print(f"reconstruction: {result.get('reconstruction_id')}  classification: {result.get('classification')}")
        print(f"output: {result.get('output_dir')}")
        for function in result.get("functions", []):
            print(f"  {function['name']:32s} {function['classification']:36s} confidence={function['confidence']:.2f}")
        if args.verify:
            print(f"verification: {result['verification'].get('status')}")
    return 0


def cmd_verify(args) -> int:
    if args.language == "java":
        result = compile_java_project(args.path)
    else:
        result = compile_c_project(args.path)
    _print(result, args.json)
    return 0 if result.get("status") == "passed" else 1


def cmd_investigate(args) -> int:
    engine = Engine()
    analysis = engine.analyze_path(args.path)
    report = engine.investigate(analysis["summary"]["analysis_id"], args.objective, args.max_steps)
    _print(report, args.json)
    return 0


def cmd_security(args) -> int:
    bundle = analyze_binary(args.path)
    findings = bundle.security_findings
    if args.json:
        _print(findings, True)
    else:
        if not findings:
            print("no evidence-linked findings in this limited scan")
        for finding in findings:
            print(f"{finding['severity']:8s} {finding['title']:40s} {finding.get('function_name') or ''} {finding.get('address') or ''}")
            print(f"  {finding['description']}")
    return 0


def cmd_dataset_generate(args) -> int:
    result = generate_dataset(args.out, fixtures=args.fixtures, compilers=args.compilers, optimizations=args.optimizations, debug_modes=[True, False], pie_modes=[True])
    if result.get("status") == "completed":
        Storage().add_dataset(result["path"], result["summary"])
        from nexus.mission.policy import register_dataset
        register_dataset(
            result["summary"]["dataset_id"], result["path"], "controlled_binary_source",
            ["binary_representation_masking", "function_boundary_prediction", "c_reconstruction"],
            "Controlled source/binary builds with compiler metadata and labels.",
            license_verified=True,
        )
    _print(result, args.json)
    return 0


def cmd_dataset_multiarche(args) -> int:
    from nexus.datasets.advanced import build_multiarche_dataset
    result = build_multiarche_dataset(args.out)
    _print(result, args.json)
    return 0 if result.get("status") == "completed" else 1


def cmd_dataset_java(args) -> int:
    from nexus.datasets.advanced import build_java_dataset
    result = build_java_dataset(args.out)
    _print(result, args.json)
    return 0 if result.get("status") == "completed" else 1


def cmd_dataset_java_samples(args) -> int:
    from nexus.data.java_corpus import build_java_samples
    result = build_java_samples(args.classes, out_dir=args.out, text_vocab_size=args.text_vocab_size)
    _print(result, args.json)
    return 0 if result.get("status") == "completed" else 1


def cmd_dataset_inspect(args) -> int:
    from nexus.datasets.advanced import validate_advanced_dataset
    manifest = Path(args.path) / "manifest.json"
    if manifest.exists():
        try:
            import json as _json
            payload = _json.loads(manifest.read_text(encoding="utf-8"))
            if str(payload.get("schema_version", "")).startswith("nexus.dataset."):
                result = validate_advanced_dataset(args.path)
                _print(result, args.json)
                return 0 if result.get("status") == "passed" else 1
        except Exception:
            pass
    _print(inspect_dataset(args.path), args.json)
    return 0


def cmd_dataset_ingest(args) -> int:
    from nexus.datasets.realworld import (
        DEFAULT_VARIANTS,
        BuildVariant,
        ingest_candidates,
    )

    if args.variants == "full":
        variants = [BuildVariant(compiler, optimization, debug) for compiler in ("gcc", "clang") for optimization in ("O0", "O2") for debug in (True, False)]
    else:
        variants = list(DEFAULT_VARIANTS)
    result = ingest_candidates(args.candidates, args.out, max_projects=args.max_projects, variants=variants, timeout=args.timeout)
    if result.get("status") == "completed":
        Storage().add_dataset(result["path"], result["summary"])
        from nexus.mission.policy import register_dataset
        register_dataset(
            result["summary"]["dataset_id"], result["path"], "real_world_binary_source",
            ["binary_representation_masking", "function_boundary_prediction", "cross_compiler_function_retrieval", "c_reconstruction"],
            "Licensed real-world projects with exact commits and controlled binaries.",
            license_verified=True,
        )
    _print(result, args.json)
    return 0


def cmd_dataset_audit(args) -> int:
    from nexus.datasets.realworld import audit_realworld_dataset

    result = audit_realworld_dataset(args.path)
    _print(result, args.json)
    return 0 if result.get("status") == "passed" else 1


def cmd_model_info(args) -> int:
    from nexus.foundation.checkpoint import inspect_checkpoint

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    result = {
        "status": "tool-first rebuild; no trained residual checkpoint",
        "architecture_version": "nexus-bi/foundation-1",
        "checkpoint": str(checkpoint) if checkpoint else None,
    }
    if checkpoint is not None and checkpoint.exists():
        info = inspect_checkpoint(checkpoint)
        result.update({
            "sha256": info["sha256"],
            "size": info["size"],
            "parameter_count": (info.get("metadata") or {}).get("parameter_count"),
            "step": (info.get("metadata") or {}).get("step"),
            "initialization": (info.get("metadata") or {}).get("initialization"),
            "pretrained_weights_loaded": (info.get("metadata") or {}).get("pretrained_weights_loaded"),
        })
    _print(result, args.json)
    return 0


def cmd_model_smoke(args) -> int:
    from nexus.foundation.smoke import smoke_test
    result = smoke_test(steps=args.steps, checkpoint_path=args.checkpoint)
    _print(result, args.json)
    return 0 if result["status"] == "passed" else 1


def cmd_mission(args) -> int:
    from nexus.mission.policy import audit_registry, load_policy, load_registry
    policy = load_policy()
    registry = load_registry()
    result = {
        "audit": audit_registry(),
        "capabilities": policy["capabilities"],
        "objectives": policy["objectives"],
        "dataset_categories": policy["dataset_categories"],
        "datasets": registry["datasets"],
    }
    _print(result, args.json)
    return 0 if result["audit"]["status"] == "passed" else 1


def cmd_benchmark(args) -> int:
    from nexus.evaluation.benchmarks import benchmark_paths

    result = benchmark_paths([args.path or "/usr/bin/ls"])
    _print(result, args.json)
    return 0 if result.get("status", "completed") in {"completed", "passed"} else 1


def cmd_validate(args) -> int:
    from nexus.validation.independent import run_validation
    binaries = args.binary or [
        str(Path("artifacts/e2e/build/simple")),
        str(Path("artifacts/e2e/build/jni/libjni_demo.so")),
        "/usr/bin/ls",
    ]
    result = run_validation(
        binaries=[path for path in binaries if Path(path).exists()],
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        output=args.out,
        fuzz_iterations=args.fuzz_iterations,
        reconstruction_limit=args.reconstruction_limit,
    )
    _print(result, args.json)
    return 0


def cmd_research(args) -> int:
    result = research_topic(args.query, include_headers=not args.no_headers, limit=args.limit)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    if args.json:
        _print(result, True)
        return 0
    for item in result["knowledge"]:
        print(f"{item['title']} [{item['id']}]")
        for fact in item["facts"]:
            print(f"  - {fact}")
    if result.get("system_headers"):
        print("system headers:")
        for match in result["system_headers"]["matches"]:
            print(f"  {match['source']}:{match.get('line') or ''}  {match['text']}")
    return 0


def cmd_report(args) -> int:
    if getattr(args, "binary", None):
        from nexus.reporting.comprehensive import build_comprehensive_report
        result = build_comprehensive_report(
            args.binary,
            java_path=args.java,
            native_path=args.native_so,
            out_dir=args.out,
            max_functions=args.max_functions,
        )
        _print(result, args.json)
        return 0
    storage = Storage()
    if args.analysis_id:
        analysis = storage.get_analysis(args.analysis_id)
        result = {"analysis": analysis}
        try:
            result["findings"] = storage.list_findings(analysis["binary_id"])
        except Exception:
            result["findings"] = []
    else:
        result = {"binaries": storage.list_binaries(), "analyses": storage.list_analyses(), "datasets": storage.list_datasets(), "models": storage.list_models()}
    _print(result, args.json)
    return 0


def cmd_serve(args) -> int:
    import uvicorn
    uvicorn.run("nexus.api.app:app", host=args.host, port=args.port, reload=False)
    return 0



def cmd_acquire_discover(args) -> int:
    from nexus.acquisition.adapters import HttpClient, discover
    client = HttpClient(timeout=args.timeout)
    candidates = discover(args.adapter, args.query, limit=args.limit, client=client)
    payload = {"adapter": args.adapter, "query": args.query, "candidates": [item.to_dict() for item in candidates], "rate_limit_state": client.rate_limit_state}
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _print(payload, args.json)
    return 0


def cmd_acquire_run(args) -> int:
    from nexus.acquisition.pipeline import run_pilot
    result = run_pilot(
        args.config,
        args.dataset,
        args.reports,
        max_projects=args.max_projects,
        resume=not args.no_resume,
        swh_check=not args.no_swh,
    )
    _print(result, args.json)
    return 0 if result.get("accepted", 0) > 0 else 1


def cmd_acquire_status(args) -> int:
    from pathlib import Path
    root = Path(args.dataset)
    result = {"acquisition_summary": json.loads((root / "acquisition_summary.json").read_text(encoding="utf-8")) if (root / "acquisition_summary.json").exists() else None, "state": json.loads((root / ".state" / "acquisition_state.json").read_text(encoding="utf-8")) if (root / ".state" / "acquisition_state.json").exists() else None}
    _print(result, args.json)
    return 0


def cmd_build_matrix(args) -> int:
    from nexus.builds.matrix import run_build_matrix
    result = run_build_matrix(args.dataset, matrix=args.matrix, max_projects=args.max_projects, jobs=args.jobs, reproduce=not args.no_reproduce, project_ids=args.project)
    _print({"builds_built": result["builds_built"], "builds_failed": result["builds_failed"], "builds_unavailable": result["builds_unavailable"], "by_compiler": result["by_compiler"], "by_optimization": result["by_optimization"], "report": str(Path(args.dataset) / "build_matrix_report.json")}, args.json)
    return 0


def cmd_align_corpus(args) -> int:
    from nexus.alignment.corpus import run_corpus_alignment
    result = run_corpus_alignment(args.dataset, max_projects=args.max_projects, transfer_stripped=not args.no_transfer)
    _print({key: result[key] for key in ("projects_total", "projects_with_alignment", "debug_indexes", "stripped_transfers", "functions", "functions_with_source", "instruction_mapping_coverage")}, args.json)
    return 0


def cmd_align_fixture(args) -> int:
    from nexus.alignment.fixtures import run_alignment_fixtures
    result = run_alignment_fixtures(args.out, args.reports)
    _print({"status": result["status"], "aggregate": result["aggregate"], "report": str(Path(args.reports) / "alignment_quality_report.json")}, args.json)
    return 0 if result["status"] == "passed" else 1


def cmd_representations(args) -> int:
    from nexus.representations.coverage import collect_representation_coverage
    result = collect_representation_coverage(args.dataset, args.reports)
    _print({key: result[key] for key in ("projects", "builds", "raw_representations", "derived_representations", "status")}, args.json)
    return 0 if result["status"] == "passed" else 1


def cmd_tasks_build(args) -> int:
    from nexus.tasks.builder import build_task_datasets
    result = build_task_datasets(args.dataset, args.reports)
    _print({"status": result["status"], "tasks": {name: item["samples"] for name, item in result["tasks"].items()}}, args.json)
    return 0 if result["status"] in {"passed", "partial"} else 1


def cmd_tools(args) -> int:
    from nexus.toolchain.registry import ToolRegistry

    registry = ToolRegistry()
    if args.write:
        registry.write_matrix(args.out)
    result = registry.matrix()
    _print(result, args.json)
    return 0 if result["status"] == "ready" else 1


def cmd_foundation_info(args) -> int:
    from nexus.foundation.checkpoint import inspect_checkpoint
    from nexus.foundation.config import profile_report
    from nexus.foundation.objectives import objective_registry
    from nexus.toolchain.registry import ToolRegistry

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    report = {
        "product": "NEXUS-BI tool-first binary intelligence",
        "model_status": "tool-first rebuild in progress; no residual-model checkpoint is loaded",
        "profiles": profile_report(),
        "objectives": objective_registry(),
        "tools": ToolRegistry().matrix(),
        "v6": {
            "pilot": "artifacts/dataset/realworld_v6_pilot2",
            "status": "real-world source+binary pilot ingestion running",
        },
        "checkpoint": inspect_checkpoint(checkpoint) if checkpoint and checkpoint.exists() else None,
        "honesty": [
            "The old v4/v5 controlled checkpoints and corpora were removed from the working tree.",
            "Tool output is authoritative only for the tool that produced it.",
            "Ghidra/JADX output is decompiler inference, not original source.",
            "No behavioral equivalence or foundation-scale capability is claimed.",
        ],
    }
    _print(report, args.json)
    return 0


def cmd_foundation_analyze(args) -> int:
    from nexus.foundation.analysis import analyze_with_foundation

    result = analyze_with_foundation(
        args.path,
        args.checkpoint,
        max_functions=args.max_functions,
        max_instructions=args.max_instructions,
    )
    _print(result, args.json)
    return 0


def cmd_foundation_init(args) -> int:
    from nexus.foundation.checkpoint import save_foundation_checkpoint
    from nexus.foundation.config import config_for_vocabulary, profile_config
    from nexus.foundation.model import build_foundation_model
    from nexus.foundation.vocab import Vocabulary

    config = config_for_vocabulary(profile_config(args.profile), Vocabulary())
    model = build_foundation_model(config, seed=args.seed)
    vocab = Vocabulary()
    saved = save_foundation_checkpoint(
        args.out,
        model,
        config,
        vocab,
        metadata={
            "status": "random_initialization",
            "trained": False,
            "steps": 0,
            "note": "Randomly initialized checkpoint. This is not a trained model.",
            "seed": args.seed,
        },
        step=0,
        include_training_state=False,
    )
    _print(saved, args.json)
    return 0


def cmd_foundation_objectives(args) -> int:
    from nexus.foundation.objectives import objective_registry

    _print(objective_registry(), args.json)
    return 0


def cmd_foundation_dataset(args) -> int:
    from nexus.foundation.dataset import dataset_source_from_registry, describe_dataset

    sources = [dataset_source_from_registry(dataset_id) for dataset_id in args.dataset_id]
    _print(describe_dataset(sources), args.json)
    return 0


def cmd_foundation_train(args) -> int:
    from nexus.foundation.launch import launch_training

    result = launch_training(
        config_path=args.config,
        init_from=args.init_from,
        resume_from=args.resume_from,
        dataset_ids=args.dataset_id or None,
        objectives=args.objective or None,
        profile=args.profile,
        run_name=args.run_name,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
        frozen_dir=args.frozen_dir,
        dry_run=args.dry_run,
        evaluate=not args.no_evaluate,
    )
    if args.dry_run:
        _print(result, args.json)
        return 0
    keys = ("status", "run_dir", "parameter_count", "steps", "checkpoint", "best_checkpoint", "parameter_update", "resources", "objective_summary", "validation", "distinction")
    _print({key: result.get(key) for key in keys if key in result}, args.json)
    return 0


def cmd_foundation_checkpoint(args) -> int:
    from nexus.foundation.checkpoint import inspect_checkpoint

    _print(inspect_checkpoint(args.path, include_state=args.include_state), args.json)
    return 0


def cmd_foundation_reconstruct(args) -> int:
    from nexus.foundation.reconstruction import UnifiedReconstructor

    reconstructor = UnifiedReconstructor(checkpoint_path=args.checkpoint, workspace=args.workspace)
    if args.project:
        result = reconstructor.reconstruct_project(args.path, language=args.language, max_functions=args.max_functions, include_neural=not args.no_neural)
    elif args.function:
        result = reconstructor.reconstruct_function(
            args.path,
            args.function,
            language=args.language,
            reference_source=Path(args.reference).read_text(encoding="utf-8", errors="replace") if args.reference else None,
            max_new_tokens=args.max_new_tokens,
            include_neural=not args.no_neural,
        )
    else:
        _print({"status": "failed", "reason": "specify --function or --project"}, args.json)
        return 2
    _print(result, args.json)
    return 0


def cmd_foundation_resume(args) -> int:
    from nexus.foundation.launch import launch_training

    result = launch_training(
        config_path=args.config,
        resume_from=args.checkpoint,
        dataset_ids=args.dataset_id or None,
        objectives=args.objective or None,
        profile=args.profile,
        run_name=args.run_name,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        output_dir=args.output_dir,
        frozen_dir=args.frozen_dir,
        evaluate=not args.no_evaluate,
    )
    keys = ("status", "run_dir", "steps", "checkpoint", "parameter_update", "resources", "distinction")
    _print({key: result.get(key) for key in keys if key in result}, args.json)
    return 0


def cmd_java_analyze(args) -> int:
    result = analyze_java_artifact(args.path)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    if args.json:
        _print(result if args.full else result.get("summary", result), True)
        return 0
    summary = result.get("summary", {})
    print(f"format: {result.get('format')}")
    for key in ("package", "name", "version", "classes", "methods", "fields", "native_library_count", "native_methods"):
        if key in summary:
            value = summary[key]
            if isinstance(value, list) and len(value) > 12:
                value = value[:12] + ["..."]
            print(f"{key}: {value}")
    limitations = result.get("class", {}).get("limitations") if result.get("format") == "class" else None
    if limitations:
        for limitation in limitations:
            print(f"limitation: {limitation}")
    return 0


def cmd_java_reconstruct(args) -> int:
    out_dir = Path(args.out) if args.out else default_paths().artifacts / "java_reconstructions" / "java_reconstruction"
    result = reconstruct_java_artifact(
        args.path,
        out_dir=out_dir,
        project_name=args.project_name,
        max_classes=args.max_classes,
    )
    if args.verify:
        result["verification"] = compile_java_project(out_dir)
    if args.json:
        _print(result, True)
    else:
        print(f"java reconstruction: {result.get('classification')}  source={result.get('source_format')}")
        print(f"output: {out_dir}")
        print(f"files: {len(result.get('files', {}))}")
        if args.verify:
            print(f"verification: {result['verification'].get('status')}")
    if args.verify and result["verification"].get("status") not in {"passed", "skipped"}:
        return 1
    return 0


def cmd_jni_correlate(args) -> int:
    result = correlate_java_native(args.java, args.native)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    if args.json:
        _print(result, True)
    else:
        summary = result["summary"]
        print(f"java native methods: {summary['java_native_methods']}  native JNI symbols: {summary['native_jni_symbols']}  matched: {summary['matched']}")
        print(f"unmatched Java: {summary['unmatched_java']}  unmatched native: {summary['unmatched_native']}  JNI_OnLoad: {summary['jni_onload']}")
        for match in result["matches"]:
            print(f"MATCH {match['class_name']}.{match['method']}{match['descriptor']} -> {match['native_symbol']} ({match['confidence']:.2f})")
        for match in result["unmatched_java_methods"]:
            print(f"MISS  {match['class_name']}.{match['method']}{match['descriptor']} expected {match['expected_symbols']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nexus", description="NEXUS-BI integrated binary intelligence platform")
    sub = parser.add_subparsers(dest="command", required=True)


    tools = sub.add_parser("tools", help="show the external tool capability matrix")
    tools.add_argument("--json", action="store_true")
    tools.add_argument("--write", action="store_true")
    tools.add_argument("--out")
    tools.set_defaults(func=cmd_tools)

    foundation = sub.add_parser("foundation")
    foundation_sub = foundation.add_subparsers(dest="target", required=True)
    foundation_info = foundation_sub.add_parser("info")
    foundation_info.add_argument("--checkpoint")
    _add_json(foundation_info)
    foundation_info.set_defaults(func=cmd_foundation_info)
    foundation_analyze = foundation_sub.add_parser("analyze")
    foundation_analyze.add_argument("path")
    foundation_analyze.add_argument("--checkpoint", required=True)
    foundation_analyze.add_argument("--max-functions", type=int, default=8)
    foundation_analyze.add_argument("--max-instructions", type=int)
    _add_json(foundation_analyze)
    foundation_analyze.set_defaults(func=cmd_foundation_analyze)
    foundation_init = foundation_sub.add_parser("init")
    foundation_init.add_argument("--profile", choices=["cpu_dev", "small", "medium", "large"], default="cpu_dev")
    foundation_init.add_argument("--seed", type=int, default=20261103)
    foundation_init.add_argument("--out", default="artifacts/initialization/random-init.pt")
    _add_json(foundation_init)
    foundation_init.set_defaults(func=cmd_foundation_init)
    foundation_objectives = foundation_sub.add_parser("objectives")
    _add_json(foundation_objectives)
    foundation_objectives.set_defaults(func=cmd_foundation_objectives)
    foundation_dataset = foundation_sub.add_parser("dataset")
    foundation_dataset.add_argument("--dataset-id", action="append", required=True)
    _add_json(foundation_dataset)
    foundation_dataset.set_defaults(func=cmd_foundation_dataset)
    foundation_train = foundation_sub.add_parser("train")
    foundation_train.add_argument("--config", default=None, help="authoritative frozen-corpus training config")
    foundation_train.add_argument("--dataset-id", action="append", help="verified member of the frozen corpus; repeatable")
    foundation_train.add_argument("--profile", choices=["compact", "cpu_dev", "small", "medium", "large"], default=None)
    foundation_train.add_argument("--run-name", default=None)
    foundation_train.add_argument("--steps", type=int, default=None)
    foundation_train.add_argument("--batch-size", type=int, default=None)
    foundation_train.add_argument("--objective", action="append")
    foundation_train.add_argument("--seed", type=int, default=None)
    foundation_train.add_argument("--output-dir", default=None)
    foundation_train.add_argument("--frozen-dir", default=None)
    init_group = foundation_train.add_mutually_exclusive_group()
    init_group.add_argument("--init-from", default=None, help="verified random initialization checkpoint; fresh optimizer/scheduler")
    init_group.add_argument("--resume-from", default=None, help="complete interrupted training checkpoint; restores full state")
    foundation_train.add_argument("--dry-run", action="store_true", help="validate launch, dataset, checkpoint and objectives without any optimizer step")
    foundation_train.add_argument("--no-evaluate", action="store_true")
    _add_json(foundation_train)
    foundation_train.set_defaults(func=cmd_foundation_train)
    foundation_checkpoint = foundation_sub.add_parser("checkpoint")
    foundation_checkpoint.add_argument("--path", required=True)
    foundation_checkpoint.add_argument("--include-state", action="store_true")
    _add_json(foundation_checkpoint)
    foundation_checkpoint.set_defaults(func=cmd_foundation_checkpoint)
    foundation_reconstruct = foundation_sub.add_parser("reconstruct")
    foundation_reconstruct.add_argument("path")
    foundation_reconstruct.add_argument("--language", choices=["c", "java", "jni"], default="c")
    foundation_reconstruct.add_argument("--function")
    foundation_reconstruct.add_argument("--project", action="store_true")
    foundation_reconstruct.add_argument("--max-functions", type=int, default=8)
    foundation_reconstruct.add_argument("--max-new-tokens", type=int, default=96)
    foundation_reconstruct.add_argument("--checkpoint")
    foundation_reconstruct.add_argument("--reference")
    foundation_reconstruct.add_argument("--workspace")
    foundation_reconstruct.add_argument("--no-neural", action="store_true")
    _add_json(foundation_reconstruct)
    foundation_reconstruct.set_defaults(func=cmd_foundation_reconstruct)

    version = sub.add_parser("version")
    version.set_defaults(func=cmd_version)

    doctor = sub.add_parser("doctor")
    _add_json(doctor)
    doctor.set_defaults(func=cmd_doctor)

    inspect = sub.add_parser("inspect")
    inspect_sub = inspect.add_subparsers(dest="target", required=True)
    inspect_binary = inspect_sub.add_parser("binary")
    inspect_binary.add_argument("path")
    _add_json(inspect_binary)
    inspect_binary.set_defaults(func=cmd_inspect)

    analyze = sub.add_parser("analyze")
    analyze_sub = analyze.add_subparsers(dest="target", required=True)
    analyze_binary_parser = analyze_sub.add_parser("binary")
    analyze_binary_parser.add_argument("path")
    analyze_binary_parser.add_argument("--out")
    _add_json(analyze_binary_parser)
    analyze_binary_parser.set_defaults(func=cmd_analyze)

    disassemble = sub.add_parser("disassemble")
    disassemble_sub = disassemble.add_subparsers(dest="target", required=True)
    disassemble_binary = disassemble_sub.add_parser("binary")
    disassemble_binary.add_argument("path")
    disassemble_binary.add_argument("--function")
    disassemble_binary.add_argument("--limit", type=int, default=500)
    _add_json(disassemble_binary)
    disassemble_binary.set_defaults(func=cmd_disassemble)

    functions = sub.add_parser("functions")
    functions_sub = functions.add_subparsers(dest="target", required=True)
    functions_binary = functions_sub.add_parser("binary")
    functions_binary.add_argument("path")
    _add_json(functions_binary)
    functions_binary.set_defaults(func=cmd_functions)

    cfg = sub.add_parser("cfg")
    cfg_sub = cfg.add_subparsers(dest="target", required=True)
    cfg_binary = cfg_sub.add_parser("binary")
    cfg_binary.add_argument("path")
    cfg_binary.add_argument("--function")
    _add_json(cfg_binary)
    cfg_binary.set_defaults(func=cmd_cfg)

    callgraph = sub.add_parser("callgraph")
    callgraph_sub = callgraph.add_subparsers(dest="target", required=True)
    callgraph_binary = callgraph_sub.add_parser("binary")
    callgraph_binary.add_argument("path")
    _add_json(callgraph_binary)
    callgraph_binary.set_defaults(func=cmd_callgraph)

    ir = sub.add_parser("ir")
    ir_sub = ir.add_subparsers(dest="target", required=True)
    ir_binary = ir_sub.add_parser("binary")
    ir_binary.add_argument("path")
    ir_binary.add_argument("--out")
    _add_json(ir_binary)
    ir_binary.set_defaults(func=cmd_ir)

    record = sub.add_parser("program-record")
    record_sub = record.add_subparsers(dest="target", required=True)
    record_binary = record_sub.add_parser("binary")
    record_binary.add_argument("path")
    record_binary.add_argument("--out")
    _add_json(record_binary)
    record_binary.set_defaults(func=cmd_program_record)

    reconstruct = sub.add_parser("reconstruct")
    reconstruct_sub = reconstruct.add_subparsers(dest="language", required=True)
    for language in ("c", "java"):
        language_parser = reconstruct_sub.add_parser(language)
        language_parser.add_argument("path")
        language_parser.add_argument("--function", action="append")
        language_parser.add_argument("--out")
        language_parser.add_argument("--verify", action="store_true")
        _add_json(language_parser)
        language_parser.set_defaults(func=cmd_reconstruct)

    verify = sub.add_parser("verify")
    verify_sub = verify.add_subparsers(dest="target", required=True)
    verify_reconstruction = verify_sub.add_parser("reconstruction")
    verify_reconstruction.add_argument("path")
    verify_reconstruction.add_argument("--language", choices=["c", "java"], default="c")
    _add_json(verify_reconstruction)
    verify_reconstruction.set_defaults(func=cmd_verify)

    investigate = sub.add_parser("investigate")
    investigate.add_argument("path")
    investigate.add_argument("--objective", required=True)
    investigate.add_argument("--max-steps", type=int, default=8)
    _add_json(investigate)
    investigate.set_defaults(func=cmd_investigate)

    security = sub.add_parser("security")
    security_sub = security.add_subparsers(dest="target", required=True)
    security_scan = security_sub.add_parser("scan")
    security_scan.add_argument("path")
    _add_json(security_scan)
    security_scan.set_defaults(func=cmd_security)

    dataset = sub.add_parser("dataset")
    dataset_sub = dataset.add_subparsers(dest="target", required=True)
    dataset_generate = dataset_sub.add_parser("generate")
    dataset_generate.add_argument("--out", required=True)
    dataset_generate.add_argument("--fixtures", nargs="*")
    dataset_generate.add_argument("--compilers", nargs="*")
    dataset_generate.add_argument("--optimizations", nargs="*")
    _add_json(dataset_generate)
    dataset_generate.set_defaults(func=cmd_dataset_generate)
    dataset_multiarche = dataset_sub.add_parser("multiarche")
    dataset_multiarche.add_argument("--out", default="artifacts/dataset/multiarche_pilot")
    _add_json(dataset_multiarche)
    dataset_multiarche.set_defaults(func=cmd_dataset_multiarche)
    dataset_java = dataset_sub.add_parser("java")
    dataset_java.add_argument("--out", default="artifacts/dataset/java_pilot")
    _add_json(dataset_java)
    dataset_java.set_defaults(func=cmd_dataset_java)
    dataset_java_samples = dataset_sub.add_parser("java-samples")
    dataset_java_samples.add_argument("--classes", default="artifacts/dataset/java_pilot/java_classes")
    dataset_java_samples.add_argument("--out", default="artifacts/dataset/java_foundation")
    dataset_java_samples.add_argument("--text-vocab-size", type=int, default=4096)
    _add_json(dataset_java_samples)
    dataset_java_samples.set_defaults(func=cmd_dataset_java_samples)
    dataset_inspect = dataset_sub.add_parser("inspect")
    dataset_inspect.add_argument("path")
    _add_json(dataset_inspect)
    dataset_inspect.set_defaults(func=cmd_dataset_inspect)
    dataset_ingest = dataset_sub.add_parser("ingest")
    dataset_ingest.add_argument("--candidates", default="configs/realworld_candidates.json")
    dataset_ingest.add_argument("--out", required=True)
    dataset_ingest.add_argument("--max-projects", type=int, default=110)
    dataset_ingest.add_argument("--variants", choices=["default", "full"], default="default")
    dataset_ingest.add_argument("--timeout", type=int, default=60)
    _add_json(dataset_ingest)
    dataset_ingest.set_defaults(func=cmd_dataset_ingest)
    dataset_audit = dataset_sub.add_parser("audit")
    dataset_audit.add_argument("path")
    _add_json(dataset_audit)
    dataset_audit.set_defaults(func=cmd_dataset_audit)

    model = sub.add_parser("model")
    model_sub = model.add_subparsers(dest="target", required=True)
    model_info = model_sub.add_parser("info")
    model_info.add_argument("--checkpoint")
    _add_json(model_info)
    model_info.set_defaults(func=cmd_model_info)
    model_smoke = model_sub.add_parser("smoke-test")
    model_smoke.add_argument("--steps", type=int, default=2)
    model_smoke.add_argument("--checkpoint")
    _add_json(model_smoke)
    model_smoke.set_defaults(func=cmd_model_smoke)

    mission = sub.add_parser("mission")
    mission_sub = mission.add_subparsers(dest="target", required=True)
    mission_audit = mission_sub.add_parser("audit")
    _add_json(mission_audit)
    mission_audit.set_defaults(func=cmd_mission)


    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--path", default="/usr/bin/ls")
    _add_json(benchmark)
    benchmark.set_defaults(func=cmd_benchmark)

    validate = sub.add_parser("validate")
    validate.add_argument("--binary", action="append")
    validate.add_argument("--dataset", default="artifacts/dataset/pilot")
    validate.add_argument("--checkpoint", default="models/smoke-test.pt")
    validate.add_argument("--out", default="artifacts/validation/phase1_5_report.json")
    validate.add_argument("--fuzz-iterations", type=int, default=200)
    validate.add_argument("--reconstruction-limit", type=int, default=4)
    _add_json(validate)
    validate.set_defaults(func=cmd_validate)

    acquire = sub.add_parser("acquire")
    acquire_sub = acquire.add_subparsers(dest="target", required=True)
    acquire_discover = acquire_sub.add_parser("discover")
    acquire_discover.add_argument("--adapter", default="github")
    acquire_discover.add_argument("--query", required=True)
    acquire_discover.add_argument("--limit", type=int, default=20)
    acquire_discover.add_argument("--timeout", type=int, default=60)
    acquire_discover.add_argument("--out")
    _add_json(acquire_discover)
    acquire_discover.set_defaults(func=cmd_acquire_discover)
    acquire_run = acquire_sub.add_parser("run")
    acquire_run.add_argument("--config", default="configs/acquisition_projects.json")
    acquire_run.add_argument("--dataset", default="artifacts/dataset/realworld_v6_pilot2")
    acquire_run.add_argument("--reports", default="artifacts/acquisition")
    acquire_run.add_argument("--max-projects", type=int, default=30)
    acquire_run.add_argument("--no-resume", action="store_true")
    acquire_run.add_argument("--no-swh", action="store_true")
    _add_json(acquire_run)
    acquire_run.set_defaults(func=cmd_acquire_run)
    acquire_status = acquire_sub.add_parser("status")
    acquire_status.add_argument("--dataset", default="artifacts/dataset/realworld_v6_pilot2")
    _add_json(acquire_status)
    acquire_status.set_defaults(func=cmd_acquire_status)

    build_matrix = sub.add_parser("build-matrix")
    build_matrix.add_argument("--dataset", default="artifacts/dataset/realworld_v6_pilot2")
    build_matrix.add_argument("--matrix", choices=["pilot", "extended", "fixture"], default="pilot")
    build_matrix.add_argument("--max-projects", type=int)
    build_matrix.add_argument("--project", action="append")
    build_matrix.add_argument("--jobs", type=int, default=2)
    build_matrix.add_argument("--no-reproduce", action="store_true")
    _add_json(build_matrix)
    build_matrix.set_defaults(func=cmd_build_matrix)

    align = sub.add_parser("align")
    align_sub = align.add_subparsers(dest="target", required=True)
    align_corpus = align_sub.add_parser("corpus")
    align_corpus.add_argument("--dataset", default="artifacts/dataset/realworld_v6_pilot2")
    align_corpus.add_argument("--max-projects", type=int)
    align_corpus.add_argument("--no-transfer", action="store_true")
    _add_json(align_corpus)
    align_corpus.set_defaults(func=cmd_align_corpus)
    align_fixture = align_sub.add_parser("fixture")
    align_fixture.add_argument("--out", default="artifacts/acquisition/alignment_fixtures")
    align_fixture.add_argument("--reports", default="artifacts/acquisition")
    _add_json(align_fixture)
    align_fixture.set_defaults(func=cmd_align_fixture)

    representations = sub.add_parser("representations")
    representations_sub = representations.add_subparsers(dest="target", required=True)
    representations_audit = representations_sub.add_parser("audit")
    representations_audit.add_argument("--dataset", default="artifacts/dataset/realworld_v6_pilot2")
    representations_audit.add_argument("--reports", default="artifacts/acquisition")
    _add_json(representations_audit)
    representations_audit.set_defaults(func=cmd_representations)

    tasks = sub.add_parser("tasks")
    tasks_sub = tasks.add_subparsers(dest="target", required=True)
    tasks_build = tasks_sub.add_parser("build")
    tasks_build.add_argument("--dataset", default="artifacts/dataset/realworld_v6_pilot2")
    tasks_build.add_argument("--reports", default="artifacts/acquisition")
    _add_json(tasks_build)
    tasks_build.set_defaults(func=cmd_tasks_build)


    java = sub.add_parser("java")
    java_sub = java.add_subparsers(dest="target", required=True)
    java_analyze = java_sub.add_parser("analyze")
    java_analyze.add_argument("path")
    java_analyze.add_argument("--full", action="store_true")
    java_analyze.add_argument("--out")
    _add_json(java_analyze)
    java_analyze.set_defaults(func=cmd_java_analyze)
    java_reconstruct = java_sub.add_parser("reconstruct")
    java_reconstruct.add_argument("path")
    java_reconstruct.add_argument("--out")
    java_reconstruct.add_argument("--project-name", default="nexus_java_reconstruction")
    java_reconstruct.add_argument("--max-classes", type=int, default=2000)
    java_reconstruct.add_argument("--verify", action="store_true")
    _add_json(java_reconstruct)
    java_reconstruct.set_defaults(func=cmd_java_reconstruct)

    jni = sub.add_parser("jni")
    jni_sub = jni.add_subparsers(dest="target", required=True)
    jni_correlate = jni_sub.add_parser("correlate")
    jni_correlate.add_argument("java")
    jni_correlate.add_argument("native")
    jni_correlate.add_argument("--out")
    _add_json(jni_correlate)
    jni_correlate.set_defaults(func=cmd_jni_correlate)

    research = sub.add_parser("research")
    research.add_argument("query")
    research.add_argument("--no-headers", action="store_true")
    research.add_argument("--limit", type=int, default=8)
    research.add_argument("--out")
    _add_json(research)
    research.set_defaults(func=cmd_research)

    report = sub.add_parser("report")
    report.add_argument("analysis_id", nargs="?")
    report.add_argument("--binary", help="build a comprehensive evidence-linked report for this native binary")
    report.add_argument("--java", help="optional Java class/DEX/APK/JAR artifact for Java reconstruction and JNI correlation")
    report.add_argument("--native-so", help="optional native library for JNI correlation")
    report.add_argument("--out", help="report output directory")
    report.add_argument("--max-functions", type=int, default=40)
    _add_json(report)
    report.set_defaults(func=cmd_report)

    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    dashboard = sub.add_parser("dashboard", help="start the web dashboard")
    dashboard.add_argument("--host", default="127.0.0.1")
    dashboard.add_argument("--port", type=int, default=8000)
    dashboard.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except NexusError as exc:
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0
    except Exception as exc:
        print(json.dumps({"code": type(exc).__name__, "message": str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
