import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nexus.datasets.factory import _split_for
from nexus.tasks.taxonomy import TASKS, eligibility_summary

TASK_ORDER = list(TASKS)
MAX_PAIRS_PER_PROJECT = 300
MAX_BLOCKS_PER_PROJECT = 800
MAX_INSTRUCTIONS_PER_PROJECT = 300
MAX_RECONSTRUCTION_PER_PROJECT = 300
MAX_TYPE_SAMPLES_PER_PROJECT = 500


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _project_split(project_id: str) -> str:
    return _split_for(project_id)


def _load_indexes(root: Path) -> dict[str, list[dict[str, Any]]]:
    summary_path = root / "alignments" / "corpus_alignment_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        projects = summary.get("projects", [])
    else:
        projects = []
    result: dict[str, list[dict[str, Any]]] = {}
    for project in projects:
        entries = []
        for item in project.get("indexes", []):
            if item.get("status") != "available":
                continue
            path = root / item["alignment_index"]
            if not path.exists():
                continue
            index = json.loads(path.read_text(encoding="utf-8"))
            entries.append({"kind": "debug", "variant_id": item.get("variant_id"), "index": index, "path": path})
        for item in project.get("stripped", []):
            if item.get("status") != "available":
                continue
            path = root / item["alignment_index"]
            if not path.exists():
                continue
            index = json.loads(path.read_text(encoding="utf-8"))
            entries.append({"kind": "stripped", "variant_id": item.get("variant_id"), "index": index, "path": path})
        result[project["project_id"]] = entries
    return result


def _function_descriptor(function: dict[str, Any], binary_sha: str, variant_id: str | None) -> dict[str, Any]:
    return {
        "binary_sha256": binary_sha,
        "variant_id": variant_id,
        "entry_va": function.get("entry_va"),
        "ranges": function.get("ranges", []),
        "size": function.get("size"),
        "instruction_count": function.get("instruction_count", len(function.get("instruction_ids", []))),
        "block_count": len(function.get("blocks", [])),
        "mnemonic_histogram": function.get("mnemonic_histogram", {}),
        "code_sha256": function.get("code_sha256"),
    }


def _key(function: dict[str, Any]) -> tuple[str, int, str] | None:
    if not function.get("source_file") or function.get("source_line") is None or not function.get("name"):
        return None
    return (function["source_file"], int(function["source_line"]), function["name"])


def _eligible_function(function: dict[str, Any], minimum: str = "DEBUG_DERIVED") -> bool:
    order = {"EXACT": 0, "DEBUG_DERIVED": 1, "SYMBOL_DERIVED": 2, "HEURISTIC": 3, "UNVERIFIED": 4, "UNAVAILABLE": 5}
    return order.get(function.get("classification"), 5) <= order.get(minimum, 1)


def _pair_confidence(first: dict[str, Any], second: dict[str, Any]) -> tuple[str, float]:
    order = {"EXACT": 0, "DEBUG_DERIVED": 1, "SYMBOL_DERIVED": 2, "HEURISTIC": 3, "UNVERIFIED": 4, "UNAVAILABLE": 5}
    worst = max(order.get(first.get("classification"), 5), order.get(second.get("classification"), 5))
    reverse = {value: key for key, value in order.items()}
    return reverse[worst], round(min(first.get("confidence", 0.5), second.get("confidence", 0.5)), 3)


def _write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(sample, sort_keys=True) + "\n" for sample in samples), encoding="utf-8")


def _source_snippet(path: Path, start_line: int, max_lines: int = 240) -> dict[str, Any] | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    if start_line < 1 or start_line > len(lines):
        return None
    begin = None
    depth = 0
    started = False
    end = None
    for index in range(start_line - 1, min(len(lines), start_line + max_lines)):
        line = lines[index]
        if begin is None and "{" in line:
            begin = index + 1
        if begin is None:
            continue
        for character in line:
            if character == "{":
                depth += 1
                started = True
            elif character == "}":
                depth -= 1
                if started and depth == 0:
                    end = index + 1
                    break
        if end is not None:
            break
    if begin is None or end is None:
        return None
    snippet = "\n".join(lines[start_line - 1 : end])
    return {"text": snippet, "start_line": start_line, "end_line": end, "line_count": end - start_line + 1}


def _parse_disassembly(path: Path) -> list[dict[str, Any]]:
    rows = []
    pattern = re.compile(r"^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2}\s)+)\s*(\S+)(?:\s+(.*))?$")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return rows
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        rows.append({"address": match.group(1), "bytes": match.group(2).strip(), "mnemonic": match.group(3), "operands": (match.group(4) or "").strip()})
    return rows


def _project_build_dir(root: Path, project_id: str, variant_id: str) -> Path:
    return root / "builds" / project_id / variant_id


def _pair_samples(project_id: str, split: str, task: str, first_index: dict[str, Any], second_index: dict[str, Any], label_source: str) -> list[dict[str, Any]]:
    first_binary = first_index.get("binary", {}).get("sha256")
    second_binary = second_index.get("binary", {}).get("sha256")
    first_map = {}
    for function in first_index.get("functions", []):
        key = _key(function)
        if key and _eligible_function(function):
            first_map.setdefault(key, function)
    second_map = {}
    for function in second_index.get("functions", []):
        key = _key(function)
        if key and _eligible_function(function):
            second_map.setdefault(key, function)
    common = sorted(set(first_map) & set(second_map))
    samples = []
    for key in common[:MAX_PAIRS_PER_PROJECT]:
        first = first_map[key]
        second = second_map[key]
        classification, confidence = _pair_confidence(first, second)
        samples.append(
            {
                "sample_id": f"{task}_{project_id}_{_hash([first_index.get('build_variant'), second_index.get('build_variant'), key])[:16]}",
                "task": task,
                "project_id": project_id,
                "split": split,
                "input": {
                    "a": _function_descriptor(first, first_binary, (first_index.get("build_variant") or {}).get("variant_id")),
                    "b": _function_descriptor(second, second_binary, (second_index.get("build_variant") or {}).get("variant_id")),
                },
                "target": {"match": True},
                "label_source": label_source,
                "label": {"source_file": key[0], "source_line": key[1], "function_name": key[2]},
                "alignment": {"level": "function", "classification": classification, "confidence": confidence},
                "known_shortcuts": ["build identity", "instruction count", "function size"],
                "leakage_risks": ["names and source locations must not be inputs"],
            }
        )
    negatives = []
    for first_key in sorted(first_map):
        if len(negatives) >= len(samples):
            break
        for second_key in sorted(second_map):
            if first_key != second_key:
                negatives.append((first_key, second_key))
                if len(negatives) >= len(samples):
                    break
    for first_key, second_key in negatives:
        first = first_map[first_key]
        second = second_map[second_key]
        classification, confidence = _pair_confidence(first, second)
        samples.append(
            {
                "sample_id": f"{task}_{project_id}_neg_{_hash([first_index.get('build_variant'), second_index.get('build_variant'), first_key, second_key])[:16]}",
                "task": task,
                "project_id": project_id,
                "split": split,
                "input": {
                    "a": _function_descriptor(first, first_binary, (first_index.get("build_variant") or {}).get("variant_id")),
                    "b": _function_descriptor(second, second_binary, (second_index.get("build_variant") or {}).get("variant_id")),
                },
                "target": {"match": False},
                "label_source": label_source,
                "label": {"a": {"source_file": first_key[0], "source_line": first_key[1], "function_name": first_key[2]}, "b": {"source_file": second_key[0], "source_line": second_key[1], "function_name": second_key[2]}},
                "alignment": {"level": "function", "classification": classification, "confidence": confidence},
                "known_shortcuts": ["build identity", "instruction count", "function size"],
                "leakage_risks": ["names and source locations must not be inputs"],
            }
        )
    return samples


def build_task_datasets(dataset_root: str | Path, reports_root: str | Path) -> dict[str, Any]:
    root = Path(dataset_root).resolve()
    reports = Path(reports_root).resolve()
    tasks_root = reports / "tasks"
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    catalog_by_id = {item["project_id"]: item for item in catalog if item.get("status") == "accepted"}
    indexes_by_project = _load_indexes(root)
    samples_by_task: dict[str, list[dict[str, Any]]] = {task: [] for task in TASK_ORDER}
    project_splits = {project_id: _project_split(project_id) for project_id in catalog_by_id}
    for project_id, entries in indexes_by_project.items():
        split = project_splits[project_id]
        project_counts = Counter()
        debug = [entry for entry in entries if entry["kind"] == "debug"]
        stripped = [entry for entry in entries if entry["kind"] == "stripped"]
        for index_entry in debug:
            index = index_entry["index"]
            binary_sha = index.get("binary", {}).get("sha256")
            for function in index.get("functions", []):
                if not _eligible_function(function):
                    continue
                if function.get("source_file") and function.get("source_line") is not None:
                    samples_by_task["source_to_binary_alignment"].append(
                        {
                            "sample_id": f"s2b_{project_id}_{_hash([index_entry.get('variant_id'), function.get('function_id')])[:16]}",
                            "task": "source_to_binary_alignment",
                            "project_id": project_id,
                            "split": split,
                            "input": {"binary_sha256": binary_sha, "binary_path": index.get("binary", {}).get("path"), "function": _function_descriptor(function, binary_sha, index_entry.get("variant_id"))},
                            "target": {"function_name": function.get("name"), "source_file": function.get("source_file"), "source_line": function.get("source_line"), "ranges": function.get("ranges", [])},
                            "label_source": "dwarf_subprogram_and_line_table",
                            "alignment": {"level": "function", "classification": function.get("classification"), "confidence": function.get("confidence")},
                            "known_shortcuts": ["function name", "source path", "instruction count"],
                            "leakage_risks": ["source text must not be an input", "project split isolation"],
                        }
                    )
            for block in index.get("basic_block_alignments", []):
                if block.get("classification") == "UNAVAILABLE" or not block.get("source_file"):
                    continue
                if project_counts["basic_block_correspondence"] >= MAX_BLOCKS_PER_PROJECT:
                    break
                project_counts["basic_block_correspondence"] += 1
                samples_by_task["basic_block_correspondence"].append(
                    {
                        "sample_id": f"bb_{project_id}_{_hash([index_entry.get('variant_id'), block.get('block_id')])[:16]}",
                        "task": "basic_block_correspondence",
                        "project_id": project_id,
                        "split": split,
                        "input": {"binary_sha256": binary_sha, "function_id": block.get("function_id"), "block_id": block.get("block_id"), "start_va": block.get("start_va"), "end_va": block.get("end_va"), "instruction_count": len(block.get("instruction_ids", []))},
                        "target": {"source_file": block.get("source_file"), "start_line": block.get("start_line"), "end_line": block.get("end_line")},
                        "label_source": "cfg_line_table_intersection",
                        "alignment": {"level": "basic_block", "classification": "HEURISTIC", "confidence": block.get("confidence", 0.5)},
                        "known_shortcuts": ["line-table over-approximation", "CFG recovery heuristic"],
                        "leakage_risks": ["debug files must not be inputs"],
                    }
                )
            parameter_positions = {}
            for variable in index.get("variables", []):
                if variable.get("role") != "parameter" or variable.get("type") in {None, "", "unknown", "void"}:
                    continue
                function = next((item for item in index.get("functions", []) if item.get("function_id") == variable.get("function_id")), None)
                if function is None:
                    continue
                position = parameter_positions.get(function.get("function_id"), 0)
                parameter_positions[function.get("function_id")] = position + 1
                if project_counts["type_dataflow_inference"] >= MAX_TYPE_SAMPLES_PER_PROJECT:
                    break
                project_counts["type_dataflow_inference"] += 1
                samples_by_task["type_dataflow_inference"].append(
                    {
                        "sample_id": f"type_{project_id}_{_hash([index_entry.get('variant_id'), variable.get('variable_id')])[:16]}",
                        "task": "type_dataflow_inference",
                        "project_id": project_id,
                        "split": split,
                        "input": {"binary_sha256": binary_sha, "function": _function_descriptor(function, binary_sha, index_entry.get("variant_id")), "parameter_index": position},
                        "target": {"name": variable.get("name"), "type": variable.get("type"), "role": variable.get("role"), "location_kind": (variable.get("location") or {}).get("kind"), "source_line": variable.get("source_line")},
                        "label_source": "dwarf_type_and_location",
                        "alignment": {"level": "variable", "classification": variable.get("classification"), "confidence": variable.get("confidence")},
                        "known_shortcuts": ["ABI width", "parameter count"],
                        "leakage_risks": ["debug dumps must not be inputs"],
                    }
                )
                if len(samples_by_task["type_dataflow_inference"]) >= MAX_TYPE_SAMPLES_PER_PROJECT * max(1, len(indexes_by_project)):
                    break
            for function in index.get("functions", []):
                if not _eligible_function(function) or not function.get("source_file") or function.get("source_line") is None:
                    continue
                source_path = root / catalog_by_id[project_id]["source_root"] / function["source_file"]
                snippet = _source_snippet(source_path, int(function["source_line"]))
                if snippet is None:
                    continue
                if project_counts["c_reconstruction"] >= MAX_RECONSTRUCTION_PER_PROJECT:
                    break
                project_counts["c_reconstruction"] += 1
                samples_by_task["c_reconstruction"].append(
                    {
                        "sample_id": f"crec_{project_id}_{_hash([index_entry.get('variant_id'), function.get('function_id')])[:16]}",
                        "task": "c_reconstruction",
                        "project_id": project_id,
                        "split": split,
                        "input": {"binary_sha256": binary_sha, "function": _function_descriptor(function, binary_sha, index_entry.get("variant_id")), "source_file": None},
                        "target": {"source_text": snippet["text"], "source_file": function.get("source_file"), "start_line": snippet["start_line"], "end_line": snippet["end_line"]},
                        "label_source": "source_text_at_aligned_span",
                        "alignment": {"level": "function", "classification": function.get("classification"), "confidence": function.get("confidence")},
                        "verification": {"status": "unverified", "reason": "no behavioral test or reconstruction compile check in this task manifest"},
                        "known_shortcuts": ["function name", "source formatting", "comments"],
                        "leakage_risks": ["source text must not be an input", "project split isolation"],
                    }
                )
            if index_entry.get("variant_id", "").startswith(("gcc_O0_debug", "clang_O0_debug")):
                disassembly_path = _project_build_dir(root, project_id, index_entry.get("variant_id", "")) / "disassembly.txt"
                rows = _parse_disassembly(disassembly_path)
                window = 16
                stride = 8
                made = 0
                for start in range(0, max(0, len(rows) - window), stride):
                    if made >= MAX_INSTRUCTIONS_PER_PROJECT:
                        break
                    chunk = rows[start : start + window]
                    mask_index = window // 2
                    tokens = [f"{item['mnemonic']} {item['operands']}".strip() for item in chunk]
                    target = tokens[mask_index]
                    tokens[mask_index] = "<mask>"
                    samples_by_task["instruction_semantics"].append(
                        {
                            "sample_id": f"ins_{project_id}_{_hash([index_entry.get('variant_id'), start])[:16]}",
                            "task": "instruction_semantics",
                            "task_variant": "masked_instruction_token",
                            "project_id": project_id,
                            "split": split,
                            "input": {"binary_sha256": binary_sha, "tokens": tokens, "mask_position": mask_index},
                            "target": {"token": target, "kind": "masked_token"},
                            "label_source": "self_supervised_input_tokens",
                            "alignment": {"level": "instruction", "classification": "UNVERIFIED", "confidence": 0.95, "basis": "disassembly_tokens", "note": "token sequence is verified by disassembly; source alignment is not claimed"},
                            "known_shortcuts": ["adjacent-token regularity", "compiler idioms"],
                            "leakage_risks": ["do not evaluate on training mask positions", "source text is not an input"],
                        }
                    )
                    made += 1
        for opt in ("O0", "O2"):
            gcc = next((entry for entry in debug if entry.get("variant_id") == f"gcc_{opt}_debug_pie_executable"), None)
            clang = next((entry for entry in debug if entry.get("variant_id") == f"clang_{opt}_debug_pie_executable"), None)
            if gcc and clang:
                samples_by_task["cross_compiler_function_matching"].extend(_pair_samples(project_id, split, "cross_compiler_function_matching", gcc["index"], clang["index"], "dwarf_and_symbol_identity"))
            gcc_shared = next((entry for entry in debug if entry.get("variant_id") == f"gcc_{opt}_debug_pie_shared"), None)
            clang_shared = next((entry for entry in debug if entry.get("variant_id") == f"clang_{opt}_debug_pie_shared"), None)
            if gcc_shared and clang_shared:
                samples_by_task["cross_compiler_function_matching"].extend(_pair_samples(project_id, split, "cross_compiler_function_matching", gcc_shared["index"], clang_shared["index"], "dwarf_and_symbol_identity"))
        for compiler in ("gcc", "clang"):
            for link in ("executable", "shared"):
                o0 = next((entry for entry in debug if entry.get("variant_id") == f"{compiler}_O0_debug_pie_{link}"), None)
                o2 = next((entry for entry in debug if entry.get("variant_id") == f"{compiler}_O2_debug_pie_{link}"), None)
                if o0 and o2:
                    samples_by_task["cross_optimization_function_matching"].extend(_pair_samples(project_id, split, "cross_optimization_function_matching", o0["index"], o2["index"], "dwarf_and_symbol_identity"))
        for entry in stripped:
            index = entry["index"]
            functions = [function for function in index.get("functions", []) if function.get("ranges")]
            if not functions:
                continue
            samples_by_task["function_boundary_detection"].append(
                {
                    "sample_id": f"boundary_{project_id}_{_hash(entry.get('variant_id'))[:16]}",
                    "task": "function_boundary_detection",
                    "project_id": project_id,
                    "split": split,
                    "input": {"binary_sha256": index.get("binary", {}).get("sha256"), "binary_path": index.get("binary", {}).get("path"), "format": "ELF", "machine": index.get("binary", {}).get("machine")},
                    "target": {"functions": [{"name": function.get("name"), "ranges": function.get("ranges"), "classification": function.get("classification")} for function in functions]},
                    "label_source": "debug_function_boundaries_transferred_by_text_identity",
                    "alignment": {"level": "function", "classification": "DEBUG_DERIVED", "confidence": 0.95, "method": (index.get("transfer") or {}).get("method")},
                    "known_shortcuts": ["startup/thunk functions", "alignment padding"],
                    "leakage_risks": ["debug companion must not be an input"],
                }
            )
        for compiler in ("gcc", "clang"):
            for link in ("executable", "shared"):
                o0 = next((entry for entry in debug if entry.get("variant_id") == f"{compiler}_O0_debug_pie_{link}"), None)
                o2 = next((entry for entry in debug if entry.get("variant_id") == f"{compiler}_O2_debug_pie_{link}"), None)
                if not (o0 and o2):
                    continue
                first_map = {_key(function): function for function in o0["index"].get("functions", []) if _key(function) and _eligible_function(function)}
                second_map = {_key(function): function for function in o2["index"].get("functions", []) if _key(function) and _eligible_function(function)}
                for key in sorted(set(first_map) & set(second_map))[:MAX_PAIRS_PER_PROJECT]:
                    first = first_map[key]
                    second = second_map[key]
                    first_hist = Counter(first.get("mnemonic_histogram") or {})
                    second_hist = Counter(second.get("mnemonic_histogram") or {})
                    delta = {name: second_hist.get(name, 0) - first_hist.get(name, 0) for name in sorted(set(first_hist) | set(second_hist))}
                    classification, confidence = _pair_confidence(first, second)
                    samples_by_task["compiler_transformation_analysis"].append(
                        {
                            "sample_id": f"xform_{project_id}_{_hash([compiler, key])[:16]}",
                            "task": "compiler_transformation_analysis",
                            "project_id": project_id,
                            "split": split,
                            "input": {"a": _function_descriptor(first, o0["index"].get("binary", {}).get("sha256"), o0.get("variant_id")), "b": _function_descriptor(second, o2["index"].get("binary", {}).get("sha256"), o2.get("variant_id"))},
                            "target": {"instruction_count_delta": second.get("instruction_count", 0) - first.get("instruction_count", 0), "block_count_delta": len(second.get("blocks", [])) - len(first.get("blocks", [])), "size_delta": (second.get("size") or 0) - (first.get("size") or 0), "mnemonic_delta": delta},
                            "label_source": "aligned_binary_differential",
                            "alignment": {"level": "function", "classification": classification, "confidence": confidence},
                            "known_shortcuts": ["optimization correlation", "size dominance"],
                            "leakage_risks": ["optimization labels must not be inputs"],
                        }
                    )

    for task in TASK_ORDER:
        _write_jsonl(tasks_root / f"{task}.jsonl", samples_by_task[task])
    summary = {
        "schema_version": "0.1.0",
        "created_at": _now(),
        "dataset_root": str(root),
        "project_splits": project_splits,
        "tasks": {},
        "cards": eligibility_summary(),
    }
    for task in TASK_ORDER:
        samples = samples_by_task[task]
        splits = Counter(sample["split"] for sample in samples)
        projects = sorted({sample["project_id"] for sample in samples})
        confidences = Counter((sample.get("alignment") or {}).get("classification", "UNAVAILABLE") for sample in samples)
        label_sources = Counter(sample.get("label_source") for sample in samples)
        if task == "function_purpose_inference":
            eligibility = "UNAVAILABLE"
        elif task in {"basic_block_correspondence", "instruction_semantics", "compiler_transformation_analysis", "type_dataflow_inference", "c_reconstruction"}:
            eligibility = "PARTIAL"
        else:
            eligibility = "PASS" if samples else "UNAVAILABLE"
        summary["tasks"][task] = {
            "samples": len(samples),
            "projects": len(projects),
            "splits": dict(sorted(splits.items())),
            "alignment_classifications": dict(sorted(confidences.items())),
            "label_sources": dict(sorted(label_sources.items())),
            "eligibility": eligibility,
            "eligibility_detail": (
                "function-level alignment PASS; instruction-level PARTIAL (DWARF line-table coverage only)"
                if task == "source_to_binary_alignment" and eligibility == "PASS"
                else "self-supervised masked-token data only; semantic category labels unavailable"
                if task == "instruction_semantics"
                else "source targets available; behavioral verification unavailable"
                if task == "c_reconstruction"
                else "heuristic or partial evidence; see task card"
                if eligibility == "PARTIAL"
                else "no verified labels; not fabricated"
                if eligibility == "UNAVAILABLE"
                else "explicit eligibility recorded in the task card"
            ),
            "card": TASKS[task],
            "path": str((tasks_root / f"{task}.jsonl").relative_to(reports)),
        }
    report = {
        "schema_version": "0.1.0",
        "created_at": _now(),
        "status": "passed" if all(summary["tasks"][task]["samples"] > 0 for task in TASK_ORDER if task != "function_purpose_inference") else "partial",
        "tasks": summary["tasks"],
        "project_splits": project_splits,
        "policy": "Eligibility is explicit per task. A sample being present does not imply that every capability is verified. Function-purpose labels are UNAVAILABLE and are not fabricated.",
        "limitations": [
            "Task manifests describe dataset eligibility, not trained-model performance.",
            "Instruction semantics is limited to self-supervised masked tokens; semantic category labels are unavailable.",
            "C reconstruction samples are unverified until compile and behavioral checks are run.",
            "Type/dataflow labels cover DWARF-declared parameters and simple locations only.",
        ],
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "task_eligibility_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
