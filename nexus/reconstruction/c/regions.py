"""Bounded region analysis used by the region-aware reconstruction strategy.

The analysis identifies dominators, natural loops and simple if-diamonds.  The
current strategy emits CFG labels/gotos for all regions; the analysis is kept
separate so the comparison can test whether region knowledge alone improves
behavior.  No source-level structure is invented.
"""

from typing import Any

from nexus.program.model import FunctionCandidate


def analyze_regions(function: FunctionCandidate) -> dict[str, Any]:
    blocks = {block.block_id: block for block in function.blocks}
    entry = next((block.block_id for block in function.blocks if block.start == function.entry), function.blocks[0].block_id)
    predecessors: dict[str, set[str]] = {block_id: set() for block_id in blocks}
    for block in blocks.values():
        for successor in block.successors:
            if successor in predecessors:
                predecessors[successor].add(block.block_id)
    dominators: dict[str, set[str]] = {block_id: set(blocks) for block_id in blocks}
    dominators[entry] = {entry}
    changed = True
    while changed:
        changed = False
        for block_id in blocks:
            if block_id == entry:
                continue
            preds = predecessors[block_id]
            incoming = set.intersection(*(dominators[pred] for pred in preds)) if preds else set()
            updated = incoming | {block_id}
            if updated != dominators[block_id]:
                dominators[block_id] = updated
                changed = True
    back_edges = []
    loops = []
    for block_id, block in blocks.items():
        for successor in block.successors:
            if successor in blocks and successor in dominators.get(block_id, set()):
                back_edges.append((block_id, successor))
                loop_nodes = {successor, block_id}
                for candidate in blocks:
                    if successor in dominators.get(candidate, set()) and candidate != successor:
                        loop_nodes.add(candidate)
                loops.append({"header": successor, "latch": block_id, "nodes": sorted(loop_nodes), "single_latch": True})
    diamonds = []
    for block_id, block in blocks.items():
        if len(block.successors) != 2:
            continue
        left, right = block.successors
        if left not in blocks or right not in blocks:
            continue
        left_successors = blocks[left].successors
        right_successors = blocks[right].successors
        if len(left_successors) == 1 and left_successors == right_successors:
            diamonds.append({"condition": block_id, "then": left, "else": right, "join": left_successors[0]})
    cyclic = {node for loop in loops for node in loop["nodes"]}
    irreducible = sorted(
        block_id for block_id in cyclic
        if any(successor in cyclic and successor not in dominators.get(block_id, set()) for successor in blocks[block_id].successors)
    )
    return {
        "blocks": len(blocks),
        "dominators": {block_id: sorted(values) for block_id, values in dominators.items()},
        "back_edges": [{"from": left, "to": right} for left, right in back_edges],
        "natural_loops": loops,
        "if_diamonds": diamonds,
        "irreducible_blocks": irreducible,
        "reducible": not irreducible,
        "emission": "cfg_labels_and_gotos",
        "limitation": "Region knowledge is measured but not rewritten into source-level if/while constructs in this strategy.",
    }
