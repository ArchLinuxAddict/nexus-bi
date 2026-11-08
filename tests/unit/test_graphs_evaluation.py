from nexus.evaluation.benchmarks import benchmark_binary, benchmark_paths
from nexus.graphs.algorithms import (
    bfs,
    reverse_adjacency,
    strongly_connected_components,
)


def test_graph_algorithms():
    graph = {"a": ["b"], "b": ["c"], "c": ["a", "d"], "d": []}
    assert bfs("a", graph) == ["a", "b", "c", "d"]
    components = strongly_connected_components(graph)
    assert sorted(sorted(component) for component in components) == [["a", "b", "c"], ["d"]]
    reversed_graph = reverse_adjacency(graph)
    assert "a" in reversed_graph["b"]


def test_benchmark_binary(simple_binary):
    result = benchmark_binary(simple_binary)
    assert result["validation"] == "passed"
    assert result["functions"] > 0
    assert result["instructions"] > 0
    assert result["analysis_seconds"] >= 0
    suite = benchmark_paths([simple_binary])
    assert suite["status"] == "completed"
    assert suite["summary"]["validation_passed"] == 1
