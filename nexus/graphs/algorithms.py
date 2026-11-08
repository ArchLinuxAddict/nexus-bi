from collections import deque
from collections.abc import Hashable, Iterable


def bfs(start: Hashable, adjacency: dict[Hashable, Iterable[Hashable]], limit: int | None = None) -> list[Hashable]:
    visited = {start}
    queue = deque([start])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        if limit is not None and len(order) >= limit:
            break
        for neighbor in adjacency.get(node, ()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order


def reachable(start: Hashable, adjacency: dict[Hashable, Iterable[Hashable]]) -> set[Hashable]:
    return set(bfs(start, adjacency))


def reverse_adjacency(adjacency: dict[Hashable, Iterable[Hashable]]) -> dict[Hashable, list[Hashable]]:
    reversed_graph: dict[Hashable, list[Hashable]] = {}
    for node, neighbors in adjacency.items():
        reversed_graph.setdefault(node, [])
        for neighbor in neighbors:
            reversed_graph.setdefault(neighbor, []).append(node)
    return reversed_graph


def strongly_connected_components(adjacency: dict[Hashable, Iterable[Hashable]]) -> list[list[Hashable]]:
    index_counter = [0]
    stack: list[Hashable] = []
    on_stack: set[Hashable] = set()
    indices: dict[Hashable, int] = {}
    lowlinks: dict[Hashable, int] = {}
    components: list[list[Hashable]] = []

    def visit(node: Hashable) -> None:
        indices[node] = index_counter[0]
        lowlinks[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack.add(node)
        for neighbor in adjacency.get(node, ()):
            if neighbor not in indices:
                visit(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])
        if lowlinks[node] == indices[node]:
            component = []
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.append(member)
                if member == node:
                    break
            components.append(component)

    for node in list(adjacency):
        if node not in indices:
            visit(node)
    return components
