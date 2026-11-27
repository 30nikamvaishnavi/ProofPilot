"""Computer-science tools: graph algorithms verified by an independent algorithm."""

from __future__ import annotations

import math
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field

from ..schemas import VerifyResult
from .base import Tool, ToolOutput

MAX_EDGES = 500
BRUTE_FORCE_NODES = 9


def _canon_edges(edges: list[list[Any]], weighted: bool = True) -> list[tuple[Any, Any, float]]:
    if not edges:
        raise ValueError("edge list is empty")
    if len(edges) > MAX_EDGES:
        raise ValueError(f"more than {MAX_EDGES} edges is not supported")
    out = []
    for e in edges:
        if weighted:
            if len(e) != 3:
                raise ValueError(f"each edge must be [u, v, weight], got {e}")
            u, v, w = e
            if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w):
                raise ValueError(f"edge weight must be a finite number, got {w!r}")
            out.append((u, v, float(w)))
        else:
            if len(e) < 2:
                raise ValueError(f"each edge must be [u, v], got {e}")
            out.append((e[0], e[1], 1.0))
    return out


def build_graph(edges: list[tuple[Any, Any, float]], directed: bool) -> nx.Graph:
    """Parallel edges keep the minimum weight (documented behaviour)."""
    best: dict[tuple[Any, Any], float] = {}
    for u, v, w in edges:
        key = (u, v) if directed else tuple(sorted((u, v), key=str))
        best[key] = min(w, best.get(key, math.inf))
    g: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    for (u, v), w in best.items():
        g.add_edge(u, v, weight=w)
    return g


def _same_node(g: nx.Graph, label: Any) -> Any:
    """Match JSON labels loosely (1 vs "1")."""
    if label in g:
        return label
    for n in g.nodes:
        if str(n) == str(label):
            return n
    raise ValueError(f"node {label!r} is not in the graph")


class ShortestPathArgs(BaseModel):
    edges: list[list[Any]] = Field(max_length=MAX_EDGES)
    source: Any
    target: Any
    directed: bool = False


class ShortestPath(Tool):
    name = "shortest_path"
    domain = "cs"
    description = "Shortest weighted path between two nodes (non-negative weights)."
    arg_hint = 'edges: [[u, v, weight], ...]; source: node; target: node; directed: bool = false'
    assumptions = ["Edge weights are non-negative; parallel edges keep the smallest weight."]
    ArgsModel = ShortestPathArgs

    def run(self, args: ShortestPathArgs) -> ToolOutput:
        edges = _canon_edges(args.edges)
        if any(w < 0 for _, _, w in edges):
            raise ValueError("Dijkstra requires non-negative weights")
        g = build_graph(edges, args.directed)
        s, t = _same_node(g, args.source), _same_node(g, args.target)
        try:
            path = nx.dijkstra_path(g, s, t, weight="weight")
        except nx.NetworkXNoPath as exc:
            raise ValueError(f"no path from {s} to {t}") from exc
        cost = float(nx.path_weight(g, path, "weight"))
        path_strings = [str(n) for n in path]
        value = {"path": path_strings, "cost": cost}
        return ToolOutput(value, f"{' -> '.join(path_strings)} (cost {cost:g})")

    def verify(self, args: ShortestPathArgs, output: ToolOutput) -> VerifyResult:
        edges = _canon_edges(args.edges)
        g = build_graph(edges, args.directed)
        s, t = _same_node(g, args.source), _same_node(g, args.target)
        by_str = {str(n): n for n in g.nodes}
        path = [by_str[p] for p in output.value["path"]]
        claimed = float(output.value["cost"])
        if path[0] != s or path[-1] != t:
            return VerifyResult(ok=False, method="endpoint check", detail="path endpoints do not match source/target")
        recomputed = 0.0
        for u, v in zip(path, path[1:], strict=False):
            if not g.has_edge(u, v):
                return VerifyResult(ok=False, method="edge existence", detail=f"edge {u}-{v} is not in the graph")
            recomputed += g[u][v]["weight"]
        bf = float(nx.bellman_ford_path_length(g, s, t, weight="weight"))
        ok = math.isclose(recomputed, claimed, abs_tol=1e-9) and math.isclose(bf, claimed, abs_tol=1e-9)
        method = "edge-sum recomputation + Bellman-Ford"
        detail = f"claimed={claimed:g}, recomputed={recomputed:g}, bellman-ford={bf:g}"
        if ok and g.number_of_nodes() <= BRUTE_FORCE_NODES:
            best = min(
                (
                    sum(g[u][v]["weight"] for u, v in zip(p, p[1:], strict=False))
                    for p in nx.all_simple_paths(g, s, t)
                ),
                default=math.inf,
            )
            method += " + brute force over all simple paths"
            detail += f", brute-force={best:g}"
            ok = math.isclose(best, claimed, abs_tol=1e-9)
        return VerifyResult(ok=ok, method=method, detail=detail)


class MSTArgs(BaseModel):
    edges: list[list[Any]] = Field(max_length=MAX_EDGES)


class MinimumSpanningTree(Tool):
    name = "minimum_spanning_tree"
    domain = "cs"
    description = "Minimum spanning tree of a connected undirected weighted graph."
    arg_hint = "edges: [[u, v, weight], ...]"
    assumptions = ["The graph is undirected and connected; parallel edges keep the smallest weight."]
    ArgsModel = MSTArgs

    def run(self, args: MSTArgs) -> ToolOutput:
        g = build_graph(_canon_edges(args.edges), directed=False)
        if not nx.is_connected(g):
            raise ValueError("graph is not connected")
        t = nx.minimum_spanning_tree(g, algorithm="kruskal")
        edges: list[list[Any]] = sorted(
            [sorted([str(u), str(v)]) + [float(d["weight"])] for u, v, d in t.edges(data=True)]
        )
        total = float(sum(float(e[2]) for e in edges))
        return ToolOutput({"edges": edges, "total_weight": total}, f"MST weight {total:g} using {len(edges)} edges")

    def verify(self, args: MSTArgs, output: ToolOutput) -> VerifyResult:
        g = build_graph(_canon_edges(args.edges), directed=False)
        prim = nx.minimum_spanning_tree(g, algorithm="prim")
        prim_total = float(prim.size(weight="weight"))
        edges = output.value["edges"]
        claimed = float(output.value["total_weight"])
        recomputed = float(sum(e[2] for e in edges))
        sub = nx.Graph()
        sub.add_nodes_from(str(n) for n in g.nodes)
        sub.add_weighted_edges_from(edges)
        tree = nx.is_tree(sub)
        ok = tree and math.isclose(prim_total, claimed, abs_tol=1e-9) and math.isclose(recomputed, claimed, abs_tol=1e-9)
        return VerifyResult(ok=ok, method="Prim cross-check + tree property", detail=f"claimed={claimed:g}, prim={prim_total:g}, is_tree={tree}")


class TopoArgs(BaseModel):
    edges: list[list[Any]] = Field(max_length=MAX_EDGES)


class TopologicalSort(Tool):
    name = "topological_sort"
    domain = "cs"
    description = "A topological ordering of a directed acyclic graph (edges are [u, v] meaning u before v)."
    arg_hint = "edges: [[u, v], ...]"
    assumptions = ["The graph must be acyclic; cycles make the step fail."]
    ArgsModel = TopoArgs

    def run(self, args: TopoArgs) -> ToolOutput:
        edges = _canon_edges(args.edges, weighted=False)
        g = nx.DiGraph()
        g.add_edges_from((u, v) for u, v, _ in edges)
        if not nx.is_directed_acyclic_graph(g):
            raise ValueError("graph contains a cycle")
        order = [str(n) for n in nx.lexicographical_topological_sort(g, key=str)]
        return ToolOutput(order, "order: " + ", ".join(order))

    def verify(self, args: TopoArgs, output: ToolOutput) -> VerifyResult:
        edges = _canon_edges(args.edges, weighted=False)
        pos = {n: i for i, n in enumerate(output.value)}
        nodes = {str(x) for u, v, _ in edges for x in (u, v)}
        if set(pos) != nodes or len(pos) != len(output.value):
            return VerifyResult(ok=False, method="edge-order check", detail="order must list every node exactly once")
        for u, v, _ in edges:
            if pos[str(u)] >= pos[str(v)]:
                return VerifyResult(ok=False, method="edge-order check", detail=f"{u} must come before {v}")
        return VerifyResult(ok=True, method="edge-order check", detail="every edge u->v has pos(u) < pos(v)")
