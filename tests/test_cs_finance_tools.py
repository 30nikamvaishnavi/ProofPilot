import pytest

from proofpilot.tools.cs_tools import MinimumSpanningTree, ShortestPath, TopologicalSort, build_graph
from proofpilot.tools.finance_tools import BlackScholes, PortfolioVariance, bs_price

EDGES = [["A", "B", 4], ["A", "C", 1], ["C", "B", 2], ["B", "D", 5]]


def run(tool, args):
    a = tool.validate(args)
    out = tool.run(a)
    return a, out, tool.verify(a, out)


def test_shortest_path():
    _, out, v = run(ShortestPath(), {"edges": EDGES, "source": "A", "target": "D"})
    assert out.value == {"path": ["A", "C", "B", "D"], "cost": 8.0} and v.ok
    assert "brute force" in v.method


def test_shortest_path_directed_blocks_reverse():
    t = ShortestPath()
    with pytest.raises(ValueError, match="no path"):
        t.run(t.validate({"edges": [["A", "B", 1]], "source": "B", "target": "A", "directed": True}))


def test_shortest_path_integer_labels_and_string_lookup():
    _, out, v = run(ShortestPath(), {"edges": [[1, 2, 3], [2, 3, 4]], "source": "1", "target": 3})
    assert out.value["cost"] == 7.0 and v.ok


def test_shortest_path_errors():
    t = ShortestPath()
    with pytest.raises(ValueError, match="non-negative"):
        t.run(t.validate({"edges": [["A", "B", -1]], "source": "A", "target": "B"}))
    with pytest.raises(ValueError, match="not in the graph"):
        t.run(t.validate({"edges": EDGES, "source": "Z", "target": "A"}))
    with pytest.raises(ValueError, match="empty"):
        t.run(t.validate({"edges": [], "source": "A", "target": "B"}))
    with pytest.raises(ValueError, match=r"\[u, v, weight\]"):
        t.run(t.validate({"edges": [["A", "B"]], "source": "A", "target": "B"}))


def test_parallel_edges_keep_minimum_weight():
    g = build_graph([("A", "B", 5.0), ("A", "B", 2.0), ("B", "A", 9.0)], directed=False)
    assert g["A"]["B"]["weight"] == 2.0


def test_shortest_path_verifier_rejects_wrong_cost_and_fake_edge():
    t = ShortestPath()
    a, out, _ = run(t, {"edges": EDGES, "source": "A", "target": "D"})
    out.value = {"path": ["A", "C", "B", "D"], "cost": 7.0}
    assert not t.verify(a, out).ok
    out.value = {"path": ["A", "D"], "cost": 1.0}
    assert not t.verify(a, out).ok


def test_mst():
    _, out, v = run(MinimumSpanningTree(), {"edges": EDGES})
    assert out.value["total_weight"] == 8.0 and len(out.value["edges"]) == 3 and v.ok


def test_mst_disconnected_raises():
    t = MinimumSpanningTree()
    with pytest.raises(ValueError, match="not connected"):
        t.run(t.validate({"edges": [["A", "B", 1], ["C", "D", 1]]}))


def test_mst_verifier_rejects_non_minimal():
    t = MinimumSpanningTree()
    a, out, _ = run(t, {"edges": EDGES})
    out.value = {"edges": [["A", "B", 4.0], ["A", "C", 1.0], ["B", "D", 5.0]], "total_weight": 10.0}
    assert not t.verify(a, out).ok


def test_topological_sort():
    _, out, v = run(TopologicalSort(), {"edges": [["a", "b"], ["b", "c"], ["a", "c"]]})
    assert out.value == ["a", "b", "c"] and v.ok


def test_topological_sort_cycle_and_bad_order():
    t = TopologicalSort()
    with pytest.raises(ValueError, match="cycle"):
        t.run(t.validate({"edges": [["a", "b"], ["b", "a"]]}))
    a, out, _ = run(t, {"edges": [["a", "b"], ["b", "c"]]})
    out.value = ["c", "b", "a"]
    assert not t.verify(a, out).ok
    out.value = ["a", "b"]
    assert not t.verify(a, out).ok


def test_black_scholes_reference_value():
    price, delta = bs_price(100, 105, 1, 0.05, 0.2, "call")
    assert abs(price - 8.0214) < 1e-3 and abs(delta - 0.5422) < 1e-3


def test_black_scholes_put_call_and_verifier():
    for kind in ("call", "put"):
        _, out, v = run(BlackScholes(), {"S": 100, "K": 105, "T": 1, "r": 0.05, "sigma": 0.2, "option_type": kind})
        assert v.ok, v.detail
    c, _ = bs_price(100, 100, 1, 0.03, 0.25, "call")
    p, _ = bs_price(100, 100, 1, 0.03, 0.25, "put")
    assert abs((c - p) - (100 - 100 * 2.718281828459045 ** -0.03)) < 1e-9


def test_black_scholes_input_validation():
    for bad in ({"S": -1}, {"sigma": 0}, {"T": 0}, {"option_type": "straddle"}):
        args = {"S": 100, "K": 105, "T": 1, "r": 0.05, "sigma": 0.2, **bad}
        with pytest.raises(ValueError):
            BlackScholes().validate(args)


def test_black_scholes_verifier_rejects_wrong_price():
    t = BlackScholes()
    a, out, _ = run(t, {"S": 100, "K": 105, "T": 1, "r": 0.05, "sigma": 0.2})
    out.value = {"price": 9.5, "delta": out.value["delta"]}
    assert not t.verify(a, out).ok


def test_portfolio_variance():
    _, out, v = run(PortfolioVariance(), {"weights": [0.5, 0.5], "cov": [[0.04, 0.01], [0.01, 0.09]]})
    assert abs(out.value["variance"] - 0.0375) < 1e-12 and v.ok


def test_portfolio_variance_errors():
    t = PortfolioVariance()
    with pytest.raises(ValueError, match="square"):
        t.run(t.validate({"weights": [1, 0], "cov": [[1]]}))
    with pytest.raises(ValueError, match="symmetric"):
        t.run(t.validate({"weights": [1, 0], "cov": [[1, 0.5], [0.1, 1]]}))
    with pytest.raises(ValueError, match="semi-definite"):
        t.run(t.validate({"weights": [1, 1], "cov": [[1, 2], [2, 1]]}))
