import pytest
from conftest_helpers import FakeLLM

from proofpilot.planner import MAX_STEPS, PlanningError, heuristic_plan, make_plan
from proofpilot.tools.base import default_registry

REG = default_registry()
GRAPH = "[['A','B',4],['A','C',1],['C','B',2],['B','D',5]]"


@pytest.mark.parametrize("problem,tool", [
    ("solve x^2 - 5x + 6 = 0", "solve_equation"),
    ("Differentiate x^2*sin(x)", "differentiate"),
    ("what is the derivative of exp(2x)", "differentiate"),
    ("integrate sin(x) from 0 to pi", "integrate"),
    ("Integrate x*exp(x) with respect to x", "integrate"),
    ("determinant of [[1,2],[3,4]]", "matrix_ops"),
    ("eigenvalues of [[2,0],[0,3]]", "matrix_ops"),
    ("is 97 prime?", "number_theory"),
    ("factorize 360", "number_theory"),
    ("gcd of 84 and 36", "number_theory"),
    ("solve A=[[2,1],[1,3]] b=[3,5]", "linear_system"),
    (f"shortest path from A to D in {GRAPH}", "shortest_path"),
    (f"minimum spanning tree of {GRAPH}", "minimum_spanning_tree"),
    ("topological order of [['a','b'],['b','c']]", "topological_sort"),
    ("price a european call S=100 K=105 T=1 r=0.05 sigma=0.2", "black_scholes"),
    ("portfolio variance weights=[0.5,0.5] cov=[[0.04,0.01],[0.01,0.09]]", "portfolio_variance"),
])
def test_heuristic_routes_to_the_right_tool(problem, tool):
    plan = heuristic_plan(problem)
    assert [s.tool for s in plan.steps] == [tool]
    REG.get(tool).validate(plan.steps[0].args)  # arguments must satisfy the tool's schema


def test_heuristic_extracts_arguments():
    s = heuristic_plan("integrate sin(x) from 0 to pi").steps[0]
    assert s.args == {"expression": "sin(x)", "variable": "x", "lower": "0", "upper": "pi"}
    s = heuristic_plan("price a put S=90 K=100 T=0.5 r=0.02 sigma=0.3").steps[0]
    assert s.args["option_type"] == "put" and s.args["T"] == 0.5
    s = heuristic_plan("solve 2y+3=7 for y").steps[0]
    assert s.args == {"equation": "2y+3=7", "variable": "y"}
    s = heuristic_plan(f"shortest path from A to D in {GRAPH} directed").steps[0]
    assert s.args["directed"] is True and s.args["source"] == "A"


def test_multi_step_split():
    plan = heuristic_plan("factorize 91; then is 91 prime")
    assert [s.id for s in plan.steps] == [1, 2]
    assert [s.args["operation"] for s in plan.steps] == ["factorize", "is_prime"]


@pytest.mark.parametrize("bad", ["", "   ", "hello world", "price a call S=100", "shortest path in [['A','B',1]]", "gcd of 5"])
def test_heuristic_errors_are_planning_errors(bad):
    with pytest.raises(PlanningError):
        heuristic_plan(bad)


def test_too_many_steps_and_too_long():
    with pytest.raises(PlanningError, match="too many"):
        heuristic_plan("; ".join(["factorize 12"] * (MAX_STEPS + 1)))
    with pytest.raises(PlanningError, match="longer"):
        make_plan("x" * 2001, REG)


def test_llm_plan_is_used_and_schema_checked():
    llm = FakeLLM({"steps": [{"tool": "number_theory", "args": {"operation": "gcd", "a": 12, "b": 18}, "description": "gcd"}]})
    plan = make_plan("anything", REG, llm)
    assert plan.planner == "llm:fake" and plan.steps[0].args["a"] == 12
    assert "untrusted data" in llm.prompts[0]


@pytest.mark.parametrize("response", [
    {"steps": [{"tool": "rm_rf", "args": {}}]},                               # unknown tool
    {"steps": [{"tool": "number_theory", "args": {"operation": "bogus"}}]},     # schema violation
    {"steps": []},                                                           # empty
    {"nothing": 1},                                                          # wrong shape
    RuntimeError("api down"),                                                # LLM failure
])
def test_bad_llm_output_falls_back_to_rules(response):
    plan = make_plan("is 97 prime", REG, FakeLLM(response))
    assert plan.planner == "heuristic" and plan.steps[0].tool == "number_theory"
