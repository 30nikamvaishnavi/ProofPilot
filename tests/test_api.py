"""HTTP API tests. Need fastapi + httpx (installed via requirements-dev.txt)."""

import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from proofpilot.api import create_app  # noqa: E402
from proofpilot.engine import Runtime  # noqa: E402


def client() -> TestClient:
    return TestClient(create_app(Runtime()))


def test_health_and_tools():
    c = client()
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["tools"] == 11
    assert len(c.get("/tools").json()) == 11


def test_solve_roundtrip_and_memory():
    c = client()
    r = c.post("/solve", json={"problem": "solve x^2-5x+6=0"})
    assert r.status_code == 200 and r.json()["status"] == "verified" and r.json()["final_answer"] == "x = 2, x = 3"
    assert c.get("/memory").json()["count"] == 1


def test_solve_with_explicit_plan():
    plan = {"problem": "p", "steps": [{"id": 1, "tool": "number_theory", "args": {"operation": "gcd", "a": 12, "b": 18}}]}
    r = client().post("/solve", json={"problem": "p", "plan": plan})
    assert r.status_code == 200 and r.json()["final_answer"] == "gcd(12, 18) = 6"


def test_validation_errors():
    c = client()
    assert c.post("/solve", json={"problem": ""}).status_code == 422
    assert c.post("/solve", json={"problem": "x" * 2001}).status_code == 422
    assert c.post("/solve", json={"problem": "is 7 prime", "orchestrator": "magic"}).status_code == 422


def test_unplannable_problem_is_a_failed_report_not_a_500():
    r = client().post("/solve", json={"problem": "tell me a joke"})
    assert r.status_code == 200 and r.json()["status"] == "failed"


def test_spec_flow_with_human_approval_gate():
    c = client()
    d = c.post("/spec/draft", json={"text": "For all integers a, b: a*b = b*a"}).json()
    sid = d["id"]
    assert d["status"] == "draft"
    assert c.get(f"/spec/{sid}/code").status_code == 409
    assert c.get(f"/spec/{sid}/check").json()["passed"] is True
    assert c.post(f"/spec/{sid}/approve", json={}).json()["status"] == "approved"
    assert "def claim(a, b)" in c.get(f"/spec/{sid}/code").json()["code"]


def test_spec_errors():
    c = client()
    assert c.post("/spec/draft", json={"text": "make me a sandwich"}).status_code == 422
    assert c.get("/spec/nope/check").status_code == 404
    assert c.post("/spec/nope/approve", json={}).status_code == 404


def test_api_key_protection():
    os.environ["PROOFPILOT_API_KEY"] = "secret"
    try:
        c = client()
        assert c.get("/health").status_code == 200
        assert c.get("/tools").status_code == 401
        assert c.get("/tools", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/tools", headers={"X-API-Key": "secret"}).status_code == 200
    finally:
        del os.environ["PROOFPILOT_API_KEY"]
