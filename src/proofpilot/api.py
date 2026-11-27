"""FastAPI service. ``uvicorn proofpilot.api:app``"""

from __future__ import annotations

import hmac
import os
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .engine import Runtime, run_engine
from .llm import get_llm
from .memory import MemoryStore
from .schemas import Plan, Report
from .spec import SpecError, SpecWorkflow
from .tools.base import default_registry


class SolveRequest(BaseModel):
    problem: str = Field(min_length=1, max_length=2000)
    plan: Plan | None = None
    orchestrator: str = "custom"  # "custom" or "langgraph"


class DraftRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1500)


class ApproveRequest(BaseModel):
    edited: dict[str, Any] | None = None


def create_app(runtime: Runtime | None = None) -> FastAPI:
    llm = get_llm()
    rt = runtime or Runtime(registry=default_registry(), memory=MemoryStore(os.environ.get("PROOFPILOT_DB", ":memory:")), llm=llm)
    specs = SpecWorkflow(rt.llm)
    app = FastAPI(title="ProofPilot", version=__version__, description="Verified multi-step STEM agent")
    api_key = os.environ.get("PROOFPILOT_API_KEY", "")
    state: dict[str, Any] = {"graph_agent": None}

    def auth(x_api_key: str = Header(default="")) -> None:
        if api_key and not hmac.compare_digest(x_api_key, api_key):
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "llm": rt.llm.name if rt.llm else None, "tools": len(rt.registry.names()), "memory_entries": rt.memory.count()}

    @app.get("/tools", dependencies=[Depends(auth)])
    def tools() -> list[dict[str, str]]:
        return rt.registry.catalog()

    @app.post("/solve", response_model=Report, dependencies=[Depends(auth)])
    def solve(req: SolveRequest) -> Report:
        if req.orchestrator == "langgraph":
            if state["graph_agent"] is None:
                from .graph import GraphAgent

                state["graph_agent"] = GraphAgent(rt)
            return state["graph_agent"].run(req.problem, req.plan)
        if req.orchestrator != "custom":
            raise HTTPException(status_code=422, detail="orchestrator must be 'custom' or 'langgraph'")
        return run_engine(req.problem, rt, req.plan)

    @app.get("/memory", dependencies=[Depends(auth)])
    def memory() -> dict[str, Any]:
        return {"count": rt.memory.count(), "entries": rt.memory.entries(50)}

    @app.post("/spec/draft", dependencies=[Depends(auth)])
    def spec_draft(req: DraftRequest) -> dict[str, Any]:
        try:
            return specs.draft(req.text).model_dump()
        except SpecError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/spec/{spec_id}/approve", dependencies=[Depends(auth)])
    def spec_approve(spec_id: str, req: ApproveRequest) -> dict[str, Any]:
        try:
            return specs.approve(spec_id, req.edited).model_dump()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown spec id") from exc
        except (SpecError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/spec/{spec_id}/check", dependencies=[Depends(auth)])
    def spec_check(spec_id: str, n: int = 1000, seed: int = 0) -> dict[str, Any]:
        try:
            return specs.check(spec_id, n, seed)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown spec id") from exc

    @app.get("/spec/{spec_id}/code", dependencies=[Depends(auth)])
    def spec_code(spec_id: str) -> dict[str, str]:
        try:
            return {"code": specs.generate(spec_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown spec id") from exc
        except SpecError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app


app = create_app()
