"""Pydantic schemas shared by every node, the engine, the API and the evaluation harness."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    id: int
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class Plan(BaseModel):
    problem: str
    steps: list[PlanStep]
    planner: str = "heuristic"


class VerifyResult(BaseModel):
    ok: bool
    method: str
    detail: str = ""


class TraceEvent(BaseModel):
    t: float
    node: str
    step_id: int | None = None
    message: str


class StepResult(BaseModel):
    step_id: int
    tool: str
    domain: str
    args: dict[str, Any] = Field(default_factory=dict)
    value: Any = None
    display: str = ""
    verified: bool = False
    method: str = ""
    detail: str = ""
    attempts: int = 0
    from_memory: bool = False
    error: str | None = None


class Report(BaseModel):
    problem: str
    status: Literal["verified", "partial", "failed"]
    final_answer: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    assumptions: list[str] = Field(default_factory=list)
    steps: list[StepResult] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    planner: str = "heuristic"
    elapsed_s: float = 0.0
