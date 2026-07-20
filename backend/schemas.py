"""Pydantic request/response schemas for the Gatekeeper API."""
from pydantic import BaseModel, Field


class EvaluateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000,
                      description="One incremental transcript fragment")


class GateDecision(BaseModel):
    text: str
    decision: str
    label: str
    confidence: float
    p_meaningful: float
    probs: dict[str, float]
    latency_ms: float
    reason: str
    similar_to: str | None = None
    context_used: list[str] = []


class ScenarioTurn(BaseModel):
    text: str
    expected: str


class Scenario(BaseModel):
    name: str
    description: str
    turns: list[ScenarioTurn]
