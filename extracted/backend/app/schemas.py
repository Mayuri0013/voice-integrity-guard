from __future__ import annotations
from pydantic import BaseModel
from typing import Optional


class CallContext(BaseModel):
    unknown_caller: bool = False
    high_value_transaction: bool = False
    privileged_request: bool = False
    caller_claimed_identity: Optional[str] = None


class RiskResult(BaseModel):
    risk_score: float
    risk_level: str
    heuristic_score: float
    ml_score: Optional[float]
    context_boost: float
    feature_breakdown: dict
    recommendation: str


class AnalyzeResponse(BaseModel):
    session_id: str
    risk_result: RiskResult
    features_extracted: int
