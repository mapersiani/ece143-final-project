from pydantic import BaseModel
from typing import Any
from uuid import UUID
from datetime import datetime


class AnalyzeRequest(BaseModel):
    file_path: str  # path in container (after upload)


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class PipelineResult(BaseModel):
    job_id: str
    status: str
    analyst_report: dict | None = None
    debate_rounds: int = 0
    approved_plan: dict | None = None
    execution_package: dict | None = None
    debate_log: dict | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None


class FeedbackRequest(BaseModel):
    action_id: str
    outcome: dict  # e.g. {"retention_rate": 0.35, "revenue_saved": 5000}


class ExperimentInfo(BaseModel):
    experiment_id: str
    name: str
    runs: list[dict]
