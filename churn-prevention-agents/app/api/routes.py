import uuid
import os
import logging
import traceback
import pandas as pd
import mlflow
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.schemas import JobResponse, PipelineResult, FeedbackRequest
from app.agents.graph import run_pipeline
from app.db.memory import get_db, update_outcome
from app.db.models import PipelineRun
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter()

UPLOAD_DIR = "/tmp/churn_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload", response_model=JobResponse)
async def upload_dataset(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    return JobResponse(job_id="", status="uploaded", message=file_path)


@router.post("/analyze", response_model=JobResponse)
async def analyze(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
):
    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    job_id = str(uuid.uuid4())
    run = PipelineRun(job_id=job_id, dataset_path=file_path, status="processing")
    db.add(run)
    db.commit()

    background_tasks.add_task(_run_pipeline_task, job_id, file_path)
    logger.info(f"[{job_id}] Pipeline queued")
    return JobResponse(job_id=job_id, status="processing", message="Pipeline started")


def _validate_uuid(job_id: str):
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid job_id '{job_id}' — must be a UUID from POST /analyze")


@router.get("/status/{job_id}", response_model=JobResponse)
def get_status(job_id: str, db: Session = Depends(get_db)):
    _validate_uuid(job_id)
    run = db.query(PipelineRun).filter(PipelineRun.job_id == job_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(job_id=job_id, status=run.status, message=f"See /results/{job_id} for full output")


@router.get("/results/{job_id}", response_model=PipelineResult)
def get_results(job_id: str, db: Session = Depends(get_db)):
    _validate_uuid(job_id)
    run = db.query(PipelineRun).filter(PipelineRun.job_id == job_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Job not found")
    return PipelineResult(
        job_id=job_id,
        status=run.status,
        analyst_report=run.analyst_report,
        debate_rounds=len(run.debate_log.get("proposals", [])) if run.debate_log else 0,
        approved_plan=run.approved_plan,
        execution_package=run.debate_log.get("execution_package") if run.debate_log else None,
        debate_log=run.debate_log,
        created_at=run.created_at,
        completed_at=run.completed_at,
    )


@router.post("/feedback")
def submit_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    update_outcome(db, req.action_id, req.outcome)
    return {"status": "ok", "message": "Outcome recorded"}


@router.get("/models")
def list_available_models():
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    models = [
        {"name": m.name, "display_name": m.display_name, "supported_methods": m.supported_generation_methods}
        for m in genai.list_models()
        if "generateContent" in m.supported_generation_methods
    ]
    return {
        "configured_models": {
            "strategist": os.getenv("GEMINI_STRATEGIST_MODEL", "gemini-2.5-flash"),
            "critic": os.getenv("GEMINI_CRITIC_MODEL", "gemini-2.5-pro"),
            "executor": os.getenv("GEMINI_EXECUTOR_MODEL", "gemini-2.0-flash-lite"),
        },
        "available_models": models,
    }


@router.get("/experiments")
def list_experiments():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    result = []
    for exp in experiments:
        runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=5)
        result.append({
            "experiment_id": exp.experiment_id,
            "name": exp.name,
            "runs": [
                {
                    "run_id": r.info.run_id,
                    "status": r.info.status,
                    "metrics": r.data.metrics,
                    "params": r.data.params,
                }
                for r in runs
            ],
        })
    return result


@router.post("/train")
async def train_endpoint(
    file: UploadFile = File(...),
    target_col: str = "churn_risk_score",
):
    from app.ml.train import train as train_model

    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        df = pd.read_csv(file_path)
        result = train_model(df, target_col=target_col)
        return result
    except Exception as e:
        logger.error(f"Training failed:\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))


def _update_run(db: Session, job_id: str, **kwargs):
    # Helper to flush intermediate state to DB so /status reflects real progress
    db.query(PipelineRun).filter(PipelineRun.job_id == job_id).update(kwargs)
    db.commit()


def _run_pipeline_task(job_id: str, file_path: str):
    from app.db.memory import SessionLocal
    db = SessionLocal()
    try:
        logger.info(f"[{job_id}] Starting analyst step")
        _update_run(db, job_id, status="analyzing")

        df = pd.read_csv(file_path)
        result = run_pipeline(df, db, job_id, _progress_cb=lambda stage: (
            logger.info(f"[{job_id}] Stage: {stage}"),
            _update_run(db, job_id, status=stage),
        ))

        logger.info(f"[{job_id}] Pipeline complete — status: {result['status']}")
        _update_run(
            db, job_id,
            status=result["status"],
            analyst_report=result["analyst_report"],
            debate_log=result["debate_log"],
            approved_plan=result.get("approved_plan"),
            completed_at=datetime.utcnow(),
        )
    except Exception:
        # Log full traceback so it shows up in docker logs
        logger.error(f"[{job_id}] Pipeline failed:\n{traceback.format_exc()}")
        _update_run(db, job_id, status="failed", debate_log={"error": traceback.format_exc()})
    finally:
        db.close()
