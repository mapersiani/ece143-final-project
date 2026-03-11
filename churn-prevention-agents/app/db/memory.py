from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from .models import Base, ActionHistory, SegmentProfile, Constraint
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://churn_user:churn_pass@localhost:5432/churn_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Fetch recent actions for given segments (used by Analyst and Critic)
def get_recent_actions(db: Session, segment_ids: list[str], limit: int = 5) -> list[dict]:
    actions = (
        db.query(ActionHistory)
        .filter(ActionHistory.segment_id.in_(segment_ids), ActionHistory.outcome.isnot(None))
        .order_by(ActionHistory.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "segment_id": a.segment_id,
            "action_type": a.action_type,
            "action_details": a.action_details,
            "outcome": a.outcome,
            "created_at": str(a.created_at),
        }
        for a in actions
    ]


# Fetch learned constraints (global + segment-specific)
def get_constraints(db: Session, segment_id: str | None = None) -> list[dict]:
    q = db.query(Constraint).filter(Constraint.active == True)
    if segment_id:
        q = q.filter((Constraint.segment_id == segment_id) | (Constraint.segment_id.is_(None)))
    return [{"rule": c.rule, "confidence": c.confidence, "segment_id": c.segment_id} for c in q.all()]


def get_segment_profile(db: Session, segment_id: str) -> dict | None:
    profile = db.query(SegmentProfile).filter(SegmentProfile.segment_id == segment_id).first()
    if not profile:
        return None
    return {
        "segment_id": profile.segment_id,
        "best_channel": profile.best_channel,
        "optimal_discount_min": profile.optimal_discount_min,
        "optimal_discount_max": profile.optimal_discount_max,
        "avg_response_rate": profile.avg_response_rate,
        "total_campaigns": profile.total_campaigns,
    }


def write_action(db: Session, action: dict) -> str:
    record = ActionHistory(**action)
    db.add(record)
    db.commit()
    return str(record.id)


def update_outcome(db: Session, action_id: str, outcome: dict):
    from datetime import datetime
    db.query(ActionHistory).filter(ActionHistory.id == action_id).update(
        {"outcome": outcome, "outcome_date": datetime.utcnow(), "status": "completed"}
    )
    db.commit()
