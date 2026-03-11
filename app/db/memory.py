from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from .models import Base, ActionHistory, SegmentProfile, Constraint
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://churn_user:churn_pass@localhost:5432/churn_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    """
    Create all database tables defined in the SQLAlchemy models.

    :return: None.
    """
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """
    Provide a SQLAlchemy session to FastAPI dependencies and ensure it is closed.

    :return: A database session yielded to the caller.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_recent_actions(db: Session, segment_ids: list[str], limit: int = 5) -> list[dict]:
    """
    Fetch recent completed actions with outcomes for a set of segments.

    :param db: Database session used to query action history.
    :param segment_ids: Segment identifiers to filter actions by.
    :param limit: Maximum number of recent actions to return.
    :return: A list of action dictionaries ordered from most to least recent.
    """
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


def get_constraints(db: Session, segment_id: str | None = None) -> list[dict]:
    """
    Retrieve active learned constraints, optionally filtered to a specific segment.

    :param db: Database session used to query constraints.
    :param segment_id: Optional segment identifier to include segment-specific and global rules.
    :return: A list of constraint dictionaries describing rules and confidence.
    """
    q = db.query(Constraint).filter(Constraint.active == True)
    if segment_id:
        q = q.filter((Constraint.segment_id == segment_id) | (Constraint.segment_id.is_(None)))
    return [{"rule": c.rule, "confidence": c.confidence, "segment_id": c.segment_id} for c in q.all()]


def get_segment_profile(db: Session, segment_id: str) -> dict | None:
    """
    Look up the aggregate profile statistics for a given segment.

    :param db: Database session used to query segment profiles.
    :param segment_id: Identifier of the segment to retrieve.
    :return: A profile dictionary for the segment, or None if not found.
    """
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
    """
    Persist a new action history record for a planned campaign.

    :param db: Database session used to insert the action.
    :param action: Dictionary of action fields to store.
    :return: The stringified UUID of the created action record.
    """
    record = ActionHistory(**action)
    db.add(record)
    db.commit()
    return str(record.id)


def update_outcome(db: Session, action_id: str, outcome: dict):
    """
    Update an action record with observed outcome metrics and mark it completed.

    :param db: Database session used to update the action.
    :param action_id: Identifier of the action history record to update.
    :param outcome: Outcome metrics payload describing campaign performance.
    :return: None.
    """
    from datetime import datetime
    db.query(ActionHistory).filter(ActionHistory.id == action_id).update(
        {"outcome": outcome, "outcome_date": datetime.utcnow(), "status": "completed"}
    )
    db.commit()
