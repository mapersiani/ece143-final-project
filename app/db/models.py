from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, JSON, Text
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.dialects.postgresql import UUID
import uuid


class Base(DeclarativeBase):
    pass


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    job_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_path = Column(String, nullable=False)
    status = Column(String, default="processing")  # processing | debating | approved | escalated | failed
    analyst_report = Column(JSON)
    debate_log = Column(JSON)
    approved_plan = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)


class ActionHistory(Base):
    __tablename__ = "actions_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), nullable=False)
    segment_id = Column(String, nullable=False)
    action_type = Column(String, nullable=False)
    action_details = Column(JSON)
    customer_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    # pending | executed | completed
    status = Column(String, default="pending")
    outcome = Column(JSON)
    outcome_date = Column(DateTime)


class SegmentProfile(Base):
    __tablename__ = "segment_profiles"

    segment_id = Column(String, primary_key=True)
    description = Column(Text)
    best_channel = Column(String)
    optimal_discount_min = Column(Float)
    optimal_discount_max = Column(Float)
    avg_response_rate = Column(Float)
    total_campaigns = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Constraint(Base):
    __tablename__ = "constraints"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # null means global constraint
    segment_id = Column(String, nullable=True)
    rule = Column(Text, nullable=False)
    confidence = Column(Float, default=0.5)
    source = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    active = Column(Boolean, default=True)
