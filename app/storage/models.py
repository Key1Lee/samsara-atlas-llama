from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship

from app.storage.database import Base


class Case(Base):
    __tablename__ = "cases"
    __table_args__ = (Index("uq_cases_idempotency_key", "idempotency_key", unique=True),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), default="pending")
    idempotency_key = Column(String(255), index=True)
    inputs_json = Column(Text)
    artifacts_json = Column(Text)
    outputs_json = Column(Text)

    steps = relationship("Step", back_populates="case")
    case_notes = relationship("CaseNote", back_populates="case")


class Step(Base):
    __tablename__ = "steps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey("cases.id"))
    name = Column(String(255))
    status = Column(String(50), default="pending")
    started_at = Column(DateTime)
    finished_at = Column(DateTime)
    attempts = Column(Integer, default=0)
    output_json = Column(Text)
    error = Column(Text)

    case = relationship("Case", back_populates="steps")


class CaseNote(Base):
    __tablename__ = "case_notes"
    __table_args__ = (Index("uq_case_notes_case_id", "case_id", unique=True),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey("cases.id"))
    employee_id = Column(String(255))
    note = Column(Text)
    # Naive datetime here vs. timezone-aware Step timestamps above - mixed
    # naive/aware audit timestamps (see P015).
    created_at = Column(DateTime, default=datetime.utcnow)

    case = relationship("Case", back_populates="case_notes")


class EmployeeRecord(Base):
    __tablename__ = "employee_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(255), unique=True)
    metadata_json = Column(Text)
    last_activities_json = Column(Text)


class CaseAuditEvent(Base):
    """Case-scoped history, committed with its corresponding business write."""
    __tablename__ = "case_audit_events"

    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False, index=True)
    request_id = Column(String(32), index=True)  # None for trusted non-HTTP calls.
    event_type = Column(String(40), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    outcome = Column(String(16), nullable=False)
    step_name = Column(String(64))
    attempt = Column(Integer)
    error_code = Column(String(32))
    note_id = Column(Integer, ForeignKey("case_notes.id"))
    case = relationship("Case")


class EmployeeDailyRequests(Base):
    __tablename__ = "employee_daily_requests"

    employee_id = Column(String(255), primary_key=True)
    day = Column(String(10), primary_key=True)
    request_count = Column(Integer, nullable=False, default=0)
