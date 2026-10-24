from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class InvestigationCase(Base):
    __tablename__ = "investigation_cases"

    __table_args__ = (
        CheckConstraint(
            "status IN ('NEW', 'UNDER_INVESTIGATION', 'CLOSED')",
            name="ck_investigation_status",
        ),
        CheckConstraint(
            "decision IS NULL OR decision IN "
            "('CONFIRMED_FRAUD', 'LEGITIMATE', 'INCONCLUSIVE')",
            name="ck_investigation_decision",
        ),
        CheckConstraint(
            "risk_score IS NULL OR "
            "(risk_score >= 0 AND risk_score <= 1)",
            name="ck_investigation_risk_score",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True
    )
    transaction_id: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True
    )
    risk_score: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    prediction: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="NEW"
    )
    notes: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    decision: Mapped[str | None] = mapped_column(
        String(30), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )