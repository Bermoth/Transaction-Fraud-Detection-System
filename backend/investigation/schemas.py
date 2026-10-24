
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


CaseStatus = Literal[
    "NEW",
    "UNDER_INVESTIGATION",
    "CLOSED",
]

CaseDecision = Literal[
    "CONFIRMED_FRAUD",
    "LEGITIMATE",
    "INCONCLUSIVE",
]

Prediction = Literal["FRAUD", "LEGITIMATE"]


class CaseCreate(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=100)
    risk_score: float | None = Field(default=None, ge=0, le=1)
    prediction: Prediction | None = None
    notes: str | None = None


class CaseUpdate(BaseModel):
    status: CaseStatus | None = None
    notes: str | None = None
    decision: CaseDecision | None = None


class CaseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    transaction_id: str
    risk_score: float | None
    prediction: str | None
    status: str
    notes: str | None
    decision: str | None
    created_at: datetime
    updated_at: datetime
