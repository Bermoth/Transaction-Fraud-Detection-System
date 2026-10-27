"""Pydantic schemas: validation of the request and shape of the response."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# A real, finite JSON number. strict=True refuses strings such as "12.5";
# allow_inf_nan=False refuses NaN and infinity.
Feature = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class PredictionRequest(BaseModel):
    """One transaction: an identifier plus the 30 model features.

    The fields are exactly the columns of feature_columns.pkl.
    'Class' is deliberately absent, and any extra field is rejected.
    """

    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(min_length=1, max_length=64, examples=["txn_1001"])

    Time: Feature
    V1: Feature
    V2: Feature
    V3: Feature
    V4: Feature
    V5: Feature
    V6: Feature
    V7: Feature
    V8: Feature
    V9: Feature
    V10: Feature
    V11: Feature
    V12: Feature
    V13: Feature
    V14: Feature
    V15: Feature
    V16: Feature
    V17: Feature
    V18: Feature
    V19: Feature
    V20: Feature
    V21: Feature
    V22: Feature
    V23: Feature
    V24: Feature
    V25: Feature
    V26: Feature
    V27: Feature
    V28: Feature
    Amount: Feature


class PredictionResponse(BaseModel):
    transaction_id: str
    risk_score: float = Field(ge=0.0, le=1.0, description="Estimated fraud probability.")
    prediction: Literal["FRAUD", "LEGITIMATE"]
    threshold: float = Field(ge=0.0, le=1.0, description="Threshold loaded from threshold.pkl.")
