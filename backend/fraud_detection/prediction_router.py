"""FastAPI router exposing POST /predict."""
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from .prediction_schemas import PredictionRequest, PredictionResponse
from .predictor import (FraudPredictor, InvalidFeaturesError, PredictorError,
                        get_predictor)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Fraud prediction"])


def verify_schema_matches_artifacts(predictor: FraudPredictor) -> None:
    """Startup check: the request schema must list exactly the trained features."""
    schema_fields = set(PredictionRequest.model_fields) - {"transaction_id"}
    trained = set(predictor.feature_columns)
    if schema_fields != trained:
        raise RuntimeError(
            "PredictionRequest does not match feature_columns.pkl. "
            f"Only in schema: {sorted(schema_fields - trained)}; "
            f"only in artifacts: {sorted(trained - schema_fields)}"
        )


def _predictor_dependency() -> FraudPredictor:
    try:
        return get_predictor()
    except PredictorError:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded.")


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest,
            predictor: FraudPredictor = Depends(_predictor_dependency)) -> PredictionResponse:
    """Score one transaction and return its fraud risk score and decision."""
    features = request.model_dump(exclude={"transaction_id"})
    try:
        result = predictor.predict(features)
    except InvalidFeaturesError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception:
        # The full traceback goes to the log; the client gets a generic message.
        logger.exception("Unexpected error while predicting transaction %s", request.transaction_id)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Internal error while computing the prediction.")
    return PredictionResponse(
        transaction_id=request.transaction_id,
        risk_score=result.risk_score,
        prediction=result.prediction,
        threshold=result.threshold,
    )
