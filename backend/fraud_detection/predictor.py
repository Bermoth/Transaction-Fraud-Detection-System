"""Model loading and prediction logic.

No FastAPI code lives here, so this file can be tested on its own.
"""
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import joblib
import numpy as np
import pandas as pd

try:  # optional: read backend/.env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# The artifacts already live in <project>/models/. We reuse them instead of
# copying them. MODEL_ARTIFACTS_DIR (in .env) can point somewhere else.
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "models"

MODEL_FILE = "xgboost_model.pkl"
COLUMNS_FILE = "feature_columns.pkl"
THRESHOLD_FILE = "threshold.pkl"

FRAUD_CLASS = 1
TARGET_COLUMN = "Class"


class PredictorError(Exception):
    """Base class for errors raised by this module."""


class ModelLoadError(PredictorError):
    """An artifact is missing, unreadable or inconsistent."""


class InvalidFeaturesError(PredictorError):
    """The input features are missing, unexpected, non-numeric or not finite."""


@dataclass(frozen=True)
class PredictionResult:
    risk_score: float
    prediction: str  # "FRAUD" or "LEGITIMATE"
    threshold: float


def resolve_artifacts_dir() -> Path:
    raw = os.getenv("MODEL_ARTIFACTS_DIR")
    if not raw:
        return DEFAULT_ARTIFACTS_DIR
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


class FraudPredictor:
    """Holds the three artifacts in memory and scores single transactions."""

    def __init__(self, artifacts_dir: Optional[Path] = None):
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else resolve_artifacts_dir()
        self._load()

    # ---------------------------------------------------------------- loading
    def _load(self) -> None:
        paths = {
            "model": self.artifacts_dir / MODEL_FILE,
            "columns": self.artifacts_dir / COLUMNS_FILE,
            "threshold": self.artifacts_dir / THRESHOLD_FILE,
        }
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            raise ModelLoadError(f"Missing artifact file(s): {missing}")

        try:
            model = joblib.load(paths["model"])
            columns = joblib.load(paths["columns"])
            threshold = joblib.load(paths["threshold"])
        except Exception as exc:
            raise ModelLoadError(f"Could not read the artifacts: {exc}") from exc

        # --- feature columns: the exact names and order used in training
        if not isinstance(columns, (list, tuple)) or not columns:
            raise ModelLoadError("feature_columns.pkl must be a non-empty list.")
        columns = [str(c) for c in columns]
        if len(set(columns)) != len(columns):
            raise ModelLoadError("feature_columns.pkl contains duplicate names.")
        if TARGET_COLUMN in columns:
            raise ModelLoadError("'Class' (the target) must never be an input feature.")

        # --- threshold: a finite probability
        try:
            threshold = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ModelLoadError("threshold.pkl is not a number.") from exc
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ModelLoadError(f"Threshold {threshold} is not a probability in [0, 1].")

        # --- model: must expose probabilities and agree with the column list
        if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
            raise ModelLoadError("The model has no predict_proba / classes_.")

        # Do not assume probability column 1 is fraud: look it up.
        classes = list(model.classes_)
        if FRAUD_CLASS not in classes:
            raise ModelLoadError(f"Model classes {classes} do not contain class 1 (fraud).")
        fraud_index = classes.index(FRAUD_CLASS)

        names = getattr(model, "feature_names_in_", None)
        if names is not None and [str(n) for n in names] != columns:
            raise ModelLoadError(
                "Feature names stored inside the model differ from feature_columns.pkl."
            )
        n_in = getattr(model, "n_features_in_", None)
        if n_in is not None and int(n_in) != len(columns):
            raise ModelLoadError(
                f"Model expects {n_in} features but feature_columns.pkl has {len(columns)}."
            )

        self.model = model
        self.feature_columns = columns
        self.threshold = threshold
        self._fraud_index = fraud_index
        logger.info(
            "Artifacts loaded from %s (%d features, threshold=%.4f, fraud column=%d)",
            self.artifacts_dir, len(columns), threshold, fraud_index,
        )

    # ------------------------------------------------------------- validation
    def _to_frame(self, features: Mapping[str, Any]) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in features]
        unexpected = [k for k in features if k not in self.feature_columns]
        problems = []
        if missing:
            problems.append(f"missing features: {missing}")
        if unexpected:
            problems.append(f"unexpected features: {unexpected}")

        row = []
        if not missing:
            for col in self.feature_columns:
                value = features[col]
                is_number = isinstance(value, (int, float, np.integer, np.floating))
                if isinstance(value, (bool, np.bool_)) or not is_number:
                    problems.append(f"'{col}' must be a number, got {type(value).__name__}")
                elif not math.isfinite(float(value)):
                    problems.append(f"'{col}' must be finite (no NaN or infinity)")
                else:
                    row.append(float(value))
        if problems:
            raise InvalidFeaturesError("; ".join(problems))

        # Columns are placed in the saved training order, whatever the input order.
        return pd.DataFrame([row], columns=self.feature_columns, dtype="float64")

    # ------------------------------------------------------------- prediction
    def predict(self, features: Mapping[str, Any]) -> PredictionResult:
        frame = self._to_frame(features)
        risk_score = float(self.model.predict_proba(frame)[0, self._fraud_index])
        if not math.isfinite(risk_score) or not 0.0 <= risk_score <= 1.0:
            raise PredictorError(f"The model returned an invalid probability: {risk_score}")
        label = "FRAUD" if risk_score >= self.threshold else "LEGITIMATE"
        return PredictionResult(risk_score=risk_score, prediction=label, threshold=self.threshold)


# One predictor for the whole process, created once at application startup.
_predictor: Optional[FraudPredictor] = None


def load_predictor(artifacts_dir: Optional[Path] = None) -> FraudPredictor:
    global _predictor
    _predictor = FraudPredictor(artifacts_dir)
    return _predictor


def get_predictor() -> FraudPredictor:
    if _predictor is None:
        raise ModelLoadError("The model has not been loaded yet.")
    return _predictor