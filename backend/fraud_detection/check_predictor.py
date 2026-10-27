"""Quick check of the predictor on real rows from the test set.

Run:  python backend/check_predictor.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # lets Python find the 'backend' package

import pandas as pd

from backend.fraud_detection.predictor import load_predictor

p = load_predictor()
print("columns:", len(p.feature_columns), "| threshold:", p.threshold)

X = pd.read_parquet(ROOT / "data" / "processed" / "X_test.parquet")
y = pd.read_parquet(ROOT / "data" / "processed" / "y_test.parquet").iloc[:, 0]

for label in (0, 1):
    for i in y[y == label].index[:3]:
        r = p.predict(X.loc[i].to_dict())
        print(f"actual={label}  score={r.risk_score:.4f}  -> {r.prediction}")
