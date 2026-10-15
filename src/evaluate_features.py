
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


def find_best_threshold(y_true, probabilities):
    """Choose the threshold that maximizes F1 on validation data."""
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    if len(thresholds) == 0:
        return 0.5

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = int(f1_scores.argmax())
    return float(thresholds[best_index])


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load training, validation, and test splits.
    X_train = pd.read_parquet(DATA_DIR / "X_train.parquet")
    X_val = pd.read_parquet(DATA_DIR / "X_val.parquet")
    X_test = pd.read_parquet(DATA_DIR / "X_test.parquet")

    y_train = pd.read_parquet(
        DATA_DIR / "y_train.parquet"
    ).iloc[:, 0]

    y_val = pd.read_parquet(
        DATA_DIR / "y_val.parquet"
    ).iloc[:, 0]

    y_test = pd.read_parquet(
        DATA_DIR / "y_test.parquet"
    ).iloc[:, 0]

    # Load the already-trained XGBoost model and its validation threshold.
    xgb_model = joblib.load(MODEL_DIR / "xgboost_model.pkl")
    xgb_threshold = float(
        joblib.load(MODEL_DIR / "threshold.pkl")
    )

    # Create the Logistic Regression baseline.
    lr_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    )

    print("Training Logistic Regression baseline...")
    lr_model.fit(X_train, y_train)

    # Select Logistic Regression's threshold using validation data only.
    lr_val_probabilities = lr_model.predict_proba(X_val)[:, 1]
    lr_threshold = find_best_threshold(
        y_val, lr_val_probabilities
    )

    print(f"XGBoost validation threshold: {xgb_threshold:.4f}")
    print(f"Logistic Regression validation threshold: {lr_threshold:.4f}")

    # Evaluate both models on the same held-out test set.
    models = {
        "XGBoost": (xgb_model, xgb_threshold),
        "Logistic Regression": (lr_model, lr_threshold),
    }

    rows = []

    for name, (model, model_threshold) in models.items():
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= model_threshold).astype(int)

        rows.append({
            "model": name,
            "threshold": model_threshold,
            "average_precision_pr_auc": float(
                average_precision_score(y_test, probabilities)
            ),
            "roc_auc": float(
                roc_auc_score(y_test, probabilities)
            ),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_test, predictions, zero_division=0)
            ),
        })

    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        RESULTS_DIR / "model_comparison.csv", index=False
    )

    with open(
        RESULTS_DIR / "model_comparison.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(rows, file, indent=4)

    print("\n=== Model Comparison on Test Set ===")
    print(comparison.round(4).to_string(index=False))

    # Save XGBoost feature importance.
    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": xgb_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(
        RESULTS_DIR / "feature_importance.csv",
        index=False,
    )

    # Plot the 15 most important features.
    top_features = importance.head(15).sort_values(
        "importance", ascending=True
    )

    plt.figure(figsize=(10, 7))
    plt.barh(
        top_features["feature"],
        top_features["importance"],
    )
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Top 15 Features - XGBoost Fraud Detection")
    plt.tight_layout()

    chart_path = RESULTS_DIR / "feature_importance.png"
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("\nEvaluation completed successfully.")
    print(f"Comparison CSV: {RESULTS_DIR / 'model_comparison.csv'}")
    print(f"Comparison JSON: {RESULTS_DIR / 'model_comparison.json'}")
    print(f"Feature importance CSV: {RESULTS_DIR / 'feature_importance.csv'}")
    print(f"Feature importance chart: {chart_path}")


if __name__ == "__main__":
    main()
